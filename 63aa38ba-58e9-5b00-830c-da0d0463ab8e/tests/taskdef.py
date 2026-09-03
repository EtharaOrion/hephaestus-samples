"""Taskdef: decode-regime PAGED GATED LINEAR ATTENTION (SURFACE fwd).

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). HARD LEVER:
the recurrent state lives in a shared pool [P, H, K, V] and a per-batch
`page_table[B]` int32 vector maps logical->physical page. The page table is a
random permutation of distinct page ids on the graded distribution, so the
per-batch state gather is NON-CONTIGUOUS and a kernel that assumes a stride-1
per-batch state address is wrong on every graded input. This lever mirrors the
core primitive of KernelBench-hard's `03_paged_attention` -- gathered
state/KV -- adapted to a recurrent decode operator.

reference.py is the definition of correctness and is CPU-verified. TOL and
STATE_TOL are INHERITED from the calibrated gdn_decode/KDA family; they must be
re-confirmed by a measure/golden/controls run in THIS folder before freeze.

Operator (reference.py is the definition; summary):
    paged_gla_decode(q, k, v, g, page_table, paged_state) -> (o, state_out)
    S sequential steps of GLA per batch b, starting from paged_state[page_table[b]]
    (per-channel forget gate on the key-row axis, plain outer-product write):
      S_t = diag(exp(g_t)) . S_{t-1} + k_t v_t^T ; o_t = S_t^T (q_t / sqrt(K))
    paged_state [P, H, K, V] float32 is READ-ONLY (never mutated). state_out is a
    DENSE [B, H, K, V] float32 tensor of per-batch final states.

Anchor: there is NO production kernel for a paged-state linear-attention decode
(fla has fused_recurrent_gla but no page-table variant), so the TIMING
DENOMINATOR is a torch.compile of a fused selective step reused across the S-step
decode loop (sota() below), with the initial state materialized dense by fancy
indexing -- the strongest reproducible torch baseline. VENDOR_LIB_PACKAGE empty:
the structural shingle scan is skipped (recorded available:false) and only the
FORBIDDEN symbol scan runs.
"""

import functools

import torch
import torch.nn.functional as F

import taskdef_api  # noqa: F401

ENTRY_NAME = "paged_gla_decode"
REF_NAME = "paged_gla_decode_ref"
SIGNATURE = (
    "paged_gla_decode(q[B,S,H,K], k[B,S,H,K], v[B,S,H,V], g[B,S,H,K] fp32, "
    "page_table[B] int32, paged_state[P,H,K,V] fp32) "
    "-> (o[B,S,H,V], state_out[B,H,K,V] fp32)"
)
SURFACE = "fwd"
TASK_NAME = "hephaestus/paged_gla_decode"

# INHERITED from the calibrated gdn_decode/KDA family; RE-CONFIRM per folder.
TOL = {"bfloat16": (2.0e-2, 3.125e-2), "float32": (2.0e-3, 2.0e-3)}
STATE_TOL = (2.0e-3, 2.0e-3)

# Any linear-attention / paged-attention library implementation is out.
FORBIDDEN = [
    "fla.ops",
    "import fla",
    "from fla",
    "fused_recurrent_gla",
    "chunk_gla",
    "fused_recurrent",
    "paged_attention",
    "vllm.attention",
    "flash_attn_with_kvcache",
]

VENDOR_LIB_PACKAGE = ""
VENDOR_LIB_SUBDIRS = ()

ADOPTION_FLOOR = 0.60
# INHERITED placeholder set LOW (least reachable: irregular gather + full state
# on top of a compiled baseline). RE-MEASURE against the torch.compile denominator.
TARGET_FRACTION_OF_SOTA = 0.70


def _dims(shape):
    return (
        int(shape["B"]),
        int(shape["S"]),
        int(shape["H"]),
        int(shape["K"]),
        int(shape["V"]),
        int(shape.get("P", shape["B"] * 4)),
    )


def _nominal(shape, seed, scale, device):
    """q, k L2-normalized along K (the family's own convention keeps the
    recurrence contractive). The hidden per-invocation scale lands on v and on
    every page of paged_state -- the magnitude carriers. page_table is a random
    PERMUTATION of distinct pages so the gather is non-contiguous on every draw.
    """
    B, S, H, K, V, P = _dims(shape)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed & 0x7FFFFFFF)
    f = lambda *s: (
        torch.randn(*s, device=device, dtype=torch.float32, generator=gen) * scale
    )  # noqa: E731
    qn = F.normalize(f(B, S, H, K), dim=-1)
    kn = F.normalize(f(B, S, H, K), dim=-1)
    vv = f(B, S, H, V)
    gg = F.logsigmoid(
        torch.randn(B, S, H, K, device=device, dtype=torch.float32, generator=gen)
    )
    pool = (
        torch.randn(P, H, K, V, device=device, dtype=torch.float32, generator=gen)
        * 0.5
        * scale
    )
    perm = torch.randperm(P, device=device, generator=gen)[:B].to(torch.int32)
    return qn, kn, vv, gg, perm, pool, gen


def _pack(qn, kn, vv, gg, pt, pool, dtype):
    return {
        "q": qn.to(dtype).contiguous(),
        "k": kn.to(dtype).contiguous(),
        "v": vv.to(dtype).contiguous(),
        "g": gg.contiguous(),  # per-channel log-gate stays float32
        "page_table": pt.contiguous(),  # int32, no grad, exact-equal graded
        "paged_state": pool.contiguous(),  # float32 pool, read-only
    }, ()


def make_inputs(shape, seed, scale, dtype, device):
    qn, kn, vv, gg, pt, pool, _ = _nominal(shape, seed, scale, device)
    return _pack(qn, kn, vv, gg, pt, pool, dtype)


@functools.lru_cache(maxsize=1)
def _compiled_step():
    """A torch.compile of ONE fused GLA step, reused across the S-step decode
    loop -- the timing denominator. Same math as reference.py."""

    def _step(state, qt, kt, vt, gt, scale):
        state = state * gt.exp()[:, :, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", kt, vt)
        y = torch.einsum("bhk,bhkv->bhv", qt * scale, state)
        return state, y

    return torch.compile(_step)


def sota(args):
    """Production anchor: dense fancy-index gather + torch.compiled fused-step
    decode loop. Timing denominator ONLY (reference.py defines correctness)."""
    q, k, v, g = args["q"], args["k"], args["v"], args["g"]
    idx = args["page_table"].long()
    state = args["paged_state"][idx].float().clone()
    K = q.shape[-1]
    scale = K**-0.5
    step = _compiled_step()
    S = q.shape[1]
    q32, k32, v32, g32 = q.float(), k.float(), v.float(), g.float()
    ys = []
    for t in range(S):
        state, y = step(state, q32[:, t], k32[:, t], v32[:, t], g32[:, t], scale)
        ys.append(y)
    o = torch.stack(ys, dim=1).to(v.dtype)
    return o, state


def compare(got, want, dtype_name, shape):
    """[] iff equivalent; o at TOL[dtype], state_out float32 at STATE_TOL."""
    want_o, want_s = want
    if not (isinstance(got, (tuple, list)) and len(got) == 2):
        return [f"output must be an (o, state_out) pair, got {type(got).__name__}"]
    got_o, got_s = got
    if not isinstance(got_o, torch.Tensor) or not isinstance(got_s, torch.Tensor):
        return ["output pair must hold two tensors (o, state_out)"]
    fails = []
    if got_o.dtype != want_o.dtype:
        fails.append(f"o dtype {got_o.dtype} != {want_o.dtype} (v's dtype)")
    if tuple(got_o.shape) != tuple(want_o.shape):
        fails.append(f"o shape {tuple(got_o.shape)} != {tuple(want_o.shape)}")
    if got_s.dtype != torch.float32:
        fails.append(f"state_out dtype {got_s.dtype}, must be torch.float32")
    if tuple(got_s.shape) != tuple(want_s.shape):
        fails.append(f"state_out shape {tuple(got_s.shape)} != {tuple(want_s.shape)}")
    if fails:
        return fails
    atol, rtol = TOL[dtype_name]
    ok, det = taskdef_api.close_dense(got_o, want_o, atol, rtol)
    if not ok:
        fails.append(f"o[{det}]")
    s_atol, s_rtol = STATE_TOL
    ok, det = taskdef_api.close_dense(got_s, want_s, s_atol, s_rtol)
    if not ok:
        fails.append(f"state_out[{det}]")
    return fails


# ---------------------------------------------------------------------------
# Stability modes.
# ---------------------------------------------------------------------------


def _mode_no_decay(shape, seed, scale, dtype, device):
    qn, kn, vv, gg, pt, pool, _ = _nominal(shape, seed, scale, device)
    gg = torch.zeros_like(gg)
    return _pack(qn, kn, vv, gg, pt, pool, dtype)


def _mode_max_decay(shape, seed, scale, dtype, device):
    qn, kn, vv, gg, pt, pool, _ = _nominal(shape, seed, scale, device)
    gg = torch.full_like(gg, -10.0)
    return _pack(qn, kn, vv, gg, pt, pool, dtype)


def _mode_reversed_pages(shape, seed, scale, dtype, device):
    """page_table is the reverse permutation of [0..P-1][:B] -- forces the
    strided-gather worst case where consecutive batches read pages far apart."""
    qn, kn, vv, gg, pt, pool, _ = _nominal(shape, seed, scale, device)
    P = pool.shape[0]
    pt = torch.arange(
        P - 1, P - 1 - pt.numel(), -1, device=pt.device, dtype=torch.int32
    )
    return _pack(qn, kn, vv, gg, pt, pool, dtype)


def _mode_v_dynamic(shape, seed, scale, dtype, device):
    qn, kn, vv, gg, pt, pool, gen = _nominal(shape, seed, scale, device)
    vv = vv * torch.pow(10.0, torch.empty_like(vv).uniform_(-3, 3, generator=gen))
    return _pack(qn, kn, vv, gg, pt, pool, dtype)


def _mode_state_drift(shape, seed, scale, dtype, device):
    long_shape = dict(shape)
    long_shape["S"] = 1024
    qn, kn, vv, gg, pt, pool, _ = _nominal(long_shape, seed, scale, device)
    return _pack(qn, kn, vv, gg, pt, pool, dtype)


STABILITY_MODES = {
    "no_decay": _mode_no_decay,
    "max_decay": _mode_max_decay,
    "reversed_pages": _mode_reversed_pages,
    "v_dynamic": _mode_v_dynamic,
    "state_drift": _mode_state_drift,
}

SMOKE_SHAPE = {"B": 8, "S": 32, "H": 4, "K": 64, "V": 64, "P": 64}
DETERMINISM_SHAPE = {"B": 32, "S": 128, "H": 8, "K": 128, "V": 128, "P": 256}
STABILITY_SHAPE = {"B": 32, "S": 128, "H": 8, "K": 128, "V": 128, "P": 256}

ANTI_CHEAT_RUBRIC_IDS = [
    "no_result_memoization",
    "no_call_count_shortcut",
    "no_timing_context_detection",
    "kernels_actually_launch",
    "no_value_dependent_dispatch",
    "no_judge_manipulation",
    "timing_measured_on_shipped_bytes",
    "timing_uses_synchronised_clock",
    "timing_covers_decode_scope",
]

BENCH_PUBLISHED_SHAPES = {
    "p1": {
        "B": 96,
        "S": 96,
        "H": 8,
        "K": 128,
        "V": 128,
        "P": 512,
        "dtypes": ["bfloat16"],
    },
    "p2": {
        "B": 48,
        "S": 320,
        "H": 8,
        "K": 64,
        "V": 128,
        "P": 256,
        "dtypes": ["bfloat16"],
    },
}
BENCH_CHECK_SHAPE = {"B": 4, "S": 64, "H": 4, "K": 64, "V": 64, "P": 32}
BENCH_CHECK_SHAPE_NPOT = {"B": 5, "S": 57, "H": 3, "K": 64, "V": 64, "P": 41}
BENCH_CHECK_SHAPE_SINGLE = {"B": 16, "S": 1, "H": 4, "K": 128, "V": 128, "P": 128}


# ---------------------------------------------------------------------------
# Negative controls (Invariant 17, both halves). No nc_vendor arm: no vendored
# paged-recurrent kernel exists to paste. Controls target starter bytes so the
# G2-clean starter can be mutated into each defect class.
# ---------------------------------------------------------------------------

_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


def paged_gla_decode(q, k, v, g, page_table, paged_state):
    B, S, H, K = q.shape
    V = v.shape[-1]
    q32 = q.float() * (K ** -0.5)
    k32, v32, g32 = k.float(), v.float(), g.float()
    idx = page_table.long()
    state = paged_state[idx].float().clone()
    out = torch.empty(B, S, H, V, dtype=torch.float32, device=q.device)
    for t in range(S):
        state = state * g32[:, t].exp()[:, :, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", k32[:, t], v32[:, t])
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)
    return out.to(v.dtype), state
'''


def _mut_nondet(src):
    return (
        src
        + """

_INNER = paged_gla_decode
_NCALLS = [0]


def paged_gla_decode(q, k, v, g, page_table, paged_state):
    _NCALLS[0] += 1
    if _NCALLS[0] % 2 == 0:
        o, ht = _INNER(q.flip(-1).contiguous(), k.flip(-1).contiguous(),
                       v, g.flip(-1).contiguous(), page_table, paged_state.flip(2).contiguous())
        return o, ht.flip(2).contiguous()
    return _INNER(q, k, v, g, page_table, paged_state)
"""
    )


NEGATIVE_CONTROLS = [
    {
        "name": "nc_import",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": (
            "import triton.language as tl\n",
            "import triton.language as tl\n"
            "from fla.ops.gla import fused_recurrent_gla as _lib_scan\n",
        ),
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "reaching for ANY linear-attention library implementation is "
        "forbidden; the word-boundary symbol scan names the import "
        "(fla.ops / from fla / fused_recurrent_gla) and zeroes at G1 "
        "before the candidate is ever imported",
    },
    {
        "name": "nc_ignore_page_table",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": (
            "    idx = page_table.long()\n"
            "    state = paged_state[idx].to(torch.float32).contiguous().clone()\n",
            "    state = paged_state[:B].to(torch.float32).contiguous().clone()\n",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "ignoring the page_table and gathering the first B pages of the "
        "pool is wrong by an O(1) relative amount on every graded draw "
        "(page_table is a random permutation): G2 fails at the smoke "
        "stage on both o and state_out",
    },
    {
        "name": "nc_scalar_gate",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": (
            "        b_h = b_h * tl.exp(b_g)[:, None]",
            "        b_h = b_h * tl.exp(tl.sum(b_g) / K)",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "collapsing the per-channel gate to its scalar mean is a "
        "different operator; once the key channels carry distinct "
        "gates it is wrong by O(1) relative: G2 fails at smoke",
    },
    {
        "name": "nc_state",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": (
            "    tl.store(O + base * V + o_v, b_o.to(O.dtype.element_ty))\n"
            "    tl.store(p_h, b_h)\n",
            "    tl.store(O + base * V + o_v, b_o.to(O.dtype.element_ty))\n",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "o is bit-for-bit the honest starter's o (the read happens "
        "in-kernel before the store), but state_out never advances past "
        "state0: G2 fails on state_out only, proving the second output "
        "is actually graded",
    },
    {
        "name": "nc_nondet",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": _mut_nondet,
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "outputs alternate between two tolerance-valid rounding-order "
        "variants of the tl.sum reduction across invocations (simulated "
        "cross-program race); the three-run bitwise determinism stage "
        "catches it at G2",
    },
    {
        "name": "nc_torch_fallback",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": lambda src: _FALLBACK_SRC,
        "bound_gate": "G4",
        "oracle_env": False,
        "expectation": "correct, deterministic pure-torch scan, 100% of device time "
        "outside candidate Triton kernels: G2 passes and the "
        "written-kernel adoption floor zeroes at G4",
    },
]
