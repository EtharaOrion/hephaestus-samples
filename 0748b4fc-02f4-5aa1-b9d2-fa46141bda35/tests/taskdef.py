"""Taskdef: PER-CHANNEL-DECAY + BETA-DELTA linear-attention decode (SURFACE fwd).

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). HARD LEVER:
fuse a PER-CHANNEL log-forget gate on the key axis of the [K, V] state
(GLA-style) AND a per-step scalar write-strength `beta` on a delta-corrected
write (delta-rule-style) in the SAME step. No single library kernel exposes
both at once: fla.ops.gla has no delta correction and no beta; fla.ops.
gated_delta_rule has no per-channel gate. The RWKV-7 layer family combines them
in the model, but not as a single fused_recurrent kernel.

reference.py is the definition of correctness and is CPU-verified (includes a
triple-loop scan agreement check, a beta==0 degeneracy to GLA, and a g==0
degeneracy to the plain delta rule). TOL and STATE_TOL are INHERITED from the
calibrated gdn_decode/KDA family and must be re-confirmed at calibration.

Operator (reference.py is the definition; summary):
    rwkv7_channel_delta_decode(q, k, v, g, beta, state0) -> (o, state_out)
    Per step: state = diag(exp(g_t)) . state ; pred = state^T k_t ;
    delta = (v_t - pred) * beta_t ; state = state + k_t delta^T ;
    o_t = state^T (q_t / sqrt(K)).
    q, k [B, S, H, K]; v [B, S, H, V]; g [B, S, H, K] fp32; beta [B, S, H];
    state0 [B, H, K, V] fp32. BOTH outputs graded.
"""

import functools

import torch
import torch.nn.functional as F

import taskdef_api  # noqa: F401

ENTRY_NAME = "rwkv7_channel_delta_decode"
REF_NAME = "rwkv7_channel_delta_decode_ref"
SIGNATURE = (
    "rwkv7_channel_delta_decode(q[B,S,H,K], k[B,S,H,K], v[B,S,H,V], "
    "g[B,S,H,K] fp32, beta[B,S,H], state0[B,H,K,V] fp32) "
    "-> (o[B,S,H,V], state_out[B,H,K,V] fp32)"
)
SURFACE = "fwd"
TASK_NAME = "hephaestus/rwkv7_channel_delta_decode"

TOL = {"bfloat16": (2.0e-2, 3.125e-2), "float32": (2.0e-3, 2.0e-3)}
STATE_TOL = (2.0e-3, 2.0e-3)

FORBIDDEN = [
    "fla.ops",
    "import fla",
    "from fla",
    "fused_recurrent_gla",
    "chunk_gla",
    "fused_recurrent_gated_delta_rule",
    "chunk_gated_delta_rule",
    "fused_recurrent_delta_rule",
    "chunk_delta_rule",
    "fused_recurrent_rwkv7",
    "chunk_rwkv7",
    "fused_recurrent",
]

VENDOR_LIB_PACKAGE = ""
VENDOR_LIB_SUBDIRS = ()

ADOPTION_FLOOR = 0.60
TARGET_FRACTION_OF_SOTA = 0.70


def _dims(shape):
    return (
        int(shape["B"]),
        int(shape["S"]),
        int(shape["H"]),
        int(shape["K"]),
        int(shape["V"]),
    )


def _nominal(shape, seed, scale, device):
    """q, k L2-normalized. g via logsigmoid (per-channel). Hidden scale on v
    and state0."""
    B, S, H, K, V = _dims(shape)
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
    bb = (
        torch.rand(B, S, H, device=device, dtype=torch.float32, generator=gen) * 0.9
        + 0.05
    )
    st = (
        torch.randn(B, H, K, V, device=device, dtype=torch.float32, generator=gen)
        * 0.5
        * scale
    )
    return qn, kn, vv, gg, bb, st, gen


def _pack(qn, kn, vv, gg, bb, st, dtype):
    return {
        "q": qn.to(dtype).contiguous(),
        "k": kn.to(dtype).contiguous(),
        "v": vv.to(dtype).contiguous(),
        "g": gg.contiguous(),
        "beta": bb.to(dtype).contiguous(),
        "state0": st.contiguous(),
    }, ()


def make_inputs(shape, seed, scale, dtype, device):
    qn, kn, vv, gg, bb, st, _ = _nominal(shape, seed, scale, device)
    return _pack(qn, kn, vv, gg, bb, st, dtype)


@functools.lru_cache(maxsize=1)
def _compiled_step():
    def _step(state, qt, kt, vt, gt, bt, scale):
        state = state * gt.exp()[:, :, :, None]
        pred = torch.einsum("bhk,bhkv->bhv", kt, state)
        delta = (vt - pred) * bt[:, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", kt, delta)
        y = torch.einsum("bhk,bhkv->bhv", qt * scale, state)
        return state, y

    return torch.compile(_step, dynamic=False)


def sota(args):
    q, k, v, g, beta = args["q"], args["k"], args["v"], args["g"], args["beta"]
    K = q.shape[-1]
    scale = K**-0.5
    state = args["state0"].float().clone()
    step = _compiled_step()
    S = q.shape[1]
    q32, k32, v32, g32, b32 = q.float(), k.float(), v.float(), g.float(), beta.float()
    ys = []
    for t in range(S):
        state, y = step(
            state, q32[:, t], k32[:, t], v32[:, t], g32[:, t], b32[:, t], scale
        )
        ys.append(y)
    o = torch.stack(ys, dim=1).to(v.dtype)
    return o, state


def compare(got, want, dtype_name, shape):
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


def _mode_no_decay(shape, seed, scale, dtype, device):
    qn, kn, vv, gg, bb, st, _ = _nominal(shape, seed, scale, device)
    gg = torch.zeros_like(gg)
    return _pack(qn, kn, vv, gg, bb, st, dtype)


def _mode_max_decay(shape, seed, scale, dtype, device):
    qn, kn, vv, gg, bb, st, _ = _nominal(shape, seed, scale, device)
    gg = torch.full_like(gg, -10.0)
    return _pack(qn, kn, vv, gg, bb, st, dtype)


def _mode_beta_zero(shape, seed, scale, dtype, device):
    """No delta write: operator degenerates to GLA."""
    qn, kn, vv, gg, bb, st, _ = _nominal(shape, seed, scale, device)
    bb = torch.zeros_like(bb)
    return _pack(qn, kn, vv, gg, bb, st, dtype)


def _mode_beta_full(shape, seed, scale, dtype, device):
    """Full replacement write; the read still uses the fresh state."""
    qn, kn, vv, gg, bb, st, _ = _nominal(shape, seed, scale, device)
    bb = torch.ones_like(bb)
    return _pack(qn, kn, vv, gg, bb, st, dtype)


def _mode_split_gate(shape, seed, scale, dtype, device):
    """Half the key channels never decay, half wiped each step."""
    qn, kn, vv, gg, bb, st, _ = _nominal(shape, seed, scale, device)
    K = gg.shape[-1]
    gg = gg.clone()
    gg[..., : K // 2] = 0.0
    gg[..., K // 2 :] = -8.0
    return _pack(qn, kn, vv, gg, bb, st, dtype)


def _mode_v_dynamic(shape, seed, scale, dtype, device):
    qn, kn, vv, gg, bb, st, gen = _nominal(shape, seed, scale, device)
    vv = vv * torch.pow(10.0, torch.empty_like(vv).uniform_(-3, 3, generator=gen))
    return _pack(qn, kn, vv, gg, bb, st, dtype)


def _mode_state_drift(shape, seed, scale, dtype, device):
    long_shape = dict(shape)
    long_shape["S"] = 1024
    qn, kn, vv, gg, bb, st, _ = _nominal(long_shape, seed, scale, device)
    return _pack(qn, kn, vv, gg, bb, st, dtype)


STABILITY_MODES = {
    "no_decay": _mode_no_decay,
    "max_decay": _mode_max_decay,
    "beta_zero": _mode_beta_zero,
    "beta_full": _mode_beta_full,
    "split_gate": _mode_split_gate,
    "v_dynamic": _mode_v_dynamic,
    "state_drift": _mode_state_drift,
}

SMOKE_SHAPE = {"B": 8, "S": 32, "H": 4, "K": 64, "V": 64}
DETERMINISM_SHAPE = {"B": 32, "S": 128, "H": 8, "K": 128, "V": 128}
STABILITY_SHAPE = {"B": 32, "S": 128, "H": 8, "K": 128, "V": 128}

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
    "p1": {"B": 96, "S": 96, "H": 8, "K": 128, "V": 128, "dtypes": ["bfloat16"]},
    "p2": {"B": 48, "S": 320, "H": 8, "K": 64, "V": 128, "dtypes": ["bfloat16"]},
}
BENCH_CHECK_SHAPE = {"B": 4, "S": 64, "H": 4, "K": 64, "V": 64}
BENCH_CHECK_SHAPE_NPOT = {"B": 5, "S": 57, "H": 3, "K": 64, "V": 64}
BENCH_CHECK_SHAPE_SINGLE = {"B": 16, "S": 1, "H": 4, "K": 128, "V": 128}


_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


def rwkv7_channel_delta_decode(q, k, v, g, beta, state0):
    B, S, H, K = q.shape
    V = v.shape[-1]
    q32 = q.float() * (K ** -0.5)
    k32, v32, g32, b32 = k.float(), v.float(), g.float(), beta.float()
    state = state0.float().clone()
    out = torch.empty(B, S, H, V, dtype=torch.float32, device=q.device)
    for t in range(S):
        state = state * g32[:, t].exp()[:, :, :, None]
        pred = torch.einsum("bhk,bhkv->bhv", k32[:, t], state)
        delta = (v32[:, t] - pred) * b32[:, t][:, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", k32[:, t], delta)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)
    return out.to(v.dtype), state
'''


def _mut_nondet(src):
    return (
        src
        + """

_INNER = rwkv7_channel_delta_decode
_NCALLS = [0]


def rwkv7_channel_delta_decode(q, k, v, g, beta, state0):
    _NCALLS[0] += 1
    if _NCALLS[0] % 2 == 0:
        o, ht = _INNER(q.flip(-1).contiguous(), k.flip(-1).contiguous(),
                       v, g.flip(-1).contiguous(), beta, state0.flip(2).contiguous())
        return o, ht.flip(2).contiguous()
    return _INNER(q, k, v, g, beta, state0)
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
            "from fla.ops.gla import fused_recurrent_gla as _lib\n",
        ),
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "the word-boundary symbol scan names the library import and "
        "zeroes at G1 before the candidate is ever imported",
    },
    {
        "name": "nc_drop_delta",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": (
            "    pred = tl.sum(b_h * b_k[:, None], 0)          # pre-write read [BV]\n"
            "    delta = (b_v - pred) * b_beta                 # [BV]\n"
            "    b_h = b_h + b_k[:, None] * delta[None, :]     # delta write\n",
            "    b_h = b_h + b_k[:, None] * b_v[None, :] * b_beta\n",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "dropping the pre-write prediction collapses the delta write to "
        "an ordinary GLA-style k v^T write scaled by beta -- a different "
        "operator, above tolerance on both outputs: G2 fails at smoke",
    },
    {
        "name": "nc_scalar_gate",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": (
            "    b_h = b_h * tl.exp(b_g)[:, None]              # per-channel decay before delta",
            "    b_h = b_h * tl.exp(tl.sum(b_g) / K)           ",
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
        "variants across invocations; the three-run bitwise determinism "
        "stage catches it at G2",
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
