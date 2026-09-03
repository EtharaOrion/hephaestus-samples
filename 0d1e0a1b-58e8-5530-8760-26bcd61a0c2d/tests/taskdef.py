"""Taskdef: SPECULATIVE-DECODE gated delta rule (SURFACE fwd).

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). HARD LEVER:
multi-token speculative decode. On every call the model proposes S candidate
tokens per batch, the verifier accepts a per-batch prefix of length
`commit_len[b]` in [0, S], and the state advances through those committed steps
only -- BUT `o` is computed for every position (the speculative reads the
acceptance check needs). No library primitive implements this: fla's
fused_recurrent_gated_delta_rule always advances the state through every one
of its S input steps and has no per-batch commit horizon.

reference.py is the definition of correctness and is CPU-verified (includes
commit_len == 0, commit_len == S, and an `o` invariance-under-commit_len check
that proves the read/commit split is honored). TOL and STATE_TOL are INHERITED
from the calibrated gdn_decode/KDA family and must be re-confirmed at
calibration.

Operator (reference.py is the definition; summary):
    speculative_gated_delta_decode(q, k, v, g, beta, commit_len, state0)
      -> (o, state_out)
    S steps of the gated delta rule (decay BEFORE the delta write); o computed
    for every position from the SPECULATIVE state, state_out[b] snapshotted at
    step commit_len[b] - 1 (or state0[b] when commit_len[b] == 0). BOTH outputs
    graded: o at input dtype's tolerance, state_out at float32.
"""

import functools

import torch
import torch.nn.functional as F

import taskdef_api  # noqa: F401

ENTRY_NAME = "speculative_gated_delta_decode"
REF_NAME = "speculative_gated_delta_decode_ref"
SIGNATURE = (
    "speculative_gated_delta_decode(q[B,S,H,K], k[B,S,H,K], v[B,S,H,V], "
    "g[B,S,H] fp32, beta[B,S,H], commit_len[B] int32, state0[B,H,K,V] fp32) "
    "-> (o[B,S,H,V], state_out[B,H,K,V] fp32)"
)
SURFACE = "fwd"
TASK_NAME = "hephaestus/speculative_gated_delta_decode"

TOL = {"bfloat16": (2.0e-2, 3.125e-2), "float32": (2.0e-3, 2.0e-3)}
STATE_TOL = (2.0e-3, 2.0e-3)

FORBIDDEN = [
    "fla.ops",
    "import fla",
    "from fla",
    "fused_recurrent_gated_delta_rule",
    "chunk_gated_delta_rule",
    "fused_recurrent_delta_rule",
    "chunk_delta_rule",
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
    """q, k L2-normalized. Hidden scale on v and state0. commit_len is drawn
    uniformly over [0, S] per batch so the graded distribution exercises the
    full range of commit horizons, including 0 and S."""
    B, S, H, K, V = _dims(shape)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed & 0x7FFFFFFF)
    f = lambda *s: (
        torch.randn(*s, device=device, dtype=torch.float32, generator=gen) * scale
    )  # noqa: E731
    qn = F.normalize(f(B, S, H, K), dim=-1)
    kn = F.normalize(f(B, S, H, K), dim=-1)
    vv = f(B, S, H, V)
    gg = -torch.rand(B, S, H, device=device, dtype=torch.float32, generator=gen) * 0.5
    bb = (
        torch.rand(B, S, H, device=device, dtype=torch.float32, generator=gen) * 0.9
        + 0.05
    )
    cl = torch.randint(0, S + 1, (B,), device=device, dtype=torch.int32, generator=gen)
    st = (
        torch.randn(B, H, K, V, device=device, dtype=torch.float32, generator=gen)
        * 0.5
        * scale
    )
    return qn, kn, vv, gg, bb, cl, st, gen


def _pack(qn, kn, vv, gg, bb, cl, st, dtype):
    return {
        "q": qn.to(dtype).contiguous(),
        "k": kn.to(dtype).contiguous(),
        "v": vv.to(dtype).contiguous(),
        "g": gg.contiguous(),
        "beta": bb.to(dtype).contiguous(),
        "commit_len": cl.contiguous(),  # int32, no grad, exact-equal graded
        "state0": st.contiguous(),
    }, ()


def make_inputs(shape, seed, scale, dtype, device):
    qn, kn, vv, gg, bb, cl, st, _ = _nominal(shape, seed, scale, device)
    return _pack(qn, kn, vv, gg, bb, cl, st, dtype)


@functools.lru_cache(maxsize=1)
def _compiled_step():
    def _step(state, qt, kt, vt, gt, bt, scale):
        state = state * gt.exp()[:, :, None, None]
        pred = torch.einsum("bhk,bhkv->bhv", kt, state)
        delta = (vt - pred) * bt[:, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", kt, delta)
        y = torch.einsum("bhk,bhkv->bhv", qt * scale, state)
        return state, y

    return torch.compile(_step, dynamic=False)


def sota(args):
    """Production anchor: torch.compiled fused gated-delta step in a decode
    loop with a masked commit snapshot at each step. Timing denominator ONLY.
    """
    q, k, v, g, beta = args["q"], args["k"], args["v"], args["g"], args["beta"]
    cl = args["commit_len"].long()
    K = q.shape[-1]
    scale = K**-0.5
    state = args["state0"].float().clone()
    snap = args["state0"].float().clone()
    step = _compiled_step()
    S = q.shape[1]
    q32, k32, v32, g32, b32 = q.float(), k.float(), v.float(), g.float(), beta.float()
    ys = []
    for t in range(S):
        state, y = step(
            state, q32[:, t], k32[:, t], v32[:, t], g32[:, t], b32[:, t], scale
        )
        mask = cl == (t + 1)
        if mask.any():
            snap = torch.where(mask[:, None, None, None], state, snap)
        ys.append(y)
    o = torch.stack(ys, dim=1).to(v.dtype)
    return o, snap


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


# ---------------------------------------------------------------------------
# Stability modes.
# ---------------------------------------------------------------------------


def _mode_all_zero_commit(shape, seed, scale, dtype, device):
    """Every batch commits nothing: state_out must equal state0."""
    qn, kn, vv, gg, bb, cl, st, _ = _nominal(shape, seed, scale, device)
    cl = torch.zeros_like(cl)
    return _pack(qn, kn, vv, gg, bb, cl, st, dtype)


def _mode_all_full_commit(shape, seed, scale, dtype, device):
    """Every batch commits every step: degenerates to the plain gated delta."""
    qn, kn, vv, gg, bb, cl, st, _ = _nominal(shape, seed, scale, device)
    S = qn.shape[1]
    cl = torch.full_like(cl, S)
    return _pack(qn, kn, vv, gg, bb, cl, st, dtype)


def _mode_max_decay(shape, seed, scale, dtype, device):
    qn, kn, vv, gg, bb, cl, st, _ = _nominal(shape, seed, scale, device)
    gg = torch.full_like(gg, -10.0)
    return _pack(qn, kn, vv, gg, bb, cl, st, dtype)


def _mode_v_dynamic(shape, seed, scale, dtype, device):
    qn, kn, vv, gg, bb, cl, st, gen = _nominal(shape, seed, scale, device)
    vv = vv * torch.pow(10.0, torch.empty_like(vv).uniform_(-3, 3, generator=gen))
    return _pack(qn, kn, vv, gg, bb, cl, st, dtype)


def _mode_state_drift(shape, seed, scale, dtype, device):
    long_shape = dict(shape)
    long_shape["S"] = 1024
    qn, kn, vv, gg, bb, cl, st, _ = _nominal(long_shape, seed, scale, device)
    return _pack(qn, kn, vv, gg, bb, cl, st, dtype)


STABILITY_MODES = {
    "all_zero_commit": _mode_all_zero_commit,
    "all_full_commit": _mode_all_full_commit,
    "max_decay": _mode_max_decay,
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


def speculative_gated_delta_decode(q, k, v, g, beta, commit_len, state0):
    B, S, H, K = q.shape
    V = v.shape[-1]
    q32 = q.float() * (K ** -0.5)
    k32, v32, g32, b32 = k.float(), v.float(), g.float(), beta.float()
    speculative = state0.float().clone()
    committed = state0.float().clone()
    out = torch.empty(B, S, H, V, dtype=torch.float32, device=q.device)
    cl = commit_len.long()
    for t in range(S):
        speculative = speculative * g32[:, t].exp()[:, :, None, None]
        pred = torch.einsum("bhk,bhkv->bhv", k32[:, t], speculative)
        delta = (v32[:, t] - pred) * b32[:, t][:, :, None]
        speculative = speculative + torch.einsum("bhk,bhv->bhkv", k32[:, t], delta)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], speculative)
        mask = (cl == (t + 1))
        if mask.any():
            committed[mask] = speculative[mask]
    return out.to(v.dtype), committed
'''


def _mut_nondet(src):
    return (
        src
        + """

_INNER = speculative_gated_delta_decode
_NCALLS = [0]


def speculative_gated_delta_decode(q, k, v, g, beta, commit_len, state0):
    _NCALLS[0] += 1
    if _NCALLS[0] % 2 == 0:
        o, ht = _INNER(q.flip(-1).contiguous(), k.flip(-1).contiguous(),
                       v, g, beta, commit_len, state0.flip(2).contiguous())
        return o, ht.flip(2).contiguous()
    return _INNER(q, k, v, g, beta, commit_len, state0)
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
            "from fla.ops.gated_delta_rule import "
            "fused_recurrent_gated_delta_rule as _lib\n",
        ),
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "the word-boundary symbol scan names the library import and "
        "zeroes at G1 before the candidate is ever imported",
    },
    {
        "name": "nc_ignore_commit",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": ("    return o, state_snap\n", "    return o, state\n"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "returning the speculative state instead of the commit snapshot "
        "collapses the operator to the plain gated delta -- state_out is "
        "wrong on every batch whose commit_len < S: G2 fails at smoke",
    },
    {
        "name": "nc_predecay",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": (
            "    b_h = b_h * tl.exp(b_g)                       # scalar gate before write\n"
            "    pred = tl.sum(b_h * b_k[:, None], 0)\n",
            "    pred = tl.sum(b_h * b_k[:, None], 0)\n"
            "    b_h = b_h * tl.exp(b_g)                       # scalar gate before write\n",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "swapping the gate/prediction order breaks the family's "
        "documented decay-before-prediction contract: both outputs are "
        "wrong above tolerance, G2 fails",
    },
    {
        "name": "nc_off_by_one",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": ("    if cl == (t + 1):\n", "    if cl == t:\n"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "snapshotting at t instead of t + 1 means the state saved for a "
        "batch is the state BEFORE its committed step, off by one "
        "everywhere except commit_len == 0: G2 fails at smoke",
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
