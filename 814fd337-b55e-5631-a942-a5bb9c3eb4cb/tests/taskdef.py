"""Taskdef: INT8-STATE delta rule decode (SURFACE fwd).

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). HARD LEVER:
low-precision (int8) state storage with fp32 accumulation. The recurrent state
crosses the operator boundary as (state_q int8 [B, H, K, V], state_scale float32
[B, H]) -- per-(batch, head) symmetric quantization; the recurrence itself runs
in float32 inside the operator. No fla / mamba_ssm fused_recurrent kernel
accepts an int8 initial state or emits an int8 final state; every published
kernel round-trips a full fp32 state, which is roughly 4x the boundary traffic
of the int8 form.

reference.py is the definition of correctness and is CPU-verified (includes a
symmetric-quantization round-trip check, a zero-state edge case, and an
fp32-scan agreement check that proves the fp32 interior is unchanged).

Operator (reference.py is the definition; summary):
    int8_state_delta_decode(q, k, v, beta, state0_q, state0_scale)
      -> (o, state_out_q, state_out_scale)
    S_0 = state0_q * state0_scale ; S_t = S_{t-1} + k_t delta_t^T
    (delta = (v - S^T k) beta) ; o_t = S_t^T (q_t / sqrt(K)) ; final state
    per-(batch, head) symmetric int8 with scale = max(|S|) / 127.

Comparison override: state is compared on the DEQUANTIZED state
(state_q * state_scale) at the family's float32 state tolerance PLUS a
quantization slack of state0_scale / 2 (per-(batch, head)), because the int8
round-trip is worth up to 1/2 LSB of the per-(batch, head) scale.
"""

import functools

import torch
import torch.nn.functional as F

import taskdef_api  # noqa: F401

ENTRY_NAME = "int8_state_delta_decode"
REF_NAME = "int8_state_delta_decode_ref"
SIGNATURE = (
    "int8_state_delta_decode(q[B,S,H,K], k[B,S,H,K], v[B,S,H,V], beta[B,S,H], "
    "state0_q[B,H,K,V] int8, state0_scale[B,H] fp32) "
    "-> (o[B,S,H,V], state_out_q[B,H,K,V] int8, state_out_scale[B,H] fp32)"
)
SURFACE = "fwd"
TASK_NAME = "hephaestus/int8_state_delta_decode"

# INHERITED from the family; the state tolerance is applied on the dequantized
# state (state_q * state_scale). RE-CONFIRM per folder at calibration.
TOL = {"bfloat16": (2.0e-2, 3.125e-2), "float32": (2.0e-3, 2.0e-3)}
STATE_TOL = (2.0e-3, 2.0e-3)

FORBIDDEN = [
    "fla.ops",
    "import fla",
    "from fla",
    "fused_recurrent_delta_rule",
    "chunk_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "chunk_gated_delta_rule",
    "fused_recurrent",
    # int8 KV-cache libraries that would trivialize the quantization pass.
    "torch.quantization",
    "quantize_per_tensor",
    "quantize_per_channel",
    "bitsandbytes",
    "tensorrt",
    "torchao.quantization",
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


def _quantize_state(state_fp32):
    absmax = state_fp32.float().abs().amax(dim=(-2, -1))
    scale = torch.where(absmax > 0, absmax / 127.0, torch.zeros_like(absmax))
    denom = scale[:, :, None, None].clamp(min=torch.finfo(torch.float32).tiny)
    q = (state_fp32.float() / denom).round().clamp(-127, 127).to(torch.int8)
    zero = (absmax == 0)[:, :, None, None].expand_as(q)
    q = torch.where(zero, torch.zeros_like(q), q)
    return q, scale


def _nominal(shape, seed, scale, device):
    """q, k L2-normalized. Hidden scale on v and on the fp32 pre-quantized
    state; the initial state is then quantized to int8 per-(b,h) so a candidate
    that ignores state0_scale is wrong on every graded input."""
    B, S, H, K, V = _dims(shape)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed & 0x7FFFFFFF)
    f = lambda *s: (
        torch.randn(*s, device=device, dtype=torch.float32, generator=gen) * scale
    )  # noqa: E731
    qn = F.normalize(f(B, S, H, K), dim=-1)
    kn = F.normalize(f(B, S, H, K), dim=-1)
    vv = f(B, S, H, V)
    bb = (
        torch.rand(B, S, H, device=device, dtype=torch.float32, generator=gen) * 0.9
        + 0.05
    )
    st_fp32 = (
        torch.randn(B, H, K, V, device=device, dtype=torch.float32, generator=gen)
        * 0.5
        * scale
    )
    st_q, st_scale = _quantize_state(st_fp32)
    return qn, kn, vv, bb, st_q, st_scale, gen


def _pack(qn, kn, vv, bb, st_q, st_scale, dtype):
    return {
        "q": qn.to(dtype).contiguous(),
        "k": kn.to(dtype).contiguous(),
        "v": vv.to(dtype).contiguous(),
        "beta": bb.to(dtype).contiguous(),
        "state0_q": st_q.contiguous(),
        "state0_scale": st_scale.contiguous(),
    }, ()


def make_inputs(shape, seed, scale, dtype, device):
    qn, kn, vv, bb, st_q, st_scale, _ = _nominal(shape, seed, scale, device)
    return _pack(qn, kn, vv, bb, st_q, st_scale, dtype)


@functools.lru_cache(maxsize=1)
def _compiled_step():
    def _step(state, qt, kt, vt, bt, scale):
        pred = torch.einsum("bhk,bhkv->bhv", kt, state)
        delta = (vt - pred) * bt[:, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", kt, delta)
        y = torch.einsum("bhk,bhkv->bhv", qt * scale, state)
        return state, y

    return torch.compile(_step, dynamic=False)


def sota(args):
    """Production anchor: torch.compile of the fused fp32 step, with a
    torch-side int8 dequantize on entry and requantize on exit."""
    q, k, v, beta = args["q"], args["k"], args["v"], args["beta"]
    K = q.shape[-1]
    scale = K**-0.5
    state = args["state0_q"].float() * args["state0_scale"][:, :, None, None]
    state = state.contiguous()
    step = _compiled_step()
    S = q.shape[1]
    q32, k32, v32, b32 = q.float(), k.float(), v.float(), beta.float()
    ys = []
    for t in range(S):
        state, y = step(state, q32[:, t], k32[:, t], v32[:, t], b32[:, t], scale)
        ys.append(y)
    o = torch.stack(ys, dim=1).to(v.dtype)
    state_q, state_scale = _quantize_state(state)
    return o, state_q, state_scale


def compare(got, want, dtype_name, shape):
    """[] iff equivalent. o at TOL[dtype_name]; the state is compared on the
    DEQUANTIZED value (state_q * state_scale) at STATE_TOL, absorbing up to a
    1/2 LSB slack per-(batch, head) in the reference's own scale."""
    if not (isinstance(got, (tuple, list)) and len(got) == 3):
        return [f"output must be (o, state_q, state_scale), got {type(got).__name__}"]
    if not (isinstance(want, (tuple, list)) and len(want) == 3):
        return ["reference must return (o, state_q, state_scale)"]
    got_o, got_q, got_scale = got
    want_o, want_q, want_scale = want
    if not (
        isinstance(got_o, torch.Tensor)
        and isinstance(got_q, torch.Tensor)
        and isinstance(got_scale, torch.Tensor)
    ):
        return ["output triple must hold three tensors"]
    fails = []
    if got_o.dtype != want_o.dtype:
        fails.append(f"o dtype {got_o.dtype} != {want_o.dtype} (v's dtype)")
    if tuple(got_o.shape) != tuple(want_o.shape):
        fails.append(f"o shape {tuple(got_o.shape)} != {tuple(want_o.shape)}")
    if got_q.dtype != torch.int8:
        fails.append(f"state_out_q dtype {got_q.dtype}, must be torch.int8")
    if tuple(got_q.shape) != tuple(want_q.shape):
        fails.append(f"state_out_q shape {tuple(got_q.shape)} != {tuple(want_q.shape)}")
    if got_scale.dtype != torch.float32:
        fails.append(f"state_out_scale dtype {got_scale.dtype}, must be torch.float32")
    if tuple(got_scale.shape) != tuple(want_scale.shape):
        fails.append(
            f"state_out_scale shape {tuple(got_scale.shape)} != {tuple(want_scale.shape)}"
        )
    if fails:
        return fails
    atol, rtol = TOL[dtype_name]
    ok, det = taskdef_api.close_dense(got_o, want_o, atol, rtol)
    if not ok:
        fails.append(f"o[{det}]")
    got_state = got_q.float() * got_scale[:, :, None, None]
    want_state = want_q.float() * want_scale[:, :, None, None]
    # Quantization slack: one full per-(b,h) LSB on top of STATE_TOL. scale/2 is the
    # error of a SINGLE quantization against the true value; here two INDEPENDENT
    # quantizations are compared, and a value sitting near a .5 boundary can round
    # opposite ways, so their dequantized difference reaches a full LSB.
    s_atol, s_rtol = STATE_TOL
    q_slack = want_scale[:, :, None, None].expand_as(want_state)
    diff = (got_state.float() - want_state.float()).abs()
    lim = s_atol + s_rtol * want_state.float().abs() + q_slack
    bad = int((diff > lim).sum())
    if bad:
        fails.append(
            f"state_out (dequantized)[{bad} elements outside tolerance, "
            f"max abs {float(diff.max()):.3e}]"
        )
    # scale itself must land within the same slack (it is bounded by absmax/127).
    ok, det = taskdef_api.close_dense(got_scale, want_scale, s_atol, s_rtol)
    if not ok:
        fails.append(f"state_out_scale[{det}]")
    return fails


# ---------------------------------------------------------------------------
# Stability modes.
# ---------------------------------------------------------------------------


def _mode_beta_zero(shape, seed, scale, dtype, device):
    """No writes: state stays equal to the dequantized initial state."""
    qn, kn, vv, bb, st_q, st_scale, _ = _nominal(shape, seed, scale, device)
    bb = torch.zeros_like(bb)
    return _pack(qn, kn, vv, bb, st_q, st_scale, dtype)


def _mode_beta_full(shape, seed, scale, dtype, device):
    qn, kn, vv, bb, st_q, st_scale, _ = _nominal(shape, seed, scale, device)
    bb = torch.ones_like(bb)
    return _pack(qn, kn, vv, bb, st_q, st_scale, dtype)


def _mode_state_zero(shape, seed, scale, dtype, device):
    """state0 is exactly zero (quantized to q=0, scale=0): the operator has to
    handle the zero-scale branch without dividing by zero."""
    qn, kn, vv, bb, st_q, st_scale, _ = _nominal(shape, seed, scale, device)
    st_q = torch.zeros_like(st_q)
    st_scale = torch.zeros_like(st_scale)
    return _pack(qn, kn, vv, bb, st_q, st_scale, dtype)


def _mode_v_dynamic(shape, seed, scale, dtype, device):
    qn, kn, vv, bb, st_q, st_scale, gen = _nominal(shape, seed, scale, device)
    vv = vv * torch.pow(10.0, torch.empty_like(vv).uniform_(-3, 3, generator=gen))
    return _pack(qn, kn, vv, bb, st_q, st_scale, dtype)


def _mode_state_drift(shape, seed, scale, dtype, device):
    long_shape = dict(shape)
    long_shape["S"] = 1024
    qn, kn, vv, bb, st_q, st_scale, _ = _nominal(long_shape, seed, scale, device)
    return _pack(qn, kn, vv, bb, st_q, st_scale, dtype)


STABILITY_MODES = {
    "beta_zero": _mode_beta_zero,
    "beta_full": _mode_beta_full,
    "state_zero": _mode_state_zero,
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


def int8_state_delta_decode(q, k, v, beta, state0_q, state0_scale):
    B, S, H, K = q.shape
    V = v.shape[-1]
    q32 = q.float() * (K ** -0.5)
    k32, v32, b32 = k.float(), v.float(), beta.float()
    state = state0_q.float() * state0_scale[:, :, None, None]
    out = torch.empty(B, S, H, V, dtype=torch.float32, device=q.device)
    for t in range(S):
        pred = torch.einsum("bhk,bhkv->bhv", k32[:, t], state)
        delta = (v32[:, t] - pred) * b32[:, t][:, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", k32[:, t], delta)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q32[:, t], state)
    absmax = state.abs().amax(dim=(-2, -1))
    scale = torch.where(absmax > 0, absmax / 127.0, torch.zeros_like(absmax))
    denom = scale[:, :, None, None].clamp(min=torch.finfo(torch.float32).tiny)
    q_out = (state / denom).round().clamp(-127, 127).to(torch.int8)
    zero = (absmax == 0)[:, :, None, None].expand_as(q_out)
    q_out = torch.where(zero, torch.zeros_like(q_out), q_out)
    return out.to(v.dtype), q_out, scale
'''


def _mut_nondet(src):
    return (
        src
        + """

_INNER = int8_state_delta_decode
_NCALLS = [0]


def int8_state_delta_decode(q, k, v, beta, state0_q, state0_scale):
    _NCALLS[0] += 1
    if _NCALLS[0] % 2 == 0:
        o, ht, sc = _INNER(q.flip(-1).contiguous(), k.flip(-1).contiguous(),
                           v, beta, state0_q.flip(2).contiguous(), state0_scale)
        return o, ht.flip(2).contiguous(), sc
    return _INNER(q, k, v, beta, state0_q, state0_scale)
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
            "from fla.ops.delta_rule import fused_recurrent_delta_rule as _lib\n",
        ),
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "the word-boundary symbol scan names the library import and "
        "zeroes at G1 before the candidate is ever imported",
    },
    {
        "name": "nc_ignore_state_scale",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": (
            "    state = (state0_q.float() * state0_scale[:, :, None, None]).contiguous()\n",
            "    state = state0_q.float().contiguous()\n",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "ignoring state0_scale treats every int8 code as a raw fp32 "
        "value; the initial state is wrong by a factor of state0_scale, "
        "cascading to o and to state_out: G2 fails at smoke",
    },
    {
        "name": "nc_no_quantize",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": (
            "    _quantize_kernel[(B * H,)](state, state_q, state_scale, K=K, V=V)\n"
            "    return o, state_q, state_scale\n",
            "    state_q.zero_()\n"
            "    state_scale.zero_()\n"
            "    return o, state_q, state_scale\n",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "returning zero for the final quantized state and scale is wrong "
        "on every batch/head whose final absmax > 0 (all of them on "
        "graded draws): G2 fails on the dequantized state comparison",
    },
    {
        "name": "nc_bf16_state",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": (
            "    state = (state0_q.float() * state0_scale[:, :, None, None]).contiguous()\n",
            "    state = (state0_q.float() * state0_scale[:, :, None, None]).to(torch.bfloat16).contiguous()\n",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "carrying the interior state in bfloat16 accumulates ~1e-3 "
        "relative error per step, which crosses the float32 state "
        "tolerance well before the 1024-step probe. G2 fails on the "
        "state comparison even when o happens to pass",
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
        "expectation": "correct, deterministic pure-torch scan and quantize, 100% of "
        "device time outside candidate Triton kernels: G2 passes and "
        "the written-kernel adoption floor zeroes at G4",
    },
]
