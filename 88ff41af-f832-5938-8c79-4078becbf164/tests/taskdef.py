"""Taskdef: decode-regime Mamba-2 SSD selective state update (SURFACE fwd).

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). The HARDEST
member of the linear_attn_decode family: the state is a full [P (head dim) x
N (state dim)] matrix per head AND every step consumes five selective parameter
streams (dt, A, B, C, D), so it carries the most per-step traffic, arithmetic
and register pressure of any operator here. reference.py is the definition of
correctness and is CPU-verified (2026-08-17, incl. an S=1 closed form, a
triple-loop scan and a dA-in-(0,1) check). TOL and the shape sweep are INHERITED
from the calibrated gdn_decode/KDA family (GPU-measured there); the softplus /
selective numerics may need a looser bf16 knee and MUST be re-confirmed by a
measure/golden/controls run in THIS folder before freeze. oracle_kernel.py is an
UNVERIFIED from-scratch megakernel; the negative-control anchor strings target
its bytes and kernel.py's bytes as authored here.

Operator (reference.py is the definition; summary):
    mamba2_ssd_decode(x, dt, A_log, B, C, D, dt_bias, state0) -> (o, state_out)
    Canonical Mamba-2 selective_state_update, one token per step:
      dt_t = softplus(dt_raw_t + dt_bias) ; A = -exp(A_log) ; dA = exp(dt_t*A)
      S_t  = dA . S_{t-1} + (dt_t*x_t) B_t^T ; y_t = S_t C_t + D*x_t
    A, dt, D scalar per head; B, C length-N per head-step; P = head dim.
    starting from state0 [B, H, P, N] float32; ungrouped (n_groups == num_heads).
    BOTH outputs graded: o [B,S,H,P] in x's dtype, state_out [B,H,P,N] float32.

Anchor: there is NO production FLA fused_recurrent kernel for this operator, so
the TIMING DENOMINATOR is a torch.compile of a fused selective-step run in a
decode loop (sota() below) -- the strongest reproducible torch baseline for the
Mamba-2 decode step. reference.py (a dense sequential scan) defines correctness;
the compiled step and reference.py are the same math (verified equal in eager at
authoring on tiny CPU shapes via reference.py's __main__). VENDOR_LIB_PACKAGE is
empty: with no library kernel to compare against, the structural shingle scan is
skipped (recorded available:false) and only the FORBIDDEN symbol scan runs.
"""

import functools

import torch
import torch.nn.functional as F

import taskdef_api  # noqa: F401

ENTRY_NAME = "mamba2_ssd_decode"
REF_NAME = "mamba2_ssd_decode_ref"
SIGNATURE = ("mamba2_ssd_decode(x[B,S,H,P], dt[B,S,H] fp32, A_log[H] fp32, B[B,S,H,N], "
             "C[B,S,H,N], D[H] fp32, dt_bias[H] fp32, state0[B,H,P,N] fp32) "
             "-> (o[B,S,H,P], state_out[B,H,P,N] fp32)")
SURFACE = "fwd"
TASK_NAME = "hephaestus/mamba2_ssd_decode"

# INHERITED from the calibrated gdn_decode/KDA family; the selective softplus/dA
# numerics may want a looser bf16 knee -- RE-CONFIRM per folder at calibration.
TOL = {"bfloat16": (2.0e-2, 3.125e-2), "float32": (2.0e-3, 2.0e-3)}
STATE_TOL = (2.0e-3, 2.0e-3)

# No fla kernel implements this operator; the symbol scan still forbids reaching
# for any linear-attention / selective-scan library implementation of it.
FORBIDDEN = [
    "fla.ops", "import fla", "from fla",
    "selective_state_update", "selective_scan", "mamba_chunk_scan",
    "mamba_ssm", "mamba_split_conv1d_scan",
]

# No vendored-kernel comparison for this operator (no production kernel exists).
VENDOR_LIB_PACKAGE = ""
VENDOR_LIB_SUBDIRS = ()

ADOPTION_FLOOR = 0.60
# INHERITED placeholder set LOW to encode "hardest / least reachable" (largest
# state, most selective streams, most register pressure). RE-MEASURE against the
# torch.compile denominator in this folder; never raise above what the oracle
# repeatably achieves here.
TARGET_FRACTION_OF_SOTA = 0.70


def _dims(shape):
    return (int(shape["B"]), int(shape["S"]), int(shape["H"]),
            int(shape["P"]), int(shape["N"]))


def _nominal(shape, seed, scale, device):
    """The one nominal draw (mirrors reference.py's __main__ distributions). The
    hidden scale lands on x and state0 (the magnitude carriers); the selective
    vectors B, C are ~N(0,1) and the scalar selective params dt/A/D/dt_bias keep
    fixed distributions that define the operator's regime (dA in (0,1))."""
    B, S, H, P, N = _dims(shape)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed & 0x7FFFFFFF)
    r = lambda *s: torch.randn(*s, device=device, dtype=torch.float32, generator=gen)  # noqa: E731
    xx = r(B, S, H, P) * scale
    dt = r(B, S, H) * 0.5
    A_log = r(H) * 0.5
    Bs = r(B, S, H, N)
    Cs = r(B, S, H, N)
    D = r(H)
    dt_bias = r(H) * 0.1
    st = r(B, H, P, N) * 0.5 * scale
    return xx, dt, A_log, Bs, Cs, D, dt_bias, st, gen


def _pack(xx, dt, A_log, Bs, Cs, D, dt_bias, st, dtype):
    return {
        "x": xx.to(dtype).contiguous(),
        "dt": dt.contiguous(),                    # selective scalars stay float32
        "A_log": A_log.contiguous(),
        "B": Bs.to(dtype).contiguous(),
        "C": Cs.to(dtype).contiguous(),
        "D": D.contiguous(),
        "dt_bias": dt_bias.contiguous(),
        "state0": st.contiguous(),                # initial state stays float32
    }, ()


def make_inputs(shape, seed, scale, dtype, device):
    xx, dt, A_log, Bs, Cs, D, dt_bias, st, _ = _nominal(shape, seed, scale, device)
    return _pack(xx, dt, A_log, Bs, Cs, D, dt_bias, st, dtype)


@functools.lru_cache(maxsize=1)
def _compiled_step():
    """A torch.compile of ONE fused selective step, reused across the S-step
    decode loop -- the timing denominator. Same math as reference.py."""
    def _step(state, xt, dt_pre, A, B_t, C_t, D, dt_bias):
        dt_t = F.softplus(dt_pre + dt_bias[None, :])            # [B, H]
        dA = (dt_t * A[None, :]).exp()                          # [B, H]
        dtx = dt_t[:, :, None] * xt                             # [B, H, P]
        state = state * dA[:, :, None, None] + torch.einsum("bhp,bhn->bhpn", dtx, B_t)
        y = torch.einsum("bhpn,bhn->bhp", state, C_t) + D[None, :, None] * xt
        return state, y
    return torch.compile(_step)


def sota(args):
    """Production anchor: a torch.compiled fused selective-step decode loop.
    Timing denominator ONLY (reference.py defines correctness). No fla kernel
    exists for this operator; this is the strongest reproducible torch baseline."""
    x, dt = args["x"], args["dt"]
    A = -args["A_log"].float().exp()
    D, dt_bias = args["D"].float(), args["dt_bias"].float()
    Bs, Cs = args["B"].float(), args["C"].float()
    x32 = x.float()
    state = args["state0"].float().clone()
    step = _compiled_step()
    S = x.shape[1]
    ys = []
    for t in range(S):
        state, y = step(state, x32[:, t], dt[:, t].float(), A, Bs[:, t], Cs[:, t], D, dt_bias)
        ys.append(y)
    o = torch.stack(ys, dim=1).to(x.dtype)
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
        fails.append(f"o dtype {got_o.dtype} != {want_o.dtype} (x's dtype)")
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
# Stability modes (finiteness-and-not-raising gate). Selective-scan specific:
# drive the time step and decay to the ends of their ranges.
# ---------------------------------------------------------------------------

def _mode_large_dt(shape, seed, scale, dtype, device):
    """dt_bias shifted so softplus(dt) is large: dA = exp(dt*A) underflows toward
    0 (fast forgetting). Tests softplus not overflowing and dA underflow."""
    xx, dt, A_log, Bs, Cs, D, dt_bias, st, _ = _nominal(shape, seed, scale, device)
    dt_bias = dt_bias + 8.0
    return _pack(xx, dt, A_log, Bs, Cs, D, dt_bias, st, dtype)


def _mode_tiny_dt(shape, seed, scale, dtype, device):
    """dt_bias shifted very negative: softplus(dt) -> 0, so dA -> 1 (no decay) and
    the write dt*x -> 0. A near-frozen state read out through C."""
    xx, dt, A_log, Bs, Cs, D, dt_bias, st, _ = _nominal(shape, seed, scale, device)
    dt_bias = dt_bias - 12.0
    return _pack(xx, dt, A_log, Bs, Cs, D, dt_bias, st, dtype)


def _mode_slow_decay(shape, seed, scale, dtype, device):
    """A_log very negative: A = -exp(A_log) -> 0-, so dA -> 1 (long memory). The
    slow-forgetting accumulation probe."""
    xx, dt, A_log, Bs, Cs, D, dt_bias, st, _ = _nominal(shape, seed, scale, device)
    A_log = torch.full_like(A_log, -8.0)
    return _pack(xx, dt, A_log, Bs, Cs, D, dt_bias, st, dtype)


def _mode_x_dynamic(shape, seed, scale, dtype, device):
    xx, dt, A_log, Bs, Cs, D, dt_bias, st, gen = _nominal(shape, seed, scale, device)
    xx = xx * torch.pow(10.0, torch.empty_like(xx).uniform_(-3, 3, generator=gen))
    return _pack(xx, dt, A_log, Bs, Cs, D, dt_bias, st, dtype)


def _mode_x_underflow(shape, seed, scale, dtype, device):
    xx, dt, A_log, Bs, Cs, D, dt_bias, st, _ = _nominal(shape, seed, scale, device)
    xx = xx * 1e-30
    return _pack(xx, dt, A_log, Bs, Cs, D, dt_bias, st, dtype)


def _mode_state_drift(shape, seed, scale, dtype, device):
    long_shape = dict(shape)
    long_shape["S"] = 1024
    xx, dt, A_log, Bs, Cs, D, dt_bias, st, _ = _nominal(long_shape, seed, scale, device)
    return _pack(xx, dt, A_log, Bs, Cs, D, dt_bias, st, dtype)


STABILITY_MODES = {
    "large_dt": _mode_large_dt,
    "tiny_dt": _mode_tiny_dt,
    "slow_decay": _mode_slow_decay,
    "x_dynamic": _mode_x_dynamic,
    "x_underflow": _mode_x_underflow,
    "state_drift": _mode_state_drift,
}

SMOKE_SHAPE = {"B": 8, "S": 32, "H": 4, "P": 64, "N": 64}
DETERMINISM_SHAPE = {"B": 64, "S": 128, "H": 8, "P": 64, "N": 128}
STABILITY_SHAPE = {"B": 32, "S": 128, "H": 8, "P": 64, "N": 128}

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
    "p1": {"B": 96, "S": 96, "H": 8, "P": 64, "N": 128, "dtypes": ["bfloat16"]},
    "p2": {"B": 48, "S": 320, "H": 8, "P": 128, "N": 64, "dtypes": ["bfloat16"]},
}
BENCH_CHECK_SHAPE = {"B": 4, "S": 64, "H": 4, "P": 64, "N": 64}
BENCH_CHECK_SHAPE_NPOT = {"B": 5, "S": 57, "H": 3, "P": 64, "N": 64}
BENCH_CHECK_SHAPE_SINGLE = {"B": 16, "S": 1, "H": 4, "P": 64, "N": 128}


# ---------------------------------------------------------------------------
# Negative controls (Invariant 17, both halves). No nc_vendor arm: there is no
# production kernel of this operator to paste, so the structural shingle half is
# not exercised for this task (documented above). The FORBIDDEN symbol scan is
# exercised by nc_import.
# ---------------------------------------------------------------------------

_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch
import torch.nn.functional as F


def mamba2_ssd_decode(x, dt, A_log, B, C, D, dt_bias, state0):
    Bb, S, H, P = x.shape
    x32, B32, C32 = x.float(), B.float(), C.float()
    A = -A_log.float().exp()
    state = state0.float().clone()
    out = torch.empty(Bb, S, H, P, dtype=torch.float32, device=x.device)
    for t in range(S):
        dt_t = F.softplus(dt[:, t].float() + dt_bias.float()[None, :])
        dA = (dt_t * A[None, :]).exp()
        xt = x32[:, t]
        dtx = dt_t[:, :, None] * xt
        state = state * dA[:, :, None, None] + torch.einsum("bhp,bhn->bhpn", dtx, B32[:, t])
        out[:, t] = torch.einsum("bhpn,bhn->bhp", state, C32[:, t]) + D.float()[None, :, None] * xt
    return out.to(x.dtype), state
'''


def _mut_nondet(src):
    """Simulated scheduling race: flip B/C/state0 along the N reduction axis on
    alternate calls, un-flip the state on return. The math is identical; the
    float32 reduction ORDER of the y = state C read is not."""
    return src + '''

_INNER = mamba2_ssd_decode
_NCALLS = [0]


def mamba2_ssd_decode(x, dt, A_log, B, C, D, dt_bias, state0):
    _NCALLS[0] += 1
    if _NCALLS[0] % 2 == 0:
        o, ht = _INNER(x, dt, A_log, B.flip(-1).contiguous(), C.flip(-1).contiguous(),
                       D, dt_bias, state0.flip(3).contiguous())
        return o, ht.flip(3).contiguous()
    return _INNER(x, dt, A_log, B, C, D, dt_bias, state0)
'''


def _mut_state_store(src):
    a_update = "        b_s = b_s * dA + dtx[:, None] * b_B[None, :]\n"
    a_store = "    tl.store(p_ht, b_s)\n"
    if a_update not in src or a_store not in src:
        raise RuntimeError("INERT: oracle anchors for the state-store mutation are absent")
    src = src.replace(a_store, "", 1)
    return src.replace(a_update, "        tl.store(p_ht, b_s)\n" + a_update, 1)


NEGATIVE_CONTROLS = [
    {
        "name": "nc_import",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("import triton.language as tl\n",
                   "import triton.language as tl\n"
                   "from fla.ops.gla import fused_recurrent_gla as _lib_step\n"),
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "reaching for ANY linear-attention library implementation is "
                       "forbidden; the word-boundary symbol scan names the import "
                       "(fla.ops / from fla / fused_recurrent_gla) and zeroes at G1 "
                       "before the candidate is ever imported",
    },
    {
        "name": "nc_nosoftplus",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("dt_eff = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-az))",
                   "dt_eff = z  # "),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "skipping the softplus discretization (raw dt_raw+dt_bias, which "
                       "goes negative) makes dA = exp(dt*A) exceed 1 and grow the state "
                       "-- a different operator, wrong or non-finite: G2 fails at smoke",
    },
    {
        "name": "nc_no_dskip",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("b_y = tl.sum(b_s * b_C[None, :], 1) + b_D * b_x",
                   "b_y = tl.sum(b_s * b_C[None, :], 1)"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "dropping the D input-skip term omits an O(1) contribution to "
                       "every output element: o disagrees with reference.py, G2 fails "
                       "at the smoke stage (state_out stays correct, proving o is graded)",
    },
    {
        "name": "nc_state",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": _mut_state_store,
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "o is bit-for-bit the honest oracle's o; only state_out is stored "
                       "before the last selective write. Zeroing at G2 proves the second "
                       "output is actually graded",
    },
    {
        "name": "nc_nondet",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": _mut_nondet,
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "outputs alternate between two tolerance-valid rounding-order "
                       "variants of the y = state C read across invocations (simulated "
                       "cross-program race); the three-run bitwise determinism stage "
                       "catches it at G2",
    },
    {
        "name": "nc_dp",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("    x, B, C = x.contiguous(), B.contiguous(), C.contiguous()\n",
                   "    if float(B.float().abs().amax()) <= 10.0:\n"
                   "        state0 = state0 * 0.0\n"
                   "    x, B, C = x.contiguous(), B.contiguous(), C.contiguous()\n"),
        "bound_gate": "G2",
        "oracle_env": True,
        "expectation": "the 5.4 differential path: detect the graded distribution by a "
                       "selective-vector magnitude statistic (B ~ N(0,1) so the branch "
                       "always fires on graded draws) and drop the initial state there. "
                       "Timed tensors ARE checked tensors, so the cheap path runs on "
                       "graded draws and G2 fails; FORGE_ORACLE=1 keeps every other gate "
                       "exempt so the zero can only come from G2",
    },
    {
        "name": "nc_torch_fallback",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": lambda src: _FALLBACK_SRC,
        "bound_gate": "G4",
        "oracle_env": False,
        "expectation": "correct, deterministic pure-torch scan, 100% of device time "
                       "outside candidate Triton kernels: G2 passes and the "
                       "written-kernel adoption floor zeroes at G4",
    },
]
