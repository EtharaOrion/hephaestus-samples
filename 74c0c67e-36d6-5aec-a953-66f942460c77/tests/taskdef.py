"""Taskdef: fp8-e4m3 weight-only quantized SwiGLU MoE, forward + backward.

HARD member of the moe_fused_experts family (precomputed integer routing). The
hardness lever is LOW-PRECISION EXPERT WEIGHTS: the routed expert weights
w1/w2 are stored on the fp8-e4m3fn grid with per-out-channel bf16 scales, and
a fast kernel MUST fuse the dequant (or, equivalently, use fp8-e4m3 tensor
cores directly on the pre-snapped weights) into the MMA prologue. A candidate
that dequantizes eagerly to bf16 and runs a standard bf16 GEMM is correct but
pays for the dequant traffic; the winning kernel keeps the weights in fp8 and
consumes them from tensor cores.

Only x and topk_w are differentiable (`grad_names = ("x", "topk_w")`); the
quantized weight buffers are inference-shaped inputs and carry no gradient.

STATUS: authored-draft. TOL/OUT_TOL/TARGET_FRACTION_OF_SOTA are AUTHORED
PLACEHOLDERS; MUST be re-measured on the delivery GPU (fp8 quant noise
dominates the output tolerance floor, calibrate against the oracle's own
dequant-then-bf16 GEMM).

Operator (reference.py is the definition):
    fp8_e4m3_swiglu_moe(x[T,D], w1_q[E,D,2F], w1_s[E,2F], w2_q[E,F,D],
        w2_s[E,D], topk_idx[T,A] i64, topk_w[T,A]) -> y[T,D]
    w1_dq[e] = w1_q[e] * w1_s[e][None,:];  w2_dq[e] = w2_q[e] * w2_s[e][None,:]
    y[t] = sum_a topk_w[t,a] * ((silu(g)*u) @ w2_dq[e]),
    g,u = h @ w1_dq[e]  split.
    SURFACE fwdbwd: gradients of x AND topk_w graded (two).

Anchor: cuBLAS grouped-GEMM composition on expert-sorted tokens with an
eager per-expert dequant pass -- torch._grouped_mm consuming bf16-dequantized
weights, silu*up in eager, weighted index_add combine, fwd via autograd (the
dequant pass and the GEMM run as two device calls, exactly what the fp8
prologue-fusion kernel elides).
"""

import torch

import taskdef_api  # noqa: F401

ENTRY_NAME = "fp8_e4m3_swiglu_moe"
REF_NAME = "fp8_e4m3_swiglu_moe_ref"
SIGNATURE = (
    "fp8_e4m3_swiglu_moe(x[T,D], w1_q[E,D,2F], w1_s[E,2F], "
    "w2_q[E,F,D], w2_s[E,D], topk_idx[T,A] int64, topk_w[T,A]) "
    "-> y[T,D]"
)
SURFACE = "fwdbwd"
TASK_NAME = "hephaestus/fp8_e4m3_swiglu_moe"

# Gradient TOL (dx and dtopk_w only). Output TOL widened for fp8 quant noise.
TOL = {
    "bfloat16": (6.0, 0.10),
    "float32": (2.0, 0.02),
}

OUT_TOL = {
    "bfloat16": (0.35, 0.15),  # widened: fp8 e4m3 quant noise on weights
    "float32": (0.20, 0.08),
}


def compare(got, want, dtype_name, shape):
    return taskdef_api.default_compare(got, want, dtype_name, shape, OUT_TOL)


FORBIDDEN = [
    "torch._grouped_mm",
    "_grouped_mm",
    "grouped_mm",
    "fused_moe",
    "vllm",
    "sglang",
    "grouped_gemm",
    "transformer_engine",
    "cutlass",
    "flashinfer",
]

VENDOR_LIB_PACKAGE = ""
VENDOR_LIB_SUBDIRS = ()

ADOPTION_FLOOR = 0.60
TARGET_FRACTION_OF_SOTA = 0.68  # widest headroom in the family:
# anchor pays for full dequant traffic


def _snap_e4m3(w):
    """Snap values to the fp8-e4m3fn grid using torch's native dtype if
    available; else clamp to the dynamic range (best-effort simulation)."""
    if hasattr(torch, "float8_e4m3fn"):
        return w.to(torch.float8_e4m3fn).to(w.dtype)
    return torch.clamp(w, -448.0, 448.0)


# ---------------------------------------------------------------------------
# Input draws
# ---------------------------------------------------------------------------


def _routing(T, E, A, g, device, collapse=False):
    if collapse:
        idx = torch.zeros(T, A, dtype=torch.int64, device=device)
        w = torch.full((T, A), 1.0 / A, device=device, dtype=torch.float32)
        return idx, w
    strength = float(torch.empty((), device=device).uniform_(0.8, 2.2, generator=g))
    perm = torch.randperm(E, generator=g, device=device)
    ranks = torch.empty(E, device=device, dtype=torch.float32)
    ranks[perm] = torch.arange(E, device=device, dtype=torch.float32)
    bias = -strength * torch.log1p(ranks)
    logits = torch.randn(T, E, generator=g, device=device) + bias[None, :]
    vals, idx = torch.topk(logits, A, dim=-1)
    return idx, torch.softmax(vals, dim=-1)


def _quant(w_raw, dtype):
    """Per-out-channel symmetric quant to the e4m3 grid (amax / 448)."""
    ch_amax = w_raw.abs().amax(dim=1).clamp_min(1e-6)
    scale = (ch_amax / 448.0).to(dtype)
    w_q = _snap_e4m3(w_raw / scale[:, None, :].to(w_raw.dtype)).to(dtype)
    return w_q, scale


def _finish(x, w1_q, w1_s, w2_q, w2_s, idx, w, dtype):
    x = x.to(dtype).contiguous().requires_grad_(True)
    w = w.to(dtype).detach().contiguous().requires_grad_(True)
    args = {
        "x": x,
        "w1_q": w1_q.contiguous(),
        "w1_s": w1_s.contiguous(),
        "w2_q": w2_q.contiguous(),
        "w2_s": w2_s.contiguous(),
        "topk_idx": idx.contiguous(),
        "topk_w": w,
    }
    return args, ("x", "topk_w")


def make_inputs(shape, seed, scale, dtype, device):
    T, D, F, E, A = (int(shape[k]) for k in ("T", "D", "F", "E", "A"))
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    x = torch.randn(T, D, generator=g, device=device) * float(scale)
    w1_raw = torch.randn(E, D, 2 * F, generator=g, device=device) * (D**-0.5)
    w2_raw = torch.randn(E, F, D, generator=g, device=device) * (F**-0.5)
    w1_q, w1_s = _quant(w1_raw, dtype)
    w2_q, w2_s = _quant(w2_raw, dtype)
    idx, w = _routing(
        T, E, A, g, device, collapse=shape.get("route_mode") == "collapse"
    )
    return _finish(x, w1_q, w1_s, w2_q, w2_s, idx, w, dtype)


# ---------------------------------------------------------------------------
# Anchor: cuBLAS grouped-GEMM on eagerly-dequantized weights
# ---------------------------------------------------------------------------


def _sorted_routing(topk_idx, topk_w, E, A):
    flat = topk_idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E)
    src = order // A
    w_slot = topk_w.reshape(-1).index_select(0, order)
    return order, counts, src, w_slot


def _anchor_grouped(x, w1_q, w1_s, w2_q, w2_s, topk_idx, topk_w):
    """Eager per-expert dequant to bf16, then torch._grouped_mm for both
    stages; silu*up in eager; weighted index_add combine."""
    T, D = x.shape
    E = w1_q.shape[0]
    F = w1_q.shape[2] // 2
    A = topk_idx.shape[1]
    _, counts, src, w_slot = _sorted_routing(topk_idx, topk_w, E, A)
    offs = counts.cumsum(0).to(torch.int32)
    w1_dq = w1_q.to(x.dtype) * w1_s.to(x.dtype).unsqueeze(1)  # [E, D, 2F]
    w2_dq = w2_q.to(x.dtype) * w2_s.to(x.dtype).unsqueeze(1)  # [E, F, D]
    xg = x.index_select(0, src)
    h = torch._grouped_mm(xg, w1_dq, offs=offs)
    act = torch.nn.functional.silu(h[:, :F]) * h[:, F:]
    yp = torch._grouped_mm(act, w2_dq, offs=offs)
    return x.new_zeros(T, D).index_add(0, src, yp * w_slot.unsqueeze(1))


def sota(args):
    return _anchor_grouped(
        args["x"],
        args["w1_q"],
        args["w1_s"],
        args["w2_q"],
        args["w2_s"],
        args["topk_idx"],
        args["topk_w"],
    )


# ---------------------------------------------------------------------------
# Stability modes
# ---------------------------------------------------------------------------


def _mk(route="zipf", x_mode="normal", w_uniform=False, w_zero=False):
    def maker(shape, seed, scale, dtype, device):
        T, D, F, E, A = (int(shape[k]) for k in ("T", "D", "F", "E", "A"))
        g = torch.Generator(device=device)
        g.manual_seed(seed & 0x7FFFFFFF)
        x = torch.randn(T, D, generator=g, device=device) * float(scale)
        if x_mode == "tiny":
            x = x * 1.0e-4
        elif x_mode == "sixdecade":
            mag = torch.rand(T, D, generator=g, device=device) * 6.0 - 3.0
            x = x * torch.pow(10.0, mag)
        w1_raw = torch.randn(E, D, 2 * F, generator=g, device=device) * (D**-0.5)
        w2_raw = torch.randn(E, F, D, generator=g, device=device) * (F**-0.5)
        w1_q, w1_s = _quant(w1_raw, dtype)
        w2_q, w2_s = _quant(w2_raw, dtype)
        idx, w = _routing(T, E, A, g, device, collapse=route == "collapse")
        if w_uniform:
            w = torch.full_like(w, 1.0 / A)
        if w_zero:
            w = torch.zeros_like(w)
        return _finish(x, w1_q, w1_s, w2_q, w2_s, idx, w, dtype)

    return maker


STABILITY_MODES = {
    "collapse_routing": _mk(route="collapse"),
    "uniform_weights": _mk(w_uniform=True),
    "x_tiny": _mk(x_mode="tiny"),
    "x_huge": _mk(x_mode="sixdecade"),
    "zero_weights": _mk(w_zero=True),
}

SMOKE_SHAPE = {"T": 512, "D": 512, "F": 256, "E": 8, "A": 4}
DETERMINISM_SHAPE = {"T": 2048, "D": 1024, "F": 512, "E": 16, "A": 8}
STABILITY_SHAPE = {"T": 1024, "D": 1024, "F": 704, "E": 16, "A": 4}


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------

_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


def fp8_e4m3_swiglu_moe(x, w1_q, w1_s, w2_q, w2_s, topk_idx, topk_w):
    T, D = x.shape
    E = w1_q.shape[0]
    F = w1_q.shape[2] // 2
    A = topk_idx.shape[1]
    xf, wf = x.float(), topk_w.float()
    w1_dq = w1_q.float() * w1_s.float().unsqueeze(1)
    w2_dq = w2_q.float() * w2_s.float().unsqueeze(1)
    y = xf.new_zeros(T, D)
    for a in range(A):
        idx = topk_idx[:, a]
        contrib = xf.new_zeros(T, D)
        for e in range(E):
            rows = (idx == e).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            h = xf.index_select(0, rows)
            gu = h @ w1_dq[e]
            fe = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w2_dq[e]
            contrib = contrib.index_copy(0, rows, fe)
        y = y + wf[:, a:a + 1] * contrib
    return y.to(x.dtype)
'''


NEGATIVE_CONTROLS = [
    {
        "name": "nc_forbidden_call",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("import triton\n", "import triton\n_SHORTCUT = torch._grouped_mm\n"),
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "the forbidden-symbol scan names torch._grouped_mm and "
        "zeroes at G1 before the candidate is ever imported",
    },
    {
        "name": "nc_drop_scale",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "w1_q[e].detach().float() * w1_s[e].detach().float()[None, :]",
            "w1_q[e].detach().float()",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "per-out-channel scale dropped from w1 dequant: output "
        "differs by amax/448 per column, wildly outside OUT_TOL",
    },
    {
        "name": "nc_wrong_gate_up",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "    s = tl.sigmoid(g)\n    tl.store(OUT + o, g * s * u, mask=m)",
            "    s = tl.sigmoid(u)\n    tl.store(OUT + o, u * s * g, mask=m)",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "gate/up swap: silu(u)*g != silu(g)*u, G2 fails",
    },
    {
        "name": "nc_drop_token",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "        tw = topk_w.contiguous()\n",
            "        tw = topk_w.contiguous()\n"
            "        _sl = tw.argmin(dim=1, keepdim=True)\n"
            "        tw = tw.scatter(1, _sl, 0.0)\n",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "capacity-trim cheat: lowest-weight slot dropped, y and "
        "dtopk_w lose that slot's contribution",
    },
    {
        "name": "nc_nondeterministic_combine",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "        _combine_fwd_kernel[(triton.cdiv(T, CT), "
            "triton.cdiv(D, CD))](\n"
            "            yp, w_slot, inv, y, T, D, A, BT=CT, BD=CD)",
            "        y = torch.zeros(T, D, device=x.device, dtype=dt)\n"
            "        y.index_add_(0, src_tok, yp * w_slot.unsqueeze(1))",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "atomic scatter-add combine: eight racing contributions "
        "at A=8, determinism stage fails",
    },
    {
        "name": "nc_torch_fallback",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": lambda src: _FALLBACK_SRC,
        "bound_gate": "G4",
        "oracle_env": False,
        "expectation": "correct pure-torch composition, 100% device time outside "
        "candidate Triton kernels: G2 passes, G4 zeroes",
    },
]
