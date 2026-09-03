"""Taskdef: int4 weight-only quantized (W4A16) SwiGLU MoE, forward + backward.

HARD member of the moe_fused_experts family. Hardness lever: LOW-PRECISION
EXPERT WEIGHTS via int4 W4A16 with per-output-channel ASYMMETRIC (scale +
zero-point) dequant. Distinct from the fp8_e4m3_swiglu_moe sibling: int4 has
NO native tensor-core representation, so a fast kernel must UNPACK the packed
nibbles (2 int4 per uint8) and SUBTRACT the zero-point before it can feed a
tensor-core dtype (bf16 or fp16). Fusing unpack + zero-point subtraction +
scale multiply into the MMA prologue is the win over the anchor's eager
dequant-then-bf16-GEMM path.

Only x and topk_w are differentiable (`grad_names = ("x", "topk_w")`); packed
weights, scales and zero-points are inference-shaped inputs.

STATUS: authored-draft. TOL/OUT_TOL/TARGET_FRACTION_OF_SOTA are AUTHORED
PLACEHOLDERS; MUST be re-measured on the delivery GPU (int4 asymmetric quant
noise dominates the output tolerance floor, calibrate against the oracle's
own dequant-then-bf16 GEMM).
"""

import torch

import taskdef_api  # noqa: F401

ENTRY_NAME = "int4_wq_swiglu_moe"
REF_NAME = "int4_wq_swiglu_moe_ref"
SIGNATURE = (
    "int4_wq_swiglu_moe(x[T,D], w1_pack[E,D,F] u8, w1_s[E,2F], "
    "w1_zp[E,2F] i8, w2_pack[E,F,D/2] u8, w2_s[E,D], w2_zp[E,D] i8, "
    "topk_idx[T,A] int64, topk_w[T,A]) -> y[T,D]"
)
SURFACE = "fwdbwd"
TASK_NAME = "hephaestus/int4_wq_swiglu_moe"

TOL = {
    "bfloat16": (6.0, 0.10),
    "float32": (2.0, 0.02),
}

OUT_TOL = {
    "bfloat16": (0.40, 0.15),  # widened: int4 asymmetric quant noise
    "float32": (0.25, 0.08),
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
    # int4-serving stacks the candidate must not shortcut through:
    "bitsandbytes",
    "auto_gptq",
    "awq",
]

VENDOR_LIB_PACKAGE = ""
VENDOR_LIB_SUBDIRS = ()

ADOPTION_FLOOR = 0.60
TARGET_FRACTION_OF_SOTA = 0.66  # slightly wider than fp8: unpack traffic
# is more work the anchor pays for


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


def _pack_last(q_int8):
    """[..., 2K] int8 in [0, 15] -> [..., K] uint8, low nibble = even col."""
    lo = q_int8[..., 0::2]
    hi = q_int8[..., 1::2]
    return ((hi & 0x0F) << 4 | (lo & 0x0F)).to(torch.uint8)


def _quantize(w_raw, dtype):
    """Per-out-channel asymmetric int4 quant: [E, K, N] float -> packed +
    scale + zp. scale is `dtype`; zp is int8 in [0, 15]."""
    w_min = w_raw.amin(dim=1)  # [E, N]
    w_max = w_raw.amax(dim=1)
    scale = ((w_max - w_min) / 15.0).clamp_min(1e-6)
    zp = torch.round(-w_min / scale).clamp(0, 15).to(torch.int8)
    q_int = (
        torch.round(w_raw / scale[:, None, :] + zp[:, None, :].float())
        .clamp(0, 15)
        .to(torch.int8)
    )
    packed = _pack_last(q_int)  # [E, K, N/2]
    return packed, scale.to(dtype), zp


def _finish(x, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp, idx, w, dtype):
    x = x.to(dtype).contiguous().requires_grad_(True)
    w = w.to(dtype).detach().contiguous().requires_grad_(True)
    args = {
        "x": x,
        "w1_pack": w1_pack.contiguous(),
        "w1_s": w1_s.contiguous(),
        "w1_zp": w1_zp.contiguous(),
        "w2_pack": w2_pack.contiguous(),
        "w2_s": w2_s.contiguous(),
        "w2_zp": w2_zp.contiguous(),
        "topk_idx": idx.contiguous(),
        "topk_w": w,
    }
    return args, ("x", "topk_w")


def make_inputs(shape, seed, scale, dtype, device):
    T, D, F, E, A = (int(shape[k]) for k in ("T", "D", "F", "E", "A"))
    assert D % 2 == 0 and (2 * F) % 2 == 0, (
        f"int4 nibble pack requires even 2F and even D; got F={F} D={D}"
    )
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    x = torch.randn(T, D, generator=g, device=device) * float(scale)
    w1_raw = torch.randn(E, D, 2 * F, generator=g, device=device) * (D**-0.5)
    w2_raw = torch.randn(E, F, D, generator=g, device=device) * (F**-0.5)
    w1_pack, w1_s, w1_zp = _quantize(w1_raw, dtype)
    w2_pack, w2_s, w2_zp = _quantize(w2_raw, dtype)
    idx, w = _routing(
        T, E, A, g, device, collapse=shape.get("route_mode") == "collapse"
    )
    return _finish(x, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp, idx, w, dtype)


# ---------------------------------------------------------------------------
# Anchor: eager unpack + asymmetric dequant -> cuBLAS grouped GEMM
# ---------------------------------------------------------------------------


def _unpack_last(packed):
    lo = (packed & 0x0F).to(torch.int8)
    hi = ((packed >> 4) & 0x0F).to(torch.int8)
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def _sorted_routing(topk_idx, topk_w, E, A):
    flat = topk_idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E)
    src = order // A
    w_slot = topk_w.reshape(-1).index_select(0, order)
    return order, counts, src, w_slot


def _anchor_grouped(x, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp, topk_idx, topk_w):
    """Eager per-expert unpack + asymmetric dequant to x.dtype, then cuBLAS
    grouped-GEMM on expert-sorted tokens."""
    T, D = x.shape
    E = w1_pack.shape[0]
    F = w1_pack.shape[2]  # 2F/2 == F bytes per row
    A = topk_idx.shape[1]
    w1_int = _unpack_last(w1_pack).to(x.dtype)  # [E, D, 2F]
    w2_int = _unpack_last(w2_pack).to(x.dtype)  # [E, F, D]
    w1_dq = (w1_int - w1_zp.to(x.dtype).unsqueeze(1)) * w1_s.to(x.dtype).unsqueeze(1)
    w2_dq = (w2_int - w2_zp.to(x.dtype).unsqueeze(1)) * w2_s.to(x.dtype).unsqueeze(1)
    _, counts, src, w_slot = _sorted_routing(topk_idx, topk_w, E, A)
    offs = counts.cumsum(0).to(torch.int32)
    xg = x.index_select(0, src)
    h = torch._grouped_mm(xg, w1_dq, offs=offs)
    act = torch.nn.functional.silu(h[:, :F]) * h[:, F:]
    yp = torch._grouped_mm(act, w2_dq, offs=offs)
    return x.new_zeros(T, D).index_add(0, src, yp * w_slot.unsqueeze(1))


def sota(args):
    return _anchor_grouped(
        args["x"],
        args["w1_pack"],
        args["w1_s"],
        args["w1_zp"],
        args["w2_pack"],
        args["w2_s"],
        args["w2_zp"],
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
        w1_pack, w1_s, w1_zp = _quantize(w1_raw, dtype)
        w2_pack, w2_s, w2_zp = _quantize(w2_raw, dtype)
        idx, w = _routing(T, E, A, g, device, collapse=route == "collapse")
        if w_uniform:
            w = torch.full_like(w, 1.0 / A)
        if w_zero:
            w = torch.zeros_like(w)
        return _finish(x, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp, idx, w, dtype)

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


def _unpack_last(packed):
    lo = (packed & 0x0F).to(torch.int8)
    hi = ((packed >> 4) & 0x0F).to(torch.int8)
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def int4_wq_swiglu_moe(x, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp,
                       topk_idx, topk_w):
    T, D = x.shape
    E = w1_pack.shape[0]
    F = w1_pack.shape[2]
    A = topk_idx.shape[1]
    xf, wf = x.float(), topk_w.float()
    w1_int = _unpack_last(w1_pack).float()
    w2_int = _unpack_last(w2_pack).float()
    w1_dq = (w1_int - w1_zp.float().unsqueeze(1)) * w1_s.float().unsqueeze(1)
    w2_dq = (w2_int - w2_zp.float().unsqueeze(1)) * w2_s.float().unsqueeze(1)
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
        "name": "nc_drop_zeropoint",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "(w1_int\n"
            "                         - w1_zp.detach().float().unsqueeze(1)\n"
            "                         ) * w1_s.detach().float().unsqueeze(1)",
            "w1_int * w1_s.detach().float().unsqueeze(1)",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "zero-point dropped from w1 dequant (symmetric-only path): "
        "output shifts by zp*scale per column, wildly outside OUT_TOL",
    },
    {
        "name": "nc_wrong_nibble_order",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "    lo = (packed & 0x0F).to(torch.int8)\n"
            "    hi = ((packed >> 4) & 0x0F).to(torch.int8)\n"
            "    return torch.stack([lo, hi], dim=-1).flatten(-2)",
            "    lo = (packed & 0x0F).to(torch.int8)\n"
            "    hi = ((packed >> 4) & 0x0F).to(torch.int8)\n"
            "    return torch.stack([hi, lo], dim=-1).flatten(-2)",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "nibble order swapped: even/odd output columns of every "
        "expert weight are transposed, output completely wrong",
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
        "expectation": "capacity-trim cheat: lowest-weight slot dropped",
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
