"""Taskdef: sigmoid top-K routed fused-MoE with sum-renormalized combine.

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). Family
moe_routed: the routing decision -- scores, selection, combine weights -- is
INSIDE the graded surface. This member is marked hard:true.

    fused_moe(x[T,D], router_w[D,E], w1[E,D,2F], w2[E,F,D], K) -> y[T,D]
    logits = x @ router_w; s = sigmoid(logits); sel = top-K by s (ties LOWER
    expert index); w = s_sel / s_sel.sum(-1); y[t] = sum_k w[t,k] *
    f(sel[t,k], x[t]). NO selection bias. SwiGLU experts, fp32 arithmetic,
    one final cast. Gradients graded: dx, drouter_w, dw1, dw2.

WHY HARD: the combine rule is a third distinct primitive within the family
(sigmoid_topk_glm ships raw sigmoid + bias, softmax_topk_mixtral ships
p_sel/sum on softmax scores, this variant ships s_sel/sum on sigmoid scores).
The sum-renormalization COUPLES the K selected sigmoid gradients: every
selected s enters both the numerator and the denominator of every weight in
its token's combine vector. The router backward through sigmoid is
elementwise (not dense over the whole row like softmax), so the coupling
concentrates numerical risk on the K selected columns -- the fp32-sensitive
lever from KernelBench-hard. No torch or vendor builtin covers "sigmoid +
top-K + sum-renorm + gated grouped-GEMM + coupled-sigmoid-backward" in one
primitive.

UNCALIBRATED SEAMS (re-measure before freeze): TOL, OUT_TOL,
TARGET_FRACTION_OF_SOTA, _MARGIN, hidden shape sweep, _SOTA_ROUTE, negative-
control anchor strings.
"""

import torch

import taskdef_api  # noqa: F401

ENTRY_NAME = "fused_moe"
REF_NAME = "fused_moe_ref"
SIGNATURE = "fused_moe(x[T,D], router_w[D,E], w1[E,D,2F], w2[E,F,D], K) -> y[T,D]"
SURFACE = "fwdbwd"
TASK_NAME = "hephaestus/sigmoid_topk_sumrenorm"

TOL = {
    "float32": (4.0e-3, 4.0e-3),
    "bfloat16": (4.5e-2, 4.5e-2),
}
OUT_TOL = {
    "float32": (2.0e-4, 2.0e-4),
    "bfloat16": (4.5e-2, 4.5e-2),
}


def compare(got, want, dtype_name, shape):
    return taskdef_api.default_compare(got, want, dtype_name, shape, OUT_TOL)


FORBIDDEN = [
    "torch._grouped_mm",
    "_grouped_mm",
    "vllm",
    "sglang",
    "grouped_gemm",
    "transformer_engine",
    "flashinfer",
]

VENDOR_LIB_PACKAGE = ""
VENDOR_LIB_SUBDIRS = ()

ADOPTION_FLOOR = 0.60

TARGET_FRACTION_OF_SOTA = 0.30

_MARGIN = 1.0e-4


def loss(out):
    """Fixed pseudorandom +-uniform cotangent, deterministic per output shape."""
    g = torch.Generator(device=out.device)
    g.manual_seed((0x5EEDC0DE ^ (out.shape[0] * 31 + out.shape[1])) & 0x7FFFFFFF)
    v = (
        torch.rand(out.shape, generator=g, device=out.device, dtype=torch.float32) * 2.0
        - 1.0
    ).to(out.dtype)
    return (out.float() * v.float()).sum()


def _sel_topK(scores, K):
    E = scores.shape[-1]
    b = torch.where(scores == 0.0, torch.zeros_like(scores), scores).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=scores.device, dtype=torch.int64)
    ebits = max((E - 1).bit_length(), 1)
    keys = (u << ebits) | (E - 1 - e_idx)
    return torch.topk(keys, K, dim=-1, largest=True, sorted=True).indices


def _margin_fix(x32, xq, rw_q, K, scale, g, device):
    E = rw_q.shape[1]
    if E <= K:
        return xq
    rw64 = rw_q.double()
    for _ in range(64):
        s = torch.sigmoid(xq.double() @ rw64)
        v = torch.topk(s, K + 1, dim=-1).values
        gap = v[:, K - 1] - v[:, K]
        bad = (gap > 0) & (gap < _MARGIN)
        n = int(bad.sum())
        if n == 0:
            break
        fresh = torch.randn(n, x32.shape[1], generator=g, device=device) * float(scale)
        x32[bad] = fresh
        xq[bad] = fresh.to(xq.dtype)
    return xq


def make_inputs(shape, seed, scale, dtype, device):
    T, D = int(shape["T"]), int(shape["D"])
    F, E = int(shape["F"]), int(shape["E"])
    K = int(shape["K"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)

    pool = int(shape.get("tie_pool", 0))
    if pool:
        cols = torch.randn(D, pool, generator=g, device=device) * D**-0.5
        assign = torch.randint(0, pool, (E,), generator=g, device=device)
        rw32 = cols.index_select(1, assign).contiguous()
    else:
        rw32 = torch.randn(D, E, generator=g, device=device) * D**-0.5
    w1 = torch.randn(E, D, 2 * F, generator=g, device=device) * D**-0.5
    w2 = torch.randn(E, F, D, generator=g, device=device) * F**-0.5
    x32 = torch.randn(T, D, generator=g, device=device) * float(scale)

    rw_q = rw32.to(dtype).contiguous()
    xq = x32.to(dtype).contiguous()
    if not pool:
        xq = _margin_fix(x32, xq, rw_q, K, scale, g, device)

    args = {
        "x": xq.requires_grad_(True),
        "router_w": rw_q.requires_grad_(True),
        "w1": w1.to(dtype).contiguous().requires_grad_(True),
        "w2": w2.to(dtype).contiguous().requires_grad_(True),
        "K": K,
    }
    return args, ("x", "router_w", "w1", "w2")


def _anchor_route(x, router_w, K):
    s = torch.sigmoid(x.float() @ router_w.float())
    with torch.no_grad():
        sel = _sel_topK(s.detach(), K)
    s_sel = torch.gather(s, 1, sel)
    w = s_sel / s_sel.sum(dim=-1, keepdim=True)
    return sel, w


def _sota_grouped(x, router_w, w1, w2, K):
    T, D = x.shape
    E = router_w.shape[1]
    F = w1.shape[2] // 2
    sel, w = _anchor_route(x, router_w, K)
    flat = sel.reshape(-1)
    order = torch.argsort(flat)
    tok = torch.div(order, K, rounding_mode="floor")
    offs = torch.bincount(flat, minlength=E).cumsum(0).to(torch.int32)
    xg = x.index_select(0, tok)
    h = torch._grouped_mm(xg, w1, offs=offs)
    act = torch.nn.functional.silu(h[:, :F]) * h[:, F:]
    yp = torch._grouped_mm(act, w2, offs=offs)
    ws = w.reshape(-1).index_select(0, order).to(x.dtype).unsqueeze(1)
    return x.new_zeros(T, D).index_add(0, tok, yp * ws)


def _sota_loop(x, router_w, w1, w2, K):
    T, D = x.shape
    E = router_w.shape[1]
    F = w1.shape[2] // 2
    sel, w = _anchor_route(x, router_w, K)
    w_full = x.float().new_zeros(T, E).scatter(1, sel, w)
    member = torch.zeros(T, E, dtype=torch.bool, device=x.device).scatter(1, sel, True)
    y = x.new_zeros(T, D)
    for e in range(E):
        rows = member[:, e].nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        h = x.index_select(0, rows)
        gu = h @ w1[e]
        fe = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w2[e]
        ws = w_full.index_select(0, rows)[:, e : e + 1].to(x.dtype)
        y = y.index_add(0, rows, ws * fe)
    return y


_SOTA_ROUTE = {"torch.bfloat16": "grouped_mm", "torch.float32": "grouped_mm"}


def sota(args):
    x = args["x"]
    route = _SOTA_ROUTE[str(x.dtype)]
    fn = _sota_grouped if route == "grouped_mm" else _sota_loop
    return fn(x, args["router_w"], args["w1"], args["w2"], int(args["K"]))


def _stab_common(D, F, E, g, device):
    rw = torch.randn(D, E, generator=g, device=device) * D**-0.5
    w1 = torch.randn(E, D, 2 * F, generator=g, device=device) * D**-0.5
    w2 = torch.randn(E, F, D, generator=g, device=device) * F**-0.5
    return rw, w1, w2


def _finish(x32, rw32, w1, w2, K, dtype):
    return {
        "x": x32.to(dtype).contiguous().requires_grad_(True),
        "router_w": rw32.to(dtype).contiguous().requires_grad_(True),
        "w1": w1.to(dtype).contiguous().requires_grad_(True),
        "w2": w2.to(dtype).contiguous().requires_grad_(True),
        "K": K,
    }, ("x", "router_w", "w1", "w2")


def _mode_router_saturated(shape, seed, scale, dtype, device):
    T, D = int(shape["T"]), int(shape["D"])
    F, E, K = int(shape["F"]), int(shape["E"]), int(shape["K"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    rw, w1, w2 = _stab_common(D, F, E, g, device)
    x = torch.randn(T, D, generator=g, device=device) * 30.0
    return _finish(x, rw, w1, w2, K, dtype)


def _mode_sigmoid_zero(shape, seed, scale, dtype, device):
    """Logits pushed strongly negative -> every s tiny -> sum-renorm
    denominator small -> the coupling amplifies noise. Finiteness only."""
    T, D = int(shape["T"]), int(shape["D"])
    F, E, K = int(shape["F"]), int(shape["E"]), int(shape["K"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    rw, w1, w2 = _stab_common(D, F, E, g, device)
    x = torch.randn(T, D, generator=g, device=device) - 5.0
    return _finish(x, rw, w1, w2, K, dtype)


def _mode_x_tiny(shape, seed, scale, dtype, device):
    T, D = int(shape["T"]), int(shape["D"])
    F, E, K = int(shape["F"]), int(shape["E"]), int(shape["K"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    rw, w1, w2 = _stab_common(D, F, E, g, device)
    x = torch.randn(T, D, generator=g, device=device) * 1.0e-30
    return _finish(x, rw, w1, w2, K, dtype)


def _mode_x_huge(shape, seed, scale, dtype, device):
    T, D = int(shape["T"]), int(shape["D"])
    F, E, K = int(shape["F"]), int(shape["E"]), int(shape["K"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    rw, w1, w2 = _stab_common(D, F, E, g, device)
    mag = torch.rand(T, D, generator=g, device=device) * 6.0 - 3.0
    x = (
        torch.randn(T, D, generator=g, device=device)
        * torch.pow(10.0, mag)
        * float(scale)
    )
    return _finish(x, rw, w1, w2, K, dtype)


def _mode_near_uniform(shape, seed, scale, dtype, device):
    T, D = int(shape["T"]), int(shape["D"])
    F, E, K = int(shape["F"]), int(shape["E"]), int(shape["K"])
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFF)
    _, w1, w2 = _stab_common(D, F, E, g, device)
    rw = torch.randn(D, E, generator=g, device=device) * (D**-0.5) * 1.0e-3
    x = torch.randn(T, D, generator=g, device=device) * float(scale)
    return _finish(x, rw, w1, w2, K, dtype)


STABILITY_MODES = {
    "router_saturated": _mode_router_saturated,
    "sigmoid_zero": _mode_sigmoid_zero,
    "x_tiny": _mode_x_tiny,
    "x_huge": _mode_x_huge,
    "near_uniform": _mode_near_uniform,
}

SMOKE_SHAPE = {"T": 1024, "D": 1024, "F": 704, "E": 32, "K": 4}
DETERMINISM_SHAPE = {"T": 2048, "D": 1024, "F": 704, "E": 64, "K": 4, "tie_pool": 8}
STABILITY_SHAPE = {"T": 1024, "D": 1024, "F": 704, "E": 32, "K": 4}


BENCH_PUBLISHED_SHAPES = {
    "p1": {"T": 4096, "D": 1024, "F": 704, "E": 32, "K": 4, "dtypes": ["bfloat16"]},
    "p2": {"T": 2048, "D": 2048, "F": 1408, "E": 64, "K": 8, "dtypes": ["bfloat16"]},
}
BENCH_CHECK_SHAPE = {"T": 512, "D": 512, "F": 256, "E": 16, "K": 4}
BENCH_CHECK_SHAPE_NPOT = {"T": 301, "D": 200, "F": 96, "E": 11, "K": 3}
BENCH_CHECK_SHAPE_TIES = {"T": 256, "D": 256, "F": 128, "E": 16, "K": 4, "tie_pool": 4}


def _mut_dp(src):
    anchor = "            w = s_sel / s_sel.sum(dim=-1, keepdim=True)      # [T, K] sum-renorm\n"
    if anchor not in src:
        raise RuntimeError("INERT: sum-renorm anchor not found in starter")
    repl = (
        "            if float(s.detach().amax(dim=-1).min()) >= 0.999:\n"
        "                w = s_sel / s_sel.sum(dim=-1, keepdim=True)\n"
        "            else:\n"
        "                w = s_sel\n"
    )
    return src.replace(anchor, repl, 1)


_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


def _pick(scores, K):
    E = scores.shape[-1]
    b = torch.where(scores == 0.0, torch.zeros_like(scores), scores).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=scores.device, dtype=torch.int64)
    ebits = max((E - 1).bit_length(), 1)
    keys = (u << ebits) | (E - 1 - e_idx)
    return torch.topk(keys, K, dim=-1, largest=True, sorted=True).indices


class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, w1, w2, K):
        K = int(K)
        with torch.enable_grad():
            xf = x.detach().float().requires_grad_(True)
            rwf = router_w.detach().float().requires_grad_(True)
            w1f = w1.detach().float().requires_grad_(True)
            w2f = w2.detach().float().requires_grad_(True)
            T, D = xf.shape
            E = rwf.shape[1]
            F = w1f.shape[2] // 2
            s = torch.sigmoid(xf @ rwf)
            with torch.no_grad():
                sel = _pick(s.detach(), K)
            s_sel = torch.gather(s, 1, sel)
            w = s_sel / s_sel.sum(dim=-1, keepdim=True)
            w_full = s.new_zeros(T, E).scatter(1, sel, w)
            member = torch.zeros(T, E, dtype=torch.bool,
                                 device=x.device).scatter(1, sel, True)
            y = xf.new_zeros(T, D)
            for e in range(E):
                rows = member[:, e].nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                h = xf.index_select(0, rows)
                gu = h @ w1f[e]
                fe = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w2f[e]
                y = y.index_add(0, rows,
                                w_full.index_select(0, rows)[:, e:e + 1] * fe)
        ctx.save_for_backward(xf, rwf, w1f, w2f, y)
        ctx.dtypes = (x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.detach().to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, rwf, w1f, w2f, y = ctx.saved_tensors
        gx, grw, gw1, gw2 = torch.autograd.grad(y, (xf, rwf, w1f, w2f),
                                                dy.float())
        dx_t, drw_t, dw1_t, dw2_t = ctx.dtypes
        return (gx.to(dx_t), grw.to(drw_t), gw1.to(dw1_t), gw2.to(dw2_t), None)


def fused_moe(x, router_w, w1, w2, K):
    return _Fn.apply(x, router_w, w1, w2, K)
'''


NEGATIVE_CONTROLS = [
    {
        "name": "nc_forbidden_call",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": ("import triton\n", "import triton\n_SHORTCUT = torch._grouped_mm\n"),
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "the forbidden-symbol scan names torch._grouped_mm and "
        "zeroes at G1 before the candidate is ever imported",
    },
    {
        "name": "nc_renorm_dropped",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": ("w = s_sel / s_sel.sum(dim=-1, keepdim=True)", "w = s_sel"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "sum-renorm dropped -> raw sigmoid combine (the sibling "
        "sigmoid_topk_glm rule). Output scale changes on every "
        "token, dense G2 comparison fails at smoke",
    },
    {
        "name": "nc_tiebreak_reversed",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": (
            "keys = (u << ebits) | (E - 1 - e_idx)",
            "keys = (u << ebits) | e_idx",
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "top-K ties broken toward the HIGHER expert index: "
        "identical on tie-free draws, wrong set on tie-forcing "
        "shape -> G2 edge-case failure",
    },
    {
        "name": "nc_differential_path",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": _mut_dp,
        "bound_gate": "G2",
        "oracle_env": True,
        "expectation": "regime-switch exploit: apply sum-renorm only under "
        "saturated sigmoid (a stability probe), and use raw "
        "sigmoid on nominal inputs. G2 fails at smoke even "
        "with FORGE_ORACLE=1",
    },
    {
        "name": "nc_torch_fallback",
        "base": "starter",
        "target_file": "kernel.py",
        "mutate": lambda src: _FALLBACK_SRC,
        "bound_gate": "G4",
        "oracle_env": False,
        "expectation": "correct pure-torch composition, 100% of device time "
        "outside candidate Triton kernels: G2 passes and G4 "
        "adoption floor zeroes",
    },
]
