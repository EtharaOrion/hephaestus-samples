"""
Starter -- shared-expert routed top-K fused MoE, fwd + bwd.

Deliberately slow: three small Triton kernels (unblocked fp32 matmul,
elementwise silu*up fwd, elementwise silu*up bwd) wrapped in a python loop
over routed experts plus a two-matmul dense shared path. Correct,
deterministic, and slow by design.

The entry point `fused_moe` and its signature are fixed:
    softmax router `p = softmax(x @ router_w)`,
    top-K by p (ties LOWER expert index),
    routed combine weights = `p_sel / p_sel.sum(-1)` (mixtral rule),
    shared expert = dense SwiGLU always-on,
    y = y_routed + y_shared, fp32 arithmetic, one final cast.
SIX graded gradients: dx, drouter_w, dw1, dw2, dw1_s, dw2_s.
"""

import torch
import triton
import triton.language as tl

BM, BN, BK = 16, 16, 16
EBLOCK = 1024


@triton.jit
def _mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    sam,
    sak,
    sbk,
    sbn,
    scm,
    scn,
    BLK_M: tl.constexpr,
    BLK_N: tl.constexpr,
    BLK_K: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M).to(tl.int64)
    rn = pid_n * BLK_N + tl.arange(0, BLK_N).to(tl.int64)
    acc = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
    for k0 in range(0, K, BLK_K):
        rk = k0 + tl.arange(0, BLK_K).to(tl.int64)
        a = tl.load(
            A + rm[:, None] * sam + rk[None, :] * sak,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + rk[:, None] * sbk + rn[None, :] * sbn,
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(
        C + rm[:, None] * scm + rn[None, :] * scn,
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _silu_mul_fwd_kernel(G, U, OUT, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    o = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < n
    g = tl.load(G + o, mask=m, other=0.0)
    u = tl.load(U + o, mask=m, other=0.0)
    s = tl.sigmoid(g)
    tl.store(OUT + o, g * s * u, mask=m)


@triton.jit
def _silu_mul_bwd_kernel(G, U, DO, DG, DU, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    o = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < n
    g = tl.load(G + o, mask=m, other=0.0)
    u = tl.load(U + o, mask=m, other=0.0)
    do = tl.load(DO + o, mask=m, other=0.0)
    s = tl.sigmoid(g)
    silu = g * s
    tl.store(DG + o, do * u * (s + silu * (1.0 - s)), mask=m)
    tl.store(DU + o, do * silu, mask=m)


def _mm_launch(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    _mm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLK_M=BM,
        BLK_N=BN,
        BLK_K=BK,
    )
    return c


class _MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _mm_launch(a, b)

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        dc = dc.contiguous()
        return _mm_launch(dc, b.t()), _mm_launch(a.t(), dc)


class _SiLUMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u):
        ctx.save_for_backward(g, u)
        out = torch.empty_like(g)
        n = g.numel()
        _silu_mul_fwd_kernel[(triton.cdiv(n, EBLOCK),)](g, u, out, n, BLOCK=EBLOCK)
        return out

    @staticmethod
    def backward(ctx, dout):
        g, u = ctx.saved_tensors
        dout = dout.contiguous()
        dg, du = torch.empty_like(g), torch.empty_like(u)
        n = g.numel()
        _silu_mul_bwd_kernel[(triton.cdiv(n, EBLOCK),)](
            g, u, dout, dg, du, n, BLOCK=EBLOCK
        )
        return dg, du


def _select_topK(scores, K):
    E = scores.shape[-1]
    b = torch.where(scores == 0.0, torch.zeros_like(scores), scores).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=scores.device, dtype=torch.int64)
    ebits = max((E - 1).bit_length(), 1)
    keys = (u << ebits) | (E - 1 - e_idx)
    return torch.topk(keys, K, dim=-1, largest=True, sorted=True).indices


class _FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, w1, w2, w1_s, w2_s, K):
        K = int(K)
        with torch.enable_grad():
            xf = x.detach().float().requires_grad_(True)
            rwf = router_w.detach().float().requires_grad_(True)
            w1f = w1.detach().float().requires_grad_(True)
            w2f = w2.detach().float().requires_grad_(True)
            w1sf = w1_s.detach().float().requires_grad_(True)
            w2sf = w2_s.detach().float().requires_grad_(True)
            T, D = xf.shape
            E = rwf.shape[1]
            F = w1f.shape[2] // 2
            Fs = w1sf.shape[1] // 2
            logits = _MM.apply(xf, rwf)
            p = torch.softmax(logits, dim=-1)
            w1s = w1f.unbind(0)
            w2s = w2f.unbind(0)
            sel = _select_topK(p.detach(), K)
            p_sel = torch.gather(p, 1, sel)
            w = p_sel / p_sel.sum(dim=-1, keepdim=True)
            w_full = p.new_zeros(T, E).scatter(1, sel, w)
            member = torch.zeros(T, E, dtype=torch.bool, device=x.device).scatter(
                1, sel, True
            )
            y = xf.new_zeros(T, D)
            # routed
            for e in range(E):
                rows = member[:, e].nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                h = xf.index_select(0, rows)
                gu = _MM.apply(h, w1s[e])
                act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
                fe = _MM.apply(act, w2s[e])
                y = y.index_add(
                    0, rows, w_full.index_select(0, rows)[:, e : e + 1] * fe
                )
            # shared (dense, unconditional)
            gs = _MM.apply(xf, w1sf)
            acts = _SiLUMul.apply(gs[:, :Fs].contiguous(), gs[:, Fs:].contiguous())
            y = y + _MM.apply(acts, w2sf)
        ctx.save_for_backward(xf, rwf, w1f, w2f, w1sf, w2sf, y)
        ctx.dtypes = (
            x.dtype,
            router_w.dtype,
            w1.dtype,
            w2.dtype,
            w1_s.dtype,
            w2_s.dtype,
        )
        return y.detach().to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, rwf, w1f, w2f, w1sf, w2sf, y = ctx.saved_tensors
        gx, grw, gw1, gw2, gw1s, gw2s = torch.autograd.grad(
            y, (xf, rwf, w1f, w2f, w1sf, w2sf), dy.float()
        )
        dx_t, drw_t, dw1_t, dw2_t, dw1s_t, dw2s_t = ctx.dtypes
        return (
            gx.to(dx_t),
            grw.to(drw_t),
            gw1.to(dw1_t),
            gw2.to(dw2_t),
            gw1s.to(dw1s_t),
            gw2s.to(dw2s_t),
            None,
        )


def fused_moe(
    x: torch.Tensor,  # [T, D]
    router_w: torch.Tensor,  # [D, E]
    w1: torch.Tensor,  # [E, D, 2F]
    w2: torch.Tensor,  # [E, F, D]
    w1_s: torch.Tensor,  # [D, 2Fs]
    w2_s: torch.Tensor,  # [Fs, D]
    K: int,  # top-K routed
) -> torch.Tensor:  # [T, D]
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _FusedMoe.apply(x, router_w, w1, w2, w1_s, w2_s, K)
