import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32
EBLOCK = 2048


@triton.jit
def _mm_kernel(A, B, C, M, N, K,
               sam, sak, sbk, sbn, scm, scn,
               BLK_M: tl.constexpr, BLK_N: tl.constexpr, BLK_K: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M).to(tl.int64)
    rn = pid_n * BLK_N + tl.arange(0, BLK_N).to(tl.int64)
    acc = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
    for k0 in range(0, K, BLK_K):
        rk = k0 + tl.arange(0, BLK_K).to(tl.int64)
        a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak,
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn,
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


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


@triton.jit
def _scatter_scaled_fwd_kernel(DST, SRC, ROWS, W, D, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= n:
        return
    row = tl.load(ROWS + pid).to(tl.int64)
    w = tl.load(W + pid)
    base_dst = row * D
    base_src = pid * D
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK).to(tl.int64)
        mask = cols < D
        s = tl.load(SRC + base_src + cols, mask=mask)
        d = tl.load(DST + base_dst + cols, mask=mask)
        tl.store(DST + base_dst + cols, d + s * w, mask=mask)


@triton.jit
def _scatter_src_grad_kernel(G_DST, ROWS, W, G_SRC, D, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= n:
        return
    row = tl.load(ROWS + pid).to(tl.int64)
    w = tl.load(W + pid)
    base_dst = row * D
    base_src = pid * D
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK).to(tl.int64)
        mask = cols < D
        g = tl.load(G_DST + base_dst + cols, mask=mask)
        tl.store(G_SRC + base_src + cols, g * w, mask=mask)


@triton.jit
def _scatter_w_grad_kernel(G_DST, SRC, ROWS, G_W, D, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= n:
        return
    row = tl.load(ROWS + pid).to(tl.int64)
    base_dst = row * D
    base_src = pid * D
    acc = 0.0
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK).to(tl.int64)
        mask = cols < D
        g = tl.load(G_DST + base_dst + cols, mask=mask, other=0.0)
        s = tl.load(SRC + base_src + cols, mask=mask, other=0.0)
        acc += tl.sum(g * s)
    tl.store(G_W + pid, acc)


def _mm_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    _mm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK,
        num_warps=4, num_stages=3)
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
        _silu_mul_bwd_kernel[(triton.cdiv(n, EBLOCK),)](g, u, dout, dg, du, n,
                                                        BLOCK=EBLOCK)
        return dg, du


class _ScatterMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_in, src, rows, w):
        # y_in: [T,D] float32, src: [n,D], rows: [n], w: [n]
        n = src.shape[0]
        D = y_in.shape[1]
        y_out = y_in.clone()
        if n > 0:
            _scatter_scaled_fwd_kernel[(n,)](y_out, src, rows, w, D, n, BLOCK=1024)
        ctx.save_for_backward(src, rows, w)
        ctx.D = D
        ctx.n = n
        return y_out

    @staticmethod
    def backward(ctx, grad_out):
        src, rows, w = ctx.saved_tensors
        D = ctx.D
        n = ctx.n
        grad_out = grad_out.contiguous()
        grad_y_in = grad_out.clone()
        grad_src = None
        grad_w = None
        if n > 0:
            grad_src = torch.empty_like(src)
            _scatter_src_grad_kernel[(n,)](grad_out, rows, w, grad_src, D, n, BLOCK=1024)
            grad_w = torch.empty_like(w)
            _scatter_w_grad_kernel[(n,)](grad_out, src, rows, grad_w, D, n, BLOCK=1024)
        return grad_y_in, grad_src, None, grad_w


def _select_topA(s: torch.Tensor, router_b: torch.Tensor, A: int) -> torch.Tensor:
    E = s.shape[-1]
    b = s + router_b
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=s.device, dtype=torch.int64)
    keys = (u << 11) | (E - 1 - e_idx)
    return torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices


class _FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        with torch.enable_grad():
            xf = x.detach().float().requires_grad_(True)
            rwf = router_w.detach().float().requires_grad_(True)
            w1f = w1.detach().float().requires_grad_(True)
            w2f = w2.detach().float().requires_grad_(True)
            T, D = xf.shape
            E = rwf.shape[1]
            F = w1f.shape[2] // 2
            s = torch.sigmoid(_MM.apply(xf, rwf))
            w1s = w1f.unbind(0)
            w2s = w2f.unbind(0)
            with torch.no_grad():
                sel = _select_topA(s.detach(), router_b.float(), A)
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
                gu = _MM.apply(h, w1s[e])
                act = _SiLUMul.apply(gu[:, :F].contiguous(),
                                     gu[:, F:].contiguous())
                fe = _MM.apply(act, w2s[e])
                w_e = w_full.index_select(0, rows)[:, e].contiguous()
                y = _ScatterMul.apply(y, fe, rows, w_e)
        ctx.save_for_backward(xf, rwf, w1f, w2f, y)
        ctx.dtypes = (x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.detach().to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, rwf, w1f, w2f, y = ctx.saved_tensors
        gx, grw, gw1, gw2 = torch.autograd.grad(y, (xf, rwf, w1f, w2f),
                                                dy.float())
        dx_t, drw_t, dw1_t, dw2_t = ctx.dtypes
        return (gx.to(dx_t), grw.to(drw_t), None,
                gw1.to(dw1_t), gw2.to(dw2_t), None)


def fused_moe(
    x: torch.Tensor,
    router_w: torch.Tensor,
    router_b: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    A: int,
) -> torch.Tensor:
    return _FusedMoe.apply(x, router_w, router_b, w1, w2, A)
