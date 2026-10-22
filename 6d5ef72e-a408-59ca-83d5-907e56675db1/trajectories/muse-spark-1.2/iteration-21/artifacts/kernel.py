import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32
EBLOCK = 2048
GBLOCK = 64
SBLOCK = 64


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
def _gather_kernel(X, IND, OUT, N, D, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    idx = tl.load(IND + pid).to(tl.int64)
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK).to(tl.int64)
        mask = cols < D
        vals = tl.load(X + idx * D + cols, mask=mask, other=0.0)
        tl.store(OUT + pid * D + cols, vals, mask=mask)


@triton.jit
def _scatter_kernel(Y, IND, SRC, N, D, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    idx = tl.load(IND + pid).to(tl.int64)
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK).to(tl.int64)
        mask = cols < D
        src = tl.load(SRC + pid * D + cols, mask=mask, other=0.0)
        y_ptr = Y + idx * D + cols
        y_old = tl.load(y_ptr, mask=mask, other=0.0)
        tl.store(y_ptr, y_old + src, mask=mask)


@triton.jit
def _scatter_grad_kernel(GX, IND, GOUT, N, D, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    idx = tl.load(IND + pid).to(tl.int64)
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK).to(tl.int64)
        mask = cols < D
        g = tl.load(GOUT + pid * D + cols, mask=mask, other=0.0)
        p = GX + idx * D + cols
        old = tl.load(p, mask=mask, other=0.0)
        tl.store(p, old + g, mask=mask)


@triton.jit
def _gather_dy_kernel(DY, IND, OUT, N, D, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    idx = tl.load(IND + pid).to(tl.int64)
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK).to(tl.int64)
        mask = cols < D
        v = tl.load(DY + idx * D + cols, mask=mask, other=0.0)
        tl.store(OUT + pid * D + cols, v, mask=mask)


def _mm_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    _mm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK)
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
        _silu_mul_bwd_kernel[(triton.cdiv(n, EBLOCK),)](g, u, dout, dg, du, n, BLOCK=EBLOCK)
        return dg, du


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ind):
        ctx.save_for_backward(ind)
        ctx.x_shape = x.shape
        N = ind.shape[0]
        D = x.shape[1]
        out = torch.empty(N, D, device=x.device, dtype=x.dtype)
        _gather_kernel[(N,)](x, ind, out, N, D, BLOCK=GBLOCK)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (ind,) = ctx.saved_tensors
        T, D = ctx.x_shape
        N = ind.shape[0]
        grad_x = torch.zeros(T, D, device=grad_out.device, dtype=grad_out.dtype)
        _scatter_grad_kernel[(N,)](grad_x, ind, grad_out.contiguous(), N, D, BLOCK=GBLOCK)
        return grad_x, None


class _Scatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, ind, src):
        ctx.save_for_backward(ind)
        ctx.src_shape = src.shape
        N = ind.shape[0]
        out = y.clone()
        _scatter_kernel[(N,)](out, ind, src.contiguous(), N, out.shape[1], BLOCK=SBLOCK)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (ind,) = ctx.saved_tensors
        N = ind.shape[0]
        grad_y = grad_out.clone()
        g = torch.zeros(ctx.src_shape, device=grad_out.device, dtype=grad_out.dtype)
        _gather_dy_kernel[(N,)](grad_out.contiguous(), ind, g, N, grad_out.shape[1], BLOCK=SBLOCK)
        return grad_y, None, g


def _select_topA(s: torch.Tensor, router_b: torch.Tensor, A: int) -> torch.Tensor:
    E = s.shape[-1]
    b = s + router_b
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=s.device, dtype=torch.int64)
    keys = (u << 10) | (E - 1 - e_idx)
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
            flat = sel.reshape(-1)
            order = torch.argsort(flat)
            tok = torch.div(order, A, rounding_mode="floor")
            cnt = torch.bincount(flat, minlength=E)
            offs = cnt.cumsum(0)
            w_sorted = w.reshape(-1).index_select(0, order)
            y = torch.zeros(T, D, device=xf.device, dtype=torch.float32)
            offs_cpu = offs.tolist()
            prev = 0
            for e in range(E):
                cur = offs_cpu[e]
                tok_e = tok[prev:cur]
                ws = w_sorted[prev:cur]
                xs = _Gather.apply(xf, tok_e)
                gu = _MM.apply(xs, w1s[e])
                act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
                fe = _MM.apply(act, w2s[e])
                fe_w = fe * ws.unsqueeze(1)
                y = _Scatter.apply(y, tok_e, fe_w)
                prev = cur
        ctx.save_for_backward(xf, rwf, w1f, w2f, y)
        ctx.dtypes = (x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.detach().to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, rwf, w1f, w2f, y = ctx.saved_tensors
        gx, grw, gw1, gw2 = torch.autograd.grad(y, (xf, rwf, w1f, w2f), dy.float())
        dx_t, drw_t, dw1_t, dw2_t = ctx.dtypes
        return (gx.to(dx_t), grw.to(drw_t), None, gw1.to(dw1_t), gw2.to(dw2_t), None)


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _FusedMoe.apply(x, router_w, router_b, w1, w2, A)
