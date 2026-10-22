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
def _gather_kernel(X, OUT, ROWS, D, sx0, sx1, so0, so1, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    row = tl.load(ROWS + pid).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    v = tl.load(X + row * sx0 + offs * sx1, mask=mask, other=0.0)
    tl.store(OUT + pid * so0 + offs * so1, v, mask=mask)


@triton.jit
def _gather_bwd_kernel(GOUT, GIN, ROWS, D, sg0, sg1, si0, si1, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    row = tl.load(ROWS + pid).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    v = tl.load(GOUT + pid * sg0 + offs * sg1, mask=mask, other=0.0)
    base = GIN + row * si0 + offs * si1
    cur = tl.load(base, mask=mask, other=0.0)
    tl.store(base, cur + v, mask=mask)


@triton.jit
def _scatter_kernel(Y, SRC, ROWS, D, sy0, sy1, ss0, ss1, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    row = tl.load(ROWS + pid).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    src = tl.load(SRC + pid * ss0 + offs * ss1, mask=mask, other=0.0)
    yptr = Y + row * sy0 + offs * sy1
    cur = tl.load(yptr, mask=mask, other=0.0)
    tl.store(yptr, cur + src, mask=mask)


@triton.jit
def _scatter_gather_kernel(GOUT, SRC_G, ROWS, D, sg0, sg1, ss0, ss1, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    row = tl.load(ROWS + pid).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    v = tl.load(GOUT + row * sg0 + offs * sg1, mask=mask, other=0.0)
    tl.store(SRC_G + pid * ss0 + offs * ss1, v, mask=mask)


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
    def forward(ctx, x, rows):
        ctx.save_for_backward(rows)
        ctx.T = x.shape[0]
        ctx.D = x.shape[1]
        n = rows.numel()
        out = torch.empty(n, ctx.D, device=x.device, dtype=x.dtype)
        if n == 0:
            return out
        D = ctx.D
        if D <= 1024:
            BD = triton.next_power_of_2(D)
            _gather_kernel[(n,)](x, out, rows, D, x.stride(0), x.stride(1), out.stride(0), out.stride(1), BLOCK_D=BD)
        else:
            _gather_kernel[(n,)](x, out, rows, 1024, x.stride(0), x.stride(1), out.stride(0), out.stride(1), BLOCK_D=1024)
            rem = D - 1024
            BR = triton.next_power_of_2(rem)
            _gather_kernel[(n,)](x[:, 1024:], out[:, 1024:], rows, rem, x.stride(0), x.stride(1), out.stride(0), out.stride(1), BLOCK_D=BR)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        rows, = ctx.saved_tensors
        T = ctx.T
        D = ctx.D
        grad_in = torch.zeros(T, D, device=grad_out.device, dtype=grad_out.dtype)
        n = rows.numel()
        if n == 0:
            return grad_in, None
        go = grad_out.contiguous()
        if D <= 1024:
            BD = triton.next_power_of_2(D)
            _gather_bwd_kernel[(n,)](go, grad_in, rows, D, go.stride(0), go.stride(1), grad_in.stride(0), grad_in.stride(1), BLOCK_D=BD)
        else:
            _gather_bwd_kernel[(n,)](go, grad_in, rows, 1024, go.stride(0), go.stride(1), grad_in.stride(0), grad_in.stride(1), BLOCK_D=1024)
            rem = D - 1024
            BR = triton.next_power_of_2(rem)
            _gather_bwd_kernel[(n,)](go[:, 1024:], grad_in[:, 1024:], rows, rem, go.stride(0), go.stride(1), grad_in.stride(0), grad_in.stride(1), BLOCK_D=BR)
        return grad_in, None


class _Scatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, src, rows):
        ctx.save_for_backward(rows)
        ctx.D = y.shape[1]
        out = y.clone()
        n = rows.numel()
        if n == 0:
            return out
        D = ctx.D
        s = src.contiguous()
        if D <= 1024:
            BD = triton.next_power_of_2(D)
            _scatter_kernel[(n,)](out, s, rows, D, out.stride(0), out.stride(1), s.stride(0), s.stride(1), BLOCK_D=BD)
        else:
            _scatter_kernel[(n,)](out, s, rows, 1024, out.stride(0), out.stride(1), s.stride(0), s.stride(1), BLOCK_D=1024)
            rem = D - 1024
            BR = triton.next_power_of_2(rem)
            _scatter_kernel[(n,)](out[:, 1024:], s[:, 1024:], rows, rem, out.stride(0), out.stride(1), s.stride(0), s.stride(1), BLOCK_D=BR)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        rows, = ctx.saved_tensors
        D = ctx.D
        n = rows.numel()
        grad_y = grad_out
        if n == 0:
            return grad_y, torch.empty(0, D, device=grad_out.device, dtype=grad_out.dtype), None
        grad_src = torch.empty(n, D, device=grad_out.device, dtype=grad_out.dtype)
        go = grad_out.contiguous()
        if D <= 1024:
            BD = triton.next_power_of_2(D)
            _scatter_gather_kernel[(n,)](go, grad_src, rows, D, go.stride(0), go.stride(1), grad_src.stride(0), grad_src.stride(1), BLOCK_D=BD)
        else:
            _scatter_gather_kernel[(n,)](go, grad_src, rows, 1024, go.stride(0), go.stride(1), grad_src.stride(0), grad_src.stride(1), BLOCK_D=1024)
            rem = D - 1024
            BR = triton.next_power_of_2(rem)
            _scatter_gather_kernel[(n,)](go[:, 1024:], grad_src[:, 1024:], rows, rem, go.stride(0), go.stride(1), grad_src.stride(0), grad_src.stride(1), BLOCK_D=BR)
        return grad_y, grad_src, None


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
            flat = sel.reshape(-1)
            w_flat = w.reshape(-1)
            order = torch.argsort(flat)
            tok_sorted = torch.div(order, A, rounding_mode="floor")
            w_sorted = w_flat.index_select(0, order)
            cnt = torch.bincount(flat, minlength=E)
            offs = torch.cumsum(cnt, 0)
            y = xf.new_zeros(T, D)
            off = 0
            for e in range(E):
                nrows = int(cnt[e].item())
                start = off
                off += nrows
                rows = tok_sorted[start:start+nrows]
                w_slice = w_sorted[start:start+nrows]
                h = _Gather.apply(xf, rows)
                gu = _MM.apply(h, w1s[e])
                act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
                fe = _MM.apply(act, w2s[e])
                weighted = fe * w_slice[:, None]
                y = _Scatter.apply(y, weighted, rows)
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
