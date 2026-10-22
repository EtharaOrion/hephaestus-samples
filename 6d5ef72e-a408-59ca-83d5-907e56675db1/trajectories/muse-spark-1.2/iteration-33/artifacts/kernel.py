import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32
EBLOCK = 2048
GM = 32
GD = 32


@triton.jit
def _mm_kernel(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn, BLK_M: tl.constexpr, BLK_N: tl.constexpr, BLK_K: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M).to(tl.int64)
    rn = pid_n * BLK_N + tl.arange(0, BLK_N).to(tl.int64)
    acc = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
    for k0 in range(0, K, BLK_K):
        rk = k0 + tl.arange(0, BLK_K).to(tl.int64)
        a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


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
def _gather_kernel(X, Idx, Out, M, D, sx0: tl.constexpr, sx1: tl.constexpr, so0: tl.constexpr, so1: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (rm[:, None] < M) & (rd[None, :] < D)
    idx = tl.load(Idx + rm, mask=rm < M, other=0)
    off_x = idx[:, None] * sx0 + rd[None, :] * sx1
    off_o = rm[:, None] * so0 + rd[None, :] * so1
    v = tl.load(X + off_x, mask=mask, other=0.0)
    tl.store(Out + off_o, v, mask=mask)


@triton.jit
def _scatter_add_kernel(Out, Idx, Inp, M, D, so0: tl.constexpr, so1: tl.constexpr, si0: tl.constexpr, si1: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (rm[:, None] < M) & (rd[None, :] < D)
    idx = tl.load(Idx + rm, mask=rm < M, other=0)
    off_in = rm[:, None] * si0 + rd[None, :] * si1
    off_out = idx[:, None] * so0 + rd[None, :] * so1
    vin = tl.load(Inp + off_in, mask=mask, other=0.0)
    vout = tl.load(Out + off_out, mask=mask, other=0.0)
    tl.store(Out + off_out, vout + vin, mask=mask)


@triton.jit
def _scatter_weighted_kernel(Y, Idx, Inp, W, M, D, sy0: tl.constexpr, sy1: tl.constexpr, si0: tl.constexpr, si1: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (rm[:, None] < M) & (rd[None, :] < D)
    idx = tl.load(Idx + rm, mask=rm < M, other=0)
    w = tl.load(W + rm, mask=rm < M, other=0.0)
    w = w[:, None]
    vin = tl.load(Inp + rm[:, None] * si0 + rd[None, :] * si1, mask=mask, other=0.0)
    y_off = idx[:, None] * sy0 + rd[None, :] * sy1
    yv = tl.load(Y + y_off, mask=mask, other=0.0)
    tl.store(Y + y_off, yv + w * vin, mask=mask)


@triton.jit
def _gather_weighted_bwd_fe_kernel(GY, Idx, W, GFe, M, D, sgy0: tl.constexpr, sgy1: tl.constexpr, sg0: tl.constexpr, sg1: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (rm[:, None] < M) & (rd[None, :] < D)
    idx = tl.load(Idx + rm, mask=rm < M, other=0)
    w = tl.load(W + rm, mask=rm < M, other=0.0)[:, None]
    gy = tl.load(GY + idx[:, None] * sgy0 + rd[None, :] * sgy1, mask=mask, other=0.0)
    tl.store(GFe + rm[:, None] * sg0 + rd[None, :] * sg1, gy * w, mask=mask)


def _mm_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    _mm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK)
    return c


def _gather_launch(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    M = idx.shape[0]
    D = x.shape[1]
    out = torch.empty(M, D, device=x.device, dtype=torch.float32)
    if M == 0 or D == 0:
        return out
    _gather_kernel[(triton.cdiv(M, GM), triton.cdiv(D, GD))](x, idx, out, M, D, x.stride(0), x.stride(1), out.stride(0), out.stride(1), BLOCK_M=GM, BLOCK_D=GD)
    return out


def _scatter_add_launch(out: torch.Tensor, idx: torch.Tensor, inp: torch.Tensor):
    M = idx.shape[0]
    D = out.shape[1]
    if M == 0:
        return
    _scatter_add_kernel[(triton.cdiv(M, GM), triton.cdiv(D, GD))](out, idx, inp, M, D, out.stride(0), out.stride(1), inp.stride(0), inp.stride(1), BLOCK_M=GM, BLOCK_D=GD)


def _scatter_weighted_launch(y: torch.Tensor, idx: torch.Tensor, inp: torch.Tensor, w: torch.Tensor):
    M = idx.shape[0]
    D = y.shape[1]
    if M == 0:
        return
    _scatter_weighted_kernel[(triton.cdiv(M, GM), triton.cdiv(D, GD))](y, idx, inp, w, M, D, y.stride(0), y.stride(1), inp.stride(0), inp.stride(1), BLOCK_M=GM, BLOCK_D=GD)


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


class _GatherOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx):
        ctx.save_for_backward(idx)
        ctx.x_shape = x.shape
        return _gather_launch(x, idx)

    @staticmethod
    def backward(ctx, grad_out):
        (idx,) = ctx.saved_tensors
        grad_x = torch.zeros(ctx.x_shape, device=grad_out.device, dtype=grad_out.dtype)
        if grad_out.shape[0] != 0:
            _scatter_add_launch(grad_x, idx, grad_out.contiguous())
        return grad_x, None


class _ScatterWeighted(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, idx, inp, w):
        ctx.save_for_backward(idx, inp, w)
        out = y.clone()
        if idx.shape[0] != 0:
            _scatter_weighted_launch(out, idx, inp, w)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        idx, inp, w = ctx.saved_tensors
        grad_y = grad_out.clone()
        M = idx.shape[0]
        if M == 0:
            grad_inp = torch.zeros_like(inp)
            grad_w = torch.zeros_like(w)
            return grad_y, None, grad_inp, grad_w
        # grad_inp = gather(grad_out, idx) * w
        grad_inp = torch.empty_like(inp)
        _gather_weighted_bwd_fe_kernel[(triton.cdiv(M, GM), triton.cdiv(inp.shape[1], GD))](grad_out, idx, w, grad_inp, M, inp.shape[1], grad_out.stride(0), grad_out.stride(1), grad_inp.stride(0), grad_inp.stride(1), BLOCK_M=GM, BLOCK_D=GD)
        # grad_w = sum_d grad_out[idx[d]] * inp
        # gather gy
        gy_gather = _gather_launch(grad_out, idx)
        grad_w = (gy_gather * inp).sum(dim=1)
        return grad_y, None, grad_inp, grad_w


def _select_topA(s: torch.Tensor, router_b: torch.Tensor, A: int) -> torch.Tensor:
    E = s.shape[-1]
    b = s + router_b
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=s.device, dtype=torch.int64)
    keys = (u << 12) | (E - 1 - e_idx)
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
            w_norm = s_sel / s_sel.sum(dim=-1, keepdim=True)
            flat = sel.reshape(-1)
            order = torch.argsort(flat, stable=True)
            flat_sorted = flat[order]
            T_rep = torch.arange(T, device=x.device).repeat_interleave(A)
            token_sorted = T_rep[order]
            w_flat = w_norm.reshape(-1)[order]
            counts = torch.bincount(flat, minlength=E)
            counts_list = counts.tolist()
            y = xf.new_zeros(T, D)
            off = 0
            for e in range(E):
                c = counts_list[e]
                start = off
                end = off + c
                off = end
                idx = token_sorted[start:end]
                wseg = w_flat[start:end]
                h = _GatherOp.apply(xf, idx)
                gu = _MM.apply(h, w1s[e])
                act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
                fe = _MM.apply(act, w2s[e])
                y = _ScatterWeighted.apply(y, idx, fe, wseg)
        ctx.save_for_backward(xf, rwf, w1f, w2f, y)
        ctx.dtypes = (x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.detach().to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, rwf, w1f, w2f, y = ctx.saved_tensors
        gx, grw, gw1, gw2 = torch.autograd.grad(y, (xf, rwf, w1f, w2f), dy.float())
        dx_t, drw_t, dw1_t, dw2_t = ctx.dtypes
        return gx.to(dx_t), grw.to(drw_t), None, gw1.to(dw1_t), gw2.to(dw2_t), None


def fused_moe(x: torch.Tensor, router_w: torch.Tensor, router_b: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, A: int) -> torch.Tensor:
    return _FusedMoe.apply(x, router_w, router_b, w1, w2, A)
