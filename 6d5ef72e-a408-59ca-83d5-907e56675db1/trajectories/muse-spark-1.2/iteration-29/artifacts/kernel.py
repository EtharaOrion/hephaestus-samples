import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32
EBLOCK = 2048

@triton.jit
def _mm_kernel(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn, BLK_M: tl.constexpr, BLK_N: tl.constexpr, BLK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M)
    rn = pid_n * BLK_N + tl.arange(0, BLK_N)
    acc = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
    for k0 in range(0, K, BLK_K):
        rk = k0 + tl.arange(0, BLK_K)
        a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

@triton.jit
def _silu_mul_fwd_kernel(G, U, OUT, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    g = tl.load(G + o, mask=m, other=0.0)
    u = tl.load(U + o, mask=m, other=0.0)
    s = tl.sigmoid(g)
    tl.store(OUT + o, g * s * u, mask=m)

@triton.jit
def _silu_mul_bwd_kernel(G, U, DO, DG, DU, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    g = tl.load(G + o, mask=m, other=0.0)
    u = tl.load(U + o, mask=m, other=0.0)
    do = tl.load(DO + o, mask=m, other=0.0)
    s = tl.sigmoid(g)
    silu = g * s
    tl.store(DG + o, do * u * (s + silu * (1.0 - s)), mask=m)
    tl.store(DU + o, do * silu, mask=m)

@triton.jit
def _gather_kernel(X, IDX, OUT, N, D, sx0, sx1, so0, so1, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < N
    mask_n = rn < D
    idx = tl.load(IDX + rm, mask=mask_m, other=0)
    x_ptr = X + idx[:, None] * sx0 + rn[None, :] * sx1
    o_ptr = OUT + rm[:, None] * so0 + rn[None, :] * so1
    mask = mask_m[:, None] & mask_n[None, :]
    vals = tl.load(x_ptr, mask=mask, other=0.0)
    tl.store(o_ptr, vals, mask=mask)

@triton.jit
def _scatter_scale_kernel(Y, IDX, FE, W, N, D, sy0, sy1, sfe0, sfe1, ws, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < N
    mask_n = rn < D
    idx = tl.load(IDX + rm, mask=mask_m, other=0)
    w = tl.load(W + rm * ws, mask=mask_m, other=0.0)
    mask = mask_m[:, None] & mask_n[None, :]
    y_ptr = Y + idx[:, None] * sy0 + rn[None, :] * sy1
    fe_ptr = FE + rm[:, None] * sfe0 + rn[None, :] * sfe1
    y = tl.load(y_ptr, mask=mask, other=0.0)
    fe = tl.load(fe_ptr, mask=mask, other=0.0)
    y = y + fe * w[:, None]
    tl.store(y_ptr, y, mask=mask)

@triton.jit
def _gather_mul_kernel(GY, IDX, W, OUT, N, D, sgy0, sgy1, so0, so1, ws, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < N
    mask_n = rn < D
    idx = tl.load(IDX + rm, mask=mask_m, other=0)
    w = tl.load(W + rm * ws, mask=mask_m, other=0.0)
    mask = mask_m[:, None] & mask_n[None, :]
    gy = tl.load(GY + idx[:, None] * sgy0 + rn[None, :] * sgy1, mask=mask, other=0.0)
    tl.store(OUT + rm[:, None] * so0 + rn[None, :] * so1, gy * w[:, None], mask=mask)

@triton.jit
def _scatter_add_kernel(DST, IDX, SRC, N, D, sd0, sd1, ss0, ss1, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < N
    mask_n = rn < D
    idx = tl.load(IDX + rm, mask=mask_m, other=0)
    mask = mask_m[:, None] & mask_n[None, :]
    src = tl.load(SRC + rm[:, None] * ss0 + rn[None, :] * ss1, mask=mask, other=0.0)
    dst_ptr = DST + idx[:, None] * sd0 + rn[None, :] * sd1
    cur = tl.load(dst_ptr, mask=mask, other=0.0)
    tl.store(dst_ptr, cur + src, mask=mask)

def _mm_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    _mm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK)
    return c

def _gather_launch(x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    N = rows.shape[0]
    D = x.shape[1]
    out = torch.empty(N, D, device=x.device, dtype=x.dtype)
    if N == 0:
        return out
    GM, GN = 32, 32
    _gather_kernel[(triton.cdiv(N, GM), triton.cdiv(D, GN))](x, rows, out, N, D, x.stride(0), x.stride(1), out.stride(0), out.stride(1), BLOCK_M=GM, BLOCK_N=GN)
    return out

def _scatter_scale_launch(y: torch.Tensor, rows: torch.Tensor, fe: torch.Tensor, w: torch.Tensor):
    N = rows.shape[0]
    if N == 0:
        return
    D = y.shape[1]
    GM, GN = 32, 32
    _scatter_scale_kernel[(triton.cdiv(N, GM), triton.cdiv(D, GN))](y, rows, fe, w, N, D, y.stride(0), y.stride(1), fe.stride(0), fe.stride(1), w.stride(0), BLOCK_M=GM, BLOCK_N=GN)

def _scatter_add_launch(dst: torch.Tensor, rows: torch.Tensor, src: torch.Tensor):
    N = rows.shape[0]
    if N == 0:
        return
    D = dst.shape[1]
    GM, GN = 32, 32
    _scatter_add_kernel[(triton.cdiv(N, GM), triton.cdiv(D, GN))](dst, rows, src, N, D, dst.stride(0), dst.stride(1), src.stride(0), src.stride(1), BLOCK_M=GM, BLOCK_N=GN)

def _gather_mul_launch(gy: torch.Tensor, rows: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    N = rows.shape[0]
    D = gy.shape[1]
    out = torch.empty(N, D, device=gy.device, dtype=gy.dtype)
    if N == 0:
        return out
    GM, GN = 32, 32
    _gather_mul_kernel[(triton.cdiv(N, GM), triton.cdiv(D, GN))](gy, rows, w, out, N, D, gy.stride(0), gy.stride(1), out.stride(0), out.stride(1), w.stride(0), BLOCK_M=GM, BLOCK_N=GN)
    return out

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
        if n == 0:
            return out
        _silu_mul_fwd_kernel[(triton.cdiv(n, EBLOCK),)](g, u, out, n, BLOCK=EBLOCK)
        return out
    @staticmethod
    def backward(ctx, dout):
        g, u = ctx.saved_tensors
        dout = dout.contiguous()
        dg, du = torch.empty_like(g), torch.empty_like(u)
        n = g.numel()
        if n == 0:
            return dg, du
        _silu_mul_bwd_kernel[(triton.cdiv(n, EBLOCK),)](g, u, dout, dg, du, n, BLOCK=EBLOCK)
        return dg, du

class _GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rows):
        ctx.save_for_backward(rows)
        ctx.T = x.shape[0]
        ctx.D = x.shape[1]
        return _gather_launch(x, rows)
    @staticmethod
    def backward(ctx, grad_out):
        rows, = ctx.saved_tensors
        T = ctx.T
        D = ctx.D
        grad_x = torch.zeros(T, D, device=grad_out.device, dtype=grad_out.dtype)
        _scatter_add_launch(grad_x, rows, grad_out.contiguous())
        return grad_x, None

class _ScatterScaleAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, rows, fe, w):
        ctx.save_for_backward(rows, fe, w)
        out = y.clone()
        _scatter_scale_launch(out, rows, fe, w)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        rows, fe, w = ctx.saved_tensors
        grad_out_c = grad_out.contiguous()
        grad_y = grad_out_c
        grad_fe = _gather_mul_launch(grad_out_c, rows, w)
        g_gather = _gather_launch(grad_out_c, rows)
        grad_w = (g_gather * fe).sum(dim=1)
        return grad_y, None, grad_fe, grad_w

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
            member = torch.zeros(T, E, dtype=torch.bool, device=x.device).scatter(1, sel, True)
            y = xf.new_zeros(T, D)
            for e in range(E):
                rows = member[:, e].nonzero(as_tuple=True)[0]
                h = _GatherRows.apply(xf, rows)
                gu = _MM.apply(h, w1s[e])
                act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
                fe = _MM.apply(act, w2s[e])
                wv = w_full.index_select(0, rows)[:, e]
                y = _ScatterScaleAdd.apply(y, rows, fe, wv)
        ctx.save_for_backward(xf, rwf, w1f, w2f, y)
        ctx.dtypes = (x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.detach().to(x.dtype)
    @staticmethod
    def backward(ctx, dy):
        xf, rwf, w1f, w2f, y = ctx.saved_tensors
        gx, grw, gw1, gw2 = torch.autograd.grad(y, (xf, rwf, w1f, w2f), dy.float())
        dx_t, drw_t, dw1_t, dw2_t = ctx.dtypes
        return (gx.to(dx_t), grw.to(drw_t), None, gw1.to(dw1_t), gw2.to(dw2_t), None)

def fused_moe(x: torch.Tensor, router_w: torch.Tensor, router_b: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, A: int) -> torch.Tensor:
    return _FusedMoe.apply(x, router_w, router_b, w1, w2, A)
