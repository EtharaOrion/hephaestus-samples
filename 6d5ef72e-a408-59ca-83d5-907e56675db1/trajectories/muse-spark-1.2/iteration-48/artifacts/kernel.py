
import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32
EBLOCK = 2048

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
def _mm_split_kernel(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn, BLK_M: tl.constexpr, BLK_N: tl.constexpr, BLK_K: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M).to(tl.int64)
    rn = pid_n * BLK_N + tl.arange(0, BLK_N).to(tl.int64)
    acc = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
    for k0 in range(0, K, BLK_K):
        rk = k0 + tl.arange(0, BLK_K).to(tl.int64)
        a_f = tl.load(A + rm[:, None] * sam + rk[None, :] * sak, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b_f = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        a_hi = a_f.to(tl.bfloat16)
        b_hi = b_f.to(tl.bfloat16)
        a_lo = (a_f - a_hi.to(tl.float32)).to(tl.bfloat16)
        b_lo = (b_f - b_hi.to(tl.float32)).to(tl.bfloat16)
        acc += tl.dot(a_hi, b_hi)
        acc += tl.dot(a_hi, b_lo)
        acc += tl.dot(a_lo, b_hi)
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm[:, None] < M) & (rn[:, None] < N))

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
def _gather_kernel(X, IDX, Y, D, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < N * D
    row = offs // D
    col = offs % D
    idx = tl.load(IDX + row, mask=row < N, other=0).to(tl.int64)
    val = tl.load(X + idx * D + col, mask=mask, other=0.0)
    tl.store(Y + offs, val, mask=mask)

@triton.jit
def _scatter_add_kernel(Y, IDX, SRC, D, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < N * D
    row = offs // D
    col = offs % D
    idx = tl.load(IDX + row, mask=row < N, other=0).to(tl.int64)
    src = tl.load(SRC + offs, mask=mask, other=0.0)
    y_ptr = Y + idx * D + col
    cur = tl.load(y_ptr, mask=mask, other=0.0)
    tl.store(y_ptr, cur + src, mask=mask)

def _mm_launch(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    _mm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK, num_warps=4, num_stages=3)
    return c

def _mm_launch_split(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    _mm_split_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK, num_warps=4, num_stages=3)
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

class _MMSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _mm_launch_split(a, b)
    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        dc = dc.contiguous()
        return _mm_launch_split(dc, b.t()), _mm_launch_split(a.t(), dc)

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
    def forward(ctx, x, idx):
        ctx.save_for_backward(idx)
        ctx.T = x.shape[0]
        ctx.D = x.shape[1]
        N = idx.shape[0]
        y = torch.empty(N, ctx.D, device=x.device, dtype=x.dtype)
        if N > 0:
            BLOCK = 2048
            _gather_kernel[(triton.cdiv(N * ctx.D, BLOCK),)](x, idx, y, ctx.D, N, BLOCK=BLOCK)
        return y
    @staticmethod
    def backward(ctx, grad_y):
        idx, = ctx.saved_tensors
        T = ctx.T
        D = ctx.D
        grad_x = torch.zeros(T, D, device=grad_y.device, dtype=grad_y.dtype)
        N = idx.shape[0]
        if N > 0:
            BLOCK = 2048
            _scatter_add_kernel[(triton.cdiv(N * D, BLOCK),)](grad_x, idx, grad_y, D, N, BLOCK=BLOCK)
        return grad_x, None

class _ScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, idx, src):
        ctx.save_for_backward(idx)
        ctx.D = y.shape[1]
        N = idx.shape[0]
        y_next = y.clone()
        if N > 0:
            BLOCK = 2048
            _scatter_add_kernel[(triton.cdiv(N * ctx.D, BLOCK),)](y_next, idx, src, ctx.D, N, BLOCK=BLOCK)
        return y_next
    @staticmethod
    def backward(ctx, grad_y):
        idx, = ctx.saved_tensors
        D = ctx.D
        N = idx.shape[0]
        grad_src = torch.empty(N, D, device=grad_y.device, dtype=grad_y.dtype)
        if N > 0:
            BLOCK = 2048
            _gather_kernel[(triton.cdiv(N * D, BLOCK),)](grad_y, idx, grad_src, D, N, BLOCK=BLOCK)
        return grad_y, None, grad_src

def _select_topA(s, router_b, A):
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
            w_sorted = w.reshape(-1)[order]
            tok_sorted = (order // A).to(torch.int64)
            cnt = torch.bincount(flat, minlength=E)
            off = torch.cat([torch.zeros(1, device=cnt.device, dtype=torch.int64), cnt.cumsum(0)])
            y = xf.new_zeros(T, D)
            for e in range(E):
                s_idx = int(off[e].item())
                e_idx = int(off[e+1].item())
                idx_seg = tok_sorted[s_idx:e_idx]
                h = _Gather.apply(xf, idx_seg)
                gu = _MM.apply(h, w1s[e])
                act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
                fe = _MM.apply(act, w2s[e])
                ws = w_sorted[s_idx:e_idx].unsqueeze(1)
                fe_w = fe * ws
                y = _ScatterAdd.apply(y, idx_seg, fe_w)
        ctx.save_for_backward(xf, rwf, w1f, w2f, y)
        ctx.dtypes = (x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.detach().to(x.dtype)
    @staticmethod
    def backward(ctx, dy):
        xf, rwf, w1f, w2f, y = ctx.saved_tensors
        gx, grw, gw1, gw2 = torch.autograd.grad(y, (xf, rwf, w1f, w2f), dy.float())
        dx_t, drw_t, dw1_t, dw2_t = ctx.dtypes
        return gx.to(dx_t), grw.to(drw_t), None, gw1.to(dw1_t), gw2.to(dw2_t), None

def fused_moe(x, router_w, router_b, w1, w2, A):
    return _FusedMoe.apply(x, router_w, router_b, w1, w2, A)
