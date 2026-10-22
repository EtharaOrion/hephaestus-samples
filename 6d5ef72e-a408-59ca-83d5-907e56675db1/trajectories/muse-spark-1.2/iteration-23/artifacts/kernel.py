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
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

@triton.jit
def _mm_precise_kernel(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn, BLK_M: tl.constexpr, BLK_N: tl.constexpr, BLK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M)
    rn = pid_n * BLK_N + tl.arange(0, BLK_N)
    acc = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
    for k0 in range(0, K, BLK_K):
        rk = k0 + tl.arange(0, BLK_K)
        a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        a_hi = a.to(tl.bfloat16)
        b_hi = b.to(tl.bfloat16)
        a_lo = (a - a_hi.to(tl.float32)).to(tl.bfloat16)
        b_lo = (b - b_hi.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(a_hi, b_hi, acc)
        acc = tl.dot(a_hi, b_lo, acc)
        acc = tl.dot(a_lo, b_hi, acc)
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
def _gather_kernel(SRC, IDX, DST, N, D, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N * D
    rows = offs // D
    cols = offs % D
    idx = tl.load(IDX + rows, mask=mask)
    vals = tl.load(SRC + idx * D + cols, mask=mask)
    tl.store(DST + offs, vals, mask=mask)

@triton.jit
def _scatter_add_kernel(SRC, IDX, DST, N, D, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N * D
    rows = offs // D
    cols = offs % D
    idx = tl.load(IDX + rows, mask=mask)
    vals = tl.load(SRC + offs, mask=mask)
    ptr = DST + idx * D + cols
    cur = tl.load(ptr, mask=mask)
    tl.store(ptr, cur + vals, mask=mask)

@triton.jit
def _gather1d_kernel(SRC, IDX, DST, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    idx = tl.load(IDX + offs, mask=mask)
    vals = tl.load(SRC + idx, mask=mask)
    tl.store(DST + offs, vals, mask=mask)

@triton.jit
def _scatter1d_kernel(SRC, IDX, DST, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    idx = tl.load(IDX + offs, mask=mask)
    vals = tl.load(SRC + offs, mask=mask)
    cur = tl.load(DST + idx, mask=mask)
    tl.store(DST + idx, cur + vals, mask=mask)

@triton.jit
def _tok_kernel(ORDER, TOK, N, A, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    ord = tl.load(ORDER + offs, mask=mask)
    tl.store(TOK + offs, ord // A, mask=mask)

def _mm_launch(a: torch.Tensor, b: torch.Tensor, precise: bool = False) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    kern = _mm_precise_kernel if precise else _mm_kernel
    kern[(triton.cdiv(M, BM), triton.cdiv(N, BN))](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK)
    return c

class _MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _mm_launch(a, b, precise=False)
    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        dc = dc.contiguous()
        return _mm_launch(dc, b.t(), precise=True), _mm_launch(a.t(), dc, precise=True)

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
    def forward(ctx, src, idx):
        ctx.save_for_backward(idx)
        ctx.T = src.shape[0]
        ctx.D = src.shape[1]
        n = idx.shape[0]
        D = ctx.D
        dst = torch.empty(n, D, device=src.device, dtype=src.dtype)
        BLOCK = 1024
        grid = (triton.cdiv(n * D, BLOCK),)
        if grid[0] > 0:
            _gather_kernel[grid](src, idx, dst, n, D, BLOCK=BLOCK)
        return dst
    @staticmethod
    def backward(ctx, grad_dst):
        idx, = ctx.saved_tensors
        T = ctx.T
        D = ctx.D
        n = idx.shape[0]
        grad_src = torch.zeros(T, D, device=grad_dst.device, dtype=grad_dst.dtype)
        BLOCK = 1024
        grid = (triton.cdiv(n * D, BLOCK),)
        if grid[0] > 0:
            _scatter_add_kernel[grid](grad_dst, idx, grad_src, n, D, BLOCK=BLOCK)
        return grad_src, None

class _Gather1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx):
        ctx.save_for_backward(idx)
        ctx.Nsrc = src.shape[0]
        n = idx.shape[0]
        dst = torch.empty(n, device=src.device, dtype=src.dtype)
        BLOCK = 2048
        grid = (triton.cdiv(n, BLOCK),)
        if grid[0] > 0:
            _gather1d_kernel[grid](src, idx, dst, n, BLOCK=BLOCK)
        return dst
    @staticmethod
    def backward(ctx, grad_dst):
        idx, = ctx.saved_tensors
        Nsrc = ctx.Nsrc
        grad_src = torch.zeros(Nsrc, device=grad_dst.device, dtype=grad_dst.dtype)
        n = idx.shape[0]
        BLOCK = 2048
        grid = (triton.cdiv(n, BLOCK),)
        if grid[0] > 0:
            _scatter1d_kernel[grid](grad_dst, idx, grad_src, n, BLOCK=BLOCK)
        return grad_src, None

class _ScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dst, idx, src):
        ctx.save_for_backward(idx)
        ctx.n = src.shape[0]
        ctx.D = src.shape[1]
        out = dst.clone()
        n = ctx.n
        D = ctx.D
        BLOCK = 1024
        grid = (triton.cdiv(n * D, BLOCK),)
        if grid[0] > 0:
            _scatter_add_kernel[grid](src, idx, out, n, D, BLOCK=BLOCK)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        idx, = ctx.saved_tensors
        n = ctx.n
        D = ctx.D
        grad_dst = grad_out.clone()
        grad_src = torch.empty(n, D, device=grad_out.device, dtype=grad_out.dtype)
        BLOCK = 1024
        grid = (triton.cdiv(n * D, BLOCK),)
        if grid[0] > 0:
            _gather_kernel[grid](grad_out, idx, grad_src, n, D, BLOCK=BLOCK)
        return grad_dst, None, grad_src

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
        Aint = int(A)
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
                sel = _select_topA(s.detach(), router_b.float(), Aint)
            s_sel = torch.gather(s, 1, sel)
            w = s_sel / s_sel.sum(dim=-1, keepdim=True)
            flat = sel.reshape(-1)
            order = torch.argsort(flat)
            N = flat.shape[0]
            tok = torch.empty(N, device=x.device, dtype=torch.int64)
            BLOCK_T = 2048
            grid_t = (triton.cdiv(N, BLOCK_T),)
            _tok_kernel[grid_t](order, tok, N, Aint, BLOCK=BLOCK_T)
            w_flat = w.reshape(-1)
            w_sorted = _Gather1D.apply(w_flat, order)
            cnt = torch.bincount(flat, minlength=E)
            off = torch.cumsum(cnt, dim=0)
            y = xf.new_zeros(T, D)
            for e in range(E):
                ce = int(cnt[e].item())
                oe = int(off[e].item())
                start = oe - ce
                rows = tok[start: start + ce]
                ws = w_sorted[start: start + ce]
                h = _Gather.apply(xf, rows)
                gu = _MM.apply(h, w1s[e])
                g = gu[:, :F].contiguous()
                u = gu[:, F:].contiguous()
                act = _SiLUMul.apply(g, u)
                fe = _MM.apply(act, w2s[e])
                fe_w = fe * ws.unsqueeze(1)
                y = _ScatterAdd.apply(y, rows, fe_w)
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
