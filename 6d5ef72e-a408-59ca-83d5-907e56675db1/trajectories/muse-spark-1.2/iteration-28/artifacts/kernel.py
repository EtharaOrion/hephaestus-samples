import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32
EBLOCK = 2048
GBLOCK = 1024

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
def _gather_kernel(X, IDX, OUT, N, D, sx0, sx1, so0, so1, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= N:
        return
    idx = tl.load(IDX + pid).to(tl.int64)
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK).to(tl.int64)
        mask = cols < D
        v = tl.load(X + idx * sx0 + cols * sx1, mask=mask, other=0.0)
        tl.store(OUT + pid * so0 + cols * so1, v, mask=mask)

@triton.jit
def _scatter_kernel(SRC, IDX, OUT, N, D, ss0, ss1, so0, so1, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= N:
        return
    idx = tl.load(IDX + pid).to(tl.int64)
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK).to(tl.int64)
        mask = cols < D
        v = tl.load(SRC + pid * ss0 + cols * ss1, mask=mask, other=0.0)
        y_ptr = OUT + idx * so0 + cols * so1
        cur = tl.load(y_ptr, mask=mask, other=0.0)
        tl.store(y_ptr, cur + v, mask=mask)

@triton.jit
def _gn_fwd_kernel(S, SEL, OUT, T, E, A, s0, s1, sel0, sel1, out0, out1, BLOCK_A: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return
    offs = tl.arange(0, BLOCK_A).to(tl.int64)
    mask = offs < A
    e_idx = tl.load(SEL + pid * sel0 + offs * sel1, mask=mask, other=0)
    e_idx = e_idx.to(tl.int64)
    s_vals = tl.load(S + pid * s0 + e_idx * s1, mask=mask, other=0.0)
    s_sum = tl.sum(s_vals, axis=0)
    w = s_vals / s_sum
    tl.store(OUT + pid * out0 + offs * out1, w, mask=mask)

@triton.jit
def _gn_bwd_kernel(S, SEL, GOUT, GIN, T, E, A, s0, s1, sel0, sel1, gout0, gout1, gin0, gin1, BLOCK_A: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return
    offs = tl.arange(0, BLOCK_A).to(tl.int64)
    mask = offs < A
    e_idx = tl.load(SEL + pid * sel0 + offs * sel1, mask=mask, other=0)
    e_idx = e_idx.to(tl.int64)
    s_vals = tl.load(S + pid * s0 + e_idx * s1, mask=mask, other=0.0)
    g_vals = tl.load(GOUT + pid * gout0 + offs * gout1, mask=mask, other=0.0)
    s_sum = tl.sum(s_vals, axis=0)
    dot = tl.sum(s_vals * g_vals, axis=0)
    inv = 1.0 / s_sum
    inv2 = inv * inv
    gs_sel = (g_vals * s_sum - dot) * inv2
    tl.store(GIN + pid * gin0 + e_idx * gin1, gs_sel, mask=mask)

def _mm_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    _mm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK)
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

class _GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx):
        ctx.save_for_backward(idx)
        ctx.x_shape = x.shape
        N = idx.shape[0]
        D = x.shape[1]
        out = torch.empty(N, D, device=x.device, dtype=x.dtype)
        if N == 0:
            return out
        BLOCK = GBLOCK if D >= GBLOCK else triton.next_power_of_2(D)
        if BLOCK > 1024:
            BLOCK = 1024
        _gather_kernel[(N,)](x, idx, out, N, D, x.stride(0), x.stride(1), out.stride(0), out.stride(1), BLOCK=BLOCK)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        idx = ctx.saved_tensors[0]
        T, D = ctx.x_shape
        grad_out = grad_out.contiguous()
        N = idx.shape[0]
        grad_in = torch.zeros(T, D, device=grad_out.device, dtype=grad_out.dtype)
        if N == 0:
            return grad_in, None
        BLOCK = GBLOCK if D >= GBLOCK else triton.next_power_of_2(D)
        if BLOCK > 1024:
            BLOCK = 1024
        _scatter_kernel[(N,)](grad_out, idx, grad_in, N, D, grad_out.stride(0), grad_out.stride(1), grad_in.stride(0), grad_in.stride(1), BLOCK=BLOCK)
        return grad_in, None

class _ScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx, T):
        ctx.save_for_backward(idx)
        ctx.T = T
        D = src.shape[1]
        out = torch.zeros(T, D, device=src.device, dtype=src.dtype)
        N = src.shape[0]
        if N == 0:
            return out
        BLOCK = GBLOCK if D >= GBLOCK else triton.next_power_of_2(D)
        if BLOCK > 1024:
            BLOCK = 1024
        _scatter_kernel[(N,)](src, idx, out, N, D, src.stride(0), src.stride(1), out.stride(0), out.stride(1), BLOCK=BLOCK)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        idx = ctx.saved_tensors[0]
        T = ctx.T
        grad_out = grad_out.contiguous()
        N = idx.shape[0]
        D = grad_out.shape[1]
        grad_src = torch.empty(N, D, device=grad_out.device, dtype=grad_out.dtype)
        if N == 0:
            return grad_src, None, None
        BLOCK = GBLOCK if D >= GBLOCK else triton.next_power_of_2(D)
        if BLOCK > 1024:
            BLOCK = 1024
        _gather_kernel[(N,)](grad_out, idx, grad_src, N, D, grad_out.stride(0), grad_out.stride(1), grad_src.stride(0), grad_src.stride(1), BLOCK=BLOCK)
        return grad_src, None, None

class _GatherNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, sel):
        T, E = s.shape
        A = sel.shape[1]
        ctx.save_for_backward(s, sel)
        out = torch.empty(T, A, device=s.device, dtype=s.dtype)
        BLOCK_A = triton.next_power_of_2(A)
        if BLOCK_A < 32:
            BLOCK_A = 32
        _gn_fwd_kernel[(T,)](s, sel, out, T, E, A, s.stride(0), s.stride(1), sel.stride(0), sel.stride(1), out.stride(0), out.stride(1), BLOCK_A=BLOCK_A)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        s, sel = ctx.saved_tensors
        T, E = s.shape
        A = sel.shape[1]
        grad_out = grad_out.contiguous()
        grad_in = torch.zeros_like(s)
        BLOCK_A = triton.next_power_of_2(A)
        if BLOCK_A < 32:
            BLOCK_A = 32
        _gn_bwd_kernel[(T,)](s, sel, grad_out, grad_in, T, E, A, s.stride(0), s.stride(1), sel.stride(0), sel.stride(1), grad_out.stride(0), grad_out.stride(1), grad_in.stride(0), grad_in.stride(1), BLOCK_A=BLOCK_A)
        return grad_in, None

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
            w = _GatherNorm.apply(s, sel)
            flat = sel.reshape(-1)
            order = torch.argsort(flat)
            tok = torch.div(order, A, rounding_mode="floor")
            counts = torch.bincount(flat, minlength=E)
            offs = counts.cumsum(0)
            w_sorted = w.reshape(-1).index_select(0, order)
            y = xf.new_zeros(T, D)
            for e in range(E):
                start = int(offs[e-1].item()) if e > 0 else 0
                end = int(offs[e].item())
                if start == end:
                    continue
                rows = tok[start:end]
                h = _GatherRows.apply(xf, rows)
                gu = _MM.apply(h, w1s[e])
                act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
                fe = _MM.apply(act, w2s[e])
                ws = w_sorted[start:end].unsqueeze(1)
                scaled = fe * ws
                contrib = _ScatterAdd.apply(scaled, rows, T)
                y = y + contrib
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
