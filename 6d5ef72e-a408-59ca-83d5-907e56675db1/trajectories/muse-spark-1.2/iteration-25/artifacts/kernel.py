import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32
EBLOCK = 2048
GBM, GBN = 32, 32
CBM, CBN = 32, 32

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
def _gather_kernel(SRC, DST, IDX, M, D, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < M
    mask_n = rn < D
    idx = tl.load(IDX + rm, mask=mask_m, other=0)
    vals = tl.load(SRC + idx[:, None] * D + rn[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    tl.store(DST + rm[:, None] * D + rn[None, :], vals, mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def _scatter_perm_kernel(SRC, DST, IDX, M, D, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < M
    mask_n = rn < D
    idx = tl.load(IDX + rm, mask=mask_m, other=0)
    vals = tl.load(SRC + rm[:, None] * D + rn[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    tl.store(DST + idx[:, None] * D + rn[None, :], vals, mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def _weight_mul_kernel(INP, W, OUT, M, D, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < M
    mask_n = rn < D
    w = tl.load(W + rm, mask=mask_m, other=0.0)
    vals = tl.load(INP + rm[:, None] * D + rn[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    tl.store(OUT + rm[:, None] * D + rn[None, :], vals * w[:, None], mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def _combine_sum_kernel(INP, OUT, T, A, D, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < T
    mask_n = rn < D
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for a in range(0, 32):
        if a < A:
            idx = rm * A + a
            vals = tl.load(INP + idx[:, None] * D + rn[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0.0)
            acc += vals
    tl.store(OUT + rm[:, None] * D + rn[None, :], acc, mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def _combine_sum_bwd_kernel(DOUT, DINP, T, A, D, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < T
    mask_n = rn < D
    g = tl.load(DOUT + rm[:, None] * D + rn[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    for a in range(0, 32):
        if a < A:
            idx = rm * A + a
            tl.store(DINP + idx[:, None] * D + rn[None, :], g, mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def _gather1d_kernel(SRC, DST, IDX, M, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    o = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < M
    idx = tl.load(IDX + o, mask=m, other=0)
    v = tl.load(SRC + idx, mask=m, other=0.0)
    tl.store(DST + o, v, mask=m)

@triton.jit
def _scatter1d_kernel(SRC, DST, IDX, M, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    o = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < M
    idx = tl.load(IDX + o, mask=m, other=0)
    v = tl.load(SRC + o, mask=m, other=0.0)
    tl.store(DST + idx, v, mask=m)

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
        if n != 0:
            _silu_mul_fwd_kernel[(triton.cdiv(n, EBLOCK),)](g, u, out, n, BLOCK=EBLOCK)
        return out
    @staticmethod
    def backward(ctx, dout):
        g, u = ctx.saved_tensors
        dout = dout.contiguous()
        dg, du = torch.empty_like(g), torch.empty_like(u)
        n = g.numel()
        if n != 0:
            _silu_mul_bwd_kernel[(triton.cdiv(n, EBLOCK),)](g, u, dout, dg, du, n, BLOCK=EBLOCK)
        return dg, du

class _Gather2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx):
        ctx.save_for_backward(idx)
        ctx.src_shape = src.shape
        M = idx.numel()
        D = src.shape[1]
        dst = torch.empty(M, D, device=src.device, dtype=src.dtype)
        if M != 0:
            _gather_kernel[(triton.cdiv(M, GBM), triton.cdiv(D, GBN))](src, dst, idx, M, D, BLOCK_M=GBM, BLOCK_N=GBN)
        return dst
    @staticmethod
    def backward(ctx, grad_dst):
        idx, = ctx.saved_tensors
        T, D = ctx.src_shape
        M = idx.numel()
        grad_src = torch.zeros(T, D, device=grad_dst.device, dtype=grad_dst.dtype)
        if M != 0:
            _scatter_perm_kernel[(triton.cdiv(M, GBM), triton.cdiv(D, GBN))](grad_dst, grad_src, idx, M, D, BLOCK_M=GBM, BLOCK_N=GBN)
        return grad_src, None

class _GatherExpand2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, tok_sorted, order, inv_order):
        ctx.save_for_backward(order, inv_order)
        ctx.src_shape = src.shape
        M = tok_sorted.numel()
        D = src.shape[1]
        dst = torch.empty(M, D, device=src.device, dtype=src.dtype)
        if M != 0:
            _gather_kernel[(triton.cdiv(M, GBM), triton.cdiv(D, GBN))](src, dst, tok_sorted, M, D, BLOCK_M=GBM, BLOCK_N=GBN)
        return dst
    @staticmethod
    def backward(ctx, grad_dst):
        order, inv_order = ctx.saved_tensors
        T, D = ctx.src_shape
        M = grad_dst.shape[0]
        A = M // T if T != 0 else 0
        tmp = torch.empty(M, D, device=grad_dst.device, dtype=grad_dst.dtype)
        if M != 0:
            _gather_kernel[(triton.cdiv(M, GBM), triton.cdiv(D, GBN))](grad_dst, tmp, inv_order, M, D, BLOCK_M=GBM, BLOCK_N=GBN)
        grad_src = torch.empty(T, D, device=grad_dst.device, dtype=grad_dst.dtype)
        if T != 0 and A != 0:
            _combine_sum_kernel[(triton.cdiv(T, CBM), triton.cdiv(D, CBN))](tmp, grad_src, T, A, D, BLOCK_M=CBM, BLOCK_N=CBN)
        else:
            grad_src.zero_()
        return grad_src, None, None, None

class _Gather1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx):
        ctx.save_for_backward(idx)
        ctx.src_len = src.numel()
        M = idx.numel()
        dst = torch.empty(M, device=src.device, dtype=src.dtype)
        if M != 0:
            _gather1d_kernel[(triton.cdiv(M, 1024),)](src, dst, idx, M, BLOCK=1024)
        return dst
    @staticmethod
    def backward(ctx, grad_dst):
        idx, = ctx.saved_tensors
        N = ctx.src_len
        M = idx.numel()
        grad_src = torch.zeros(N, device=grad_dst.device, dtype=grad_dst.dtype)
        if M != 0:
            _scatter1d_kernel[(triton.cdiv(M, 1024),)](grad_dst, grad_src, idx, M, BLOCK=1024)
        return grad_src, None

@triton.jit
def _weight_mul_wbwd_kernel(DOUT, INP, DW, M, D, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rm = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = rm < M
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(0, D, 32):
        rk = k + tl.arange(0, 32).to(tl.int64)
        m2 = rk < D
        dout = tl.load(DOUT + rm[:, None] * D + rk[None, :], mask=mask[:, None] & m2[None, :], other=0.0)
        inp = tl.load(INP + rm[:, None] * D + rk[None, :], mask=mask[:, None] & m2[None, :], other=0.0)
        acc += tl.sum(dout * inp, axis=1)
    tl.store(DW + rm, acc, mask=mask)

class _WeightMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, w):
        ctx.save_for_backward(inp, w)
        M, D = inp.shape
        out = torch.empty_like(inp)
        if M != 0:
            _weight_mul_kernel[(triton.cdiv(M, GBM), triton.cdiv(D, GBN))](inp, w, out, M, D, BLOCK_M=GBM, BLOCK_N=GBN)
        return out
    @staticmethod
    def backward(ctx, dout):
        inp, w = ctx.saved_tensors
        dout = dout.contiguous()
        M, D = dout.shape
        dinp = torch.empty_like(dout)
        if M != 0:
            _weight_mul_kernel[(triton.cdiv(M, GBM), triton.cdiv(D, GBN))](dout, w, dinp, M, D, BLOCK_M=GBM, BLOCK_N=GBN)
        dw = torch.empty(M, device=dout.device, dtype=dout.dtype)
        if M != 0:
            _weight_mul_wbwd_kernel[(triton.cdiv(M, 1024),)](dout, inp, dw, M, D, BLOCK=1024)
        return dinp, dw

class _CombineSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, A):
        ctx.A = A
        T = inp.shape[0] // A if A != 0 else 0
        D = inp.shape[1]
        out = torch.empty(T, D, device=inp.device, dtype=inp.dtype)
        if T != 0:
            _combine_sum_kernel[(triton.cdiv(T, CBM), triton.cdiv(D, CBN))](inp, out, T, A, D, BLOCK_M=CBM, BLOCK_N=CBN)
        return out
    @staticmethod
    def backward(ctx, dout):
        A = ctx.A
        T, D = dout.shape
        dinp = torch.empty(T * A, D, device=dout.device, dtype=dout.dtype)
        if T != 0:
            _combine_sum_bwd_kernel[(triton.cdiv(T, CBM), triton.cdiv(D, CBN))](dout, dinp, T, A, D, BLOCK_M=CBM, BLOCK_N=CBN)
        return dinp, None

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
            inv_order = torch.argsort(order)
            tok_sorted = torch.div(order, A, rounding_mode="floor")
            w_flat = w.reshape(-1)
            w_sorted = _Gather1D.apply(w_flat, order)
            x_sorted = _GatherExpand2D.apply(xf, tok_sorted, order, inv_order)
            sorted_flat = _Gather1D.apply(flat.to(torch.float32), order).to(torch.int64)
            # counts via bincount on sorted_flat is same as flat, use flat
            offs = torch.bincount(flat, minlength=E).cumsum(0)
            offs_cpu = offs.detach().cpu().tolist()
            starts = [0] + offs_cpu[:-1]
            ends = offs_cpu
            out_sorted = torch.empty(T * A, D, device=xf.device, dtype=torch.float32)
            for e in range(E):
                st = starts[e]
                en = ends[e]
                seg = x_sorted[st:en]
                gu = _MM.apply(seg, w1s[e])
                act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
                fe = _MM.apply(act, w2s[e])
                out_sorted[st:en] = fe
            out_weighted = _WeightMul.apply(out_sorted, w_sorted)
            out_perm = _Gather2D.apply(out_weighted, inv_order)
            y = _CombineSum.apply(out_perm, A)
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
