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
def _mm_hilo_kernel(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn, BLK_M: tl.constexpr, BLK_N: tl.constexpr, BLK_K: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M).to(tl.int64)
    rn = pid_n * BLK_N + tl.arange(0, BLK_N).to(tl.int64)
    acc = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
    for k0 in range(0, K, BLK_K):
        rk = k0 + tl.arange(0, BLK_K).to(tl.int64)
        a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        a_hi = a.to(tl.bfloat16)
        b_hi = b.to(tl.bfloat16)
        a_lo = (a - a_hi.to(tl.float32)).to(tl.bfloat16)
        b_lo = (b - b_hi.to(tl.float32)).to(tl.bfloat16)
        acc += tl.dot(a_hi, b_hi)
        acc += tl.dot(a_hi, b_lo)
        acc += tl.dot(a_lo, b_hi)
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
def _gather_kernel(SRC, DST, IDX, D: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    row = tl.load(IDX + pid_m).to(tl.int64)
    offs = pid_n * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    mask = offs < D
    v = tl.load(SRC + row * D + offs, mask=mask, other=0.0)
    tl.store(DST + pid_m * D + offs, v, mask=mask)

@triton.jit
def _gather_bwd_kernel(GRAD_SRC, GRAD_DST, IDX, D: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    row = tl.load(IDX + pid_m).to(tl.int64)
    offs = pid_n * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    mask = offs < D
    g = tl.load(GRAD_DST + pid_m * D + offs, mask=mask, other=0.0)
    cur = tl.load(GRAD_SRC + row * D + offs, mask=mask, other=0.0)
    tl.store(GRAD_SRC + row * D + offs, cur + g, mask=mask)

@triton.jit
def _weight_norm_kernel(S, SEL, W, T: tl.constexpr, E: tl.constexpr, A: tl.constexpr, BA: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BA).to(tl.int64)
    mask = offs < A
    sel = tl.load(SEL + pid * A + offs, mask=mask, other=0)
    svals = tl.load(S + pid * E + sel, mask=mask, other=0.0).to(tl.float32)
    ssum = tl.sum(svals, axis=0)
    w = svals / ssum
    tl.store(W + pid * A + offs, w, mask=mask)

@triton.jit
def _weight_ds_kernel(DW, S, SEL, DS_SEL, T: tl.constexpr, E: tl.constexpr, A: tl.constexpr, BA: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BA).to(tl.int64)
    mask = offs < A
    sel = tl.load(SEL + pid * A + offs, mask=mask, other=0)
    svals = tl.load(S + pid * E + sel, mask=mask, other=0.0).to(tl.float32)
    dw = tl.load(DW + pid * A + offs, mask=mask, other=0.0).to(tl.float32)
    ssum = tl.sum(svals, axis=0)
    dot = tl.sum(dw * svals, axis=0)
    ds = (dw * ssum - dot) / (ssum * ssum)
    tl.store(DS_SEL + pid * A + offs, ds, mask=mask)

@triton.jit
def _weight_scatter_kernel(DS_SEL, SEL, GRAD_S, T: tl.constexpr, E: tl.constexpr, A: tl.constexpr, BA: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    for i in range(BA):
        if i < A:
            s_idx = tl.load(SEL + pid * A + i).to(tl.int64)
            v = tl.load(DS_SEL + pid * A + i)
            tl.store(GRAD_S + pid * E + s_idx, v)

@triton.jit
def _scale_kernel(FE, W, OUT, D: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    w = tl.load(W + pid_m).to(tl.float32)
    offs = pid_n * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    mask = offs < D
    fe = tl.load(FE + pid_m * D + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + pid_m * D + offs, fe * w, mask=mask)

@triton.jit
def _scale_bwd_fe_kernel(GO, GFE, W, D: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    w = tl.load(W + pid_m).to(tl.float32)
    offs = pid_n * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    mask = offs < D
    go = tl.load(GO + pid_m * D + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(GFE + pid_m * D + offs, go * w, mask=mask)

@triton.jit
def _scale_bwd_w_kernel(GO, FE, GW, D: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    acc = 0.0
    offs = tl.arange(0, BLOCK_D).to(tl.int64)
    for bo in range(0, D, BLOCK_D):
        cur = bo + offs
        mask = cur < D
        go = tl.load(GO + pid * D + cur, mask=mask, other=0.0).to(tl.float32)
        fe = tl.load(FE + pid * D + cur, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(go * fe, axis=0)
    tl.store(GW + pid, acc)

@triton.jit
def _accum_kernel(Y, TMP, IDX, D: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    row = tl.load(IDX + pid_m).to(tl.int64)
    offs = pid_n * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    mask = offs < D
    t = tl.load(TMP + pid_m * D + offs, mask=mask, other=0.0).to(tl.float32)
    cur = tl.load(Y + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * D + offs, cur + t, mask=mask)

def _mm_launch(a: torch.Tensor, b: torch.Tensor, hilo: bool = False) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    if hilo:
        _mm_hilo_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK, num_warps=4, num_stages=3)
    else:
        _mm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK, num_warps=4, num_stages=3)
    return c

class _MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _mm_launch(a, b, hilo=False)
    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        dc = dc.contiguous()
        return _mm_launch(dc, b.t(), hilo=False), _mm_launch(a.t(), dc, hilo=True)

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
        ctx.src_shape = src.shape
        n = idx.numel()
        D = src.shape[1]
        if n == 0:
            return torch.empty(0, D, device=src.device, dtype=src.dtype)
        dst = torch.empty(n, D, device=src.device, dtype=src.dtype)
        BD = 1024
        _gather_kernel[(n, triton.cdiv(D, BD))](src, dst, idx, D, BLOCK_D=BD)
        return dst
    @staticmethod
    def backward(ctx, grad_out):
        (idx,) = ctx.saved_tensors
        src_shape = ctx.src_shape
        grad_src = torch.zeros(src_shape, device=grad_out.device, dtype=grad_out.dtype)
        n = idx.numel()
        D = src_shape[1]
        if n != 0:
            BD = 1024
            _gather_bwd_kernel[(n, triton.cdiv(D, BD))](grad_src, grad_out.contiguous(), idx, D, BLOCK_D=BD)
        return grad_src, None

class _Weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, sel):
        ctx.save_for_backward(s, sel)
        T, E = s.shape
        A = sel.shape[1]
        w = torch.empty(T, A, device=s.device, dtype=torch.float32)
        BA = triton.next_power_of_2(A)
        if BA < 16:
            BA = 16
        _weight_norm_kernel[(T,)](s, sel, w, T, E, A, BA)
        return w
    @staticmethod
    def backward(ctx, grad_w):
        s, sel = ctx.saved_tensors
        T, E = s.shape
        A = sel.shape[1]
        BA = triton.next_power_of_2(A)
        if BA < 16:
            BA = 16
        ds_sel = torch.empty(T, A, device=s.device, dtype=torch.float32)
        _weight_ds_kernel[(T,)](grad_w.contiguous(), s, sel, ds_sel, T, E, A, BA)
        grad_s = torch.zeros_like(s)
        _weight_scatter_kernel[(T,)](ds_sel, sel, grad_s, T, E, A, BA)
        return grad_s, None

class _Scale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fe, w):
        ctx.save_for_backward(fe, w)
        n, D = fe.shape
        out = torch.empty_like(fe)
        BD = 1024
        _scale_kernel[(n, triton.cdiv(D, BD))](fe.contiguous(), w.contiguous(), out, D, BLOCK_D=BD)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        fe, w = ctx.saved_tensors
        n, D = grad_out.shape
        BD = 1024
        grad_fe = torch.empty_like(grad_out)
        _scale_bwd_fe_kernel[(n, triton.cdiv(D, BD))](grad_out.contiguous(), grad_fe, w.contiguous(), D, BLOCK_D=BD)
        grad_w = torch.empty(n, device=fe.device, dtype=fe.dtype)
        _scale_bwd_w_kernel[(n,)](grad_out.contiguous(), fe.contiguous(), grad_w, D, BLOCK_D=BD)
        return grad_fe, grad_w

class _Accum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, tmp, idx):
        ctx.save_for_backward(idx)
        n = idx.numel()
        D = y.shape[1]
        out = y.clone()
        if n != 0:
            BD = 1024
            _accum_kernel[(n, triton.cdiv(D, BD))](out, tmp.contiguous(), idx, D, BLOCK_D=BD)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        (idx,) = ctx.saved_tensors
        n = idx.numel()
        D = grad_out.shape[1]
        grad_y = grad_out
        if n == 0:
            return grad_y, None, None
        grad_tmp = torch.empty(n, D, device=grad_out.device, dtype=grad_out.dtype)
        BD = 1024
        _gather_kernel[(n, triton.cdiv(D, BD))](grad_out.contiguous(), grad_tmp, idx, D, BLOCK_D=BD)
        return grad_y, grad_tmp, None

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
            w = _Weight.apply(s, sel)
            w_full = s.new_zeros(T, E).scatter(1, sel, w)
            member = torch.zeros(T, E, dtype=torch.bool, device=x.device).scatter(1, sel, True)
            y = xf.new_zeros(T, D)
            for e in range(E):
                rows = member[:, e].nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                h = _Gather.apply(xf, rows)
                gu = _MM.apply(h, w1s[e])
                act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
                fe = _MM.apply(act, w2s[e])
                wt = w_full.index_select(0, rows)[:, e]
                fe_w = _Scale.apply(fe, wt)
                y = _Accum.apply(y, fe_w, rows)
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
