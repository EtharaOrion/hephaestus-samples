import torch
import triton
import triton.language as tl
torch.use_deterministic_algorithms(True)

BM, BN, BK = 64, 128, 32
EBLOCK = 1024

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
        a_hi = a.to(tl.bfloat16)
        b_hi = b.to(tl.bfloat16)
        a_hi_f = a_hi.to(tl.float32)
        b_hi_f = b_hi.to(tl.float32)
        a_lo = (a - a_hi_f).to(tl.bfloat16)
        b_lo = (b - b_hi_f).to(tl.bfloat16)
        acc += tl.dot(a_hi, b_hi)
        acc += tl.dot(a_hi, b_lo)
        acc += tl.dot(a_lo, b_hi)
        acc += tl.dot(a_lo, b_lo)
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
def _gather_kernel(src, idx, dst, N, D, s_s0, s_s1, s_d0, s_d1, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_n = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    m = (rows[:, None] < N) & (cols[None, :] < D)
    rr = tl.load(idx + rows, mask=rows < N, other=0).to(tl.int64)
    v = tl.load(src + rr[:, None] * s_s0 + cols[None, :] * s_s1, mask=m, other=0.0)
    tl.store(dst + rows[:, None] * s_d0 + cols[None, :] * s_d1, v, mask=m)

@triton.jit
def _scale_kernel(inp, ws, out, N, D, s_i0, s_i1, s_o0, s_o1, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_n = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    m = (rows[:, None] < N) & (cols[None, :] < D)
    v = tl.load(inp + rows[:, None] * s_i0 + cols[None, :] * s_i1, mask=m, other=0.0)
    w = tl.load(ws + rows, mask=rows < N, other=0.0)
    tl.store(out + rows[:, None] * s_o0 + cols[None, :] * s_o1, v * w[:, None], mask=m)









@triton.jit
def _gmm_kernel(A, B, C, block_expert, block_m, block_n, offs, N, K, Nout, s_am, s_ak, s_be, s_bk, s_bn, s_cm, s_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    e = tl.load(block_expert + pid).to(tl.int64)
    m_idx = tl.load(block_m + pid).to(tl.int64)
    n_idx = tl.load(block_n + pid).to(tl.int64)
    off = tl.load(offs + e).to(tl.int64)
    off_next = tl.load(offs + e + 1).to(tl.int64)
    row_start = off + m_idx * BLOCK_M
    col_start = n_idx * BLOCK_N
    rm = row_start + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = col_start + tl.arange(0, BLOCK_N).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
        a = tl.load(A + rm[:, None] * s_am + rk[None, :] * s_ak, mask=(rm[:, None] < off_next) & (rm[:, None] >= off) & (rk[None, :] < K), other=0.0)
        b = tl.load(B + e * s_be + rk[:, None] * s_bk + rn[None, :] * s_bn, mask=(rk[:, None] < K) & (rn[None, :] < Nout), other=0.0)
        a_hi = a.to(tl.bfloat16)
        b_hi = b.to(tl.bfloat16)
        a_hi_f = a_hi.to(tl.float32)
        b_hi_f = b_hi.to(tl.float32)
        a_lo = (a - a_hi_f).to(tl.bfloat16)
        b_lo = (b - b_hi_f).to(tl.bfloat16)
        acc += tl.dot(a_hi, b_hi)
        acc += tl.dot(a_hi, b_lo)
        acc += tl.dot(a_lo, b_hi)
        acc += tl.dot(a_lo, b_lo)
    tl.store(C + rm[:, None] * s_cm + rn[None, :] * s_cn, acc, mask=(rm[:, None] < off_next) & (rm[:, None] >= off) & (rn[None, :] < Nout))

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



class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx):
        ctx.save_for_backward(idx)
        ctx.src_shape = src.shape
        N = idx.numel()
        D = src.shape[1]
        dst = torch.empty(N, D, device=src.device, dtype=src.dtype)
        _gather_kernel[(triton.cdiv(N, 32), triton.cdiv(D, 64))](src, idx, dst, N, D, src.stride(0), src.stride(1), dst.stride(0), dst.stride(1), BLOCK_N=32, BLOCK_D=64)
        return dst
    @staticmethod
    def backward(ctx, grad_out):
        idx, = ctx.saved_tensors
        T, D = ctx.src_shape
        grad_in = torch.zeros(T, D, device=grad_out.device, dtype=grad_out.dtype)
        grad_in = grad_in.index_add(0, idx, grad_out)
        return grad_in, None

class _Scale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, ws):
        ctx.save_for_backward(inp, ws)
        N, D = inp.shape
        out = torch.empty(N, D, device=inp.device, dtype=inp.dtype)
        _scale_kernel[(triton.cdiv(N, 32), triton.cdiv(D, 64))](inp, ws, out, N, D, inp.stride(0), inp.stride(1), out.stride(0), out.stride(1), BLOCK_N=32, BLOCK_D=64)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        inp, ws = ctx.saved_tensors
        grad_inp = grad_out * ws.unsqueeze(1)
        grad_ws = (grad_out * inp).sum(dim=1)
        return grad_inp, grad_ws







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

def _select_topA(s: torch.Tensor, router_b: torch.Tensor, A: int) -> torch.Tensor:
    E = s.shape[-1]
    b = s + router_b
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=s.device, dtype=torch.int64)
    keys = (u << 10) | (E - 1 - e_idx)
    return torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices

def _gmm(a_sorted, w, offs):
    if a_sorted.shape[0] == 0:
        return torch.empty(a_sorted.shape[0], w.shape[2], device=a_sorted.device, dtype=torch.float32)
    N, K = a_sorted.shape
    E, K2, Nout = w.shape
    assert K == K2
    offs_cpu = offs.cpu().tolist()
    block_expert = []
    block_m = []
    block_n = []
    for e in range(E):
        n_e = int(offs_cpu[e+1] - offs_cpu[e])
        if n_e == 0:
            continue
        bm = (n_e + BM - 1)//BM
        bn = (Nout + BN - 1)//BN
        for im in range(bm):
            for jn in range(bn):
                block_expert.append(e)
                block_m.append(im)
                block_n.append(jn)
    if len(block_expert) == 0:
        return torch.empty(N, Nout, device=a_sorted.device, dtype=torch.float32)
    be_t = torch.tensor(block_expert, device=a_sorted.device, dtype=torch.int32)
    bm_t = torch.tensor(block_m, device=a_sorted.device, dtype=torch.int32)
    bn_t = torch.tensor(block_n, device=a_sorted.device, dtype=torch.int32)
    c = torch.empty(N, Nout, device=a_sorted.device, dtype=torch.float32)
    _gmm_kernel[(len(block_expert),)](a_sorted, w, c, be_t, bm_t, bn_t, offs, N, K, Nout, a_sorted.stride(0), a_sorted.stride(1), w.stride(0), w.stride(1), w.stride(2), c.stride(0), c.stride(1), BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    return c

class _GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_sorted, w, offs):
        ctx.save_for_backward(a_sorted, w, offs)
        return _gmm(a_sorted, w, offs)
    @staticmethod
    def backward(ctx, dC):
        a_sorted, w, offs = ctx.saved_tensors
        dC = dC.contiguous()
        offs_cpu = offs.cpu().tolist()
        E = w.shape[0]
        dA = torch.empty_like(a_sorted)
        dB = torch.zeros_like(w)
        for e in range(E):
            s = int(offs_cpu[e]); e2 = int(offs_cpu[e+1]); n_e = e2 - s
            if n_e == 0:
                continue
            a_e = a_sorted[s:e2]
            dc_e = dC[s:e2]
            dA[s:e2] = _mm_launch(dc_e, w[e].t())
            dB[e] = _mm_launch(a_e.t(), dc_e)
        return dA, dB, None

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
            with torch.no_grad():
                sel = _select_topA(s.detach(), router_b.float(), A)
            s_sel = torch.gather(s, 1, sel)
            w = s_sel / s_sel.sum(dim=-1, keepdim=True)
            flat = sel.reshape(-1)
            order = torch.argsort(flat, stable=True)
            tok = torch.div(order, A, rounding_mode="floor")
            counts = torch.bincount(flat, minlength=E)
            offs = torch.zeros(E+1, device=x.device, dtype=torch.int32)
            offs[1:] = counts.cumsum(0).to(torch.int32)
            N = tok.numel()
            xg = _Gather.apply(xf, tok)
            h = _GroupedMM.apply(xg, w1f, offs)
            act = _SiLUMul.apply(h[:, :F].contiguous(), h[:, F:].contiguous())
            yp = _GroupedMM.apply(act, w2f, offs)
            ws = w.reshape(-1).index_select(0, order).to(torch.float32)
            yp_w = _Scale.apply(yp, ws)
            y = xf.new_zeros(T, D)
            y = y.index_add(0, tok, yp_w)
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
