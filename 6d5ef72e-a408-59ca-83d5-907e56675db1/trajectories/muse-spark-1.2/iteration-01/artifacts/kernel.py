import torch
import triton
import triton.language as tl

# tuned tiles for H100
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 32
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
        acc = tl.dot(a, b, acc, input_precision="ieee")
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

# per-expert tiled matmul over sorted tokens
@triton.jit
def _gmm_fwd_kernel(X, W, Y, OFFS, K, Nout,
                    sxm, sxk, swk, swn, sym, syn,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    pid_e = tl.program_id(2).to(tl.int64)
    off = tl.load(OFFS + pid_e).to(tl.int64)
    prev = tl.where(pid_e > 0, tl.load(OFFS + pid_e - 1).to(tl.int64), 0)
    cnt = off - prev
    if pid_m * BLOCK_M >= cnt:
        return
    if pid_n * BLOCK_N >= Nout:
        return
    rm = prev + pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    # base of W for this expert: W is [E, K, Nout] contiguous
    w_base = W + pid_e * K * Nout
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K, BLOCK_K):
        rk = kk + tl.arange(0, BLOCK_K).to(tl.int64)
        a = tl.load(X + rm[:, None] * sxm + rk[None, :] * sxk,
                    mask=(rm[:, None] < off) & (rk[None, :] < K), other=0.0)
        b = tl.load(w_base + rk[:, None] * swk + rn[None, :] * swn,
                    mask=(rk[:, None] < K) & (rn[None, :] < Nout), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(Y + rm[:, None] * sym + rn[None, :] * syn, acc,
             mask=(rm[:, None] < off) & (rn[None, :] < Nout))

@triton.jit
def _gmm_dx_kernel(DY, W, DX, OFFS, K, Nout,
                   sdm, sdn, swk, swn, sxm, sxk,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    pid_e = tl.program_id(2).to(tl.int64)
    off = tl.load(OFFS + pid_e).to(tl.int64)
    prev = tl.where(pid_e > 0, tl.load(OFFS + pid_e - 1).to(tl.int64), 0)
    cnt = off - prev
    if pid_m * BLOCK_M >= cnt:
        return
    if pid_n * BLOCK_N >= K:
        return
    rm = prev + pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    w_base = W + pid_e * K * Nout
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, Nout, BLOCK_K):
        rk = kk + tl.arange(0, BLOCK_K).to(tl.int64)
        a = tl.load(DY + rm[:, None] * sdm + rk[None, :] * sdn,
                    mask=(rm[:, None] < off) & (rk[None, :] < Nout), other=0.0)
        b = tl.load(w_base + rn[:, None] * swk + rk[None, :] * swn,
                    mask=(rn[:, None] < K) & (rk[None, :] < Nout), other=0.0)
        b_t = tl.trans(b)
        acc = tl.dot(a, b_t, acc, input_precision="ieee")
    tl.store(DX + rm[:, None] * sxm + rn[None, :] * sxk, acc,
             mask=(rm[:, None] < off) & (rn[None, :] < K))

@triton.jit
def _gmm_dw_kernel(X, DY, DW, OFFS, K, Nout,
                   sxm, sxk, sdm, sdn, sdwk, sdwn,
                   BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_k = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    pid_e = tl.program_id(2).to(tl.int64)
    off = tl.load(OFFS + pid_e).to(tl.int64)
    prev = tl.where(pid_e > 0, tl.load(OFFS + pid_e - 1).to(tl.int64), 0)
    cnt = off - prev
    if pid_k * BLOCK_K >= K:
        return
    if pid_n * BLOCK_N >= Nout:
        return
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for m0 in range(0, 2048, BLOCK_M):
        rm = prev + m0 + tl.arange(0, BLOCK_M).to(tl.int64)
        cur = m0 < cnt
        a = tl.load(X + rm[:, None] * sxm + rk[None, :] * sxk,
                    mask=(rm[:, None] < off) & (rk[None, :] < K) & cur, other=0.0)
        b = tl.load(DY + rm[:, None] * sdm + rn[None, :] * sdn,
                    mask=(rm[:, None] < off) & (rn[None, :] < Nout) & cur, other=0.0)
        a_t = tl.trans(a)
        acc = tl.dot(a_t, b, acc, input_precision="ieee")
    dw_base = DW + pid_e * K * Nout
    tl.store(dw_base + rk[:, None] * sdwk + rn[None, :] * sdwn, acc,
             mask=(rk[:, None] < K) & (rn[None, :] < Nout))

def _mm_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b

class _MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b
    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        dc = dc.contiguous()
        return dc @ b.t(), a.t() @ dc

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

def _gmm_forward(x_perm, w, offs):
    N, K = x_perm.shape
    E = w.shape[0]
    Nout = w.shape[2] if w.dim()==3 else w.shape[1]
    out = torch.empty(N, Nout, device=x_perm.device, dtype=torch.float32)
    if N == 0:
        return out
    offs_cpu = offs.tolist()
    prev = 0
    max_cnt = 0
    for e in range(E):
        c = offs_cpu[e] - prev
        if c > max_cnt: max_cnt = c
        prev = offs_cpu[e]
    if max_cnt == 0:
        max_cnt = 32
    grid_m = triton.cdiv(max_cnt, BLOCK_M)
    grid_n = triton.cdiv(Nout, BLOCK_N)
    grid = (grid_m, grid_n, E)
    _gmm_fwd_kernel[grid](x_perm, w, out, offs, K, Nout,
                          x_perm.stride(0), x_perm.stride(1),
                          w.stride(1), w.stride(2),
                          out.stride(0), out.stride(1),
                          BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return out

class _GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_perm, w, offs):
        ctx.save_for_backward(x_perm, w, offs)
        return _gmm_forward(x_perm, w, offs)
    @staticmethod
    def backward(ctx, dY):
        x_perm, w, offs = ctx.saved_tensors
        dY = dY.contiguous()
        E = w.shape[0]
        K = w.shape[1]
        Nout = w.shape[2]
        N = x_perm.shape[0]
        dX = torch.empty_like(x_perm)
        if N>0:
            offs_cpu = offs.tolist()
            prev = 0
            max_cnt = 0
            for e in range(E):
                c = offs_cpu[e] - prev
                if c > max_cnt: max_cnt = c
                prev = offs_cpu[e]
            if max_cnt == 0:
                max_cnt = 32
            grid_m = triton.cdiv(max_cnt, BLOCK_M)
            grid_n = triton.cdiv(K, BLOCK_N)
            grid = (grid_m, grid_n, E)
            _gmm_dx_kernel[grid](dY, w, dX, offs, K, Nout,
                                 dY.stride(0), dY.stride(1),
                                 w.stride(1), w.stride(2),
                                 dX.stride(0), dX.stride(1),
                                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        else:
            dX = torch.empty_like(x_perm)
        dW = torch.zeros_like(w)
        if N>0:
            grid_k = triton.cdiv(K, BLOCK_K)
            grid_n = triton.cdiv(Nout, BLOCK_N)
            grid = (grid_k, grid_n, E)
            _gmm_dw_kernel[grid](x_perm, dY, dW, offs, K, Nout,
                                 x_perm.stride(0), x_perm.stride(1),
                                 dY.stride(0), dY.stride(1),
                                 dW.stride(1), dW.stride(2),
                                 BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M)
        return dX, dW, None

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
            w1s = w1f # keep stacked
            w2s = w2f
            with torch.no_grad():
                sel = _select_topA(s.detach(), router_b.float(), A)
            s_sel = torch.gather(s, 1, sel)
            w = s_sel / s_sel.sum(dim=-1, keepdim=True)
            # routing metadata
            N = T * A
            flat = sel.reshape(-1)  # [N]
            order = torch.argsort(flat, stable=True)  # sort by expert
            tok_exp = torch.arange(T, device=x.device).repeat_interleave(A)
            tok_perm = tok_exp[order]  # [N]
            w_flat = w.reshape(-1)
            w_perm = w_flat[order]
            cnt = torch.bincount(flat, minlength=E)
            offs = cnt.cumsum(0).to(torch.int32)
            x_perm = xf.index_select(0, tok_perm)
            h = _GroupedMM.apply(x_perm, w1s, offs)  # [N,2F]
            g = h[:, :F].contiguous()
            u = h[:, F:].contiguous()
            act = _SiLUMul.apply(g, u)  # [N,F]
            y_perm = _GroupedMM.apply(act, w2s, offs)  # [N,D]
            y_perm = y_perm * w_perm.unsqueeze(1)
            # deterministic combine: per-expert index_add (each destination hit at most once per expert)
            y = torch.zeros(T, D, device=xf.device, dtype=torch.float32)
            # iterate over experts in order, using offs slices
            offs_cpu = offs.tolist()
            prev = 0
            for e in range(E):
                off = offs_cpu[e]
                if off > prev:
                    y.index_add_(0, tok_perm[prev:off], y_perm[prev:off])
                prev = off
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
