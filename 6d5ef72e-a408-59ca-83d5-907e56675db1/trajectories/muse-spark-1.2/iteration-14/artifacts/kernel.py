import torch
import triton
import triton.language as tl

BM = 64
BN = 64
BK = 64
EBLOCK = 4096

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

def _mm_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _mm_kernel[grid](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK,
        num_warps=4, num_stages=3)
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

# ---- grouped kernels ----
@triton.jit
def _grouped_fwd_kernel(A, B, C,
                        starts, counts,
                        tile_expert, tile_m, tile_n,
                        K, N,
                        stride_a_m, stride_a_k,
                        stride_b_e, stride_b_k, stride_b_n,
                        stride_c_m, stride_c_n,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    e = tl.load(tile_expert + pid).to(tl.int64)
    pm = tl.load(tile_m + pid).to(tl.int64)
    pn = tl.load(tile_n + pid).to(tl.int64)
    start = tl.load(starts + e).to(tl.int64)
    cnt = tl.load(counts + e).to(tl.int64)
    rm = start + pm * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask_m = rm < (start + cnt)
    mask_n = rn < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    b_base = B + e * stride_b_e
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
        mask_k = rk < K
        a = tl.load(A + rm[:, None] * stride_a_m + rk[None, :] * stride_a_k,
                    mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        b = tl.load(b_base + rk[:, None] * stride_b_k + rn[None, :] * stride_b_n,
                    mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b, input_precision="tf32")
    tl.store(C + rm[:, None] * stride_c_m + rn[None, :] * stride_c_n, acc,
             mask=(mask_m[:, None] & mask_n[None, :]))

@triton.jit
def _grouped_bwd_dx_kernel(G, B, DX,
                           starts, counts,
                           tile_expert, tile_m, tile_n,
                           K, N,
                           stride_g_m, stride_g_k,
                           stride_b_e, stride_b_k, stride_b_n,
                           stride_dx_m, stride_dx_k,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    e = tl.load(tile_expert + pid).to(tl.int64)
    pm = tl.load(tile_m + pid).to(tl.int64)
    pn = tl.load(tile_n + pid).to(tl.int64)
    start = tl.load(starts + e).to(tl.int64)
    cnt = tl.load(counts + e).to(tl.int64)
    rm = start + pm * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rk = pn * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask_m = rm < (start + cnt)
    mask_k = rk < K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    b_base = B + e * stride_b_e
    for n0 in range(0, N, BLOCK_K):
        rn = n0 + tl.arange(0, BLOCK_K).to(tl.int64)
        mask_n = rn < N
        g = tl.load(G + rm[:, None] * stride_g_m + rn[None, :] * stride_g_k,
                    mask=(mask_m[:, None] & mask_n[None, :]), other=0.0)
        b = tl.load(b_base + rk[None, :] * stride_b_k + rn[:, None] * stride_b_n,
                    mask=(mask_k[None, :] & mask_n[:, None]), other=0.0)
        # b is [N_chunk, K_tile] need [K_chunk? actually g[M,N] @ B^T[N,K] => bT is [N,K]
        # our b load gives shape [BLOCK_K(N), BLOCK_N(K)] with rows rn, cols rk -> [N_chunk, K_tile]
        # g is [M, N_chunk], b is [N_chunk, K_tile] -> good
        # but we loaded b with rk[None,:] stride_b_k + rn[:,None] stride_b_n => that is transposed layout
        # Need to verify: element at (rn_i, rk_j) = b_base + rk_j*stride_b_k + rn_i*stride_b_n
        # matrix with row rn_i col rk_j shape [N_chunk, K_tile] correct
        acc += tl.dot(g, b, input_precision="tf32")
    tl.store(DX + rm[:, None] * stride_dx_m + rk[None, :] * stride_dx_k, acc,
             mask=(mask_m[:, None] & mask_k[None, :]))

@triton.jit
def _grouped_bwd_dw_kernel(A, G, DW,
                           starts, counts,
                           tile_expert, tile_k, tile_n,
                           K, N,
                           stride_a_m, stride_a_k,
                           stride_g_m, stride_g_k,
                           stride_dw_e, stride_dw_k, stride_dw_n,
                           BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    e = tl.load(tile_expert + pid).to(tl.int64)
    pk = tl.load(tile_k + pid).to(tl.int64)
    pn = tl.load(tile_n + pid).to(tl.int64)
    cnt = tl.load(counts + e).to(tl.int64)
    rk = pk * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask_k = rk < K
    mask_n = rn < N
    if cnt == 0:
        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
        tl.store(DW + e * stride_dw_e + rk[:, None] * stride_dw_k + rn[None, :] * stride_dw_n, acc,
                 mask=(mask_k[:, None] & mask_n[None, :]))
        return
    start = tl.load(starts + e).to(tl.int64)
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for m0 in range(0, cnt, BLOCK_M):
        rm = start + m0 + tl.arange(0, BLOCK_M).to(tl.int64)
        mask_m = rm < (start + cnt)
        # A chunk: [M, K] tile [M_block, K_tile]
        a = tl.load(A + rm[:, None] * stride_a_m + rk[None, :] * stride_a_k,
                    mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        at = tl.trans(a)
        g = tl.load(G + rm[:, None] * stride_g_m + rn[None, :] * stride_g_k,
                    mask=(mask_m[:, None] & mask_n[None, :]), other=0.0)
        acc += tl.dot(at, g, input_precision="ieee")
    tl.store(DW + e * stride_dw_e + rk[:, None] * stride_dw_k + rn[None, :] * stride_dw_n, acc,
             mask=(mask_k[:, None] & mask_n[None, :]))


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, idx, starts, counts):
        ctx.save_for_backward(idx, starts, counts)
        ctx.inp_shape = inp.shape
        return inp.index_select(0, idx)
    @staticmethod
    def backward(ctx, grad_out):
        idx, starts, counts = ctx.saved_tensors
        grad_in = torch.zeros(ctx.inp_shape, device=grad_out.device, dtype=grad_out.dtype)
        E = starts.shape[0]
        starts_l = starts.tolist(); counts_l = counts.tolist()
        for e in range(E):
            s = starts_l[e]; c = counts_l[e]
            if c==0: continue
            grad_in.index_add_(0, idx[s:s+c], grad_out[s:s+c])
        return grad_in, None, None, None

def _build_fwd_tiles(counts, N):
    experts = []
    m_ids = []
    n_ids = []
    for e, cnt in enumerate(counts.tolist()):
        if cnt == 0:
            continue
        nm = triton.cdiv(int(cnt), BM)
        nn = triton.cdiv(int(N), BN)
        for m in range(nm):
            for n_ in range(nn):
                experts.append(e)
                m_ids.append(m)
                n_ids.append(n_)
    if not experts:
        return None, None, None
    return (torch.tensor(experts, device='cuda', dtype=torch.int32),
            torch.tensor(m_ids, device='cuda', dtype=torch.int32),
            torch.tensor(n_ids, device='cuda', dtype=torch.int32))

def _build_dw_tiles(E, K, N):
    experts = []
    k_ids = []
    n_ids = []
    nk = triton.cdiv(int(K), BK)
    nn = triton.cdiv(int(N), BN)
    for e in range(E):
        for k in range(nk):
            for n_ in range(nn):
                experts.append(e)
                k_ids.append(k)
                n_ids.append(n_)
    return (torch.tensor(experts, device='cuda', dtype=torch.int32),
            torch.tensor(k_ids, device='cuda', dtype=torch.int32),
            torch.tensor(n_ids, device='cuda', dtype=torch.int32))

class _GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, starts, counts, fwd_expert, fwd_m, fwd_n, dw_expert, dw_k, dw_n, K, N):
        TA = a.shape[0]
        out = torch.empty(TA, N, device=a.device, dtype=torch.float32)
        if fwd_expert is not None and fwd_expert.numel() > 0:
            grid = (fwd_expert.numel(),)
            _grouped_fwd_kernel[grid](a, b, out,
                starts, counts, fwd_expert, fwd_m, fwd_n,
                K, N,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1), b.stride(2),
                out.stride(0), out.stride(1),
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=4, num_stages=3)
        else:
            out.zero_()
        ctx.save_for_backward(a, b, starts, counts, fwd_expert, fwd_m, fwd_n, dw_expert, dw_k, dw_n)
        ctx.K = K
        ctx.N = N
        ctx.E = b.shape[0]
        return out
    @staticmethod
    def backward(ctx, grad_out):
        a, b, starts, counts, fwd_expert, fwd_m, fwd_n, dw_expert, dw_k, dw_n = ctx.saved_tensors
        # grad_out: [TA, N]
        K = ctx.K
        N = ctx.N
        E = ctx.E
        grad_out = grad_out.contiguous()
        # dA
        dA = None
        if ctx.needs_input_grad[0]:
            dA = torch.empty_like(a)
            if fwd_expert is not None and fwd_expert.numel() > 0:
                # tiles for dA are same as forward but with K as output dim
                # need tiles where N in fwd becomes K for dA? Actually dA is [TA, K]
                # forward tiles partition N; dA tiles partition K
                # we need separate tiles for dA: partition K dimension
                # Reuse forward tiles if K==N? Not generally. Build correct tiles for dA.
                # For dA, we need tiles (m over rows per expert, n over K)
                # Our fwd tiles were built for N, not K. So we need to recompute.
                # Instead build dA tiles on the fly from counts and K
                dA_expert, dA_m, dA_n = _build_fwd_tiles(counts, K)
                if dA_expert is not None:
                    grid = (dA_expert.numel(),)
                    _grouped_bwd_dx_kernel[grid](grad_out, b, dA,
                        starts, counts, dA_expert, dA_m, dA_n,
                        K, N,
                        grad_out.stride(0), grad_out.stride(1),
                        b.stride(0), b.stride(1), b.stride(2),
                        dA.stride(0), dA.stride(1),
                        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=4, num_stages=3)
                else:
                    dA.zero_()
            else:
                dA.zero_()
        dB = None
        if ctx.needs_input_grad[1]:
            dB = torch.empty(E, K, N, device=a.device, dtype=torch.float32)
            grid = (dw_expert.numel(),)
            _grouped_bwd_dw_kernel[grid](a, grad_out, dB,
                starts, counts, dw_expert, dw_k, dw_n,
                K, N,
                a.stride(0), a.stride(1),
                grad_out.stride(0), grad_out.stride(1),
                dB.stride(0), dB.stride(1), dB.stride(2),
                BLOCK_K=BK, BLOCK_N=BN, BLOCK_M=BM, num_warps=4, num_stages=2)
        return dA, dB, None, None, None, None, None, None, None, None, None, None

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
            w1s_dummy = w1f  # to keep graph
            w2s_dummy = w2f
            with torch.no_grad():
                sel = _select_topA(s.detach(), router_b.float(), A)
            s_sel = torch.gather(s, 1, sel)
            w = s_sel / s_sel.sum(dim=-1, keepdim=True)
            flat = sel.reshape(-1)
            order = torch.argsort(flat, stable=True)
            tok = torch.div(order, A, rounding_mode="floor")
            # counts and starts
            counts = torch.bincount(flat, minlength=E).to(torch.int32)
            starts = torch.cat([torch.zeros(1, device=x.device, dtype=torch.int32), counts.cumsum(0)[:-1]])
            # ensure int32
            starts = starts.to(torch.int32)
            # permuted inputs via deterministic gather
            xg = _Gather.apply(xf, tok, starts, counts)
            w_sorted = w.reshape(-1).index_select(0, order)
            # grouped GEMMs with tile precomputation
            # first GEMM: D -> 2F
            N1 = 2*F
            fwd1_e, fwd1_m, fwd1_n = _build_fwd_tiles(counts, N1)
            dw1_e, dw1_k, dw1_n = _build_dw_tiles(E, D, N1)
            gu = _GroupedMM.apply(xg, w1f, starts, counts, fwd1_e, fwd1_m, fwd1_n, dw1_e, dw1_k, dw1_n, D, N1)
            act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
            N2 = D
            fwd2_e, fwd2_m, fwd2_n = _build_fwd_tiles(counts, N2)
            dw2_e, dw2_k, dw2_n = _build_dw_tiles(E, F, N2)
            yp = _GroupedMM.apply(act, w2f, starts, counts, fwd2_e, fwd2_m, fwd2_n, dw2_e, dw2_k, dw2_n, F, N2)
            ws = w_sorted.to(xf.dtype).unsqueeze(1) if False else w_sorted.unsqueeze(1)
            y_grouped_scaled = yp * ws
            y = xf.new_zeros(T, D)
            # deterministic per-expert scatter
            starts_l = starts.tolist(); counts_l = counts.tolist()
            for e in range(E):
                s_ = starts_l[e]; c = counts_l[e]
                if c==0: continue
                idx = tok[s_:s_+c]
                y.index_add_(0, idx, y_grouped_scaled[s_:s_+c])
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
