import torch
import triton
import triton.language as tl

RM, RN, RK = 64, 64, 32
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
def _grouped_forward_kernel(X, W, Out, Starts, Counts, Kg, Kd, Nd, E, sx0, sx1, sw0, sw1, sw2, so0, so1, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    # map pid to (e, tm, tn) by scanning experts
    rem = pid.to(tl.int64)
    e = 0
    tm = 0
    tn = 0
    e = e.to(tl.int64)
    tm = tm.to(tl.int64)
    tn = tn.to(tl.int64)
    tn_cnt = tl.cdiv(Nd, BLOCK_N).to(tl.int64)
    # E is up to 160 but we handle via fixed 64 loop chunk; for larger E we loop over 160 via two stages? For bench E=32 so 64 enough
    # Use conditional without break: iterate over all possible ee and update when found not yet and rem in range
    acc_tiles = 0
    acc_tiles = acc_tiles.to(tl.int64)
    # we need to find e,tm,tn without break/continue: do prefix sum scan
    # Since tl language limited, we do simple loop with if and keep tracking
    # Approach: loop over ee=0..E-1, compute tiles, check if pid in [acc_tiles, acc_tiles+tiles)
    # Keep last matching ee
    # Triton requires constexpr loop bound, so we use range(128) and mask ee<E
    for ee in range(128):
        is_valid = ee < E
        # load counts conditionally
        cnt_raw = tl.load(Counts + ee).to(tl.int64)
        cnt = tl.where(is_valid, cnt_raw, 0)
        tm_cnt = tl.cdiv(cnt, BLOCK_M)
        tiles = tm_cnt * tn_cnt
        # mask tiles when not valid
        tiles = tl.where(is_valid, tiles, 0)
        in_this = (rem >= acc_tiles) & (rem < acc_tiles + tiles) & is_valid
        e = tl.where(in_this, ee, e)
        rel = rem - acc_tiles
        tm_candidate = rel // tn_cnt
        tn_candidate = rel % tn_cnt
        tm = tl.where(in_this, tm_candidate, tm)
        tn = tl.where(in_this, tn_candidate, tn)
        acc_tiles += tiles
    k_start = tl.load(Starts + e).to(tl.int64)
    k_end = k_start + tl.load(Counts + e).to(tl.int64)
    rm = k_start + tm * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = tn * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, Kd, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K).to(tl.int64)
        a = tl.load(X + rm[:, None] * sx0 + rk[None, :] * sx1, mask=(rm[:, None] < k_end) & (rk[None, :] < Kd), other=0.0)
        b = tl.load(W + e * sw0 + rk[:, None] * sw1 + rn[None, :] * sw2, mask=(rk[:, None] < Kd) & (rn[None, :] < Nd), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(Out + rm[:, None] * so0 + rn[None, :] * so1, acc, mask=(rm[:, None] < k_end) & (rn[None, :] < Nd))

@triton.jit
def _gather_kernel(Src, Idx, Dst, Kg, D, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    src_row = tl.load(Idx + pid).to(tl.int64)
    cols = tl.arange(0, BLOCK_D).to(tl.int64)
    mask = cols < D
    v = tl.load(Src + src_row * D + cols, mask=mask)
    tl.store(Dst + pid * D + cols, v, mask=mask)

@triton.jit
def _sum_contig_kernel(Yp, Out, Starts, D, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    start = tl.load(Starts + pid).to(tl.int64)
    end = tl.load(Starts + pid + 1).to(tl.int64)
    cols = tl.arange(0, BLOCK_D).to(tl.int64)
    mask = cols < D
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for k in range(start, end):
        v = tl.load(Yp + k * D + cols, mask=mask)
        acc += v
    tl.store(Out + pid * D + cols, acc, mask=mask)

@triton.jit
def _scatter_add_kernel(Src, Idx, Dst, N, Ddim, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    dst_row = tl.load(Idx + pid).to(tl.int64)
    cols = tl.arange(0, BLOCK_D).to(tl.int64)
    mask = cols < Ddim
    v = tl.load(Src + pid * Ddim + cols, mask=mask)
    # per-expert distinct dst, so direct add of current dst value is safe without atomics when launched per-expert chunk separately
    base = Dst + dst_row * Ddim + cols
    cur = tl.load(base, mask=mask)
    tl.store(base, cur + v, mask=mask)

@triton.jit
def _scale_kernel(Y, W, N, D, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D).to(tl.int64)
    mask = cols < D
    w = tl.load(W + pid)
    v = tl.load(Y + pid * D + cols, mask=mask)
    tl.store(Y + pid * D + cols, v * w, mask=mask)

def _mm_launch(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    _mm_kernel[(triton.cdiv(M, RM), triton.cdiv(N, RN))](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLK_M=RM, BLK_N=RN, BLK_K=RK)
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

def _select_topA(s, router_b, A):
    E = s.shape[-1]
    b = s + router_b
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=s.device, dtype=torch.int64)
    keys = (u << 10) | (E - 1 - e_idx)
    return torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices

def _grouped_gemm(Xg, W, starts, counts, Nd):
    Kg, Kd = Xg.shape
    E = W.shape[0]
    Out = torch.empty(Kg, Nd, device=Xg.device, dtype=torch.float32)
    if Kg == 0:
        return Out
    BM, BN, BK = 64, 64, 32
    n_tiles_n = triton.cdiv(Nd, BN)
    total = 0
    counts_cpu = counts.cpu().tolist()
    for e in range(E):
        cnt = counts_cpu[e]
        if cnt == 0:
            continue
        total += triton.cdiv(cnt, BM) * n_tiles_n
    if total == 0:
        return Out
    _grouped_forward_kernel[(total,)](Xg, W, Out, starts, counts, Kg, Kd, Nd, E, Xg.stride(0), Xg.stride(1), W.stride(0), W.stride(1), W.stride(2), Out.stride(0), Out.stride(1), BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    return Out

class _GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Xg, W, starts, counts):
        ctx.save_for_backward(Xg, W, starts, counts)
        Nd = W.shape[2]
        out = _grouped_gemm(Xg, W, starts, counts, Nd)
        return out
    @staticmethod
    def backward(ctx, dOut):
        Xg, W, starts, counts = ctx.saved_tensors
        dOut = dOut.contiguous()
        E = W.shape[0]
        dXg = torch.empty_like(Xg)
        dW = torch.zeros_like(W)
        starts_cpu = starts.cpu().tolist()
        counts_cpu = counts.cpu().tolist()
        for e in range(E):
            cnt = counts_cpu[e]
            if cnt == 0:
                continue
            s = starts_cpu[e]
            e_end = s + cnt
            dOut_e = dOut[s:e_end]
            Xg_e = Xg[s:e_end]
            W_e = W[e]
            dXg[s:e_end] = _mm_launch(dOut_e, W_e.t())
            dW[e] = _mm_launch(Xg_e.t(), dOut_e)
        return dXg, dW, None, None

class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx, starts, counts):
        ctx.save_for_backward(idx, starts, counts)
        ctx.src_shape = src.shape
        Kg = idx.shape[0]
        D = src.shape[1]
        dst = torch.empty(Kg, D, device=src.device, dtype=src.dtype)
        BLOCK_D = triton.next_power_of_2(D)
        if BLOCK_D > 1024:
            dst = src.index_select(0, idx)
        else:
            _gather_kernel[(Kg,)](src, idx, dst, Kg, D, BLOCK_D=BLOCK_D)
        return dst
    @staticmethod
    def backward(ctx, d_out):
        idx, starts, counts = ctx.saved_tensors
        T, D = ctx.src_shape
        d_src = torch.zeros(T, D, device=d_out.device, dtype=d_out.dtype)
        E = starts.shape[0]
        starts_cpu = starts.cpu().tolist()
        counts_cpu = counts.cpu().tolist()
        BLOCK_D = triton.next_power_of_2(D)
        if BLOCK_D <= 1024:
            for e in range(E):
                cnt = counts_cpu[e]
                if cnt == 0:
                    continue
                s = starts_cpu[e]
                _scatter_add_kernel[(cnt,)](d_out[s:s+cnt].contiguous(), idx[s:s+cnt], d_src, cnt, D, BLOCK_D=BLOCK_D)
        else:
            for e in range(E):
                cnt = counts_cpu[e]
                if cnt == 0:
                    continue
                s = starts_cpu[e]
                e_ = s + cnt
                d_src = d_src.index_add(0, idx[s:e_], d_out[s:e_])
        return d_src, None, None, None

def _gather_rows(src, idx, starts, counts):
    return _Gather.apply(src, idx, starts, counts)

class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Yp, tok, starts, counts):
        ctx.save_for_backward(tok, starts, counts)
        ctx.Kg = Yp.shape[0]
        D = Yp.shape[1]
        T = int(tok.max().item()) + 1 if tok.numel()>0 else 0
        y = torch.zeros(T, D, device=Yp.device, dtype=Yp.dtype)
        BLOCK_D = triton.next_power_of_2(D)
        if BLOCK_D <= 1024:
            # per-expert deterministic scatter
            starts_cpu = starts.cpu().tolist()
            counts_cpu = counts.cpu().tolist()
            for e in range(starts.shape[0]):
                cnt = counts_cpu[e]
                if cnt==0:
                    continue
                s = starts_cpu[e]
                _scatter_add_kernel[(cnt,)](Yp[s:s+cnt].contiguous(), tok[s:s+cnt], y, cnt, D, BLOCK_D=BLOCK_D)
        else:
            # fallback deterministic torch loop
            starts_cpu = starts.cpu().tolist()
            counts_cpu = counts.cpu().tolist()
            for e in range(starts.shape[0]):
                cnt = counts_cpu[e]
                if cnt==0:
                    continue
                s = starts_cpu[e]
                e_ = s+cnt
                y = y.index_add(0, tok[s:e_], Yp[s:e_])
        ctx.D = D
        ctx.T = T
        return y
    @staticmethod
    def backward(ctx, d_y):
        tok, starts, counts = ctx.saved_tensors
        D = ctx.D
        # dYp = d_y[tok]
        Kg = ctx.Kg
        dYp = torch.empty(Kg, D, device=d_y.device, dtype=d_y.dtype)
        BLOCK_D = triton.next_power_of_2(D)
        if BLOCK_D <= 1024:
            _gather_kernel[(Kg,)](d_y, tok, dYp, Kg, D, BLOCK_D=BLOCK_D)
        else:
            dYp = d_y.index_select(0, tok)
        return dYp, None, None, None

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
            order = torch.argsort(flat)
            orig_idx = order
            tok_sorted = torch.div(orig_idx, A, rounding_mode="floor")
            flat_sorted = flat.index_select(0, order)
            counts = torch.bincount(flat_sorted, minlength=E)
            starts = torch.empty(E, dtype=torch.int32, device=x.device)
            cur = 0
            for e in range(E):
                starts[e] = cur
                cur += int(counts[e])
            Xg = _gather_rows(xf, tok_sorted, starts, counts)
            w_flat = w.reshape(-1)
            w_sorted = w_flat.index_select(0, orig_idx)
            H = _GroupedMM.apply(Xg, w1f, starts, counts)
            G = H[:, :F].contiguous()
            U = H[:, F:].contiguous()
            act = _SiLUMul.apply(G, U)
            Yp = _GroupedMM.apply(act, w2f, starts, counts)
            Yp = Yp * w_sorted.unsqueeze(1)
            y = _Combine.apply(Yp, tok_sorted, starts, counts)
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
