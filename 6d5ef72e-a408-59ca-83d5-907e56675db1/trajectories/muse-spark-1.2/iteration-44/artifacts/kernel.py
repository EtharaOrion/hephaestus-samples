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
def _mm_bf16_kernel(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn, BLK_M: tl.constexpr, BLK_N: tl.constexpr, BLK_K: tl.constexpr):
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
        a_hi_f32 = a_hi.to(tl.float32)
        b_hi_f32 = b_hi.to(tl.float32)
        a_lo = (a - a_hi_f32).to(tl.bfloat16)
        b_lo = (b - b_hi_f32).to(tl.bfloat16)
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
def _gather_kernel(IN, IDX, OUT, N, D, s0_in, s1_in, s0_out, s1_out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < N
    mask_n = rn < D
    idx = tl.load(IDX + rm, mask=mask_m, other=0)
    val = tl.load(IN + idx[:, None] * s0_in + rn[None, :] * s1_in, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    tl.store(OUT + rm[:, None] * s0_out + rn[None, :] * s1_out, val, mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def _scatter_kernel(Y, IDX, SRC, N, D, s0_y, s1_y, s0_src, s1_src, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < N
    mask_n = rn < D
    idx = tl.load(IDX + rm, mask=mask_m, other=0)
    src = tl.load(SRC + rm[:, None] * s0_src + rn[None, :] * s1_src, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    y_old = tl.load(Y + idx[:, None] * s0_y + rn[None, :] * s1_y, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    tl.store(Y + idx[:, None] * s0_y + rn[None, :] * s1_y, y_old + src, mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def _gather_1d_kernel(IN, IDX, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    idx = tl.load(IDX + off, mask=mask, other=0)
    val = tl.load(IN + idx, mask=mask, other=0.0)
    tl.store(OUT + off, val, mask=mask)

@triton.jit
def _scatter_add_1d_kernel(Y, IDX, SRC, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    idx = tl.load(IDX + off, mask=mask, other=0)
    src = tl.load(SRC + off, mask=mask, other=0.0)
    y_old = tl.load(Y + idx, mask=mask, other=0.0)
    tl.store(Y + idx, y_old + src, mask=mask)

@triton.jit
def _gather_gemm_bf16_kernel(X, W, C, TOK, ROW_START, CNT, EXPERT, TM, TN, K, N, sX0, sX1, sW0, sW1, sW2, sC0, sC1, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    e = tl.load(EXPERT + pid).to(tl.int64)
    rs = tl.load(ROW_START + pid).to(tl.int64)
    cnt = tl.load(CNT + pid).to(tl.int64)
    tm = tl.load(TM + pid).to(tl.int64)
    tn = tl.load(TN + pid).to(tl.int64)
    rm = rs + tm * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = tn * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask_m = rm < rs + cnt
    mask_n = rn < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
        mask_k = rk < K
        tok = tl.load(TOK + rm, mask=mask_m, other=0)
        a = tl.load(X + tok[:, None] * sX0 + rk[None, :] * sX1, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w_base = W + e * sW0
        b = tl.load(w_base + rk[:, None] * sW1 + rn[None, :] * sW2, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        a_hi = a.to(tl.bfloat16)
        b_hi = b.to(tl.bfloat16)
        a_hi_f32 = a_hi.to(tl.float32)
        b_hi_f32 = b_hi.to(tl.float32)
        a_lo = (a - a_hi_f32).to(tl.bfloat16)
        b_lo = (b - b_hi_f32).to(tl.bfloat16)
        acc += tl.dot(a_hi, b_hi)
        acc += tl.dot(a_hi, b_lo)
        acc += tl.dot(a_lo, b_hi)
    tl.store(C + rm[:, None] * sC0 + rn[None, :] * sC1, acc, mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def _grouped_bf16_kernel(A, W, C, ROW_START, CNT, EXPERT, TM, TN, K, N, sA0, sA1, sW0, sW1, sW2, sC0, sC1, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    e = tl.load(EXPERT + pid).to(tl.int64)
    rs = tl.load(ROW_START + pid).to(tl.int64)
    cnt = tl.load(CNT + pid).to(tl.int64)
    tm = tl.load(TM + pid).to(tl.int64)
    tn = tl.load(TN + pid).to(tl.int64)
    rm = rs + tm * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = tn * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask_m = rm < rs + cnt
    mask_n = rn < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
        mask_k = rk < K
        a = tl.load(A + rm[:, None] * sA0 + rk[None, :] * sA1, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w_base = W + e * sW0
        b = tl.load(w_base + rk[:, None] * sW1 + rn[None, :] * sW2, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        a_hi = a.to(tl.bfloat16)
        b_hi = b.to(tl.bfloat16)
        a_hi_f32 = a_hi.to(tl.float32)
        b_hi_f32 = b_hi.to(tl.float32)
        a_lo = (a - a_hi_f32).to(tl.bfloat16)
        b_lo = (b - b_hi_f32).to(tl.bfloat16)
        acc += tl.dot(a_hi, b_hi)
        acc += tl.dot(a_hi, b_lo)
        acc += tl.dot(a_lo, b_hi)
    tl.store(C + rm[:, None] * sC0 + rn[None, :] * sC1, acc, mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def _combine_fwd_kernel(Y, YP, W, INV, T, A, D, sY0, sY1, sYP0, sYP1, BLOCK_D: tl.constexpr):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    mask = off < D
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for a in range(160):
        if a < A:
            flat = pid_t * A + a
            pos = tl.load(INV + flat).to(tl.int64)
            w = tl.load(W + flat)
            yp = tl.load(YP + pos * sYP0 + off * sYP1, mask=mask, other=0.0)
            acc += w * yp
    tl.store(Y + pid_t * sY0 + off * sY1, acc, mask=mask)

def _mm_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    _mm_bf16_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK)
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

class _GatherGatherMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, tok, counts, offs):
        G = tok.shape[0]
        E, K, N = w.shape
        c = torch.empty(G, N, device=x.device, dtype=torch.float32)
        ctx.save_for_backward(x, w, tok, counts, offs)
        BMg, BNg, BKg = 64, 128, 32
        row_starts=[]; cnts=[]; experts=[]; tms=[]; tns=[]
        offs_list = offs.tolist()
        counts_list = counts.tolist()
        for e in range(E):
            cnt = counts_list[e]
            rs = offs_list[e] - cnt
            nm = triton.cdiv(cnt, BMg)
            nn = triton.cdiv(N, BNg)
            for tm in range(nm):
                for tn in range(nn):
                    row_starts.append(rs)
                    cnts.append(cnt)
                    experts.append(e)
                    tms.append(tm)
                    tns.append(tn)
        rs_t = torch.tensor(row_starts, device=x.device, dtype=torch.int64)
        cnt_t = torch.tensor(cnts, device=x.device, dtype=torch.int64)
        ex_t = torch.tensor(experts, device=x.device, dtype=torch.int64)
        tm_t = torch.tensor(tms, device=x.device, dtype=torch.int64)
        tn_t = torch.tensor(tns, device=x.device, dtype=torch.int64)
        _gather_gemm_bf16_kernel[(max(len(experts),1),)](x, w, c, tok, rs_t, cnt_t, ex_t, tm_t, tn_t, K, N, x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.stride(2), c.stride(0), c.stride(1), BLOCK_M=BMg, BLOCK_N=BNg, BLOCK_K=BKg)
        return c
    @staticmethod
    def backward(ctx, dc):
        x, w, tok, counts, offs = ctx.saved_tensors
        E, K, N = w.shape
        dc_c = dc.contiguous()
        dx = torch.zeros_like(x)
        dw = torch.zeros_like(w)
        offs_list = offs.tolist()
        counts_list = counts.tolist()
        for e in range(E):
            cnt = counts_list[e]
            rs = offs_list[e] - cnt
            dc_slice = dc_c[rs:rs+cnt]
            tok_slice = tok[rs:rs+cnt]
            da_slice = _mm_launch(dc_slice, w[e].t())
            BMG, BNG = 32, 32
            grid = (triton.cdiv(max(cnt,1), BMG), triton.cdiv(K, BNG))
            _scatter_kernel[grid](dx, tok_slice, da_slice, cnt, K, dx.stride(0), dx.stride(1), da_slice.stride(0), da_slice.stride(1), BLOCK_M=BMG, BLOCK_N=BNG)
            x_slice = torch.empty(cnt, K, device=x.device, dtype=x.dtype)
            grid2 = (triton.cdiv(max(cnt,1), BMG), triton.cdiv(K, BNG))
            _gather_kernel[grid2](x, tok_slice, x_slice, cnt, K, x.stride(0), x.stride(1), x_slice.stride(0), x_slice.stride(1), BLOCK_M=BMG, BLOCK_N=BNG)
            dw[e] = _mm_launch(x_slice.t(), dc_slice)
        return dx, dw, None, None, None

class _GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, w, counts, offs):
        G, K = a.shape
        E, _, N = w.shape
        c = torch.empty(G, N, device=a.device, dtype=torch.float32)
        ctx.save_for_backward(a, w, counts, offs)
        BMg, BNg, BKg = 64, 128, 32
        row_starts=[]; cnts=[]; experts=[]; tms=[]; tns=[]
        offs_list = offs.tolist()
        counts_list = counts.tolist()
        for e in range(E):
            cnt = counts_list[e]
            rs = offs_list[e] - cnt
            nm = triton.cdiv(cnt, BMg)
            nn = triton.cdiv(N, BNg)
            for tm in range(nm):
                for tn in range(nn):
                    row_starts.append(rs)
                    cnts.append(cnt)
                    experts.append(e)
                    tms.append(tm)
                    tns.append(tn)
        rs_t = torch.tensor(row_starts, device=a.device, dtype=torch.int64)
        cnt_t = torch.tensor(cnts, device=a.device, dtype=torch.int64)
        ex_t = torch.tensor(experts, device=a.device, dtype=torch.int64)
        tm_t = torch.tensor(tms, device=a.device, dtype=torch.int64)
        tn_t = torch.tensor(tns, device=a.device, dtype=torch.int64)
        _grouped_bf16_kernel[(max(len(experts),1),)](a, w, c, rs_t, cnt_t, ex_t, tm_t, tn_t, K, N, a.stride(0), a.stride(1), w.stride(0), w.stride(1), w.stride(2), c.stride(0), c.stride(1), BLOCK_M=BMg, BLOCK_N=BNg, BLOCK_K=BKg)
        return c
    @staticmethod
    def backward(ctx, dc):
        a, w, counts, offs = ctx.saved_tensors
        dc_c = dc.contiguous()
        da = torch.empty_like(a)
        dw = torch.zeros_like(w)
        offs_list = offs.tolist()
        counts_list = counts.tolist()
        for e in range(w.shape[0]):
            cnt = counts_list[e]
            rs = offs_list[e] - cnt
            dc_slice = dc_c[rs:rs+cnt]
            a_slice = a[rs:rs+cnt]
            da[rs:rs+cnt] = _mm_launch(dc_slice, w[e].t())
            dw[e] = _mm_launch(a_slice.t(), dc_slice)
        return da, dw, None, None

class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, yp, w_flat, inv, T, A):
        G, D = yp.shape
        y = torch.empty(T, D, device=yp.device, dtype=yp.dtype)
        ctx.save_for_backward(w_flat, inv, yp)
        ctx.T = T
        ctx.A = A
        BLOCK_D = 32
        grid = (T, triton.cdiv(D, BLOCK_D))
        _combine_fwd_kernel[grid](y, yp, w_flat, inv, T, A, D, y.stride(0), y.stride(1), yp.stride(0), yp.stride(1), BLOCK_D=BLOCK_D)
        return y
    @staticmethod
    def backward(ctx, dy):
        w_flat, inv, yp = ctx.saved_tensors
        T = ctx.T
        A = ctx.A
        G, D = yp.shape
        dy_c = dy.contiguous()
        yp_c = yp.contiguous()
        pos2flat = torch.argsort(inv)
        t_idx = pos2flat // A
        w_sorted = w_flat[pos2flat]
        dyp = dy_c[t_idx] * w_sorted.unsqueeze(1)
        flat_yp = yp_c[inv]
        dy_flat = dy_c[torch.arange(G, device=dy.device)//A]
        dw = (flat_yp * dy_flat).sum(dim=1)
        return dyp, dw, None, None, None

class _Gather1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx):
        N = idx.shape[0]
        out = torch.empty(N, device=x.device, dtype=x.dtype)
        ctx.save_for_backward(idx)
        ctx.N0 = x.shape[0]
        _gather_1d_kernel[(triton.cdiv(max(N,1),1024),)](x, idx, out, N, BLOCK=1024)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        (idx,) = ctx.saved_tensors
        N = idx.shape[0]
        T = ctx.N0
        grad_x = torch.zeros(T, device=grad_out.device, dtype=grad_out.dtype)
        _scatter_add_1d_kernel[(triton.cdiv(max(N,1),1024),)](grad_x, idx, grad_out, N, BLOCK=1024)
        return grad_x, None

def _select_topA(s: torch.Tensor, router_b: torch.Tensor, A: int) -> torch.Tensor:
    E = s.shape[-1]
    b = s + router_b
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=s.device, dtype=torch.int64)
    keys = (u << 16) | (E - 1 - e_idx)
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
            with torch.no_grad():
                sel = _select_topA(s.detach(), router_b.float(), A)
            s_sel = torch.gather(s, 1, sel)
            w = s_sel / s_sel.sum(dim=-1, keepdim=True)
            flat = sel.reshape(-1)
            perm = torch.argsort(flat)
            tok = perm // A
            w_flat = w.reshape(-1)
            counts = torch.bincount(flat, minlength=E)
            offs = counts.cumsum(0)
            inv = torch.argsort(perm)
            gu = _GatherGatherMM.apply(xf, w1f, tok, counts, offs)
            g = gu[:, :F].contiguous()
            u = gu[:, F:].contiguous()
            act = _SiLUMul.apply(g, u)
            yp = _GroupedMM.apply(act, w2f, counts, offs)
            y = _Combine.apply(yp, w_flat, inv, T, A)
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
