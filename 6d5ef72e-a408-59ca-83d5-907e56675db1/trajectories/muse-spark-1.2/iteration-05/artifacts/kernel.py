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
        acc = tl.dot(a, b, acc, input_precision="tf32")
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
def _gather_rows_kernel(Src, Idx, Dst, D, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    row = pid_m
    src_row = tl.load(Idx + row).to(tl.int64)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    mask = offs_d < D
    v = tl.load(Src + src_row * D + offs_d, mask=mask)
    tl.store(Dst + row * D + offs_d, v, mask=mask)


@triton.jit
def _gather_scores_kernel(S_ptr, Sel_ptr, Dst_ptr, T, E, A, BLOCK_A: tl.constexpr):
    pid_t = tl.program_id(0).to(tl.int64)
    offs_a = tl.arange(0, BLOCK_A).to(tl.int64)
    mask = offs_a < A
    sel = tl.load(Sel_ptr + pid_t * A + offs_a, mask=mask, other=0).to(tl.int64)
    vals = tl.load(S_ptr + pid_t * E + sel, mask=mask)
    tl.store(Dst_ptr + pid_t * A + offs_a, vals, mask=mask)


@triton.jit
def _scatter_scores_kernel(GradSel, Sel_ptr, GradS, T, E, A, BLOCK_A: tl.constexpr):
    pid_t = tl.program_id(0).to(tl.int64)
    offs_a = tl.arange(0, BLOCK_A).to(tl.int64)
    mask = offs_a < A
    sel = tl.load(Sel_ptr + pid_t * A + offs_a, mask=mask, other=0).to(tl.int64)
    g = tl.load(GradSel + pid_t * A + offs_a, mask=mask)
    tl.store(GradS + pid_t * E + sel, g, mask=mask)


@triton.jit
def _grouped_mm_kernel(A_ptr, B_ptr, C_ptr, Offs_ptr, M_total, N, K, E, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    pid_e = tl.program_id(2).to(tl.int64)
    off_start = tl.where(pid_e == 0, 0, tl.load(Offs_ptr + pid_e - 1)).to(tl.int64)
    off_end = tl.load(Offs_ptr + pid_e).to(tl.int64)
    cur_m = off_end - off_start
    rm_base = pid_m * BLOCK_M
    if rm_base >= cur_m:
        return
    rm = rm_base + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
        a = tl.load(A_ptr + (off_start + rm)[:, None] * stride_am + rk[None, :] * stride_ak, mask=(rm[:, None] < cur_m) & (rk[None, :] < K), other=0.0)
        b_off = pid_e * K * N
        b = tl.load(B_ptr + b_off + rk[:, None] * stride_bk + rn[None, :] * stride_bn, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="tf32")
    tl.store(C_ptr + (off_start + rm)[:, None] * stride_cm + rn[None, :] * stride_cn, acc, mask=(rm[:, None] < cur_m) & (rn[None, :] < N))


@triton.jit
def _grouped_mm_dB_kernel(A_ptr, DC_ptr, DB_ptr, Offs_ptr, M_total, N, K, E, stride_am, stride_ak, stride_cm, stride_cn, stride_dbk, stride_dbn, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_k = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    pid_e = tl.program_id(2).to(tl.int64)
    off_start = tl.where(pid_e == 0, 0, tl.load(Offs_ptr + pid_e - 1)).to(tl.int64)
    off_end = tl.load(Offs_ptr + pid_e).to(tl.int64)
    cur_m = off_end - off_start
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for m0 in range(0, 2048, BLOCK_M):
        rm = m0 + tl.arange(0, BLOCK_M).to(tl.int64)
        a = tl.load(A_ptr + (off_start + rm)[:, None] * stride_am + rk[None, :] * stride_ak, mask=(rm[:, None] < cur_m) & (rk[None, :] < K), other=0.0)
        dc = tl.load(DC_ptr + (off_start + rm)[:, None] * stride_cm + rn[None, :] * stride_cn, mask=(rm[:, None] < cur_m) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(tl.trans(a), dc, acc, input_precision="tf32")
    db_off = pid_e * K * N
    tl.store(DB_ptr + db_off + rk[:, None] * stride_dbk + rn[None, :] * stride_dbn, acc, mask=(rk[:, None] < K) & (rn[None, :] < N))


@triton.jit
def _grouped_mm_dA_kernel(DC_ptr, B_ptr, DA_ptr, Offs_ptr, M_total, N, K, E, stride_cm, stride_cn, stride_bk, stride_bn, stride_am, stride_an, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    pid_e = tl.program_id(2).to(tl.int64)
    off_start = tl.where(pid_e == 0, 0, tl.load(Offs_ptr + pid_e - 1)).to(tl.int64)
    off_end = tl.load(Offs_ptr + pid_e).to(tl.int64)
    cur_m = off_end - off_start
    rm_base = pid_m * BLOCK_M
    if rm_base >= cur_m:
        return
    rm = rm_base + tl.arange(0, BLOCK_M).to(tl.int64)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N).to(tl.int64)
        dc = tl.load(DC_ptr + (off_start + rm)[:, None] * stride_cm + rn[None, :] * stride_cn, mask=(rm[:, None] < cur_m) & (rn[None, :] < N), other=0.0)
        b_off = pid_e * K * N
        b = tl.load(B_ptr + b_off + rk[:, None] * stride_bk + rn[None, :] * stride_bn, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(dc, tl.trans(b), acc, input_precision="tf32")
    tl.store(DA_ptr + (off_start + rm)[:, None] * stride_am + rk[None, :] * stride_an, acc, mask=(rm[:, None] < cur_m) & (rk[None, :] < K))


@triton.jit
def _combine_gather_kernel(PosPtr, SortedOut, Weights, Y, T, D, A, BLOCK_D: tl.constexpr):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    mask = offs_d < D
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for a in range(0, 64):
        pos = tl.load(PosPtr + pid_t * A + a, mask=a < A, other=0).to(tl.int64)
        w = tl.load(Weights + pos, mask=a < A, other=0.0)
        row = tl.load(SortedOut + pos * D + offs_d, mask=mask & (a < A))
        acc += tl.where(a < A, row * w, 0.0)
    tl.store(Y + pid_t * D + offs_d, acc, mask=mask)


@triton.jit
def _gather_bwd_kernel(PosPtr, GradOutPtr, GradSrcPtr, T, D, A, BLOCK_D: tl.constexpr):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    mask = offs_d < D
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for a in range(0, 64):
        pos = tl.load(PosPtr + pid_t * A + a, mask=a < A, other=0).to(tl.int64)
        row = tl.load(GradOutPtr + pos * D + offs_d, mask=mask & (a < A))
        acc += tl.where(a < A, row, 0.0)
    tl.store(GradSrcPtr + pid_t * D + offs_d, acc, mask=mask)


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


class _DetGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, sel):
        ctx.save_for_backward(sel)
        ctx.T = s.shape[0]; ctx.E = s.shape[1]; ctx.A = sel.shape[1]
        out = torch.empty(s.shape[0], sel.shape[1], device=s.device, dtype=s.dtype)
        BLOCK_A = triton.next_power_of_2(sel.shape[1])
        if BLOCK_A < 16: BLOCK_A = 16
        _gather_scores_kernel[(s.shape[0],)](s, sel, out, s.shape[0], s.shape[1], sel.shape[1], BLOCK_A=BLOCK_A)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        sel = ctx.saved_tensors[0]
        T, E, A = ctx.T, ctx.E, ctx.A
        grad_s = torch.zeros(T, E, device=grad_out.device, dtype=grad_out.dtype)
        BLOCK_A = triton.next_power_of_2(A)
        if BLOCK_A < 16: BLOCK_A = 16
        _scatter_scores_kernel[(T,)](grad_out, sel, grad_s, T, E, A, BLOCK_A=BLOCK_A)
        return grad_s, None


class _GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx, pos):
        ctx.save_for_backward(pos)
        ctx.T = src.shape[0]; ctx.D = src.shape[1]
        M = idx.numel(); D = src.shape[1]
        dst = torch.empty(M, D, device=src.device, dtype=src.dtype)
        if M > 0:
            BLOCK_D = 64
            if D < 64: BLOCK_D = triton.next_power_of_2(D)
            _gather_rows_kernel[(M, triton.cdiv(D, BLOCK_D))](src, idx, dst, D, BLOCK_D=BLOCK_D)
        return dst
    @staticmethod
    def backward(ctx, grad_out):
        pos = ctx.saved_tensors[0]
        T = ctx.T; D = ctx.D; A = pos.shape[1]
        grad_src = torch.zeros(T, D, device=grad_out.device, dtype=grad_out.dtype)
        BLOCK_D = 64
        if D < 64: BLOCK_D = triton.next_power_of_2(D)
        _gather_bwd_kernel[(T, triton.cdiv(D, BLOCK_D))](pos, grad_out, grad_src, T, D, A, BLOCK_D=BLOCK_D)
        return grad_src, None, None


class _GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_sorted, b_stack, offs):
        ctx.save_for_backward(a_sorted, b_stack, offs)
        M, K = a_sorted.shape; E, K2, N = b_stack.shape
        c = torch.empty(M, N, device=a_sorted.device, dtype=a_sorted.dtype)
        if M == 0 or N == 0: return c
        BLOCK_M = 64; BLOCK_N = 64; BLOCK_K = 32
        max_m = int(offs.max().item()) if E > 0 else M
        if E > 1:
            diffs = offs[1:] - offs[:-1]
            max_m = int(diffs.max().item())
            first = int(offs[0].item())
            if first > max_m: max_m = first
        else:
            max_m = int(offs[0].item()) if E else M
        if max_m == 0: max_m = 64
        grid_m = triton.cdiv(max_m, BLOCK_M)
        if grid_m == 0: grid_m = 1
        _grouped_mm_kernel[(grid_m, triton.cdiv(N, BLOCK_N), E)](a_sorted, b_stack, c, offs, M, N, K, E, a_sorted.stride(0), a_sorted.stride(1), b_stack.stride(1), b_stack.stride(2), c.stride(0), c.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        return c
    @staticmethod
    def backward(ctx, dc):
        a_sorted, b_stack, offs = ctx.saved_tensors
        dc = dc.contiguous(); M, K = a_sorted.shape; E, K2, N = b_stack.shape
        BLOCK_M = 64; BLOCK_N = 64; BLOCK_K = 32; BLOCK_M2 = 32
        max_m = int(offs.max().item()) if E > 0 else M
        if E > 1:
            diffs = offs[1:] - offs[:-1]
            max_m = int(diffs.max().item())
            first = int(offs[0].item())
            if first > max_m: max_m = first
        else:
            max_m = int(offs[0].item()) if E else M
        if max_m == 0: max_m = 64
        grid_m = triton.cdiv(max_m, BLOCK_M)
        if grid_m == 0: grid_m = 1
        da = torch.empty_like(a_sorted)
        _grouped_mm_dA_kernel[(grid_m, triton.cdiv(K, BLOCK_K), E)](dc, b_stack, da, offs, M, N, K, E, dc.stride(0), dc.stride(1), b_stack.stride(1), b_stack.stride(2), da.stride(0), da.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        db = torch.zeros_like(b_stack)
        _grouped_mm_dB_kernel[(triton.cdiv(K, BLOCK_K), triton.cdiv(N, BLOCK_N), E)](a_sorted, dc, db, offs, M, N, K, E, a_sorted.stride(0), a_sorted.stride(1), dc.stride(0), dc.stride(1), db.stride(1), db.stride(2), BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M2)
        return da, db, None


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, sorted_out, weights):
        ctx.save_for_backward(pos, weights, sorted_out)
        T, A = pos.shape; D = sorted_out.shape[1]
        y = torch.empty(T, D, device=sorted_out.device, dtype=sorted_out.dtype)
        BLOCK_D = 64
        if D < 64: BLOCK_D = triton.next_power_of_2(D)
        _combine_gather_kernel[(T, triton.cdiv(D, BLOCK_D))](pos, sorted_out, weights, y, T, D, A, BLOCK_D=BLOCK_D)
        return y
    @staticmethod
    def backward(ctx, dy):
        pos, weights, sorted_out = ctx.saved_tensors
        T, A = pos.shape; M, D = sorted_out.shape
        dy = dy.contiguous()
        d_sorted = torch.zeros_like(sorted_out)
        d_weights = torch.empty_like(weights)
        for a in range(A):
            p = pos[:, a]
            w = weights[p].unsqueeze(1)
            d_sorted.index_copy_(0, p, dy * w)
        for a in range(A):
            p = pos[:, a]
            d_weights[p] = (dy * sorted_out[p]).sum(dim=1)
        return None, d_sorted, d_weights


def _select_topA(s: torch.Tensor, router_b: torch.Tensor, A: int) -> torch.Tensor:
    E = s.shape[-1]
    b = s + router_b
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=s.device, dtype=torch.int64)
    bits = max(1, (E - 1).bit_length())
    keys = (u << bits) | (E - 1 - e_idx)
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
            T, D = xf.shape; E = rwf.shape[1]; F = w1f.shape[2] // 2
            s = torch.sigmoid(_MM.apply(xf, rwf))
            with torch.no_grad():
                sel = _select_topA(s.detach(), router_b.float(), A)
            s_sel = _DetGather.apply(s, sel)
            w = s_sel / s_sel.sum(dim=-1, keepdim=True)
            sel_flat = sel.reshape(-1)
            w_flat = w.reshape(-1)
            order = torch.argsort(sel_flat)
            w_sorted = w_flat[order]
            tok_sorted = order // A
            offs = torch.bincount(sel_flat, minlength=E).cumsum(0).to(torch.int32)
            inv_order = torch.argsort(order)
            pos = inv_order.view(T, A)
            x_sorted = _GatherRows.apply(xf, tok_sorted, pos)
            gu_sorted = _GroupedMM.apply(x_sorted, w1f, offs)
            act_sorted = _SiLUMul.apply(gu_sorted[:, :F].contiguous(), gu_sorted[:, F:].contiguous())
            out_sorted = _GroupedMM.apply(act_sorted, w2f, offs)
            y = _Combine.apply(pos, out_sorted, w_sorted)
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
