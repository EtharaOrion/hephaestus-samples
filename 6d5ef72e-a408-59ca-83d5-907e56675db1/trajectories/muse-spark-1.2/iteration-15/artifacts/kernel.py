
import torch
torch.use_deterministic_algorithms(True)
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32
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
        acc = tl.dot(a, b, acc, input_precision="ieee")
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
def _grouped_fwd_kernel(A, B, C, Offs, Exp, Row, Col, N, K, s_am, s_ak, s_b0, s_bk, s_bn, s_cm, s_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    e = tl.load(Exp + pid)
    r = tl.load(Row + pid)
    cc = tl.load(Col + pid)
    off_end = tl.load(Offs + e)
    rm = r + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = cc + tl.arange(0, BLOCK_N).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
        a = tl.load(A + rm[:, None] * s_am + rk[None, :] * s_ak, mask=(rm[:, None] < off_end) & (rk[None, :] < K), other=0.0)
        b = tl.load(B + e * s_b0 + rk[:, None] * s_bk + rn[None, :] * s_bn, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(C + rm[:, None] * s_cm + rn[None, :] * s_cn, acc, mask=(rm[:, None] < off_end) & (rn[None, :] < N))

@triton.jit
def _grouped_fwd_kernel_ieee(A, B, C, Offs, Exp, Row, Col, N, K, s_am, s_ak, s_b0, s_bk, s_bn, s_cm, s_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    e = tl.load(Exp + pid)
    r = tl.load(Row + pid)
    cc = tl.load(Col + pid)
    off_end = tl.load(Offs + e)
    rm = r + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = cc + tl.arange(0, BLOCK_N).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
        a = tl.load(A + rm[:, None] * s_am + rk[None, :] * s_ak, mask=(rm[:, None] < off_end) & (rk[None, :] < K), other=0.0)
        b = tl.load(B + e * s_b0 + rk[:, None] * s_bk + rn[None, :] * s_bn, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(C + rm[:, None] * s_cm + rn[None, :] * s_cn, acc, mask=(rm[:, None] < off_end) & (rn[None, :] < N))

@triton.jit
def _grouped_bwd_da_kernel(DC, B, DA, Offs, Exp, Row, Col, N, K, s_dcm, s_dcn, s_b0, s_bk, s_bn, s_dam, s_dak, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    e = tl.load(Exp + pid)
    r = tl.load(Row + pid)
    kk = tl.load(Col + pid)
    off_end = tl.load(Offs + e)
    rm = r + tl.arange(0, BLOCK_M).to(tl.int64)
    rk = kk + tl.arange(0, BLOCK_K).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N).to(tl.int64)
        dc = tl.load(DC + rm[:, None] * s_dcm + rn[None, :] * s_dcn, mask=(rm[:, None] < off_end) & (rn[None, :] < N), other=0.0)
        b = tl.load(B + e * s_b0 + rk[:, None] * s_bk + rn[None, :] * s_bn, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(dc, tl.trans(b), acc)
    tl.store(DA + rm[:, None] * s_dam + rk[None, :] * s_dak, acc, mask=(rm[:, None] < off_end) & (rk[None, :] < K))

@triton.jit
def _grouped_bwd_da_kernel_ieee(DC, B, DA, Offs, Exp, Row, Col, N, K, s_dcm, s_dcn, s_b0, s_bk, s_bn, s_dam, s_dak, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    e = tl.load(Exp + pid)
    r = tl.load(Row + pid)
    kk = tl.load(Col + pid)
    off_end = tl.load(Offs + e)
    rm = r + tl.arange(0, BLOCK_M).to(tl.int64)
    rk = kk + tl.arange(0, BLOCK_K).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N).to(tl.int64)
        dc = tl.load(DC + rm[:, None] * s_dcm + rn[None, :] * s_dcn, mask=(rm[:, None] < off_end) & (rn[None, :] < N), other=0.0)
        b = tl.load(B + e * s_b0 + rk[:, None] * s_bk + rn[None, :] * s_bn, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(dc, tl.trans(b), acc, input_precision="ieee")
    tl.store(DA + rm[:, None] * s_dam + rk[None, :] * s_dak, acc, mask=(rm[:, None] < off_end) & (rk[None, :] < K))

@triton.jit
def _grouped_bwd_db_kernel(A, DC, DB, Offs, M, N, K, s_am, s_ak, s_dcm, s_dcn, s_db0, s_dbk, s_dbn, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_e = tl.program_id(0).to(tl.int64)
    pid_t = tl.program_id(1).to(tl.int64)
    e = pid_e
    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_k = pid_t // n_tiles
    pid_n = pid_t % n_tiles
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    off_start = 0
    if e > 0:
        off_start = tl.load(Offs + e - 1)
    off_end = tl.load(Offs + e)
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for m0 in range(off_start, off_end, BLOCK_M):
        rm = m0 + tl.arange(0, BLOCK_M).to(tl.int64)
        mask_m = rm < off_end
        a = tl.load(A + rm[:, None] * s_am + rk[None, :] * s_ak, mask=mask_m[:, None] & (rk[None, :] < K), other=0.0)
        dc = tl.load(DC + rm[:, None] * s_dcm + rn[None, :] * s_dcn, mask=mask_m[:, None] & (rn[None, :] < N), other=0.0)
        acc = tl.dot(tl.trans(a), dc, acc)
    tl.store(DB + e * s_db0 + rk[:, None] * s_dbk + rn[None, :] * s_dbn, acc, mask=(rk[:, None] < K) & (rn[None, :] < N))

@triton.jit
def _grouped_bwd_db_kernel_ieee(A, DC, DB, Offs, M, N, K, s_am, s_ak, s_dcm, s_dcn, s_db0, s_dbk, s_dbn, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_e = tl.program_id(0).to(tl.int64)
    pid_t = tl.program_id(1).to(tl.int64)
    e = pid_e
    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_k = pid_t // n_tiles
    pid_n = pid_t % n_tiles
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    off_start = 0
    if e > 0:
        off_start = tl.load(Offs + e - 1)
    off_end = tl.load(Offs + e)
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for m0 in range(off_start, off_end, BLOCK_M):
        rm = m0 + tl.arange(0, BLOCK_M).to(tl.int64)
        mask_m = rm < off_end
        a = tl.load(A + rm[:, None] * s_am + rk[None, :] * s_ak, mask=mask_m[:, None] & (rk[None, :] < K), other=0.0)
        dc = tl.load(DC + rm[:, None] * s_dcm + rn[None, :] * s_dcn, mask=mask_m[:, None] & (rn[None, :] < N), other=0.0)
        acc = tl.dot(tl.trans(a), dc, acc, input_precision="ieee")
    tl.store(DB + e * s_db0 + rk[:, None] * s_dbk + rn[None, :] * s_dbn, acc, mask=(rk[:, None] < K) & (rn[None, :] < N))


@triton.jit
def _combine_kernel(SRC, W, INV, Y, T: tl.constexpr, A: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    offs = pid_d * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < D
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for a in range(8):
        cond = a < A
        flat = pid_t * A + a
        pos = tl.load(INV + flat)
        w = tl.load(W + pos)
        src = tl.load(SRC + pos * D + offs, mask=mask, other=0.0)
        acc = tl.where(cond, acc + src * w, acc)
    tl.store(Y + pid_t * D + offs, acc, mask=mask)



@triton.jit
def _gather_kernel(SRC, IDX, DST, D, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    src_row = tl.load(IDX + pid_m).to(tl.int64)
    offs = pid_d * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < D
    vals = tl.load(SRC + src_row * D + offs, mask=mask, other=0.0)
    tl.store(DST + pid_m * D + offs, vals, mask=mask)

@triton.jit
def _gather_kernel_bwd(DOUT, IDX, DIN, D, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    dst_row = tl.load(IDX + pid).to(tl.int64)
    offs = tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < D
    vals = tl.load(DOUT + pid * D + offs, mask=mask, other=0.0)
    # This backward would need atomics if we scatter to DIN, but we will not use triton for backward gather; use torch scatter_add with deterministic.
    tl.store(DIN + dst_row * D + offs, vals, mask=mask)


class _GatherTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx, D):
        M = idx.shape[0]
        out=torch.empty(M, D, device=src.device, dtype=src.dtype)
        BLOCKG=64 if D>=64 else 32
        grid=(M, triton.cdiv(D, BLOCKG))
        _gather_kernel[grid](src, idx, out, D, BLOCK=BLOCKG)
        ctx.save_for_backward(idx)
        ctx.D=D
        ctx.src_shape=src.shape
        return out
    @staticmethod
    def backward(ctx, dout):
        idx, = ctx.saved_tensors
        D=ctx.D
        src_shape=ctx.src_shape
        # deterministic scatter via index_add (no atomics across runs, order is fixed)
        din=torch.zeros(src_shape, device=dout.device, dtype=dout.dtype)
        din = din.index_add(0, idx, dout)
        return din, None, None

@triton.jit
def _combine_bwd_dyp_kernel(DY, W, INV, DYP, T: tl.constexpr, A: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    offs = pid_d * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < D
    t = pid_m // A
    w = tl.load(W + pid_m)
    dy = tl.load(DY + t * D + offs, mask=mask, other=0.0)
    tl.store(DYP + pid_m * D + offs, dy * w, mask=mask)

class _CombineTrition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, yp, w_sorted, inv_order, T, A, D):
        BLOCKC=64 if D>=64 else 32
        y=torch.empty(T, D, device=yp.device, dtype=torch.float32)
        grid=(T, triton.cdiv(D, BLOCKC))
        ctx.save_for_backward(w_sorted, inv_order, yp)
        ctx.T=T; ctx.A=A; ctx.D=D
        if A<=8:
            _combine_kernel[grid](yp, w_sorted, inv_order, y, T, A, D, BLOCK=BLOCKC)
        else:
            _combine_kernel_general[grid](yp, w_sorted, inv_order, y, T, A, D, BLOCK=BLOCKC)
        return y
    @staticmethod
    def backward(ctx, dy):
        w_sorted, inv_order, yp = ctx.saved_tensors
        T=ctx.T; A=ctx.A; D=ctx.D
        M=T*A
        # dyp: first compute dyp_sorted = dy[t] * w  for each pos (need t per pos)
        # t per pos = inv_order ? flat = t*A+a, pos=inv[flat], so t = flat // A where flat = order[pos]
        # We have inv, need order = argsort(inv). Compute order via torch for backward (small, deterministic)
        order = torch.argsort(inv_order)
        # order[pos] = flat, t = flat // A
        # We can do: flat_of_pos = order, t_of_pos = flat_of_pos // A
        # So gather dy[t_of_pos]
        # Use triton kernel for dyp: we have pid_m = pos (0..M-1), load flat=order[pos], t=flat//A, load dy[t], w[pos]
        # Instead of extra kernel we use simple triton for dyp
        BLOCKC=64 if D>=64 else 32
        dyp=torch.empty(M, D, device=dy.device, dtype=torch.float32)
        # we need to map pos -> t via order
        # Build Tpos array on device for kernel: we can reuse inv? Instead we scatter
        # Use kernel that takes order directly
        # Fallback to torch for correctness and determinism:
        flat = order
        t_of_pos = flat // A
        dy_gather = dy[t_of_pos]  # [M,D]
        dyp = dy_gather * w_sorted.unsqueeze(1)
        # dw = sum_d dy[t] * yp[pos,d]  -> per pos scalar
        # dw per flat? w_sorted is per pos, gradient for w per pos
        # dw_sorted = sum_d dy[t] * yp[pos,d]
        # Compute via elementwise multiply and sum
        dw = (dy_gather * yp).sum(dim=1)  # [M]
        # Need to return gradients for yp, w_sorted, inv_order, T,A,D
        # yp gradient is dyp but need to be in original sorted order? dyp we computed is already per pos (sorted position)
        # That's correct: dyp corresponds to yp input which is sorted order yp input.
        # For w_sorted, gradient is dw per pos
        return dyp, dw, None, None, None, None

@triton.jit
def _combine_kernel_general(SRC, W, INV, Y, T: tl.constexpr, A: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    offs = pid_d * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < D
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for a in range(64):
        cond = a < A
        flat = pid_t * A + a
        pos = tl.load(INV + flat)
        w = tl.load(W + pos)
        src = tl.load(SRC + pos * D + offs, mask=mask, other=0.0)
        acc = tl.where(cond, acc + src * w, acc)
    tl.store(Y + pid_t * D + offs, acc, mask=mask)

def _mm_launch(a, b):
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

def _select_topA(s, router_b, A):
    E = s.shape[-1]
    b = s + router_b
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=s.device, dtype=torch.int64)
    keys = (u << 10) | (E - 1 - e_idx)
    return torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices

def _grouped_forward(a, b, offs, is_float32):
    M, K = a.shape
    E, K2, N = b.shape
    assert K == K2
    if M == 0:
        return torch.empty(M, N, device=a.device, dtype=torch.float32)
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    exps=[]; rows=[]; cols=[]
    prev=0
    for e in range(E):
        cur=int(offs[e].item())
        cnt=cur-prev
        if cnt>0:
            mt=triton.cdiv(cnt, BM)
            nt=triton.cdiv(N, BN)
            for mi in range(mt):
                r=prev+mi*BM
                for ni in range(nt):
                    exps.append(e); rows.append(r); cols.append(ni*BN)
        prev=cur
    total=len(exps)
    if total==0:
        c.zero_()
        return c
    exp_t=torch.tensor(exps, dtype=torch.int32, device=a.device)
    row_t=torch.tensor(rows, dtype=torch.int32, device=a.device)
    col_t=torch.tensor(cols, dtype=torch.int32, device=a.device)
    if is_float32:
        _grouped_fwd_kernel_ieee[(total,)](a, b, c, offs, exp_t, row_t, col_t, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), b.stride(2), c.stride(0), c.stride(1), BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=4, num_stages=2)
    else:
        _grouped_fwd_kernel[(total,)](a, b, c, offs, exp_t, row_t, col_t, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), b.stride(2), c.stride(0), c.stride(1), BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=4, num_stages=2)
    return c

class _GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, offs, is_float32):
        ctx.save_for_backward(a, b, offs)
        ctx.is_f32 = is_float32
        return _grouped_forward(a, b, offs, is_float32)
    @staticmethod
    def backward(ctx, dc):
        a, b, offs = ctx.saved_tensors
        is_f32 = ctx.is_f32
        M, K = a.shape
        E, K2, N = b.shape
        prev=0; cnts=[]
        for e in range(E):
            cur=int(offs[e].item())
            cnts.append(cur-prev)
            prev=cur
        # dA tiles
        exps=[]; rows=[]; kcols=[]
        prev2=0
        for e,cnt in enumerate(cnts):
            if cnt>0:
                mt=triton.cdiv(cnt, BM)
                kt=triton.cdiv(K, BK)
                for mi in range(mt):
                    rr=prev2+mi*BM
                    for ki in range(kt):
                        exps.append(e); rows.append(rr); kcols.append(ki*BK)
            prev2+=cnt
        total_da=len(exps)
        da=torch.empty_like(a, dtype=torch.float32)
        if total_da>0:
            exp_t=torch.tensor(exps, dtype=torch.int32, device=a.device)
            row_t=torch.tensor(rows, dtype=torch.int32, device=a.device)
            kcol_t=torch.tensor(kcols, dtype=torch.int32, device=a.device)
            if is_f32:
                _grouped_bwd_da_kernel_ieee[(total_da,)](dc, b, da, offs, exp_t, row_t, kcol_t, N, K, dc.stride(0), dc.stride(1), b.stride(0), b.stride(1), b.stride(2), da.stride(0), da.stride(1), BLOCK_M=BM, BLOCK_K=BK, BLOCK_N=BN, num_warps=4, num_stages=2)
            else:
                _grouped_bwd_da_kernel[(total_da,)](dc, b, da, offs, exp_t, row_t, kcol_t, N, K, dc.stride(0), dc.stride(1), b.stride(0), b.stride(1), b.stride(2), da.stride(0), da.stride(1), BLOCK_M=BM, BLOCK_K=BK, BLOCK_N=BN, num_warps=4, num_stages=2)
        else:
            da.zero_()
        db=torch.zeros_like(b, dtype=torch.float32)
        if M>0:
            grid=(E, triton.cdiv(K, BK)*triton.cdiv(N, BN))
            if is_f32:
                _grouped_bwd_db_kernel_ieee[grid](a, dc, db, offs, M, N, K, a.stride(0), a.stride(1), dc.stride(0), dc.stride(1), db.stride(0), db.stride(1), db.stride(2), BLOCK_K=BK, BLOCK_N=BN, BLOCK_M=BM)
            else:
                _grouped_bwd_db_kernel[grid](a, dc, db, offs, M, N, K, a.stride(0), a.stride(1), dc.stride(0), dc.stride(1), db.stride(0), db.stride(1), db.stride(2), BLOCK_K=BK, BLOCK_N=BN, BLOCK_M=BM)
        return da, db, None, None

class _FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A=int(A)
        with torch.enable_grad():
            xf=x.detach().float().requires_grad_(True)
            rwf=router_w.detach().float().requires_grad_(True)
            w1f=w1.detach().float().requires_grad_(True)
            w2f=w2.detach().float().requires_grad_(True)
            T,D=xf.shape
            E=rwf.shape[1]
            F=w1f.shape[2]//2
            s=torch.sigmoid(_MM.apply(xf, rwf))
            with torch.no_grad():
                sel=_select_topA(s.detach(), router_b.float(), A)
            s_sel=torch.gather(s,1,sel)
            w=s_sel / s_sel.sum(dim=-1, keepdim=True)
            flat=sel.reshape(-1)
            order=torch.argsort(flat, stable=True)
            tok=order // A
            offs=torch.bincount(flat, minlength=E).cumsum(0).to(torch.int32)
            xg=_GatherTriton.apply(xf, tok, D)
            is_f32=(x.dtype==torch.float32)
            hg=_GroupedMM.apply(xg, w1f, offs, is_f32)
            act=_SiLUMul.apply(hg[:,:F].contiguous(), hg[:,F:].contiguous())
            yp=_GroupedMM.apply(act, w2f, offs, is_f32)
            # triton combine instead of index_add (deterministic, fast)
            M=T*A
            inv_order=torch.empty_like(order)
            inv_order[order]=torch.arange(M, device=x.device)
            w_sorted=w.reshape(-1).index_select(0, order).to(torch.float32)
            y=_CombineTrition.apply(yp, w_sorted, inv_order, T, A, D)
        ctx.save_for_backward(xf, rwf, w1f, w2f, y)
        ctx.dtypes=(x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.detach().to(x.dtype)
    @staticmethod
    def backward(ctx, dy):
        xf,rwf,w1f,w2f,y=ctx.saved_tensors
        gx,grw,gw1,gw2=torch.autograd.grad(y, (xf,rwf,w1f,w2f), dy.float())
        dx_t,drw_t,dw1_t,dw2_t=ctx.dtypes
        return (gx.to(dx_t), grw.to(drw_t), None, gw1.to(dw1_t), gw2.to(dw2_t), None)

def fused_moe(x, router_w, router_b, w1, w2, A):
    return _FusedMoe.apply(x, router_w, router_b, w1, w2, A)
