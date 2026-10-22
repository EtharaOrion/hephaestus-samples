import torch
import triton
import triton.language as tl

EBLOCK = 1024

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

@triton.jit
def _mm_fast_kernel(A, B, C, M, N, K,
               sam, sak, sbk, sbn, scm, scn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak,
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn,
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))

def _mm_fast(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0 or K == 0:
        return c
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    if M < 64:
        BLOCK_M = 32
    if N < 64:
        BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _mm_fast_kernel[grid](a, b, c, M, N, K,
                          a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                          c.stride(0), c.stride(1),
                          BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c

class _FastMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _mm_fast(a, b)
    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        dc = dc.contiguous()
        da = _mm_fast(dc, b.t().contiguous())
        db = _mm_fast(a.t().contiguous(), dc)
        return da, db

# ---------- grouped forward / dA kernels ----------
@triton.jit
def _grouped_forward_kernel(A, B, C,
                            stride_am, stride_ak,
                            stride_be, stride_bk, stride_bn,
                            stride_cm, stride_cn,
                            offs_ptr, exp_ids_ptr, block_starts_ptr,
                            K, N,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(exp_ids_ptr + pid_m)
    bstart = tl.load(block_starts_ptr + pid_m)
    off_e = tl.load(offs_ptr + e)
    rows = bstart + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_row = rows < off_e
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        kcur = k + tl.arange(0, BLOCK_K)
        mask_k = kcur < K
        a = tl.load(A + rows[:, None] * stride_am + kcur[None, :] * stride_ak,
                    mask=mask_row[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(B + e * stride_be + kcur[:, None] * stride_bk + cols[None, :] * stride_bn,
                    mask=mask_k[:, None] & (cols[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(C + rows[:, None] * stride_cm + cols[None, :] * stride_cn, acc,
             mask=mask_row[:, None] & (cols[None, :] < N))

@triton.jit
def _grouped_dA_kernel(DC, B, DA,
                       stride_dcm, stride_dcn,
                       stride_be, stride_bk, stride_bn,
                       stride_dam, stride_dak,
                       offs_ptr, exp_ids_ptr, block_starts_ptr,
                       K, N,
                       BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    e = tl.load(exp_ids_ptr + pid_m)
    bstart = tl.load(block_starts_ptr + pid_m)
    off_e = tl.load(offs_ptr + e)
    rows = bstart + tl.arange(0, BLOCK_M)
    kcols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_row = rows < off_e
    mask_k = kcols < K
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n in range(0, N, BLOCK_N):
        ncur = n + tl.arange(0, BLOCK_N)
        mask_n = ncur < N
        dc = tl.load(DC + rows[:, None] * stride_dcm + ncur[None, :] * stride_dcn,
                     mask=mask_row[:, None] & mask_n[None, :], other=0.0)
        bt = tl.load(B + e * stride_be + kcols[None, :] * stride_bk + ncur[:, None] * stride_bn,
                     mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(dc, bt, acc, input_precision="ieee")
    tl.store(DA + rows[:, None] * stride_dam + kcols[None, :] * stride_dak, acc,
             mask=mask_row[:, None] & mask_k[None, :])

@triton.jit
def _grouped_dB_kernel(A, DC, DB,
                       stride_am, stride_ak,
                       stride_dcm, stride_dcn,
                       stride_dbe, stride_dbk, stride_dbn,
                       offs_ptr,
                       K, N, E,
                       BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    nKB = tl.cdiv(K, BLOCK_K)
    e = pid_m // nKB
    kb = pid_m % nKB
    k_off = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_k = k_off < K
    mask_n = n_off < N
    if e >= E:
        return
    start = tl.where(e == 0, 0, tl.load(offs_ptr + e - 1))
    end = tl.load(offs_ptr + e)
    m_len = end - start
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for m in range(0, m_len, BLOCK_M):
        rows = start + m + tl.arange(0, BLOCK_M)
        mask_m = rows < end
        a = tl.load(A + rows[:, None] * stride_am + k_off[None, :] * stride_ak,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        dc = tl.load(DC + rows[:, None] * stride_dcm + n_off[None, :] * stride_dcn,
                     mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        a_t = tl.trans(a)
        acc = tl.dot(a_t, dc, acc, input_precision="ieee")
    tl.store(DB + e * stride_dbe + k_off[:, None] * stride_dbk + n_off[None, :] * stride_dbn, acc,
             mask=mask_k[:, None] & mask_n[None, :])

def _build_grouped_meta(offs: torch.Tensor, BLOCK_M: int):
    # offs: [E] int32 cumsum
    E = offs.numel()
    offs_list = offs.tolist()
    exp_ids = []
    block_starts = []
    prev = 0
    for e, cur in enumerate(offs_list):
        cnt = cur - prev
        prev = cur
        if cnt == 0:
            continue
        start = cur - cnt
        nb = (cnt + BLOCK_M - 1) // BLOCK_M
        for b in range(nb):
            exp_ids.append(e)
            block_starts.append(start + b * BLOCK_M)
    if len(exp_ids) == 0:
        return None, None, 0
    exp_ids_t = torch.tensor(exp_ids, dtype=torch.int32, device=offs.device)
    block_starts_t = torch.tensor(block_starts, dtype=torch.int32, device=offs.device)
    return exp_ids_t, block_starts_t, len(exp_ids)

class _GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, offs):
        M_total, K = a.shape
        E, Kb, N = b.shape
        assert Kb == K
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        if M_total < 64:
            BLOCK_M = 32
        if N < 64:
            BLOCK_N = 32
        if K < 32:
            BLOCK_K = 16
        exp_ids, block_starts, num_blocks = _build_grouped_meta(offs, BLOCK_M)
        c = torch.empty(M_total, N, device=a.device, dtype=torch.float32)
        if M_total == 0 or N == 0 or num_blocks == 0:
            ctx.save_for_backward(a, b, offs)
            ctx.meta = (exp_ids, block_starts, BLOCK_M, BLOCK_N, BLOCK_K)
            ctx.shapes = (M_total, K, N, E)
            if num_blocks == 0:
                c.zero_()
            return c
        grid = (num_blocks, triton.cdiv(N, BLOCK_N))
        _grouped_forward_kernel[grid](a, b, c,
                                      a.stride(0), a.stride(1),
                                      b.stride(0), b.stride(1), b.stride(2),
                                      c.stride(0), c.stride(1),
                                      offs, exp_ids, block_starts,
                                      K, N,
                                      BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        ctx.save_for_backward(a, b, offs)
        ctx.meta = (exp_ids, block_starts, BLOCK_M, BLOCK_N, BLOCK_K)
        ctx.shapes = (M_total, K, N, E)
        return c

    @staticmethod
    def backward(ctx, dc):
        a, b, offs = ctx.saved_tensors
        exp_ids, block_starts, BLOCK_M, BLOCK_N, BLOCK_K = ctx.meta
        M_total, K, N, E = ctx.shapes
        dc = dc.contiguous()
        da = torch.empty_like(a)
        db = torch.zeros_like(b)
        offs_list = offs.tolist()
        prev = 0
        for e, cur in enumerate(offs_list):
            cnt = cur - prev
            start = prev
            prev = cur
            if cnt == 0:
                continue
            dc_e = dc[start:cur]
            a_e = a[start:cur]
            bt = b[e].t().contiguous()
            da[start:cur] = _mm_fast(dc_e, bt)
            at = a_e.t().contiguous()
            db[e] = _mm_fast(at, dc_e)
        if M_total == 0 or exp_ids is None:
            da.zero_()
        return da, db, None

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
            # router matmul in float via torch (small)
            s = torch.sigmoid(torch.matmul(xf, rwf))
            with torch.no_grad():
                sel = _select_topA(s.detach(), router_b.float(), A)
            s_sel = torch.gather(s, 1, sel)
            w = s_sel / s_sel.sum(dim=-1, keepdim=True)
            # build routing metadata for grouped compute
            sel_flat = sel.reshape(-1)  # [M_total]
            w_flat = w.reshape(-1)  # [M_total]
            order = torch.argsort(sel_flat, stable=True)  # sort by expert
            counts = torch.bincount(sel_flat, minlength=E)
            offs = torch.cumsum(counts, 0).to(torch.int32)
            x_rep = xf.repeat_interleave(A, dim=0)  # [M_total, D]
            xg = x_rep[order]  # [M_total, D]
            w_sorted = w_flat[order]  # [M_total]
            # grouped expert compute
            # w1: [E, D, 2F] -> need contiguous; group mm expects [E, K, N] where K=D, N=2F
            gu = _GroupedMM.apply(xg, w1f, offs)  # [M_total, 2F]
            act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())  # [M_total, F]
            # w2: [E, F, D] => K=F, N=D
            yp_sorted = _GroupedMM.apply(act, w2f, offs)  # [M_total, D]
            # weight and scatter back
            yp_w_sorted = yp_sorted * w_sorted.unsqueeze(1)
            # unsort to token-major
            inv_order = torch.argsort(order)
            yp_w = yp_w_sorted[inv_order]  # [M_total, D]
            # sum per token: index_add would be racy; use contiguous view sum (deterministic)
            # dx routing uses repeat_interleave's deterministic scatter-add, no atomics
            y = yp_w.view(T, A, D).sum(dim=1)  # [T, D]
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
