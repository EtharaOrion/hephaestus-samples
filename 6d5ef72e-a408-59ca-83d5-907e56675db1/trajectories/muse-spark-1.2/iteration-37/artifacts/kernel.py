import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32
EBLOCK = 2048
GBLOCK = 1024

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
def _gather_scores_kernel(S, SEL, OUT, E, A: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < A
    sel = tl.load(SEL + pid * A + offs, mask=mask)
    v = tl.load(S + pid * E + sel, mask=mask, other=0.0)
    tl.store(OUT + pid * A + offs, v, mask=mask)

@triton.jit
def _scatter_scores_kernel(DOUT, SEL, DS, E, A: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < A
    sel = tl.load(SEL + pid * A + offs, mask=mask)
    d = tl.load(DOUT + pid * A + offs, mask=mask, other=0.0)
    tl.store(DS + pid * E + sel, d, mask=mask)

@triton.jit
def _gather_rows_kernel(SRC, DST, IDX, D: tl.constexpr, BLOCK: tl.constexpr):
    pid_row = tl.program_id(0).to(tl.int64)
    pid_blk = tl.program_id(1).to(tl.int64)
    idx = tl.load(IDX + pid_row).to(tl.int64)
    offs = pid_blk * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < D
    v = tl.load(SRC + idx * D + offs, mask=mask, other=0.0)
    tl.store(DST + pid_row * D + offs, v, mask=mask)

@triton.jit
def _combine_fwd_kernel(YP, W, OUT, D: tl.constexpr, A: tl.constexpr, BLOCK: tl.constexpr):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_blk = tl.program_id(1).to(tl.int64)
    offs = pid_blk * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < D
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for a in range(A):
        w = tl.load(W + pid_t * A + a)
        yp = tl.load(YP + (pid_t * A + a) * D + offs, mask=mask, other=0.0)
        acc += w * yp
    tl.store(OUT + pid_t * D + offs, acc, mask=mask)

@triton.jit
def _reduce_rows_kernel(SRC, DST, D: tl.constexpr, A: tl.constexpr, BLOCK: tl.constexpr):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_blk = tl.program_id(1).to(tl.int64)
    offs = pid_blk * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < D
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for a in range(A):
        v = tl.load(SRC + (pid_t * A + a) * D + offs, mask=mask, other=0.0)
        acc += v
    tl.store(DST + pid_t * D + offs, acc, mask=mask)

@triton.jit
def _bwd_yp_kernel(DY, W, DYP, D: tl.constexpr, A: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    pid_blk = tl.program_id(1).to(tl.int64)
    offs = pid_blk * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < D
    t = (pid // A).to(tl.int64)
    w = tl.load(W + pid)
    dy = tl.load(DY + t * D + offs, mask=mask, other=0.0)
    tl.store(DYP + pid * D + offs, dy * w, mask=mask)


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

def _select_topA(s: torch.Tensor, router_b: torch.Tensor, A: int) -> torch.Tensor:
    E = s.shape[-1]
    b = s + router_b
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=s.device, dtype=torch.int64)
    keys = (u << 20) | (E - 1 - e_idx)
    return torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices

def _gather_scores_launch(s, sel):
    T, E = s.shape
    A = sel.shape[1]
    out = torch.empty(T, A, device=s.device, dtype=s.dtype)
    BLOCK = triton.next_power_of_2(A)
    if BLOCK < 16:
        BLOCK = 16
    _gather_scores_kernel[(T,)](s, sel, out, E, A, BLOCK=BLOCK)
    return out

def _scatter_scores_launch(dout, sel, E):
    T, A = dout.shape
    ds = torch.zeros(T, E, device=dout.device, dtype=dout.dtype)
    BLOCK = triton.next_power_of_2(A)
    if BLOCK < 16:
        BLOCK = 16
    _scatter_scores_kernel[(T,)](dout, sel, ds, E, A, BLOCK=BLOCK)
    return ds

def _gather_rows_launch(src, idx):
    N = idx.shape[0]
    D = src.shape[1]
    dst = torch.empty(N, D, device=src.device, dtype=src.dtype)
    BLOCK = GBLOCK
    if D <= 512:
        BLOCK = 512
    elif D <= 1024:
        BLOCK = 1024
    else:
        BLOCK = 1024
    grid = (N, triton.cdiv(D, BLOCK))
    _gather_rows_kernel[grid](src, dst, idx, D, BLOCK=BLOCK)
    return dst

def _combine_fwd_launch(yp_unsorted, w, T, D, A):
    out = torch.empty(T, D, device=yp_unsorted.device, dtype=yp_unsorted.dtype)
    BLOCK = 1024
    if D < 512:
        BLOCK = 512
    grid = (T, triton.cdiv(D, BLOCK))
    _combine_fwd_kernel[grid](yp_unsorted, w, out, D, A, BLOCK=BLOCK)
    return out

def _reduce_rows_launch(src, T, D, A):
    dst = torch.empty(T, D, device=src.device, dtype=src.dtype)
    BLOCK = 1024
    if D < 512:
        BLOCK = 512
    grid = (T, triton.cdiv(D, BLOCK))
    _reduce_rows_kernel[grid](src, dst, D, A, BLOCK=BLOCK)
    return dst

def _bwd_yp_launch(dy, w, N, D, A):
    dst = torch.empty(N, D, device=dy.device, dtype=dy.dtype)
    BLOCK = 1024
    if D < 512:
        BLOCK = 512
    grid = (N, triton.cdiv(D, BLOCK))
    _bwd_yp_kernel[grid](dy, w, dst, D, A, BLOCK=BLOCK)
    return dst

class _FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        xf = x.detach().float()
        rwf = router_w.detach().float()
        w1f = w1.detach().float()
        w2f = w2.detach().float()
        T, D = xf.shape
        E = rwf.shape[1]
        F = w1f.shape[2] // 2
        # router
        logits = _MM.apply(xf, rwf)
        s = torch.sigmoid(logits)
        with torch.no_grad():
            sel = _select_topA(s.detach(), router_b.float(), A)
        s_sel = _gather_scores_launch(s, sel)
        w = s_sel / s_sel.sum(dim=-1, keepdim=True)
        # sorted dispatch
        flat = sel.reshape(-1)
        order = torch.argsort(flat, stable=True)
        inv_order = torch.argsort(order, stable=True)
        # tok for sorted
        tok_sorted = (order // A).to(torch.int64)
        xg_sorted = _gather_rows_launch(xf, tok_sorted)
        # per-expert compute with offsets
        cnt = torch.bincount(flat, minlength=E)
        offs = cnt.cumsum(0)
        # build yp_sorted
        N = T * A
        yp_sorted = torch.empty(N, D, device=xf.device, dtype=torch.float32)
        w1s = w1f.unbind(0)
        w2s = w2f.unbind(0)
        # need to keep intermediates for backward
        gu_list = []
        act_list = []
        xg_list = []
        # iterate
        start = 0
        for e in range(E):
            n_e = int(cnt[e].item())
            if n_e == 0:
                gu_list.append(None)
                act_list.append(None)
                xg_list.append(None)
                continue
            xg_e = xg_sorted.narrow(0, start, n_e)
            xg_list.append(xg_e)
            gu = _MM.apply(xg_e, w1s[e])
            gu_list.append(gu)
            act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
            act_list.append(act)
            yp_e = _MM.apply(act, w2s[e])
            yp_sorted.narrow(0, start, n_e).copy_(yp_e)
            start += n_e
        # unsort
        yp_unsorted = _gather_rows_launch(yp_sorted, inv_order)
        # combine
        y = _combine_fwd_launch(yp_unsorted, w, T, D, A)
        # save for backward
        ctx.save_for_backward(xf, rwf, w1f, w2f, s, sel, s_sel, w, order, inv_order, tok_sorted, xg_sorted, yp_sorted, yp_unsorted, offs, cnt)
        ctx.gu_list = gu_list
        ctx.act_list = act_list
        ctx.xg_list = xg_list
        ctx.T = T; ctx.D = D; ctx.E = E; ctx.F = F; ctx.A = A; ctx.N = N
        ctx.dtypes = (x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, rwf, w1f, w2f, s, sel, s_sel, w, order, inv_order, tok_sorted, xg_sorted, yp_sorted, yp_unsorted, offs, cnt = ctx.saved_tensors
        gu_list = ctx.gu_list
        act_list = ctx.act_list
        xg_list = ctx.xg_list  # noqa: F841
        T = ctx.T; D = ctx.D; E = ctx.E; F = ctx.F; A = ctx.A; N = ctx.N
        dy_f = dy.float().contiguous()
        # combine backward
        # d_yp_unsorted = dy * w  (broadcast)
        d_yp_unsorted = _bwd_yp_launch(dy_f, w.reshape(-1), N, D, A)
        # d_w
        # dw = sum_d dy * yp_unsorted  per (t,a)
        dw_flat = (dy_f.repeat_interleave(A, dim=0) * yp_unsorted).sum(dim=1) if False else ( (dy_f.unsqueeze(1).expand(T, A, D).reshape(N, D) * yp_unsorted).sum(dim=1) )
        # Above uses repeat which is framework but small; deterministic sum order is deterministic
        # compute dw_flat via torch (deterministic)
        dy_expanded = dy_f.unsqueeze(1).expand(T, A, D).reshape(N, D)
        dw_flat = (dy_expanded * yp_unsorted).sum(dim=1)
        dw = dw_flat.view(T, A)
        # permute d_yp to sorted
        d_yp_sorted = _gather_rows_launch(d_yp_unsorted, order)
        # expert backward
        dw1 = torch.zeros_like(w1f)
        dw2 = torch.zeros_like(w2f)
        d_xg_sorted = torch.empty_like(xg_sorted)
        w1s = w1f.unbind(0)
        w2s = w2f.unbind(0)
        start = 0
        for e in range(E):
            n_e = int(cnt[e].item())
            if n_e == 0:
                continue
            dyp_e = d_yp_sorted.narrow(0, start, n_e)
            act_e = act_list[e]
            gu_e = gu_list[e]
            xg_e = xg_sorted.narrow(0, start, n_e)
            # dw2
            dw2[e].copy_(_mm_launch(act_e.t(), dyp_e))
            # d_act
            d_act = _mm_launch(dyp_e, w2s[e].t())
            # silu backward
            g = gu_e[:, :F].contiguous()
            u = gu_e[:, F:].contiguous()
            # use silu bwd kernel via _SiLUMul backward? Compute manually via kernel launch
            # we can call _SiLUMul backward via autograd trick: create tensors and use kernel
            n = g.numel()
            dg = torch.empty_like(g)
            du = torch.empty_like(u)
            _silu_mul_bwd_kernel[(triton.cdiv(n, EBLOCK),)](g, u, d_act, dg, du, n, BLOCK=EBLOCK)
            d_gu = torch.cat([dg, du], dim=1)
            dw1[e].copy_(_mm_launch(xg_e.t(), d_gu))
            dxg_e = _mm_launch(d_gu, w1s[e].t())
            d_xg_sorted.narrow(0, start, n_e).copy_(dxg_e)
            start += n_e
        # dx from expert path: reduce d_xg_sorted -> dx_expert
        d_xg_unsorted = _gather_rows_launch(d_xg_sorted, inv_order)
        dx_expert = _reduce_rows_launch(d_xg_unsorted, T, D, A)
        # router path
        # dw -> d_s_sel
        # dw is gradient w.r.t w, need d_s_sel
        # per token: S = sum s_sel, dot = sum dw * w
        S = s_sel.sum(dim=1, keepdim=True)  # [T,1]
        dot = (dw * w).sum(dim=1, keepdim=True)  # [T,1]
        d_s_sel = (dw - dot) / S  # [T,A]
        # scatter to d_s
        d_s = _scatter_scores_launch(d_s_sel, sel, E)
        # sigmoid derivative
        dsig = s * (1 - s)
        d_logits = d_s * dsig
        # dx_router = d_logits @ rw.T
        dx_router = _mm_launch(d_logits, rwf.t())
        # drw = xf.T @ d_logits
        drw = _mm_launch(xf.t(), d_logits)
        dx_total = dx_expert + dx_router
        # w1/w2 already computed, dw1/dw2 done
        dx_t, drw_t, dw1_t, dw2_t = ctx.dtypes
        return dx_total.to(dx_t), drw.to(drw_t), None, dw1.to(dw1_t), dw2.to(dw2_t), None

def fused_moe(x, router_w, router_b, w1, w2, A):
    return _FusedMoe.apply(x, router_w, router_b, w1, w2, A)
