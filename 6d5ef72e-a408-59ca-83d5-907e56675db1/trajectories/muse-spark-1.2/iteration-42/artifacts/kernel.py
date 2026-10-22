import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32
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
        acc += tl.dot(a, b, input_precision="ieee")
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


@triton.jit
def _gather_norm_kernel(S, Sel, W, T, E, A, BLOCK_A: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return
    offs_a = tl.arange(0, BLOCK_A)
    mask_a = offs_a < A
    sel = tl.load(Sel + pid * A + offs_a, mask=mask_a, other=0)
    s_vals = tl.load(S + pid * E + sel, mask=mask_a, other=0.0)
    s_sum = tl.sum(s_vals, axis=0)
    w = s_vals * (1.0 / s_sum)
    tl.store(W + pid * A + offs_a, w, mask=mask_a)


@triton.jit
def _gather_norm_bwd_kernel(Sel, W, S, DW, DG_S, T, E, A, BLOCK_A: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return
    offs_a = tl.arange(0, BLOCK_A)
    mask_a = offs_a < A
    sel = tl.load(Sel + pid * A + offs_a, mask=mask_a, other=0)
    w_vals = tl.load(W + pid * A + offs_a, mask=mask_a, other=0.0)
    dw = tl.load(DW + pid * A + offs_a, mask=mask_a, other=0.0)
    s_vals = tl.load(S + pid * E + sel, mask=mask_a, other=0.0)
    s_sum = tl.sum(s_vals, axis=0)
    inv = 1.0 / s_sum
    dot = tl.sum(dw * w_vals, axis=0)
    grad_sel = (dw - dot) * inv
    tl.store(DG_S + pid * A + offs_a, grad_sel, mask=mask_a)


@triton.jit
def _gather_rows_kernel(SRC, IDX, DST, D, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    row = tl.load(IDX + pid).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    vals = tl.load(SRC + row * D + offs, mask=mask, other=0.0)
    tl.store(DST + pid * D + offs, vals, mask=mask)


@triton.jit
def _scale_rows_kernel(X, W, OUT, N, D, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    w = tl.load(W + pid)
    x = tl.load(X + pid * D + offs, mask=mask, other=0.0)
    tl.store(OUT + pid * D + offs, x * w, mask=mask)


@triton.jit
def _scatter_add_kernel(SRC, IDX, DST, D, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    row = tl.load(IDX + pid).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    src = tl.load(SRC + pid * D + offs, mask=mask, other=0.0)
    dst_ptr = DST + row * D + offs
    cur = tl.load(dst_ptr, mask=mask, other=0.0)
    tl.store(dst_ptr, cur + src, mask=mask)


@triton.jit
def _gather_weight_kernel(Sel, W, Rows, Out, E_IDX: tl.constexpr, A: tl.constexpr, BLOCK_A: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    row = tl.load(Rows + pid).to(tl.int64)
    offs = tl.arange(0, BLOCK_A)
    mask = offs < A
    sel_vals = tl.load(Sel + row * A + offs, mask=mask, other=-1)
    w_vals = tl.load(W + row * A + offs, mask=mask, other=0.0)
    hit = sel_vals == E_IDX
    w = tl.sum(tl.where(hit, w_vals, 0.0), axis=0)
    tl.store(Out + pid, w)


@triton.jit
def _member_kernel(Sel, Member, T, E, A, BLOCK_A: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return
    offs_a = tl.arange(0, BLOCK_A)
    mask_a = offs_a < A
    sel = tl.load(Sel + pid * A + offs_a, mask=mask_a, other=0)
    for a in range(0, BLOCK_A):
        if a < A:
            e = tl.load(Sel + pid * A + a)
            tl.store(Member + pid * E + e, True)


@triton.jit
def _fill_rows_kernel(Member, Rows, E_IDX: tl.constexpr, T, E):
    pid = tl.program_id(0).to(tl.int64)
    # This kernel is launched with 1 program that loops over T to fill rows sequentially
    # Use single program for deterministic order
    cnt = 0
    for t in range(0, T):
        v = tl.load(Member + t * E + E_IDX)
        if v:
            tl.store(Rows + cnt, t)
            cnt += 1


def _mm_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    _mm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a, b, c, M, N, K,
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


class _GatherNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, sel):
        ctx.save_for_backward(s, sel)
        T = s.shape[0]
        A = sel.shape[1]
        w = torch.empty(T, A, device=s.device, dtype=torch.float32)
        BLOCK_A = triton.next_power_of_2(A)
        if BLOCK_A < 32:
            BLOCK_A = 32
        _gather_norm_kernel[(T,)](s, sel, w, T, s.shape[1], A, BLOCK_A=BLOCK_A)
        ctx.w = w
        return w
    @staticmethod
    def backward(ctx, dw):
        s, sel = ctx.saved_tensors
        w = ctx.w
        dw = dw.contiguous()
        T, E = s.shape
        A = sel.shape[1]
        BLOCK_A = triton.next_power_of_2(A)
        if BLOCK_A < 32:
            BLOCK_A = 32
        grad_sel = torch.empty(T, A, device=s.device, dtype=s.dtype)
        _gather_norm_bwd_kernel[(T,)](sel, w, s, dw, grad_sel, T, E, A, BLOCK_A=BLOCK_A)
        grad_s = torch.zeros_like(s)
        rows = torch.arange(T, device=s.device).repeat_interleave(A)
        full_idx = rows * E + sel.reshape(-1)
        grad_s.view(-1).scatter_add_(0, full_idx, grad_sel.reshape(-1))
        return grad_s, None


class _GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx):
        ctx.save_for_backward(idx)
        ctx.src_shape = src.shape
        N = idx.shape[0]
        D = src.shape[1]
        dst = torch.empty(N, D, device=src.device, dtype=src.dtype)
        if N == 0:
            return dst
        BLOCK_D = triton.next_power_of_2(D)
        if BLOCK_D > 1024:
            BLOCK_D = 1024
        _gather_rows_kernel[(N,)](src, idx, dst, D, BLOCK_D=BLOCK_D)
        return dst
    @staticmethod
    def backward(ctx, grad_out):
        (idx,) = ctx.saved_tensors
        T, D = ctx.src_shape
        N = idx.shape[0]
        grad_src = torch.zeros(T, D, device=grad_out.device, dtype=grad_out.dtype)
        if N == 0:
            return grad_src, None
        BLOCK_D = triton.next_power_of_2(D)
        if BLOCK_D > 1024:
            BLOCK_D = 1024
        _scatter_add_kernel[(N,)](grad_out.contiguous(), idx, grad_src, D, BLOCK_D=BLOCK_D)
        return grad_src, None


class _ScaleRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, w):
        ctx.save_for_backward(src, w)
        N, D = src.shape
        out = torch.empty_like(src)
        if N == 0:
            return out
        BLOCK_D = triton.next_power_of_2(D)
        if BLOCK_D > 1024:
            BLOCK_D = 1024
        _scale_rows_kernel[(N,)](src, w, out, N, D, BLOCK_D=BLOCK_D)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        src, w = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        N, D = src.shape
        BLOCK_D = triton.next_power_of_2(D)
        if BLOCK_D > 1024:
            BLOCK_D = 1024
        grad_src = torch.empty_like(src)
        if N != 0:
            _scale_rows_kernel[(N,)](grad_out, w, grad_src, N, D, BLOCK_D=BLOCK_D)
        else:
            grad_src = torch.empty_like(src)
        grad_w = (grad_out * src).sum(dim=1) if N != 0 else torch.empty(0, device=src.device, dtype=src.dtype)
        return grad_src, grad_w


class _ScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dst, src, idx):
        ctx.save_for_backward(idx)
        ctx.dst_shape = dst.shape
        out = dst.clone()
        N = idx.shape[0]
        D = dst.shape[1]
        if N != 0:
            BLOCK_D = triton.next_power_of_2(D)
            if BLOCK_D > 1024:
                BLOCK_D = 1024
            _scatter_add_kernel[(N,)](src.contiguous(), idx, out, D, BLOCK_D=BLOCK_D)
        ctx.N = N
        return out
    @staticmethod
    def backward(ctx, grad_out):
        (idx,) = ctx.saved_tensors
        N = ctx.N
        if N == 0:
            return grad_out, None, None
        D = grad_out.shape[1]
        BLOCK_D = triton.next_power_of_2(D)
        if BLOCK_D > 1024:
            BLOCK_D = 1024
        grad_src = torch.empty(N, D, device=grad_out.device, dtype=grad_out.dtype)
        _gather_rows_kernel[(N,)](grad_out.contiguous(), idx, grad_src, D, BLOCK_D=BLOCK_D)
        return grad_out, grad_src, None


class _GatherWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sel, w, rows, e):
        ctx.save_for_backward(sel, w, rows)
        ctx.e = e
        N = rows.shape[0]
        A = sel.shape[1]
        out = torch.empty(N, device=sel.device, dtype=w.dtype)
        if N == 0:
            return out
        BLOCK_A = triton.next_power_of_2(A)
        if BLOCK_A < 32:
            BLOCK_A = 32
        _gather_weight_kernel[(N,)](sel, w, rows, out, E_IDX=e, A=A, BLOCK_A=BLOCK_A)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        sel, w, rows = ctx.saved_tensors
        e = ctx.e
        grad_w = torch.zeros_like(w)
        N = rows.shape[0]
        if N != 0:
            sel_rows = sel.index_select(0, rows)
            mask = sel_rows == e
            col = mask.float().argmax(dim=1)
            flat_idx = rows * w.shape[1] + col
            grad_w.view(-1).scatter_(0, flat_idx, grad_out)
        return None, grad_w, None, None


def _select_topA(s: torch.Tensor, router_b: torch.Tensor, A: int) -> torch.Tensor:
    E = s.shape[-1]
    b = s + router_b
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=s.device, dtype=torch.int64)
    shift = max(1, (E - 1).bit_length())
    keys = (u << shift) | (E - 1 - e_idx)
    return torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices



@triton.jit
def _gather_weight_scale_kernel(FE, Sel, W, Rows, Out, D, E_IDX: tl.constexpr, A: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_A: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    row = tl.load(Rows + pid).to(tl.int64)
    offs_a = tl.arange(0, BLOCK_A)
    mask_a = offs_a < A
    sel_vals = tl.load(Sel + row * A + offs_a, mask=mask_a, other=-1)
    w_vals = tl.load(W + row * A + offs_a, mask=mask_a, other=0.0)
    hit = sel_vals == E_IDX
    w = tl.sum(tl.where(hit, w_vals, 0.0), axis=0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    fe = tl.load(FE + pid * D + offs_d, mask=mask_d, other=0.0)
    tl.store(Out + pid * D + offs_d, fe * w, mask=mask_d)

class _GatherWeightScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fe, sel, w, rows, e):
        ctx.save_for_backward(fe, sel, w, rows)
        ctx.e = e
        N, D = fe.shape
        A = sel.shape[1]
        out = torch.empty_like(fe)
        if N == 0:
            return out
        BLOCK_D = triton.next_power_of_2(D)
        if BLOCK_D > 1024: BLOCK_D = 1024
        BLOCK_A = triton.next_power_of_2(A)
        if BLOCK_A < 32: BLOCK_A = 32
        _gather_weight_scale_kernel[(N,)](fe, sel, w, rows, out, D, E_IDX=e, A=A, BLOCK_D=BLOCK_D, BLOCK_A=BLOCK_A)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        fe, sel, w, rows = ctx.saved_tensors
        e = ctx.e
        N, D = fe.shape
        A = sel.shape[1]
        grad_out = grad_out.contiguous()
        BLOCK_D = triton.next_power_of_2(D)
        if BLOCK_D > 1024: BLOCK_D = 1024
        BLOCK_A = triton.next_power_of_2(A)
        if BLOCK_A < 32: BLOCK_A = 32
        w_e = torch.empty(N, device=sel.device, dtype=w.dtype)
        if N != 0:
            _gather_weight_kernel[(N,)](sel, w, rows, w_e, E_IDX=e, A=A, BLOCK_A=BLOCK_A)
        grad_fe = torch.empty_like(fe)
        if N != 0:
            _scale_rows_kernel[(N,)](grad_out, w_e, grad_fe, N, D, BLOCK_D=BLOCK_D)
        else:
            grad_fe = torch.empty_like(fe)
        grad_w = torch.zeros_like(w)
        if N != 0:
            gw_row = (grad_out * fe).sum(dim=1)
            sel_rows = sel.index_select(0, rows)
            mask = sel_rows == e
            col = mask.float().argmax(dim=1)
            flat_idx = rows * w.shape[1] + col
            grad_w.view(-1).scatter_(0, flat_idx, gw_row)
        return grad_fe, None, grad_w, None, None

def _build_member(sel: torch.Tensor, E: int) -> torch.Tensor:
    T, A = sel.shape
    member = torch.zeros(T, E, dtype=torch.bool, device=sel.device)
    BLOCK_A = triton.next_power_of_2(A)
    if BLOCK_A < 32:
        BLOCK_A = 32
    _member_kernel[(T,)](sel, member, T, E, A, BLOCK_A=BLOCK_A)
    return member


def _rows_for_expert(member: torch.Tensor, e: int) -> torch.Tensor:
    T, E = member.shape
    cnt = int(member[:, e].sum().item())
    rows = torch.empty(cnt, dtype=torch.int64, device=member.device)
    if cnt != 0:
        _fill_rows_kernel[(1,)](member, rows, E_IDX=e, T=T, E=E)
    return rows


class _FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        with torch.enable_grad():
            xf = x.detach().float().requires_grad_(True)
            rwf = router_w.detach().float().requires_grad_(True)
            w1f = w1.detach().float().requires_grad_(True)
            w2f = w2.detach().float().requires_grad_(True)
            s = torch.sigmoid(_MM.apply(xf, rwf))
            w1s = w1f.unbind(0)
            w2s = w2f.unbind(0)
            with torch.no_grad():
                sel = _select_topA(s.detach(), router_b.float(), A)
            w = _GatherNorm.apply(s, sel)
            member = _build_member(sel, s.shape[1])
            y = xf.new_zeros(xf.shape[0], xf.shape[1])
            for e in range(s.shape[1]):
                rows = _rows_for_expert(member, e)
                if rows.numel() == 0:
                    continue
                h = _GatherRows.apply(xf, rows)
                gu = _MM.apply(h, w1s[e])
                F = gu.shape[1] // 2
                act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
                fe = _MM.apply(act, w2s[e])
                fe_scaled = _GatherWeightScale.apply(fe, sel, w, rows, e)
                y = _ScatterAdd.apply(y, fe_scaled, rows)
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
