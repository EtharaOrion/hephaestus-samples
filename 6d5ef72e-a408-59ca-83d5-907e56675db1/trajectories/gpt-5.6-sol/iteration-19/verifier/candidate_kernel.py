import torch
import triton
import triton.language as tl

BM = 64
BN = 64
BK = 32


@triton.jit
def _router_kernel(X, W, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                   BM_: tl.constexpr, BN_: tl.constexpr, BK_: tl.constexpr):
    rm = tl.program_id(0) * BM_ + tl.arange(0, BM_)
    rn = tl.program_id(1) * BN_ + tl.arange(0, BN_)
    acc = tl.zeros((BM_, BN_), tl.float32)
    for k in range(0, D, BK_):
        rk = k + tl.arange(0, BK_)
        a = tl.load(X + rm[:, None] * D + rk[None, :], mask=(rm[:, None] < T) & (rk[None, :] < D), other=0.0).to(tl.float32)
        b = tl.load(W + rk[:, None] * E + rn[None, :], mask=(rk[:, None] < D) & (rn[None, :] < E), other=0.0).to(tl.float32)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(S + rm[:, None] * E + rn[None, :], tl.sigmoid(acc), mask=(rm[:, None] < T) & (rn[None, :] < E))


@triton.jit
def _gather_kernel(X, TOK, OUT, N: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    q = tl.program_id(0)
    d = tl.program_id(1) * B + tl.arange(0, B)
    t = tl.load(TOK + q)
    v = tl.load(X + t * D + d, mask=d < D, other=0.0)
    tl.store(OUT + q * D + d, v, mask=d < D)


@triton.jit
def _ungather_kernel(DR, INV, DX, T: tl.constexpr, D: tl.constexpr,
                     A: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.zeros((B,), tl.float32)
    for a in range(0, A):
        q = tl.load(INV + t * A + a)
        acc += tl.load(DR + q * D + d, mask=d < D, other=0.0).to(tl.float32)
    tl.store(DX + t * D + d, acc, mask=d < D)


@triton.jit
def _make_tiles(OFF, TE, TS, E: tl.constexpr, CAP: tl.constexpr,
                BM_: tl.constexpr):
    out = 0
    for e in range(0, E):
        begin = tl.load(OFF + e)
        end = tl.load(OFF + e + 1)
        count = tl.cdiv(end - begin, BM_)
        for j in range(0, count):
            tl.store(TE + out, e)
            tl.store(TS + out, begin + j * BM_)
            out += 1
    for j in range(out, CAP):
        tl.store(TE + j, -1)
        tl.store(TS + j, 0)


@triton.jit
def _seg_fwd(A, W, OFF, TE, TS, C, N: tl.constexpr, K: tl.constexpr,
             M: tl.constexpr, E: tl.constexpr, BM_: tl.constexpr,
             BN_: tl.constexpr, BK_: tl.constexpr, BF: tl.constexpr,
             SPLIT: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TE + tile)
    rn = tl.program_id(1) * BN_ + tl.arange(0, BN_)
    rm = tl.load(TS + tile) + tl.arange(0, BM_)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    acc = tl.zeros((BM_, BN_), tl.float32)
    for k in range(0, K, BK_):
        rk = k + tl.arange(0, BK_)
        a = tl.load(A + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < end) & (rk[None, :] < K), other=0.0)
        b = tl.load(W + e * K * M + rk[:, None] * M + rn[None, :], mask=(e >= 0) & (rk[:, None] < K) & (rn[None, :] < M), other=0.0)
        if BF:
            acc += tl.dot(a.to(tl.bfloat16), b)
        elif SPLIT:
            ah = a.to(tl.bfloat16)
            al = (a.to(tl.float32) - ah.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(ah, b) + tl.dot(al, b)
        else:
            acc += tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="tf32x3")
    tl.store(C + rm[:, None] * M + rn[None, :], acc, mask=(rm[:, None] < end) & (rn[None, :] < M))


@triton.jit
def _seg_da(DC, W, OFF, TE, TS, DA, N: tl.constexpr, K: tl.constexpr,
            M: tl.constexpr, E: tl.constexpr, BM_: tl.constexpr,
            BN_: tl.constexpr, BK_: tl.constexpr, BF: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TE + tile)
    rk = tl.program_id(1) * BN_ + tl.arange(0, BN_)
    rm = tl.load(TS + tile) + tl.arange(0, BM_)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    acc = tl.zeros((BM_, BN_), tl.float32)
    for j in range(0, M, BK_):
        rj = j + tl.arange(0, BK_)
        a = tl.load(DC + rm[:, None] * M + rj[None, :], mask=(rm[:, None] < end) & (rj[None, :] < M), other=0.0)
        b = tl.load(W + e * K * M + rk[:, None] * M + rj[None, :], mask=(e >= 0) & (rk[:, None] < K) & (rj[None, :] < M), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            acc += tl.dot(ah, tl.trans(b)) + tl.dot((a - ah.to(tl.float32)).to(tl.bfloat16), tl.trans(b))
        else:
            acc += tl.dot(a.to(tl.float32), tl.trans(b.to(tl.float32)), input_precision="tf32x3")
    tl.store(DA + rm[:, None] * K + rk[None, :], acc, mask=(rm[:, None] < end) & (rk[None, :] < K))


@triton.jit
def _seg_dw(A, DC, OFF, DW, N: tl.constexpr, K: tl.constexpr, M: tl.constexpr,
            E: tl.constexpr, BM_: tl.constexpr, BN_: tl.constexpr,
            BK_: tl.constexpr, BF: tl.constexpr):
    e = tl.program_id(0)
    rk = tl.program_id(1) * BM_ + tl.arange(0, BM_)
    rn = tl.program_id(2) * BN_ + tl.arange(0, BN_)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM_, BN_), tl.float32)
    for q in range(begin, end, BK_):
        rq = q + tl.arange(0, BK_)
        a = tl.load(A + rq[:, None] * K + rk[None, :], mask=(rq[:, None] < end) & (rk[None, :] < K), other=0.0)
        b = tl.load(DC + rq[:, None] * M + rn[None, :], mask=(rq[:, None] < end) & (rn[None, :] < M), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            bh = b.to(tl.bfloat16)
            al = (a.to(tl.float32) - ah.to(tl.float32)).to(tl.bfloat16)
            bl = (b.to(tl.float32) - bh.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(tl.trans(ah), bh) + tl.dot(tl.trans(al), bh) + tl.dot(tl.trans(ah), bl)
        else:
            acc += tl.dot(tl.trans(a.to(tl.float32)), b.to(tl.float32), input_precision="tf32x3")
    tl.store(DW + e * K * M + rk[:, None] * M + rn[None, :], acc, mask=(rk[:, None] < K) & (rn[None, :] < M))


@triton.jit
def _act_fwd(GU, Z, N: tl.constexpr, F: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < N * F
    r = i // F
    c = i - r * F
    g = tl.load(GU + r * (2 * F) + c, mask=m, other=0.0)
    u = tl.load(GU + r * (2 * F) + F + c, mask=m, other=0.0)
    tl.store(Z + i, g * tl.sigmoid(g) * u, mask=m)


@triton.jit
def _act_bwd(GU, DZ, DGU, N: tl.constexpr, F: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < N * F
    r = i // F
    c = i - r * F
    g = tl.load(GU + r * (2 * F) + c, mask=m, other=0.0)
    u = tl.load(GU + r * (2 * F) + F + c, mask=m, other=0.0)
    dz = tl.load(DZ + i, mask=m, other=0.0)
    s = tl.sigmoid(g)
    tl.store(DGU + r * (2 * F) + c, dz * u * (s + g * s * (1.0 - s)), mask=m)
    tl.store(DGU + r * (2 * F) + F + c, dz * g * s, mask=m)


@triton.jit
def _combine_fwd(YP, W, INV, Y, T: tl.constexpr, D: tl.constexpr,
                 A: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.zeros((B,), tl.float32)
    for a in range(0, A):
        p = t * A + a
        q = tl.load(INV + p)
        acc += tl.load(W + p) * tl.load(YP + q * D + d, mask=d < D, other=0.0)
    tl.store(Y + t * D + d, acc, mask=d < D)


@triton.jit
def _combine_bwd(DY, YP, W, INV, DYP, DW, T: tl.constexpr, D: tl.constexpr,
                 A: tl.constexpr, B: tl.constexpr):
    p = tl.program_id(0)
    t = p // A
    q = tl.load(INV + p)
    d = tl.arange(0, B)
    acc = tl.zeros((B,), tl.float32)
    w = tl.load(W + p)
    for j in range(0, D, B):
        jj = j + d
        dy = tl.load(DY + t * D + jj, mask=jj < D, other=0.0).to(tl.float32)
        yp = tl.load(YP + q * D + jj, mask=jj < D, other=0.0)
        tl.store(DYP + q * D + jj, dy * w, mask=jj < D)
        acc += dy * yp
    tl.store(DW + p, tl.sum(acc, axis=0))


class _Router(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        T, D = x.shape
        E = w.shape[1]
        s = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _router_kernel[(triton.cdiv(T, 32), triton.cdiv(E, 32))](x, w, s, T, D, E, BM_=32, BN_=32, BK_=32)
        ctx.save_for_backward(x, w, s)
        return s

    @staticmethod
    def backward(ctx, ds):
        x, w, s = ctx.saved_tensors
        dz = ds * s * (1.0 - s)
        return dz @ w.float().t(), x.float().t() @ dz


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tok, inv, A):
        N, D = tok.numel(), x.shape[1]
        out = torch.empty((N, D), device=x.device, dtype=x.dtype)
        _gather_kernel[(N, triton.cdiv(D, 256))](x, tok, out, N, D, B=256)
        ctx.save_for_backward(inv)
        ctx.shape = (x.shape[0], D, int(A))
        return out

    @staticmethod
    def backward(ctx, dr):
        (inv,) = ctx.saved_tensors
        T, D, A = ctx.shape
        dx = torch.empty((T, D), device=dr.device, dtype=dr.dtype)
        _ungather_kernel[(T, triton.cdiv(D, 256))](dr.contiguous(), inv, dx, T, D, A, B=256)
        return dx, None, None, None


class _SegmentMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, w, off):
        N, K = a.shape
        E, _, M = w.shape
        cap = triton.cdiv(N, BM) + E
        te = torch.empty(cap, device=a.device, dtype=torch.int32)
        ts = torch.empty_like(te)
        _make_tiles[(1,)](off, te, ts, E, cap, BM_=BM, num_warps=1)
        c = torch.empty((N, M), device=a.device, dtype=torch.float32)
        bf = a.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
        split = a.dtype == torch.float32 and w.dtype == torch.bfloat16
        _seg_fwd[(cap, triton.cdiv(M, BN))](a, w, off, te, ts, c, N, K, M, E, BM_=BM, BN_=BN, BK_=BK, BF=bf, SPLIT=split)
        ctx.save_for_backward(a, w, off, te, ts)
        ctx.dims = (N, K, M, E, w.dtype == torch.bfloat16, cap)
        return c

    @staticmethod
    def backward(ctx, dc):
        a, w, off, te, ts = ctx.saved_tensors
        N, K, M, E, bf, cap = ctx.dims
        dc = dc.contiguous()
        da = torch.empty_like(a)
        dw = torch.empty_like(w)
        _seg_da[(cap, triton.cdiv(K, BN))](dc, w, off, te, ts, da, N, K, M, E, BM_=BM, BN_=BN, BK_=BK, BF=bf)
        _seg_dw[(E, triton.cdiv(K, BM), triton.cdiv(M, BN))](a, dc, off, dw, N, K, M, E, BM_=BM, BN_=BN, BK_=BK, BF=bf)
        return da, dw, None


class _Act(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gu):
        N, F2 = gu.shape
        F = F2 // 2
        z = torch.empty((N, F), device=gu.device, dtype=torch.float32)
        _act_fwd[(triton.cdiv(N * F, 256),)](gu, z, N, F, B=256)
        ctx.save_for_backward(gu)
        return z

    @staticmethod
    def backward(ctx, dz):
        (gu,) = ctx.saved_tensors
        N, F2 = gu.shape
        F = F2 // 2
        dgu = torch.empty_like(gu)
        _act_bwd[(triton.cdiv(N * F, 256),)](gu, dz.contiguous(), dgu, N, F, B=256)
        return dgu


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, yp, w, inv, T, A):
        D = yp.shape[1]
        y = torch.empty((T, D), device=yp.device, dtype=torch.float32)
        _combine_fwd[(T, triton.cdiv(D, 256))](yp, w, inv, y, T, D, A, B=256)
        ctx.save_for_backward(yp, w, inv)
        ctx.dims = (T, D, A)
        return y

    @staticmethod
    def backward(ctx, dy):
        yp, w, inv = ctx.saved_tensors
        T, D, A = ctx.dims
        dyp = torch.empty_like(yp)
        dw = torch.empty_like(w)
        _combine_bwd[(T * A,)](dy.contiguous(), yp, w, inv, dyp, dw, T, D, A, B=256)
        return dyp, dw, None, None, None


def _select(s, rb, A):
    E = s.shape[1]
    b = s + rb
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    ei = torch.arange(E, device=s.device, dtype=torch.int64)
    return torch.topk((u << 10) | (E - 1 - ei), A, dim=1, largest=True, sorted=True).indices


def fused_moe(x, router_w, router_b, w1, w2, A):
    A = int(A)
    T = x.shape[0]
    E = router_w.shape[1]
    s = _Router.apply(x, router_w)
    with torch.no_grad():
        sel = _select(s.detach(), router_b, A)
        flat = sel.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok = torch.div(order, A, rounding_mode="floor")
        inv = torch.argsort(order)
        counts = torch.bincount(flat, minlength=E)
        off = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(torch.int32)
    ss = torch.gather(s, 1, sel)
    weights = (ss / ss.sum(1, keepdim=True)).reshape(-1)
    xg = _Gather.apply(x, tok, inv, A)
    gu = _SegmentMM.apply(xg, w1, off)
    z = _Act.apply(gu)
    yp = _SegmentMM.apply(z, w2, off)
    return _Combine.apply(yp, weights, inv, T, A).to(x.dtype)
