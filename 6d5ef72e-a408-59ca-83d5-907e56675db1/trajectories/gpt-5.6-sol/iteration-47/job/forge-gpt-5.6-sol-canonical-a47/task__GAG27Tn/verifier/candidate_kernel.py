import torch
import triton
import triton.language as tl

R = 64
C = 64
KB = 32


@triton.jit
def _router(X, W, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    im = tl.program_id(0) * BM + tl.arange(0, BM)
    jn = tl.program_id(1) * BN + tl.arange(0, BN)
    z = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(X + im[:, None] * D + kk[None, :], mask=(im[:, None] < T) & (kk[None, :] < D), other=0.).to(tl.float32)
        b = tl.load(W + kk[:, None] * E + jn[None, :], mask=(kk[:, None] < D) & (jn[None, :] < E), other=0.).to(tl.float32)
        z += tl.dot(a, b, input_precision="ieee")
    tl.store(S + im[:, None] * E + jn[None, :], tl.sigmoid(z), mask=(im[:, None] < T) & (jn[None, :] < E))


@triton.jit
def _keys(S, B, KOUT, n: tl.constexpr, E: tl.constexpr, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    m = q < n
    e = q % E
    v = tl.load(S + q, mask=m, other=0.) + tl.load(B + e, mask=m, other=0.)
    v = tl.where(v == 0., 0., v)
    i = v.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(i >= 0, i ^ 0x80000000, (~i) & 0xffffffff)
    tl.store(KOUT + q, (u << 10) | (E - 1 - e), mask=m)


@triton.jit
def _weights(S, SEL, W, A: tl.constexpr, BA: tl.constexpr, T: tl.constexpr, E: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BA)
    m = a < A
    e = tl.load(SEL + t * A + a, mask=m, other=0)
    s = tl.load(S + t * E + e, mask=m, other=0.)
    den = tl.sum(tl.where(m, s, 0.))
    tl.store(W + t * A + a, s / den, mask=m)


@triton.jit
def _offsets(SE, OFF, N: tl.constexpr, E: tl.constexpr, IT: tl.constexpr):
    e = tl.program_id(0)
    lo = 0
    hi = N
    for _ in range(IT):
        md = (lo + hi) // 2
        v = tl.load(SE + md, mask=md < N, other=E)
        take = v < e
        lo = tl.where(take, md + 1, lo)
        hi = tl.where(take, hi, md)
    tl.store(OFF + e, lo)


@triton.jit
def _prefix(OFF, PRE, E: tl.constexpr, ROW: tl.constexpr):
    z = tl.full((), 0, tl.int64)
    tl.store(PRE, z)
    for e in range(E):
        b = tl.load(OFF + e)
        en = tl.load(OFF + e + 1)
        z += tl.cdiv(en - b, ROW)
        tl.store(PRE + e + 1, z)


@triton.jit
def _inverse(ORDER, INV, N: tl.constexpr, BS: tl.constexpr):
    p = tl.program_id(0) * BS + tl.arange(0, BS)
    m = p < N
    q = tl.load(ORDER + p, mask=m, other=0)
    tl.store(INV + q, p, mask=m)


@triton.jit
def _loc(pid, PRE, OFF, E: tl.constexpr, IT: tl.constexpr, ROW: tl.constexpr):
    lo = 0
    hi = E
    for _ in range(IT):
        md = (lo + hi) // 2
        v = tl.load(PRE + md + 1)
        take = v <= pid
        lo = tl.where(take, md + 1, lo)
        hi = tl.where(take, hi, md)
    e = lo
    first = tl.load(PRE + e, mask=e < E, other=0)
    beg = tl.load(OFF + e, mask=e < E, other=0)
    end = tl.load(OFF + e + 1, mask=e < E, other=0)
    p = beg + (pid - first) * ROW
    return e, p, end


@triton.jit
def _ein(X, W1, ORDER, PRE, OFF, GU, N: tl.constexpr, D: tl.constexpr,
         O: tl.constexpr, E: tl.constexpr, A: tl.constexpr, IT: tl.constexpr,
         BF: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e, p0, end = _loc(tl.program_id(0), PRE, OFF, E, IT, BM)
    rr = tl.arange(0, BM)
    cc = tl.program_id(1) * BN + tl.arange(0, BN)
    pp = p0 + rr
    pm = (pp < end) & (pp < N) & (e < E)
    q = tl.load(ORDER + pp, mask=pm, other=0)
    tok = q // A
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(X + tok[:, None] * D + kk[None, :], mask=pm[:, None] & (kk[None, :] < D), other=0.)
        b = tl.load(W1 + e * D * O + kk[:, None] * O + cc[None, :], mask=(e < E) & (kk[:, None] < D) & (cc[None, :] < O), other=0.)
        if BF:
            acc += tl.dot(a, b)
        else:
            acc += tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="tf32x3")
    tl.store(GU + pp[:, None] * O + cc[None, :], acc, mask=pm[:, None] & (cc[None, :] < O))


@triton.jit
def _activate(GU, ACT, N: tl.constexpr, F: tl.constexpr, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    m = q < N * F
    r = q // F
    f = q % F
    g = tl.load(GU + r * (2 * F) + f, mask=m, other=0.)
    u = tl.load(GU + r * (2 * F) + F + f, mask=m, other=0.)
    tl.store(ACT + q, g * tl.sigmoid(g) * u, mask=m)


@triton.jit
def _seg(A0, B0, PRE, OFF, OUT, N: tl.constexpr, K0: tl.constexpr,
         O: tl.constexpr, E: tl.constexpr, IT: tl.constexpr, BF: tl.constexpr,
         SPLIT_A: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e, p0, end = _loc(tl.program_id(0), PRE, OFF, E, IT, BM)
    rr = tl.arange(0, BM)
    cc = tl.program_id(1) * BN + tl.arange(0, BN)
    pp = p0 + rr
    pm = (pp < end) & (pp < N) & (e < E)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K0, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(A0 + pp[:, None] * K0 + kk[None, :], mask=pm[:, None] & (kk[None, :] < K0), other=0.)
        b = tl.load(B0 + e * K0 * O + kk[:, None] * O + cc[None, :], mask=(e < E) & (kk[:, None] < K0) & (cc[None, :] < O), other=0.)
        if BF:
            ah = a.to(tl.bfloat16)
            acc += tl.dot(ah, b)
            if SPLIT_A:
                acc += tl.dot((a - ah).to(tl.bfloat16), b)
        else:
            acc += tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="tf32x3")
    tl.store(OUT + pp[:, None] * O + cc[None, :], acc, mask=pm[:, None] & (cc[None, :] < O))


@triton.jit
def _combine(FE, INV, W, Y, T: tl.constexpr, D: tl.constexpr, A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    z = tl.zeros((BD,), tl.float32)
    for a in range(A):
        q = t * A + a
        p = tl.load(INV + q)
        w = tl.load(W + q)
        z += w * tl.load(FE + p * D + d, mask=d < D, other=0.)
    tl.store(Y + t * D + d, z, mask=d < D)


@triton.jit
def _dz(DY, W, ORDER, DZ, N: tl.constexpr, D: tl.constexpr, A: tl.constexpr, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    m = q < N * D
    p = q // D
    d = q % D
    orig = tl.load(ORDER + p, mask=m, other=0)
    tok = orig // A
    w = tl.load(W + orig, mask=m, other=0.)
    v = tl.load(DY + tok * D + d, mask=m, other=0.).to(tl.float32)
    tl.store(DZ + q, v * w, mask=m)


@triton.jit
def _dgu(GU, DA, DGU, N: tl.constexpr, F: tl.constexpr, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    m = q < N * F
    p = q // F
    f = q % F
    g = tl.load(GU + p * (2 * F) + f, mask=m, other=0.)
    u = tl.load(GU + p * (2 * F) + F + f, mask=m, other=0.)
    da = tl.load(DA + q, mask=m, other=0.)
    s = tl.sigmoid(g)
    tl.store(DGU + p * (2 * F) + f, da * u * (s + g * s * (1. - s)), mask=m)
    tl.store(DGU + p * (2 * F) + F + f, da * g * s, mask=m)


@triton.jit
def _segbt(A0, B0, PRE, OFF, OUT, N: tl.constexpr, K0: tl.constexpr,
           O: tl.constexpr, E: tl.constexpr, IT: tl.constexpr, BF: tl.constexpr,
           SPLIT_A: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e, p0, end = _loc(tl.program_id(0), PRE, OFF, E, IT, BM)
    rr = tl.arange(0, BM)
    cc = tl.program_id(1) * BN + tl.arange(0, BN)
    pp = p0 + rr
    pm = (pp < end) & (pp < N) & (e < E)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K0, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(A0 + pp[:, None] * K0 + kk[None, :], mask=pm[:, None] & (kk[None, :] < K0), other=0.)
        b = tl.load(B0 + e * O * K0 + cc[None, :] * K0 + kk[:, None], mask=(e < E) & (cc[None, :] < O) & (kk[:, None] < K0), other=0.)
        if BF:
            ah = a.to(tl.bfloat16)
            acc += tl.dot(ah, b)
            if SPLIT_A:
                acc += tl.dot((a - ah).to(tl.bfloat16), b)
        else:
            acc += tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="tf32x3")
    tl.store(OUT + pp[:, None] * O + cc[None, :], acc, mask=pm[:, None] & (cc[None, :] < O))


@triton.jit
def _dw(A0, B0, ORDER, OFF, OUT, N: tl.constexpr, K0: tl.constexpr,
        O: tl.constexpr, E: tl.constexpr, ASEL: tl.constexpr, GATHER: tl.constexpr,
        BF: tl.constexpr, STRICT: tl.constexpr, SPLIT_A: tl.constexpr,
        SPLIT_B: tl.constexpr, DEEP_B: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(0)
    ii = tl.program_id(1) * BM + tl.arange(0, BM)
    jj = tl.program_id(2) * BN + tl.arange(0, BN)
    beg = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    z = tl.zeros((BM, BN), tl.float32)
    for p0 in range(beg, end, BK):
        pp = p0 + tl.arange(0, BK)
        pm = pp < end
        if GATHER:
            q = tl.load(ORDER + pp, mask=pm, other=0)
            row = q // ASEL
        else:
            row = pp
        aa = tl.load(A0 + row[None, :] * K0 + ii[:, None], mask=(ii[:, None] < K0) & pm[None, :], other=0.)
        bb = tl.load(B0 + pp[:, None] * O + jj[None, :], mask=pm[:, None] & (jj[None, :] < O), other=0.)
        if STRICT:
            z += tl.dot(aa.to(tl.float32), bb.to(tl.float32), input_precision="tf32x3")
        elif BF:
            ah = aa.to(tl.bfloat16)
            bh = bb.to(tl.bfloat16)
            z += tl.dot(ah, bh)
            if SPLIT_A:
                z += tl.dot((aa - ah).to(tl.bfloat16), bh)
            if SPLIT_B:
                bl = (bb - bh).to(tl.bfloat16)
                z += tl.dot(ah, bl)
                if DEEP_B:
                    z += tl.dot(ah, (bb - bh - bl).to(tl.bfloat16))
        else:
            z += tl.dot(aa.to(tl.float32), bb.to(tl.float32), input_precision="tf32x3")
    tl.store(OUT + e * K0 * O + ii[:, None] * O + jj[None, :], z, mask=(ii[:, None] < K0) & (jj[None, :] < O))


@triton.jit
def _route(DY, FE, Y, S, SEL, INV, DS, T: tl.constexpr, D: tl.constexpr,
           E: tl.constexpr, A: tl.constexpr, BA: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BA)
    am = a < A
    e = tl.load(SEL + t * A + a, mask=am, other=0)
    ss = tl.load(S + t * E + e, mask=am, other=0.)
    den = tl.sum(tl.where(am, ss, 0.))
    dots = tl.zeros((BA,), tl.float32)
    yd = 0.
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        dy = tl.load(DY + t * D + d, mask=d < D, other=0.).to(tl.float32)
        yy = tl.load(Y + t * D + d, mask=d < D, other=0.)
        yd += tl.sum(dy * yy)
        p = tl.load(INV + t * A + a, mask=am, other=0)
        fe = tl.load(FE + p[:, None] * D + d[None, :], mask=am[:, None] & (d[None, :] < D), other=0.)
        dots += tl.sum(fe * dy[None, :], axis=1)
    v = (dots - yd) / den * ss * (1. - ss)
    tl.store(DS + t * E + e, v, mask=am)


@triton.jit
def _zero(X, n: tl.constexpr, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    tl.store(X + q, 0., mask=q < n)


@triton.jit
def _dense(A0, B0, OUT, M: tl.constexpr, K0: tl.constexpr, O: tl.constexpr,
           TA: tl.constexpr, TB: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    i = tl.program_id(0) * BM + tl.arange(0, BM)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    z = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K0, BK):
        k = k0 + tl.arange(0, BK)
        if TA:
            a = tl.load(A0 + k[None, :] * M + i[:, None], mask=(i[:, None] < M) & (k[None, :] < K0), other=0.)
        else:
            a = tl.load(A0 + i[:, None] * K0 + k[None, :], mask=(i[:, None] < M) & (k[None, :] < K0), other=0.)
        if TB:
            b = tl.load(B0 + j[None, :] * K0 + k[:, None], mask=(k[:, None] < K0) & (j[None, :] < O), other=0.)
        else:
            b = tl.load(B0 + k[:, None] * O + j[None, :], mask=(k[:, None] < K0) & (j[None, :] < O), other=0.)
        z += tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="ieee")
    tl.store(OUT + i[:, None] * O + j[None, :], z, mask=(i[:, None] < M) & (j[None, :] < O))


@triton.jit
def _sum_dx(DXP, INV, DXR, DX, T: tl.constexpr, D: tl.constexpr, A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    z = tl.load(DXR + t * D + d, mask=d < D, other=0.)
    for a in range(A):
        p = tl.load(INV + t * A + a)
        z += tl.load(DXP + p * D + d, mask=d < D, other=0.)
    tl.store(DX + t * D + d, z, mask=d < D)


def _meta(sel, A, E):
    flat = sel.reshape(-1)
    order = torch.argsort(flat, stable=True)
    se = flat[order]
    n = flat.numel()
    off = torch.empty(E + 1, device=sel.device, dtype=torch.int64)
    itn = max(1, (n + 1).bit_length())
    _offsets[(E + 1,)](se, off, N=n, E=E, IT=itn, num_warps=1)
    pre = torch.empty(E + 1, device=sel.device, dtype=torch.int64)
    _prefix[(1,)](off, pre, E=E, ROW=R, num_warps=1)
    inv = torch.empty_like(order)
    _inverse[(triton.cdiv(n, 256),)](order, inv, N=n, BS=256)
    return order, off, pre, inv, triton.cdiv(n, R) + E


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rw, rb, w1, w2, A):
        A = int(A)
        T, D = x.shape
        E = rw.shape[1]
        F = w2.shape[1]
        N = T * A
        bf = x.dtype == torch.bfloat16
        it = max(1, E.bit_length())
        s = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _router[(triton.cdiv(T, 32), triton.cdiv(E, 32))](x, rw, s, T=T, D=D, E=E, BM=32, BN=32, BK=32)
        keys = torch.empty((T, E), device=x.device, dtype=torch.int64)
        _keys[(triton.cdiv(T * E, 256),)](s, rb, keys, n=T * E, E=E, BS=256)
        sel = torch.topk(keys, A, dim=1, largest=True, sorted=True).indices
        weights = torch.empty((T, A), device=x.device, dtype=torch.float32)
        ba = triton.next_power_of_2(A)
        _weights[(T,)](s, sel, weights, A=A, BA=ba, T=T, E=E)
        order, off, pre, inv, cap = _meta(sel, A, E)
        gu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _ein[(cap, triton.cdiv(2 * F, C))](x, w1, order, pre, off, gu, N=N, D=D, O=2 * F, E=E, A=A, IT=it, BF=bf, BM=R, BN=C, BK=KB, num_warps=4, num_stages=3)
        act = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _activate[(triton.cdiv(N * F, 256),)](gu, act, N=N, F=F, BS=256)
        fe = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _seg[(cap, triton.cdiv(D, C))](act, w2, pre, off, fe, N=N, K0=F, O=D, E=E, IT=it, BF=bf, SPLIT_A=bf, BM=R, BN=C, BK=KB, num_warps=4, num_stages=3)
        y = torch.empty((T, D), device=x.device, dtype=torch.float32)
        _combine[(T, triton.cdiv(D, 256))](fe, inv, weights, y, T=T, D=D, A=A, BD=256)
        ctx.save_for_backward(x, rw, w1, w2, s, sel, weights, order, off, pre, inv, gu, act, fe, y)
        ctx.params = (A, cap, it, bf)
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        x, rw, w1, w2, s, sel, weights, order, off, pre, inv, gu, act, fe, y = ctx.saved_tensors
        A, cap, it, bf = ctx.params
        T, D = x.shape
        E = rw.shape[1]
        F = w2.shape[1]
        N = T * A
        dz = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _dz[(triton.cdiv(N * D, 256),)](dy, weights, order, dz, N=N, D=D, A=A, BS=256)
        da = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _segbt[(cap, triton.cdiv(F, C))](dz, w2, pre, off, da, N=N, K0=D, O=F, E=E, IT=it, BF=bf, SPLIT_A=bf, BM=R, BN=C, BK=KB, num_warps=4, num_stages=3)
        dgu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _dgu[(triton.cdiv(N * F, 256),)](gu, da, dgu, N=N, F=F, BS=256)
        dxp = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _segbt[(cap, triton.cdiv(D, C))](dgu, w1, pre, off, dxp, N=N, K0=2 * F, O=D, E=E, IT=it, BF=bf, SPLIT_A=bf, BM=R, BN=C, BK=KB, num_warps=4, num_stages=3)
        gw2 = torch.empty_like(w2)
        _dw[(E, triton.cdiv(F, 128), triton.cdiv(D, 64))](act, dz, order, off, gw2, N=N, K0=F, O=D, E=E, ASEL=A, GATHER=False, BF=bf, STRICT=False, SPLIT_A=bf, SPLIT_B=bf, DEEP_B=False, BM=128, BN=64, BK=32, num_warps=4)
        gw1 = torch.empty_like(w1)
        _dw[(E, triton.cdiv(D, 128), triton.cdiv(2 * F, 64))](x, dgu, order, off, gw1, N=N, K0=D, O=2 * F, E=E, ASEL=A, GATHER=True, BF=bf, STRICT=False, SPLIT_A=False, SPLIT_B=bf, DEEP_B=bf, BM=128, BN=64, BK=32, num_warps=4)
        ds = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _zero[(triton.cdiv(T * E, 256),)](ds, n=T * E, BS=256)
        _route[(T,)](dy, fe, y, s, sel, inv, ds, T=T, D=D, E=E, A=A, BA=triton.next_power_of_2(A), BD=128)
        grw = torch.empty_like(rw)
        _dense[(triton.cdiv(D, 64), triton.cdiv(E, 64))](x, ds, grw, M=D, K0=T, O=E, TA=True, TB=False, BM=64, BN=64, BK=32, num_warps=4)
        dxr = torch.empty((T, D), device=x.device, dtype=torch.float32)
        _dense[(triton.cdiv(T, 64), triton.cdiv(D, 64))](ds, rw, dxr, M=T, K0=E, O=D, TA=False, TB=True, BM=64, BN=64, BK=32, num_warps=4)
        dx = torch.empty_like(x)
        _sum_dx[(T, triton.cdiv(D, 256))](dxp, inv, dxr, dx, T=T, D=D, A=A, BD=256)
        return dx, grw, None, gw1, gw2, None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
