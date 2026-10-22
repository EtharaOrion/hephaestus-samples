import torch
import triton
import triton.language as tl

ROW = 64
COL = 64
KB = 32


@triton.jit
def _router(X, RW, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
            B: tl.constexpr):
    pt = tl.program_id(0)
    pe = tl.program_id(1)
    rt = pt * B + tl.arange(0, B)
    re = pe * B + tl.arange(0, B)
    acc = tl.zeros((B, B), tl.float32)
    for k in range(0, D, B):
        rk = k + tl.arange(0, B)
        x = tl.load(X + rt[:, None] * D + rk[None, :],
                    mask=(rt[:, None] < T) & (rk[None, :] < D), other=0.)
        w = tl.load(RW + rk[:, None] * E + re[None, :],
                    mask=(rk[:, None] < D) & (re[None, :] < E), other=0.)
        acc += tl.dot(x, w, input_precision="ieee")
    tl.store(S + rt[:, None] * E + re[None, :], tl.sigmoid(acc),
             mask=(rt[:, None] < T) & (re[None, :] < E))


@triton.jit
def _keys(S, RB, K, T: tl.constexpr, E: tl.constexpr, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    m = q < T * E
    e = q % E
    v = tl.load(S + q, mask=m, other=0.) + tl.load(RB + e, mask=m, other=0.)
    v = tl.where(v == 0., 0., v)
    i = v.to(tl.int32, bitcast=True)
    u = tl.where(i >= 0, i.to(tl.int64) ^ 0x80000000,
                 (~i).to(tl.int64) & 0xFFFFFFFF)
    tl.store(K + q, (u << 10) | (E - 1 - e), mask=m)


@triton.jit
def _route_data(S, SEL, W, FLAT, T: tl.constexpr, E: tl.constexpr,
                A: tl.constexpr, BS: tl.constexpr):
    t = tl.program_id(0) * BS + tl.arange(0, BS)
    mt = t < T
    den = tl.zeros((BS,), tl.float32)
    for a in range(A):
        e = tl.load(SEL + t * A + a, mask=mt, other=0)
        den += tl.load(S + t * E + e, mask=mt, other=0.)
    for a in range(A):
        p = t * A + a
        e = tl.load(SEL + p, mask=mt, other=0)
        ss = tl.load(S + t * E + e, mask=mt, other=0.)
        tl.store(W + p, ss / den, mask=mt)
        tl.store(FLAT + p, e, mask=mt)


@triton.jit
def _offsets(FLAT, ORDER, OFF, N: tl.constexpr, E: tl.constexpr):
    e = tl.program_id(0)
    lo = 0
    hi = N
    for _ in range(32):
        mid = (lo + hi) // 2
        p = tl.load(ORDER + mid, mask=mid < N, other=0)
        v = tl.load(FLAT + p, mask=mid < N, other=E)
        lo = tl.where((mid < N) & (v < e), mid + 1, lo)
        hi = tl.where((mid < N) & (v < e), hi, mid)
    tl.store(OFF + e, lo)


@triton.jit
def _prefix(OFF, PRE, E: tl.constexpr, R: tl.constexpr):
    total = 0
    tl.store(PRE, 0)
    for e in range(E):
        b = tl.load(OFF + e)
        z = tl.load(OFF + e + 1)
        total += tl.cdiv(z - b, R)
        tl.store(PRE + e + 1, total)


@triton.jit
def _inverse(ORDER, INV, N: tl.constexpr, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    m = q < N
    p = tl.load(ORDER + q, mask=m, other=0)
    tl.store(INV + p, q, mask=m)


@triton.jit
def _loc(pid, OFF, PRE, E: tl.constexpr, R: tl.constexpr):
    lo = 0
    hi = E
    for _ in range(10):
        mid = (lo + hi) // 2
        v = tl.load(PRE + mid + 1, mask=mid < E, other=0x7FFFFFFF)
        hi = tl.where(pid < v, mid, hi)
        lo = tl.where(pid < v, lo, mid + 1)
    e = lo
    start = tl.load(OFF + e, mask=e < E, other=0) + (pid - tl.load(PRE + e, mask=e < E, other=0)) * R
    end = tl.load(OFF + e + 1, mask=e < E, other=0)
    return e, start, end


@triton.jit
def _expert_in(X, W1, ORDER, OFF, PRE, GU, N: tl.constexpr, D: tl.constexpr,
               F2: tl.constexpr, E: tl.constexpr, A: tl.constexpr,
               BF: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
               K: tl.constexpr):
    pid = tl.program_id(0)
    pn = tl.program_id(1)
    e, start, end = _loc(pid, OFF, PRE, E, R)
    rr = start + tl.arange(0, R)
    cc = pn * C + tl.arange(0, C)
    pair = tl.load(ORDER + rr, mask=rr < end, other=0)
    tok = pair // A
    acc = tl.zeros((R, C), tl.float32)
    for k0 in range(0, D, K):
        kk = k0 + tl.arange(0, K)
        x = tl.load(X + tok[:, None] * D + kk[None, :],
                    mask=(rr[:, None] < end) & (kk[None, :] < D), other=0.)
        w = tl.load(W1 + e * D * F2 + kk[:, None] * F2 + cc[None, :],
                    mask=(e < E) & (kk[:, None] < D) & (cc[None, :] < F2), other=0.)
        if BF:
            acc += tl.dot(x.to(tl.bfloat16), w.to(tl.bfloat16))
        else:
            acc += tl.dot(x, w, input_precision="tf32x3")
    tl.store(GU + rr[:, None] * F2 + cc[None, :], acc,
             mask=(rr[:, None] < end) & (cc[None, :] < F2))


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
def _segment(A0, B0, ORDER, OFF, PRE, OUT, N: tl.constexpr,
             KSIZE: tl.constexpr, OSIZE: tl.constexpr, E: tl.constexpr,
             EA: tl.constexpr, ASEL: tl.constexpr, GATHER: tl.constexpr,
             TRANS: tl.constexpr, BF: tl.constexpr, SPLIT: tl.constexpr,
             R: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    pn = tl.program_id(1)
    e, start, end = _loc(pid, OFF, PRE, E, R)
    rr = start + tl.arange(0, R)
    cc = pn * C + tl.arange(0, C)
    if GATHER:
        pair = tl.load(ORDER + rr, mask=rr < end, other=0)
        ar = pair // ASEL
    else:
        ar = rr
    acc = tl.zeros((R, C), tl.float32)
    for k0 in range(0, KSIZE, K):
        kk = k0 + tl.arange(0, K)
        a = tl.load(A0 + ar[:, None] * KSIZE + kk[None, :],
                    mask=(rr[:, None] < end) & (kk[None, :] < KSIZE), other=0.)
        if TRANS:
            w = tl.load(B0 + e * OSIZE * KSIZE + cc[None, :] * KSIZE + kk[:, None],
                        mask=(e < E) & (cc[None, :] < OSIZE) & (kk[:, None] < KSIZE), other=0.)
        else:
            w = tl.load(B0 + e * KSIZE * OSIZE + kk[:, None] * OSIZE + cc[None, :],
                        mask=(e < E) & (kk[:, None] < KSIZE) & (cc[None, :] < OSIZE), other=0.)
        if BF and SPLIT:
            ah = a.to(tl.bfloat16)
            al = (a - ah).to(tl.bfloat16)
            wh = w.to(tl.bfloat16)
            acc += tl.dot(ah, wh) + tl.dot(al, wh)
        elif BF:
            acc += tl.dot(a.to(tl.bfloat16), w.to(tl.bfloat16))
        else:
            acc += tl.dot(a, w, input_precision="tf32x3")
    tl.store(OUT + rr[:, None] * OSIZE + cc[None, :], acc,
             mask=(rr[:, None] < end) & (cc[None, :] < OSIZE))


@triton.jit
def _combine(Z, W, INV, Y, T: tl.constexpr, D: tl.constexpr,
             A: tl.constexpr, C: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1) * C + tl.arange(0, C)
    acc = tl.zeros((C,), tl.float32)
    for a in range(A):
        p = t * A + a
        r = tl.load(INV + p)
        ww = tl.load(W + p)
        acc += ww * tl.load(Z + r * D + c, mask=c < D, other=0.)
    tl.store(Y + t * D + c, acc, mask=c < D)


@triton.jit
def _dz(DY, W, ORDER, DZ, N: tl.constexpr, D: tl.constexpr,
        A: tl.constexpr, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    m = q < N * D
    r = q // D
    d = q % D
    p = tl.load(ORDER + r, mask=m, other=0)
    t = p // A
    ww = tl.load(W + p, mask=m, other=0.)
    v = tl.load(DY + t * D + d, mask=m, other=0.)
    tl.store(DZ + q, ww * v, mask=m)


@triton.jit
def _dgu(GU, DA, DG, N: tl.constexpr, F: tl.constexpr, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    m = q < N * F
    r = q // F
    f = q % F
    g = tl.load(GU + r * (2 * F) + f, mask=m, other=0.)
    u = tl.load(GU + r * (2 * F) + F + f, mask=m, other=0.)
    da = tl.load(DA + q, mask=m, other=0.)
    s = tl.sigmoid(g)
    sg = g * s
    tl.store(DG + r * (2 * F) + f, da * u * (s + sg * (1. - s)), mask=m)
    tl.store(DG + r * (2 * F) + F + f, da * sg, mask=m)


@triton.jit
def _dw(A0, B0, ORDER, OFF, OUT, KSIZE: tl.constexpr, OSIZE: tl.constexpr,
        N: tl.constexpr, E: tl.constexpr, ASEL: tl.constexpr,
        GATHER: tl.constexpr, BF: tl.constexpr, BM: tl.constexpr,
        BN: tl.constexpr, K: tl.constexpr):
    e = tl.program_id(0)
    pm = tl.program_id(1)
    pn = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for p0 in range(begin, end, K):
        p = p0 + tl.arange(0, K)
        if GATHER:
            pair = tl.load(ORDER + p, mask=p < end, other=0)
            ar = pair // ASEL
        else:
            ar = p
        a = tl.load(A0 + ar[None, :] * KSIZE + rm[:, None],
                    mask=(p[None, :] < end) & (rm[:, None] < KSIZE), other=0.)
        b = tl.load(B0 + p[:, None] * OSIZE + rn[None, :],
                    mask=(p[:, None] < end) & (rn[None, :] < OSIZE), other=0.)
        if BF:
            ah = a.to(tl.bfloat16)
            al = (a - ah).to(tl.bfloat16)
            bh = b.to(tl.bfloat16)
            bl = (b - bh).to(tl.bfloat16)
            acc += tl.dot(ah, bh) + tl.dot(al, bh) + tl.dot(ah, bl)
        else:
            acc += tl.dot(a, b, input_precision="ieee")
    tl.store(OUT + e * KSIZE * OSIZE + rm[:, None] * OSIZE + rn[None, :], acc,
             mask=(rm[:, None] < KSIZE) & (rn[None, :] < OSIZE))


@triton.jit
def _pair_dot(Z, INV, DY, Q, N: tl.constexpr, D: tl.constexpr,
              A: tl.constexpr, BS: tl.constexpr):
    p = tl.program_id(0) * BS + tl.arange(0, BS)
    mp = p < N
    t = p // A
    r = tl.load(INV + p, mask=mp, other=0)
    q = tl.zeros((BS,), tl.float32)
    for d0 in range(0, D, 32):
        d = d0 + tl.arange(0, 32)
        yy = tl.load(DY + t[:, None] * D + d[None, :],
                     mask=mp[:, None] & (d[None, :] < D), other=0.)
        zz = tl.load(Z + r[:, None] * D + d[None, :],
                     mask=mp[:, None] & (d[None, :] < D), other=0.)
        q += tl.sum(yy * zz, axis=1)
    tl.store(Q + p, q, mask=mp)


@triton.jit
def _route_grad(S, SEL, W, Q, DL, T: tl.constexpr, E: tl.constexpr,
                A: tl.constexpr, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    mq = q < T * E
    t = q // E
    e0 = q % E
    den = tl.zeros((BS,), tl.float32)
    mean = tl.zeros((BS,), tl.float32)
    val = tl.zeros((BS,), tl.float32)
    for a in range(A):
        p = t * A + a
        e = tl.load(SEL + p, mask=mq, other=0)
        ss = tl.load(S + t * E + e, mask=mq, other=0.)
        qq = tl.load(Q + p, mask=mq, other=0.)
        den += ss
        mean += tl.load(W + p, mask=mq, other=0.) * qq
    for a in range(A):
        p = t * A + a
        e = tl.load(SEL + p, mask=mq, other=0)
        ss = tl.load(S + t * E + e, mask=mq, other=0.)
        qq = tl.load(Q + p, mask=mq, other=0.)
        val = tl.where(e == e0, (qq - mean) / den * ss * (1. - ss), val)
    tl.store(DL + q, val, mask=mq)


@triton.jit
def _dense_mm(A0, B0, C0, M: tl.constexpr, N: tl.constexpr,
              KSIZE: tl.constexpr, TRANS_A: tl.constexpr,
              TRANS_B: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
              K: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, KSIZE, K):
        rk = k0 + tl.arange(0, K)
        if TRANS_A:
            a = tl.load(A0 + rk[None, :] * M + rm[:, None],
                        mask=(rm[:, None] < M) & (rk[None, :] < KSIZE), other=0.)
        else:
            a = tl.load(A0 + rm[:, None] * KSIZE + rk[None, :],
                        mask=(rm[:, None] < M) & (rk[None, :] < KSIZE), other=0.)
        if TRANS_B:
            b = tl.load(B0 + rn[None, :] * KSIZE + rk[:, None],
                        mask=(rk[:, None] < KSIZE) & (rn[None, :] < N), other=0.)
        else:
            b = tl.load(B0 + rk[:, None] * N + rn[None, :],
                        mask=(rk[:, None] < KSIZE) & (rn[None, :] < N), other=0.)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="ieee")
    tl.store(C0 + rm[:, None] * N + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _combine_dx(DXR, DQR, INV, DX, T: tl.constexpr, D: tl.constexpr,
                A: tl.constexpr, C: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * C + tl.arange(0, C)
    v = tl.load(DQR + t * D + d, mask=d < D, other=0.)
    for a in range(A):
        r = tl.load(INV + t * A + a)
        v += tl.load(DXR + r * D + d, mask=d < D, other=0.)
    tl.store(DX + t * D + d, v, mask=d < D)


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rw, rb, w1, w2, A):
        A = int(A)
        T, D = x.shape
        E = rw.shape[1]
        F = w2.shape[1]
        N = T * A
        bf = x.dtype == torch.bfloat16
        s = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _router[(triton.cdiv(T, 32), triton.cdiv(E, 32))](x, rw, s, T, D, E, B=32)
        keys = torch.empty((T, E), device=x.device, dtype=torch.int64)
        _keys[(triton.cdiv(T * E, 256),)](s, rb, keys, T, E, BS=256)
        sel = torch.topk(keys, A, dim=1, largest=True, sorted=True).indices
        w = torch.empty((N,), device=x.device, dtype=torch.float32)
        flat = torch.empty((N,), device=x.device, dtype=torch.int64)
        _route_data[(triton.cdiv(T, 256),)](s, sel, w, flat, T, E, A, BS=256)
        order = torch.argsort(flat)
        off = torch.empty((E + 1,), device=x.device, dtype=torch.int32)
        _offsets[(E + 1,)](flat, order, off, N, E)
        pre = torch.empty((E + 1,), device=x.device, dtype=torch.int32)
        _prefix[(1,)](off, pre, E, R=ROW, num_warps=1)
        inv = torch.empty_like(order)
        _inverse[(triton.cdiv(N, 256),)](order, inv, N, BS=256)
        cap = triton.cdiv(N, ROW) + E
        gu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _expert_in[(cap, triton.cdiv(2 * F, COL))](x, w1, order, off, pre, gu,
            N, D, 2 * F, E, A, bf, R=ROW, C=COL, K=KB, num_warps=4, num_stages=3)
        act = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _activate[(triton.cdiv(N * F, 256),)](gu, act, N, F, BS=256)
        z = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _segment[(cap, triton.cdiv(D, COL))](act, w2, order, off, pre, z,
            N, F, D, E, E, A, False, False, bf, True, R=ROW, C=COL, K=KB,
            num_warps=4, num_stages=3)
        y = torch.empty((T, D), device=x.device, dtype=x.dtype)
        _combine[(T, triton.cdiv(D, 256))](z, w, inv, y, T, D, A, C=256)
        ctx.save_for_backward(x, rw, w1, w2, s, sel, w, order, off, pre, inv, gu, act, z)
        ctx.meta = (A, bf)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, rw, w1, w2, s, sel, w, order, off, pre, inv, gu, act, z = ctx.saved_tensors
        A, bf = ctx.meta
        dy = dy.contiguous()
        T, D = x.shape
        E = rw.shape[1]
        F = w2.shape[1]
        N = T * A
        cap = triton.cdiv(N, ROW) + E
        dz = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _dz[(triton.cdiv(N * D, 256),)](dy, w, order, dz, N, D, A, BS=256)
        da = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _segment[(cap, triton.cdiv(F, COL))](dz, w2, order, off, pre, da,
            N, D, F, E, E, A, False, True, bf, False, R=ROW, C=COL, K=KB,
            num_warps=4, num_stages=3)
        dgu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _dgu[(triton.cdiv(N * F, 256),)](gu, da, dgu, N, F, BS=256)
        dxr = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _segment[(cap, triton.cdiv(D, COL))](dgu, w1, order, off, pre, dxr,
            N, 2 * F, D, E, E, A, False, True, bf, False, R=ROW, C=COL, K=KB,
            num_warps=4, num_stages=3)
        gw2 = torch.empty_like(w2)
        _dw[(E, triton.cdiv(F, 128), triton.cdiv(D, 64))](act, dz, order, off, gw2,
            F, D, N, E, A, False, bf, BM=128, BN=64, K=KB,
            num_warps=4, num_stages=3)
        gw1 = torch.empty_like(w1)
        _dw[(E, triton.cdiv(D, 128), triton.cdiv(2 * F, 64))](x, dgu, order, off, gw1,
            D, 2 * F, N, E, A, True, bf, BM=128, BN=64, K=KB,
            num_warps=4, num_stages=3)
        q = torch.empty((N,), device=x.device, dtype=torch.float32)
        _pair_dot[(triton.cdiv(N, 32),)](z, inv, dy, q, N, D, A, BS=32)
        dl = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _route_grad[(triton.cdiv(T * E, 256),)](s, sel, w, q, dl,
            T, E, A, BS=256)
        grw = torch.empty_like(rw)
        _dense_mm[(triton.cdiv(D, 64), triton.cdiv(E, 64))](x, dl, grw,
            D, E, T, True, False, BM=64, BN=64, K=32, num_warps=4, num_stages=3)
        dqr = torch.empty((T, D), device=x.device, dtype=torch.float32)
        _dense_mm[(triton.cdiv(T, 64), triton.cdiv(D, 64))](dl, rw, dqr,
            T, D, E, False, True, BM=64, BN=64, K=32, num_warps=4, num_stages=3)
        dx = torch.empty_like(x)
        _combine_dx[(T, triton.cdiv(D, 256))](dxr, dqr, inv, dx, T, D, A, C=256)
        return dx, grw, None, gw1, gw2, None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
