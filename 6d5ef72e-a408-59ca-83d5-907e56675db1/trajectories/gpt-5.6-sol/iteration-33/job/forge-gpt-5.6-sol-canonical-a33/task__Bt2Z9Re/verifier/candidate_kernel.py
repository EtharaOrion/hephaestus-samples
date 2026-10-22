import torch
import triton
import triton.language as tl

ROW = 64
COL = 64
KB = 32


@triton.jit
def _router(X, RW, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    im = tl.program_id(0) * BM + tl.arange(0, BM)
    ie = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        ik = k0 + tl.arange(0, BK)
        x = tl.load(X + im[:, None] * D + ik[None, :],
                    mask=(im[:, None] < T) & (ik[None, :] < D), other=0.0).to(tl.float32)
        w = tl.load(RW + ik[:, None] * E + ie[None, :],
                    mask=(ik[:, None] < D) & (ie[None, :] < E), other=0.0).to(tl.float32)
        acc += tl.dot(x, w, input_precision="ieee")
    tl.store(S + im[:, None] * E + ie[None, :], tl.sigmoid(acc),
             mask=(im[:, None] < T) & (ie[None, :] < E))


@triton.jit
def _keys(S, B, K, n: tl.constexpr, E: tl.constexpr, BS: tl.constexpr):
    i = tl.program_id(0) * BS + tl.arange(0, BS)
    m = i < n
    e = i % E
    v = tl.load(S + i, mask=m, other=0.0) + tl.load(B + e, mask=m, other=0.0)
    v = tl.where(v == 0.0, 0.0, v)
    bits = v.to(tl.int32, bitcast=True).to(tl.int64)
    ordered = tl.where(bits >= 0, bits ^ 0x80000000, (~bits) & 0xFFFFFFFF)
    tl.store(K + i, (ordered << 10) | (E - 1 - e).to(tl.int64), mask=m)


@triton.jit
def _weights(S, SEL, W, T: tl.constexpr, E: tl.constexpr, A: tl.constexpr,
             BS: tl.constexpr):
    t = tl.program_id(0) * BS + tl.arange(0, BS)
    m = t < T
    den = tl.zeros((BS,), tl.float32)
    for a in range(A):
        e = tl.load(SEL + t * A + a, mask=m, other=0)
        den += tl.load(S + t * E + e, mask=m, other=0.0)
    for a in range(A):
        e = tl.load(SEL + t * A + a, mask=m, other=0)
        v = tl.load(S + t * E + e, mask=m, other=0.0)
        tl.store(W + t * A + a, v / den, mask=m)


@triton.jit
def _offsets(FLAT, ORDER, OFF, N: tl.constexpr, E: tl.constexpr):
    e = tl.program_id(0)
    lo = 0
    hi = N
    while lo < hi:
        mid = (lo + hi) // 2
        q = tl.load(ORDER + mid)
        v = tl.load(FLAT + q)
        take = v < e
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(take, hi, mid)
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
    p = tl.program_id(0) * BS + tl.arange(0, BS)
    m = p < N
    q = tl.load(ORDER + p, mask=m, other=0)
    tl.store(INV + q, p, mask=m)


@triton.jit
def _location(pid, PRE, OFF, E: tl.constexpr, R: tl.constexpr):
    lo = 0
    hi = E
    while lo < hi:
        mid = (lo + hi) // 2
        v = tl.load(PRE + mid + 1)
        take = pid >= v
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(take, hi, mid)
    e = tl.minimum(lo, E - 1)
    tile = pid - tl.load(PRE + e)
    start = tl.load(OFF + e) + tile * R
    end = tl.load(OFF + e + 1)
    return e, start, end


@triton.jit
def _expert_in(X, W1, ORDER, OFF, PRE, GU,
               N: tl.constexpr, A: tl.constexpr, D: tl.constexpr,
               F2: tl.constexpr, E: tl.constexpr, CAP: tl.constexpr,
               BF: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
               K: tl.constexpr):
    pid = tl.program_id(0)
    j = tl.program_id(1) * C + tl.arange(0, C)
    valid_tile = pid < tl.load(PRE + E)
    e, begin, end = _location(pid, PRE, OFF, E, R)
    p = begin + tl.arange(0, R)
    mp = valid_tile & (p < end) & (p < N)
    q = tl.load(ORDER + p, mask=mp, other=0)
    t = q // A
    acc = tl.zeros((R, C), tl.float32)
    for k0 in range(0, D, K):
        k = k0 + tl.arange(0, K)
        x = tl.load(X + t[:, None] * D + k[None, :],
                    mask=mp[:, None] & (k[None, :] < D), other=0.0)
        w = tl.load(W1 + e * D * F2 + k[:, None] * F2 + j[None, :],
                    mask=(k[:, None] < D) & (j[None, :] < F2), other=0.0)
        if BF:
            acc += tl.dot(x, w)
        else:
            acc += tl.dot(x, w, input_precision="tf32x3")
    tl.store(GU + p[:, None] * F2 + j[None, :], acc,
             mask=mp[:, None] & (j[None, :] < F2))


@triton.jit
def _activate(GU, ACT, N: tl.constexpr, F: tl.constexpr, BS: tl.constexpr):
    i = tl.program_id(0) * BS + tl.arange(0, BS)
    m = i < N * F
    p = i // F
    f = i % F
    g = tl.load(GU + p * (2 * F) + f, mask=m, other=0.0)
    u = tl.load(GU + p * (2 * F) + F + f, mask=m, other=0.0)
    tl.store(ACT + i, g * tl.sigmoid(g) * u, mask=m)


@triton.jit
def _segment(AIN, B, ORDER, OFF, PRE, OUT,
             N: tl.constexpr, KSIZE: tl.constexpr, OSIZE: tl.constexpr,
             E: tl.constexpr, BF: tl.constexpr, SPLIT: tl.constexpr,
             TRANS_B: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
             K: tl.constexpr):
    pid = tl.program_id(0)
    j = tl.program_id(1) * C + tl.arange(0, C)
    valid_tile = pid < tl.load(PRE + E)
    e, begin, end = _location(pid, PRE, OFF, E, R)
    p = begin + tl.arange(0, R)
    mp = valid_tile & (p < end) & (p < N)
    acc = tl.zeros((R, C), tl.float32)
    for k0 in range(0, KSIZE, K):
        k = k0 + tl.arange(0, K)
        a = tl.load(AIN + p[:, None] * KSIZE + k[None, :],
                    mask=mp[:, None] & (k[None, :] < KSIZE), other=0.0).to(tl.float32)
        if TRANS_B:
            bp = B + e * KSIZE * OSIZE + j[None, :] * KSIZE + k[:, None]
        else:
            bp = B + e * KSIZE * OSIZE + k[:, None] * OSIZE + j[None, :]
        b = tl.load(bp, mask=(k[:, None] < KSIZE) & (j[None, :] < OSIZE), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            if SPLIT:
                al = (a - ah.to(tl.float32)).to(tl.bfloat16)
                acc += tl.dot(ah, b) + tl.dot(al, b)
            else:
                acc += tl.dot(ah, b)
        else:
            acc += tl.dot(a, b.to(tl.float32), input_precision="tf32x3")
    tl.store(OUT + p[:, None] * OSIZE + j[None, :], acc,
             mask=mp[:, None] & (j[None, :] < OSIZE))


@triton.jit
def _combine(FE, W, INV, Y, T: tl.constexpr, D: tl.constexpr,
             A: tl.constexpr, C: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * C + tl.arange(0, C)
    acc = tl.zeros((C,), tl.float32)
    for a in range(A):
        q = t * A + a
        p = tl.load(INV + q)
        w = tl.load(W + q)
        v = tl.load(FE + p * D + d, mask=d < D, other=0.0)
        acc += w * v
    tl.store(Y + t * D + d, acc, mask=d < D)


@triton.jit
def _dz(DY, W, ORDER, DZ, N: tl.constexpr, A: tl.constexpr,
        D: tl.constexpr, BS: tl.constexpr):
    i = tl.program_id(0) * BS + tl.arange(0, BS)
    m = i < N * D
    p = i // D
    d = i % D
    q = tl.load(ORDER + p, mask=m, other=0)
    t = q // A
    w = tl.load(W + q, mask=m, other=0.0)
    dy = tl.load(DY + t * D + d, mask=m, other=0.0).to(tl.float32)
    tl.store(DZ + i, dy * w, mask=m)


@triton.jit
def _dactivate(GU, DA, DGU, N: tl.constexpr, F: tl.constexpr,
               BS: tl.constexpr):
    i = tl.program_id(0) * BS + tl.arange(0, BS)
    m = i < N * F
    p = i // F
    f = i % F
    g = tl.load(GU + p * (2 * F) + f, mask=m, other=0.0)
    u = tl.load(GU + p * (2 * F) + F + f, mask=m, other=0.0)
    da = tl.load(DA + i, mask=m, other=0.0)
    s = tl.sigmoid(g)
    silu = g * s
    tl.store(DGU + p * (2 * F) + f, da * u * (s + silu * (1.0 - s)), mask=m)
    tl.store(DGU + p * (2 * F) + F + f, da * silu, mask=m)


@triton.jit
def _dw1(X, DGU, ORDER, OFF, OUT,
         N: tl.constexpr, A: tl.constexpr, D: tl.constexpr,
         F2: tl.constexpr, E: tl.constexpr, BF: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, K: tl.constexpr):
    e = tl.program_id(0)
    im = tl.program_id(1) * BM + tl.arange(0, BM)
    jn = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for p0 in range(begin, end, K):
        p = p0 + tl.arange(0, K)
        mp = p < end
        q = tl.load(ORDER + p, mask=mp, other=0)
        t = q // A
        x = tl.load(X + t[None, :] * D + im[:, None],
                    mask=mp[None, :] & (im[:, None] < D), other=0.0)
        g = tl.load(DGU + p[:, None] * F2 + jn[None, :],
                    mask=mp[:, None] & (jn[None, :] < F2), other=0.0).to(tl.float32)
        if BF:
            gh = g.to(tl.bfloat16)
            gl = (g - gh.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(x, gh) + tl.dot(x, gl)
        else:
            acc += tl.dot(x.to(tl.float32), g, input_precision="tf32x3")
    tl.store(OUT + e * D * F2 + im[:, None] * F2 + jn[None, :], acc,
             mask=(im[:, None] < D) & (jn[None, :] < F2))


@triton.jit
def _dw2(ACT, DZ, OFF, OUT,
         N: tl.constexpr, F: tl.constexpr, D: tl.constexpr,
         E: tl.constexpr, BF: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, K: tl.constexpr):
    e = tl.program_id(0)
    im = tl.program_id(1) * BM + tl.arange(0, BM)
    jn = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for p0 in range(begin, end, K):
        p = p0 + tl.arange(0, K)
        mp = p < end
        a = tl.load(ACT + p[None, :] * F + im[:, None],
                    mask=mp[None, :] & (im[:, None] < F), other=0.0).to(tl.float32)
        z = tl.load(DZ + p[:, None] * D + jn[None, :],
                    mask=mp[:, None] & (jn[None, :] < D), other=0.0).to(tl.float32)
        if BF:
            ah = a.to(tl.bfloat16)
            al = (a - ah.to(tl.float32)).to(tl.bfloat16)
            zh = z.to(tl.bfloat16)
            zl = (z - zh.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(ah, zh) + tl.dot(al, zh) + tl.dot(ah, zl)
        else:
            acc += tl.dot(a, z, input_precision="tf32x3")
    tl.store(OUT + e * F * D + im[:, None] * D + jn[None, :], acc,
             mask=(im[:, None] < F) & (jn[None, :] < D))


@triton.jit
def _pair_dot(DY, FE, ORDER, PD, N: tl.constexpr, A: tl.constexpr,
              D: tl.constexpr, C: tl.constexpr):
    p = tl.program_id(0)
    q = tl.load(ORDER + p)
    t = q // A
    acc = tl.zeros((C,), tl.float32)
    for d0 in range(0, D, C):
        d = d0 + tl.arange(0, C)
        dy = tl.load(DY + t * D + d, mask=d < D, other=0.0).to(tl.float32)
        fe = tl.load(FE + p * D + d, mask=d < D, other=0.0)
        acc += dy * fe
    tl.store(PD + p, tl.sum(acc, axis=0))


@triton.jit
def _route_grad(S, SEL, W, INV, PD, DL,
                T: tl.constexpr, E: tl.constexpr, A: tl.constexpr,
                BE: tl.constexpr):
    t = tl.program_id(0)
    eb = tl.program_id(1) * BE + tl.arange(0, BE)
    mean = 0.0
    den = 0.0
    for a in range(A):
        q = t * A + a
        p = tl.load(INV + q)
        mean += tl.load(W + q) * tl.load(PD + p)
        e = tl.load(SEL + q)
        den += tl.load(S + t * E + e)
    out = tl.zeros((BE,), tl.float32)
    for a in range(A):
        q = t * A + a
        e = tl.load(SEL + q)
        p = tl.load(INV + q)
        ds = (tl.load(PD + p) - mean) / den
        raw = tl.load(S + t * E + e)
        out = tl.where(eb == e, ds * raw * (1.0 - raw), out)
    tl.store(DL + t * E + eb, out, mask=eb < E)


@triton.jit
def _dense_mm(AIN, B, OUT, M: tl.constexpr, N: tl.constexpr,
              KSIZE: tl.constexpr, SAM: tl.constexpr, SAK: tl.constexpr,
              SBK: tl.constexpr, SBN: tl.constexpr, BM: tl.constexpr,
              BN: tl.constexpr, K: tl.constexpr):
    im = tl.program_id(0) * BM + tl.arange(0, BM)
    jn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, KSIZE, K):
        ik = k0 + tl.arange(0, K)
        a = tl.load(AIN + im[:, None] * SAM + ik[None, :] * SAK,
                    mask=(im[:, None] < M) & (ik[None, :] < KSIZE), other=0.0).to(tl.float32)
        b = tl.load(B + ik[:, None] * SBK + jn[None, :] * SBN,
                    mask=(ik[:, None] < KSIZE) & (jn[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(OUT + im[:, None] * N + jn[None, :], acc,
             mask=(im[:, None] < M) & (jn[None, :] < N))


@triton.jit
def _combine_dx(DXP, DXR, INV, DX, T: tl.constexpr, D: tl.constexpr,
                A: tl.constexpr, C: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * C + tl.arange(0, C)
    acc = tl.load(DXR + t * D + d, mask=d < D, other=0.0)
    for a in range(A):
        p = tl.load(INV + t * A + a)
        acc += tl.load(DXP + p * D + d, mask=d < D, other=0.0)
    tl.store(DX + t * D + d, acc, mask=d < D)


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rw, rb, w1, w2, A):
        A = int(A)
        T, D = x.shape
        E = rw.shape[1]
        F = w1.shape[2] // 2
        N = T * A
        bf = x.dtype == torch.bfloat16
        s = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _router[(triton.cdiv(T, 32), triton.cdiv(E, 32))](x, rw, s, T, D, E, 32, 32, 32)
        keys = torch.empty((T, E), device=x.device, dtype=torch.int64)
        _keys[(triton.cdiv(T * E, 256),)](s, rb, keys, T * E, E, 256)
        sel = torch.topk(keys, A, dim=1, largest=True, sorted=True).indices.contiguous()
        weights = torch.empty((N,), device=x.device, dtype=torch.float32)
        _weights[(triton.cdiv(T, 256),)](s, sel, weights, T, E, A, 256)
        flat = sel.reshape(-1)
        order = torch.argsort(flat).contiguous()
        off = torch.empty((E + 1,), device=x.device, dtype=torch.int32)
        _offsets[(E + 1,)](flat, order, off, N, E + 1)
        pre = torch.empty((E + 1,), device=x.device, dtype=torch.int32)
        _prefix[(1,)](off, pre, E, ROW, num_warps=1)
        inv = torch.empty((N,), device=x.device, dtype=torch.int64)
        _inverse[(triton.cdiv(N, 256),)](order, inv, N, 256)
        cap = triton.cdiv(N, ROW) + E
        gu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _expert_in[(cap, triton.cdiv(2 * F, COL))](x, w1, order, off, pre, gu,
            N, A, D, 2 * F, E, cap, bf, ROW, COL, KB, num_warps=4, num_stages=3)
        act = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _activate[(triton.cdiv(N * F, 256),)](gu, act, N, F, 256)
        fe = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _segment[(cap, triton.cdiv(D, COL))](act, w2, order, off, pre, fe,
            N, F, D, E, bf, bf, False, ROW, COL, KB, num_warps=4, num_stages=3)
        y = torch.empty((T, D), device=x.device, dtype=x.dtype)
        _combine[(T, triton.cdiv(D, 256))](fe, weights, inv, y, T, D, A, 256)
        ctx.save_for_backward(x, rw, w1, w2, s, sel, weights, order, off, pre, inv, gu, act, fe)
        ctx.meta = (A, T, D, E, F, bf, cap)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, rw, w1, w2, s, sel, weights, order, off, pre, inv, gu, act, fe = ctx.saved_tensors
        A, T, D, E, F, bf, cap = ctx.meta
        N = T * A
        dy = dy.contiguous()
        dz = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _dz[(triton.cdiv(N * D, 256),)](dy, weights, order, dz, N, A, D, 256)
        dw2 = torch.empty_like(w2)
        _dw2[(E, triton.cdiv(F, 128), triton.cdiv(D, 64))](act, dz, off, dw2,
            N, F, D, E, bf, 128, 64, KB, num_warps=4, num_stages=3)
        da = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _segment[(cap, triton.cdiv(F, COL))](dz, w2, order, off, pre, da,
            N, D, F, E, bf, bf, True, ROW, COL, KB, num_warps=4, num_stages=3)
        dgu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _dactivate[(triton.cdiv(N * F, 256),)](gu, da, dgu, N, F, 256)
        dw1 = torch.empty_like(w1)
        _dw1[(E, triton.cdiv(D, 128), triton.cdiv(2 * F, 64))](x, dgu, order, off, dw1,
            N, A, D, 2 * F, E, bf, 128, 64, KB, num_warps=4, num_stages=3)
        dxp = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _segment[(cap, triton.cdiv(D, COL))](dgu, w1, order, off, pre, dxp,
            N, 2 * F, D, E, bf, bf, True, ROW, COL, KB, num_warps=4, num_stages=3)
        pd = torch.empty((N,), device=x.device, dtype=torch.float32)
        _pair_dot[(N,)](dy, fe, order, pd, N, A, D, 256, num_warps=8)
        dl = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _route_grad[(T, triton.cdiv(E, 64))](s, sel, weights, inv, pd, dl, T, E, A, 64)
        drw = torch.empty_like(rw)
        _dense_mm[(triton.cdiv(D, 32), triton.cdiv(E, 32))](x, dl, drw,
            D, E, T, 1, D, E, 1, 32, 32, 32, num_warps=4, num_stages=3)
        dxr = torch.empty((T, D), device=x.device, dtype=torch.float32)
        _dense_mm[(triton.cdiv(T, 32), triton.cdiv(D, 32))](dl, rw, dxr,
            T, D, E, E, 1, 1, E, 32, 32, 32, num_warps=4, num_stages=3)
        dx = torch.empty_like(x)
        _combine_dx[(T, triton.cdiv(D, 256))](dxp, dxr, inv, dx, T, D, A, 256)
        return dx, drw, None, dw1, dw2, None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
