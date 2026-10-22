import torch
import triton
import triton.language as tl



@triton.jit
def _router(X, RW, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
            BM: tl.constexpr = 32, BN: tl.constexpr = 32, BK: tl.constexpr = 32):
    im = tl.program_id(0) * BM + tl.arange(0, BM)
    ie = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        ik = k + tl.arange(0, BK)
        x = tl.load(X + im[:, None] * D + ik[None, :], mask=(im[:, None] < T) & (ik[None, :] < D), other=0.0).to(tl.float32)
        w = tl.load(RW + ik[:, None] * E + ie[None, :], mask=(ik[:, None] < D) & (ie[None, :] < E), other=0.0).to(tl.float32)
        acc += tl.dot(x, w, input_precision="ieee")
    tl.store(S + im[:, None] * E + ie[None, :], tl.sigmoid(acc), mask=(im[:, None] < T) & (ie[None, :] < E))


@triton.jit
def _keys(S, B, K, n: tl.constexpr, E: tl.constexpr, BS: tl.constexpr = 256):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    m = q < n
    e = q % E
    v = tl.load(S + q, mask=m, other=0.0) + tl.load(B + e, mask=m, other=0.0)
    v = tl.where(v == 0.0, 0.0, v)
    i = v.to(tl.int32, bitcast=True)
    j = i.to(tl.int64) & 0xFFFFFFFF
    u = tl.where(i >= 0, j ^ 0x80000000, (~j) & 0xFFFFFFFF)
    tl.store(K + q, (u << 10) | (E - 1 - e), mask=m)


@triton.jit
def _weights(S, SEL, W, E: tl.constexpr, A: tl.constexpr, BA: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BA)
    m = a < A
    e = tl.load(SEL + t * A + a, mask=m, other=0)
    s = tl.load(S + t * E + e, mask=m, other=0.0)
    tl.store(W + t * A + a, s / tl.sum(s, axis=0), mask=m)


@triton.jit
def _offsets(SEL, ORDER, OFF, N: tl.constexpr):
    e = tl.program_id(0)
    lo = 0
    hi = N
    while lo < hi:
        md = (lo + hi) // 2
        p = tl.load(ORDER + md)
        v = tl.load(SEL + p)
        take = v < e
        lo = tl.where(take, md + 1, lo)
        hi = tl.where(take, hi, md)
    tl.store(OFF + e, lo)


@triton.jit
def _prefix(OFF, PREF, E: tl.constexpr, BE: tl.constexpr):
    e = tl.arange(0, BE)
    m = e < E
    lo = tl.load(OFF + e, mask=m, other=0)
    hi = tl.load(OFF + e + 1, mask=m, other=0)
    count = tl.where(m, (hi - lo + 64 - 1) // 64, 0)
    tl.store(PREF + e + 1, tl.cumsum(count, axis=0), mask=m)
    tl.store(PREF, 0)


@triton.jit
def _tile_map(PREF, OFF, TE, TS, E: tl.constexpr):
    q = tl.program_id(0)
    lo = 0
    hi = E
    while lo < hi:
        md = (lo + hi) // 2
        take = tl.load(PREF + md + 1) <= q
        lo = tl.where(take, md + 1, lo)
        hi = tl.where(take, hi, md)
    valid = lo < E
    base = tl.load(PREF + lo, mask=valid, other=0)
    start = tl.load(OFF + lo, mask=valid, other=0) + (q - base) * 64
    tl.store(TE + q, tl.where(valid, lo, -1))
    tl.store(TS + q, start)


@triton.jit
def _inverse(ORDER, INV, N: tl.constexpr, BS: tl.constexpr = 256):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    m = q < N
    p = tl.load(ORDER + q, mask=m, other=0)
    tl.store(INV + p, q, mask=m)


@triton.jit
def _expert_in(X, W1, ORDER, OFF, TE, TS, GU, D: tl.constexpr, F2: tl.constexpr,
               A: tl.constexpr, N: tl.constexpr, BF: tl.constexpr,
               BK: tl.constexpr = 32, BN: tl.constexpr = 64):
    tile = tl.program_id(0)
    e = tl.load(TE + tile)
    start = tl.load(TS + tile)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    ir = start + tl.arange(0, 64)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    p = tl.load(ORDER + ir, mask=(e >= 0) & (ir < end), other=0)
    tok = p // A
    acc = tl.zeros((64, BN), tl.float32)
    for k in range(0, D, BK):
        ik = k + tl.arange(0, BK)
        x = tl.load(X + tok[:, None] * D + ik[None, :], mask=(e >= 0) & (ir[:, None] < end) & (ik[None, :] < D), other=0.0)
        w = tl.load(W1 + e * D * F2 + ik[:, None] * F2 + j[None, :], mask=(e >= 0) & (ik[:, None] < D) & (j[None, :] < F2), other=0.0)
        if BF:
            acc += tl.dot(x, w)
        else:
            acc += tl.dot(x.to(tl.float32), w.to(tl.float32), input_precision="tf32x3")
    tl.store(GU + ir[:, None] * F2 + j[None, :], acc, mask=(e >= 0) & (ir[:, None] < end) & (j[None, :] < F2))


@triton.jit
def _activate(GU, ACT, N: tl.constexpr, F: tl.constexpr, BS: tl.constexpr = 256):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    m = q < N * F
    r = q // F
    f = q % F
    g = tl.load(GU + r * (2 * F) + f, mask=m, other=0.0)
    u = tl.load(GU + r * (2 * F) + F + f, mask=m, other=0.0)
    tl.store(ACT + q, g * tl.sigmoid(g) * u, mask=m)


@triton.jit
def _segment(AIN, EW, OFF, TE, TS, OUT, K: tl.constexpr, O: tl.constexpr,
             N: tl.constexpr, BF: tl.constexpr, SPLIT: tl.constexpr,
             BK: tl.constexpr = 32, BN: tl.constexpr = 64):
    tile = tl.program_id(0)
    e = tl.load(TE + tile)
    start = tl.load(TS + tile)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    ir = start + tl.arange(0, 64)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((64, BN), tl.float32)
    for k in range(0, K, BK):
        ik = k + tl.arange(0, BK)
        a = tl.load(AIN + ir[:, None] * K + ik[None, :], mask=(e >= 0) & (ir[:, None] < end) & (ik[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(EW + e * K * O + ik[:, None] * O + j[None, :], mask=(e >= 0) & (ik[:, None] < K) & (j[None, :] < O), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            acc += tl.dot(ah, b)
            if SPLIT:
                acc += tl.dot((a - ah).to(tl.bfloat16), b)
        else:
            acc += tl.dot(a, b.to(tl.float32), input_precision="tf32x3")
    tl.store(OUT + ir[:, None] * O + j[None, :], acc, mask=(e >= 0) & (ir[:, None] < end) & (j[None, :] < O))


@triton.jit
def _combine(FE, W, INV, Y, D: tl.constexpr, A: tl.constexpr,
             BD: tl.constexpr = 256):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(0, A):
        p = t * A + a
        r = tl.load(INV + p)
        acc += tl.load(W + p) * tl.load(FE + r * D + d, mask=d < D, other=0.0)
    tl.store(Y + t * D + d, acc, mask=d < D)


@triton.jit
def _dz(DY, W, ORDER, DZ, N: tl.constexpr, D: tl.constexpr, A: tl.constexpr,
        BS: tl.constexpr = 256):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    m = q < N * D
    r = q // D
    d = q % D
    p = tl.load(ORDER + r, mask=m, other=0)
    tok = p // A
    tl.store(DZ + q, tl.load(DY + tok * D + d, mask=m, other=0.0) * tl.load(W + p, mask=m, other=0.0), mask=m)


@triton.jit
def _dgu(DZ, W2, GU, OFF, TE, TS, DGU, N: tl.constexpr, D: tl.constexpr,
         F: tl.constexpr, BF: tl.constexpr, BK: tl.constexpr = 32,
         BN: tl.constexpr = 64):
    tile = tl.program_id(0)
    e = tl.load(TE + tile)
    start = tl.load(TS + tile)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    ir = start + tl.arange(0, 64)
    f = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((64, BN), tl.float32)
    for k in range(0, D, BK):
        d = k + tl.arange(0, BK)
        z = tl.load(DZ + ir[:, None] * D + d[None, :], mask=(e >= 0) & (ir[:, None] < end) & (d[None, :] < D), other=0.0).to(tl.float32)
        b = tl.load(W2 + e * F * D + f[None, :] * D + d[:, None], mask=(e >= 0) & (f[None, :] < F) & (d[:, None] < D), other=0.0)
        if BF:
            zh = z.to(tl.bfloat16)
            acc += tl.dot(zh, b)
        else:
            acc += tl.dot(z, b.to(tl.float32), input_precision="tf32x3")
    mask = (e >= 0) & (ir[:, None] < end) & (f[None, :] < F)
    g = tl.load(GU + ir[:, None] * (2 * F) + f[None, :], mask=mask, other=0.0)
    u = tl.load(GU + ir[:, None] * (2 * F) + F + f[None, :], mask=mask, other=0.0)
    s = tl.sigmoid(g)
    silu = g * s
    tl.store(DGU + ir[:, None] * (2 * F) + f[None, :], acc * u * (s + silu * (1.0 - s)), mask=mask)
    tl.store(DGU + ir[:, None] * (2 * F) + F + f[None, :], acc * silu, mask=mask)


@triton.jit
def _dw2(ACT, DZ, OFF, DW, F: tl.constexpr, D: tl.constexpr,
         BF: tl.constexpr, BM: tl.constexpr = 128, BN: tl.constexpr = 64,
         BK: tl.constexpr = 32):
    e = tl.program_id(0)
    f = tl.program_id(1) * BM + tl.arange(0, BM)
    d = tl.program_id(2) * BN + tl.arange(0, BN)
    lo = tl.load(OFF + e)
    hi = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(lo, hi, BK):
        r = k + tl.arange(0, BK)
        a = tl.load(ACT + f[:, None] + r[None, :] * F, mask=(f[:, None] < F) & (r[None, :] < hi), other=0.0).to(tl.float32)
        z = tl.load(DZ + r[:, None] * D + d[None, :], mask=(r[:, None] < hi) & (d[None, :] < D), other=0.0).to(tl.float32)
        if BF:
            ah = a.to(tl.bfloat16); al = (a - ah).to(tl.bfloat16)
            zh = z.to(tl.bfloat16); zl = (z - zh).to(tl.bfloat16)
            acc += tl.dot(ah, zh) + tl.dot(ah, zl) + tl.dot(al, zh)
        else:
            acc += tl.dot(a, z, input_precision="tf32x3")
    tl.store(DW + e * F * D + f[:, None] * D + d[None, :], acc, mask=(f[:, None] < F) & (d[None, :] < D))


@triton.jit
def _dw1(X, DGU, ORDER, OFF, DW, D: tl.constexpr, F2: tl.constexpr,
         A: tl.constexpr, BF: tl.constexpr, BM: tl.constexpr = 128,
         BN: tl.constexpr = 64, BK: tl.constexpr = 32):
    e = tl.program_id(0)
    d = tl.program_id(1) * BM + tl.arange(0, BM)
    f = tl.program_id(2) * BN + tl.arange(0, BN)
    lo = tl.load(OFF + e)
    hi = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(lo, hi, BK):
        r = k + tl.arange(0, BK)
        p = tl.load(ORDER + r, mask=r < hi, other=0)
        tok = p // A
        x = tl.load(X + d[:, None] + tok[None, :] * D, mask=(d[:, None] < D) & (r[None, :] < hi), other=0.0)
        g = tl.load(DGU + r[:, None] * F2 + f[None, :], mask=(r[:, None] < hi) & (f[None, :] < F2), other=0.0).to(tl.float32)
        if BF:
            gh = g.to(tl.bfloat16)
            acc += tl.dot(x, gh) + tl.dot(x, (g - gh).to(tl.bfloat16))
        else:
            acc += tl.dot(x.to(tl.float32), g, input_precision="tf32x3")
    tl.store(DW + e * D * F2 + d[:, None] * F2 + f[None, :], acc, mask=(d[:, None] < D) & (f[None, :] < F2))


@triton.jit
def _pair_dot(DY, FE, ORDER, P, N: tl.constexpr, D: tl.constexpr,
              A: tl.constexpr, BD: tl.constexpr = 256):
    r = tl.program_id(0)
    p = tl.load(ORDER + r)
    tok = p // A
    acc = 0.0
    for k in range(0, D, BD):
        d = k + tl.arange(0, BD)
        acc += tl.sum(tl.load(DY + tok * D + d, mask=d < D, other=0.0).to(tl.float32) * tl.load(FE + r * D + d, mask=d < D, other=0.0), axis=0)
    tl.store(P + p, acc)


@triton.jit
def _route_ds(S, SEL, W, P, DSP, E: tl.constexpr, A: tl.constexpr,
              BA: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BA)
    m = a < A
    e = tl.load(SEL + t * A + a, mask=m, other=0)
    s = tl.load(S + t * E + e, mask=m, other=0.0)
    w = tl.load(W + t * A + a, mask=m, other=0.0)
    p = tl.load(P + t * A + a, mask=m, other=0.0)
    ds = (p - tl.sum(w * p, axis=0)) / tl.sum(s, axis=0) * s * (1.0 - s)
    tl.store(DSP + t * A + a, ds, mask=m)


@triton.jit
def _dense_ds(SEL, DSP, DS, E: tl.constexpr, A: tl.constexpr,
              BE: tl.constexpr = 64):
    t = tl.program_id(0)
    e = tl.program_id(1) * BE + tl.arange(0, BE)
    v = tl.zeros((BE,), tl.float32)
    for a in range(0, A):
        se = tl.load(SEL + t * A + a)
        v += tl.where(e == se, tl.load(DSP + t * A + a), 0.0)
    tl.store(DS + t * E + e, v, mask=e < E)


@triton.jit
def _router_rw(X, DS, DRW, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
               BM: tl.constexpr = 32, BN: tl.constexpr = 32, BK: tl.constexpr = 32):
    d = tl.program_id(0) * BM + tl.arange(0, BM)
    e = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, T, BK):
        t = k + tl.arange(0, BK)
        x = tl.load(X + t[None, :] * D + d[:, None], mask=(t[None, :] < T) & (d[:, None] < D), other=0.0).to(tl.float32)
        z = tl.load(DS + t[:, None] * E + e[None, :], mask=(t[:, None] < T) & (e[None, :] < E), other=0.0)
        acc += tl.dot(x, z, input_precision="ieee")
    tl.store(DRW + d[:, None] * E + e[None, :], acc, mask=(d[:, None] < D) & (e[None, :] < E))


@triton.jit
def _dx_rows(DGU, W1, OFF, TE, TS, DXP, N: tl.constexpr, D: tl.constexpr,
             F2: tl.constexpr, BF: tl.constexpr, BK: tl.constexpr = 32,
             BN: tl.constexpr = 64):
    tile = tl.program_id(0)
    e = tl.load(TE + tile)
    start = tl.load(TS + tile)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    ir = start + tl.arange(0, 64)
    d = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((64, BN), tl.float32)
    for k in range(0, F2, BK):
        f = k + tl.arange(0, BK)
        g = tl.load(DGU + ir[:, None] * F2 + f[None, :], mask=(e >= 0) & (ir[:, None] < end) & (f[None, :] < F2), other=0.0).to(tl.float32)
        b = tl.load(W1 + e * D * F2 + d[None, :] * F2 + f[:, None], mask=(e >= 0) & (d[None, :] < D) & (f[:, None] < F2), other=0.0)
        if BF:
            gh = g.to(tl.bfloat16)
            acc += tl.dot(gh, b)
        else:
            acc += tl.dot(g, b.to(tl.float32), input_precision="tf32x3")
    tl.store(DXP + ir[:, None] * D + d[None, :], acc, mask=(e >= 0) & (ir[:, None] < end) & (d[None, :] < D))


@triton.jit
def _combine_dx(DXP, INV, W, DS, RW, DX, D: tl.constexpr, E: tl.constexpr,
                A: tl.constexpr, BD: tl.constexpr = 256):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(0, A):
        p = t * A + a
        r = tl.load(INV + p)
        acc += tl.load(DXP + r * D + d, mask=d < D, other=0.0)
    for e0 in range(0, E, 32):
        e = e0 + tl.arange(0, 32)
        z = tl.load(DS + t * E + e, mask=e < E, other=0.0)
        b = tl.load(RW + d[:, None] * E + e[None, :], mask=(d[:, None] < D) & (e[None, :] < E), other=0.0).to(tl.float32)
        acc += tl.sum(b * z[None, :], axis=1)
    tl.store(DX + t * D + d, acc, mask=d < D)


def _metadata(sel, order, E, N, device):
    off = torch.empty(E + 1, device=device, dtype=torch.int32)
    _offsets[(E + 1,)](sel, order, off, N=N, num_warps=1)
    pref = torch.empty(E + 1, device=device, dtype=torch.int32)
    _prefix[(1,)](off, pref, E=E, BE=triton.next_power_of_2(E), num_warps=1)
    cap = triton.cdiv(N, 64) + E
    te = torch.empty(cap, device=device, dtype=torch.int32)
    ts = torch.empty(cap, device=device, dtype=torch.int32)
    _tile_map[(cap,)](pref, off, te, ts, E=E, num_warps=1)
    inv = torch.empty(N, device=device, dtype=torch.int64)
    _inverse[(triton.cdiv(N, 256),)](order, inv, N=N)
    return off, te, ts, inv, cap


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        T, D = x.shape
        E, F, A = router_w.shape[1], w2.shape[1], int(A)
        N = T * A
        bf = x.dtype == torch.bfloat16
        s = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _router[(triton.cdiv(T, 32), triton.cdiv(E, 32))](x, router_w, s, T=T, D=D, E=E)
        keys = torch.empty((T, E), device=x.device, dtype=torch.int64)
        _keys[(triton.cdiv(T * E, 256),)](s, router_b, keys, n=T * E, E=E)
        sel = torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices.contiguous()
        w = torch.empty((T, A), device=x.device, dtype=torch.float32)
        _weights[(T,)](s, sel, w, E=E, A=A, BA=triton.next_power_of_2(A))
        order = torch.argsort(sel.reshape(-1)).contiguous()
        off, te, ts, inv, cap = _metadata(sel, order, E, N, x.device)
        gu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _expert_in[(cap, triton.cdiv(2 * F, 64))](x, w1, order, off, te, ts, gu, D=D, F2=2 * F, A=A, N=N, BF=bf, num_warps=4, num_stages=3)
        act = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _activate[(triton.cdiv(N * F, 256),)](gu, act, N=N, F=F)
        fe = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _segment[(cap, triton.cdiv(D, 64))](act, w2, off, te, ts, fe, K=F, O=D, N=N, BF=bf, SPLIT=bf, num_warps=4, num_stages=3)
        y = torch.empty((T, D), device=x.device, dtype=x.dtype)
        _combine[(T, triton.cdiv(D, 256))](fe, w, inv, y, D=D, A=A)
        ctx.save_for_backward(x, router_w, w1, w2, s, sel, w, order, off, te, ts, inv, gu, act, fe)
        ctx.meta = T, D, E, F, A, N, cap, bf
        return y

    @staticmethod
    def backward(ctx, dy):
        x, rw, w1, w2, s, sel, w, order, off, te, ts, inv, gu, act, fe = ctx.saved_tensors
        T, D, E, F, A, N, cap, bf = ctx.meta
        dy = dy.contiguous()
        dz = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _dz[(triton.cdiv(N * D, 256),)](dy, w, order, dz, N=N, D=D, A=A)
        dgu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _dgu[(cap, triton.cdiv(F, 64))](dz, w2, gu, off, te, ts, dgu, N=N, D=D, F=F, BF=bf, num_warps=4, num_stages=3)
        dw2 = torch.empty_like(w2)
        _dw2[(E, triton.cdiv(F, 128), triton.cdiv(D, 64))](act, dz, off, dw2, F=F, D=D, BF=bf, num_warps=4, num_stages=3)
        dw1 = torch.empty_like(w1)
        _dw1[(E, triton.cdiv(D, 128), triton.cdiv(2 * F, 64))](x, dgu, order, off, dw1, D=D, F2=2 * F, A=A, BF=bf, num_warps=4, num_stages=3)
        pd = torch.empty(N, device=x.device, dtype=torch.float32)
        _pair_dot[(N,)](dy, fe, order, pd, N=N, D=D, A=A, num_warps=4)
        dsp = torch.empty(N, device=x.device, dtype=torch.float32)
        _route_ds[(T,)](s, sel, w, pd, dsp, E=E, A=A, BA=triton.next_power_of_2(A))
        ds = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _dense_ds[(T, triton.cdiv(E, 64))](sel, dsp, ds, E=E, A=A)
        drw = torch.empty_like(rw)
        _router_rw[(triton.cdiv(D, 32), triton.cdiv(E, 32))](x, ds, drw, T=T, D=D, E=E)
        dxp = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _dx_rows[(cap, triton.cdiv(D, 64))](dgu, w1, off, te, ts, dxp, N=N, D=D, F2=2 * F, BF=bf, num_warps=4, num_stages=3)
        dx = torch.empty_like(x)
        _combine_dx[(T, triton.cdiv(D, 256))](dxp, inv, w, ds, rw, dx, D=D, E=E, A=A)
        return dx, drw, None, dw1, dw2, None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
