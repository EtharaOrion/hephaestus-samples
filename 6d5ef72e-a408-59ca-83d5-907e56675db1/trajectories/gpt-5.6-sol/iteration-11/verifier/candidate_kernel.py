import torch
import triton
import triton.language as tl

ROW = 64
KBLK = 32


@triton.jit
def _router(X, RW, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        rk = k + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * D + rk[None, :],
                    mask=(rm[:, None] < T) & (rk[None, :] < D), other=0.0)
        w = tl.load(RW + rk[:, None] * E + rn[None, :],
                    mask=(rk[:, None] < D) & (rn[None, :] < E), other=0.0)
        acc += tl.dot(x.to(tl.float32), w.to(tl.float32), input_precision="ieee")
    tl.store(S + rm[:, None] * E + rn[None, :], tl.sigmoid(acc),
             mask=(rm[:, None] < T) & (rn[None, :] < E))


@triton.jit
def _keys(S, B, K, T: tl.constexpr, E: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < T * E
    e = o % E
    v = tl.load(S + o, mask=m) + tl.load(B + e, mask=m)
    v = tl.where(v == 0.0, 0.0, v)
    ib = v.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    tl.store(K + o, (u << 10) | (E - 1 - e), mask=m)


@triton.jit
def _route_weights(S, SEL, W, T: tl.constexpr, E: tl.constexpr,
                   A: tl.constexpr, BA: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BA)
    e = tl.load(SEL + t * A + a, mask=a < A, other=0)
    q = tl.load(S + t * E + e, mask=a < A, other=0.0)
    den = tl.sum(q, axis=0)
    tl.store(W + t * A + a, q / den, mask=a < A)


@triton.jit
def _offsets(SEL, ORDER, OFF, N: tl.constexpr, E: tl.constexpr,
             A: tl.constexpr):
    e = tl.program_id(0)
    lo = 0
    hi = N
    while lo < hi:
        mid = (lo + hi) // 2
        p = tl.load(ORDER + mid)
        v = tl.load(SEL + p)
        take = v < e
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(take, hi, mid)
    tl.store(OFF + e, lo)
    if e == E - 1:
        tl.store(OFF + E, N)


@triton.jit
def _tiles(OFF, TE, TS, E: tl.constexpr, CAP: tl.constexpr,
           BM: tl.constexpr):
    cursor = 0
    for e in range(E):
        begin = tl.load(OFF + e)
        end = tl.load(OFF + e + 1)
        nt = (end - begin + BM - 1) // BM
        j = 0
        while j < nt:
            tl.store(TE + cursor, e)
            tl.store(TS + cursor, begin + j * BM)
            cursor += 1
            j += 1
    while cursor < CAP:
        tl.store(TE + cursor, -1)
        cursor += 1


@triton.jit
def _inverse(ORDER, INV, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    p = tl.load(ORDER + i, mask=m)
    tl.store(INV + p, i, mask=m)


@triton.jit
def _expert_in(X, W1, ORDER, OFF, TE, TS, GU, N: tl.constexpr, D: tl.constexpr,
               F2: tl.constexpr, A: tl.constexpr, BF: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TE + tile)
    pos = tl.load(TS + tile) + tl.arange(0, BM)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    pair = tl.load(ORDER + pos, mask=(e >= 0) & (pos < end), other=0)
    tok = pair // A
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        rk = k + tl.arange(0, BK)
        x = tl.load(X + tok[:, None] * D + rk[None, :],
                    mask=(e >= 0) & (pos[:, None] < end) & (rk[None, :] < D), other=0.0)
        w = tl.load(W1 + e * D * F2 + rk[:, None] * F2 + col[None, :],
                    mask=(e >= 0) & (rk[:, None] < D) & (col[None, :] < F2), other=0.0)
        if BF:
            acc += tl.dot(x, w)
        else:
            acc += tl.dot(x, w, input_precision="tf32x3")
    tl.store(GU + pos[:, None] * F2 + col[None, :], acc,
             mask=(e >= 0) & (pos[:, None] < end) & (col[None, :] < F2))


@triton.jit
def _activate(GU, ACT, N: tl.constexpr, F: tl.constexpr,
              BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N * F
    r = o // F
    c = o % F
    g = tl.load(GU + r * (2 * F) + c, mask=m)
    u = tl.load(GU + r * (2 * F) + F + c, mask=m)
    tl.store(ACT + o, g * tl.sigmoid(g) * u, mask=m)


@triton.jit
def _expert_out(ACT, W2, OFF, TE, TS, Z, N: tl.constexpr, F: tl.constexpr,
                D: tl.constexpr, BF: tl.constexpr, BM: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TE + tile)
    row = tl.load(TS + tile) + tl.arange(0, BM)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, F, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(ACT + row[:, None] * F + rk[None, :],
                    mask=(e >= 0) & (row[:, None] < end) & (rk[None, :] < F), other=0.0)
        w = tl.load(W2 + e * F * D + rk[:, None] * D + col[None, :],
                    mask=(e >= 0) & (rk[:, None] < F) & (col[None, :] < D), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            al = (a - ah.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(ah, w) + tl.dot(al, w)
        else:
            acc += tl.dot(a, w, input_precision="tf32x3")
    tl.store(Z + row[:, None] * D + col[None, :], acc,
             mask=(e >= 0) & (row[:, None] < end) & (col[None, :] < D))


@triton.jit
def _combine(Z, W, INV, Y, T: tl.constexpr, D: tl.constexpr,
             A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(A):
        p = t * A + a
        r = tl.load(INV + p)
        wa = tl.load(W + p)
        acc += wa * tl.load(Z + r * D + c, mask=c < D, other=0.0)
    tl.store(Y + t * D + c, acc, mask=c < D)


@triton.jit
def _dz(DY, W, ORDER, DZ, N: tl.constexpr, D: tl.constexpr,
        A: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BD + tl.arange(0, BD)
    p = tl.load(ORDER + r)
    t = p // A
    wa = tl.load(W + p)
    v = tl.load(DY + t * D + c, mask=c < D, other=0.0)
    tl.store(DZ + r * D + c, wa * v, mask=c < D)


@triton.jit
def _pair_dot(DY, Z, ORDER, DW, N: tl.constexpr, D: tl.constexpr,
              A: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BD)
    p = tl.load(ORDER + r)
    t = p // A
    dy = tl.load(DY + t * D + c, mask=c < D, other=0.0).to(tl.float32)
    z = tl.load(Z + r * D + c, mask=c < D, other=0.0)
    tl.store(DW + r, tl.sum(dy * z, axis=0))


@triton.jit
def _route_grad(S, SEL, W, INV, DW, DL, T: tl.constexpr, E: tl.constexpr,
                A: tl.constexpr, BE: tl.constexpr, BA: tl.constexpr):
    t = tl.program_id(0)
    ee = tl.arange(0, BE)
    aa = tl.arange(0, BA)
    se = tl.load(SEL + t * A + aa, mask=aa < A, other=0)
    q = tl.load(S + t * E + se, mask=aa < A, other=0.0)
    wa = tl.load(W + t * A + aa, mask=aa < A, other=0.0)
    sr = tl.load(INV + t * A + aa, mask=aa < A, other=0)
    dw = tl.load(DW + sr, mask=aa < A, other=0.0)
    center = tl.sum(wa * dw, axis=0)
    ds = (dw - center) / tl.sum(q, axis=0)
    out = tl.zeros((BE,), tl.float32)
    for a in range(A):
        chosen = tl.load(SEL + t * A + a)
        sv = tl.load(S + t * E + chosen)
        dsv = tl.sum(tl.where(aa == a, ds, 0.0), axis=0)
        out = tl.where(ee == chosen, dsv * sv * (1.0 - sv), out)
    tl.store(DL + t * E + ee, out, mask=ee < E)


@triton.jit
def _dact(DZ, W2, OFF, TE, TS, DA, N: tl.constexpr, F: tl.constexpr,
          D: tl.constexpr, BF: tl.constexpr, BM: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TE + tile)
    row = tl.load(TS + tile) + tl.arange(0, BM)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        rk = k + tl.arange(0, BK)
        dz = tl.load(DZ + row[:, None] * D + rk[None, :],
                     mask=(e >= 0) & (row[:, None] < end) & (rk[None, :] < D), other=0.0)
        w = tl.load(W2 + e * F * D + col[None, :] * D + rk[:, None],
                    mask=(e >= 0) & (col[None, :] < F) & (rk[:, None] < D), other=0.0)
        if BF:
            acc += tl.dot(dz.to(tl.bfloat16), w)
        else:
            acc += tl.dot(dz, w, input_precision="tf32x3")
    tl.store(DA + row[:, None] * F + col[None, :], acc,
             mask=(e >= 0) & (row[:, None] < end) & (col[None, :] < F))


@triton.jit
def _dgu(GU, DA, DGU, N: tl.constexpr, F: tl.constexpr,
         BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N * F
    r = o // F
    c = o % F
    g = tl.load(GU + r * (2 * F) + c, mask=m)
    u = tl.load(GU + r * (2 * F) + F + c, mask=m)
    da = tl.load(DA + o, mask=m)
    s = tl.sigmoid(g)
    silu = g * s
    tl.store(DGU + r * (2 * F) + c,
             da * u * (s + silu * (1.0 - s)), mask=m)
    tl.store(DGU + r * (2 * F) + F + c, da * silu, mask=m)


@triton.jit
def _dx_rows(DGU, W1, OFF, TE, TS, DXR, N: tl.constexpr, D: tl.constexpr,
             F2: tl.constexpr, BF: tl.constexpr, BM: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TE + tile)
    row = tl.load(TS + tile) + tl.arange(0, BM)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, F2, BK):
        rk = k + tl.arange(0, BK)
        dg = tl.load(DGU + row[:, None] * F2 + rk[None, :],
                     mask=(e >= 0) & (row[:, None] < end) & (rk[None, :] < F2), other=0.0)
        w = tl.load(W1 + e * D * F2 + col[None, :] * F2 + rk[:, None],
                    mask=(e >= 0) & (col[None, :] < D) & (rk[:, None] < F2), other=0.0)
        if BF:
            acc += tl.dot(dg.to(tl.bfloat16), w)
        else:
            acc += tl.dot(dg, w, input_precision="tf32x3")
    tl.store(DXR + row[:, None] * D + col[None, :], acc,
             mask=(e >= 0) & (row[:, None] < end) & (col[None, :] < D))


@triton.jit
def _dx_combine(DXR, DL, RW, INV, DX, T: tl.constexpr, D: tl.constexpr,
                E: tl.constexpr, A: tl.constexpr, BF: tl.constexpr,
                BD: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(A):
        r = tl.load(INV + t * A + a)
        acc += tl.load(DXR + r * D + c, mask=c < D, other=0.0)
    for k in range(0, E, BK):
        rk = k + tl.arange(0, BK)
        dl = tl.load(DL + t * E + rk, mask=rk < E, other=0.0)
        rw = tl.load(RW + c[:, None] * E + rk[None, :],
                     mask=(c[:, None] < D) & (rk[None, :] < E), other=0.0)
        acc += tl.sum(rw.to(tl.float32) * dl[None, :], axis=1)
    tl.store(DX + t * D + c, acc, mask=c < D)


@triton.jit
def _dw2(ACT, DZ, OFF, OUT, N: tl.constexpr, E: tl.constexpr,
         F: tl.constexpr, D: tl.constexpr, BF: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(0)
    rf = tl.program_id(1) * BM + tl.arange(0, BM)
    rd = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    k = begin
    while k < end:
        rr = k + tl.arange(0, BK)
        a = tl.load(ACT + rr[:, None] * F + rf[None, :],
                    mask=(rr[:, None] < end) & (rf[None, :] < F), other=0.0)
        z = tl.load(DZ + rr[:, None] * D + rd[None, :],
                    mask=(rr[:, None] < end) & (rd[None, :] < D), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16); al = (a - ah.to(tl.float32)).to(tl.bfloat16)
            zh = z.to(tl.bfloat16); zl = (z - zh.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(tl.trans(ah), zh) + tl.dot(tl.trans(al), zh) + tl.dot(tl.trans(ah), zl)
        else:
            acc += tl.dot(tl.trans(a), z, input_precision="tf32x3")
        k += BK
    tl.store(OUT + e * F * D + rf[:, None] * D + rd[None, :], acc,
             mask=(rf[:, None] < F) & (rd[None, :] < D))


@triton.jit
def _dw1(X, DGU, ORDER, OFF, OUT, N: tl.constexpr, E: tl.constexpr,
         D: tl.constexpr, F2: tl.constexpr, A: tl.constexpr,
         BF: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         BK: tl.constexpr):
    e = tl.program_id(0)
    rd = tl.program_id(1) * BM + tl.arange(0, BM)
    rf = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    k = begin
    while k < end:
        rr = k + tl.arange(0, BK)
        p = tl.load(ORDER + rr, mask=rr < end, other=0)
        tok = p // A
        x = tl.load(X + tok[:, None] * D + rd[None, :],
                    mask=(rr[:, None] < end) & (rd[None, :] < D), other=0.0)
        g = tl.load(DGU + rr[:, None] * F2 + rf[None, :],
                    mask=(rr[:, None] < end) & (rf[None, :] < F2), other=0.0)
        if BF:
            gh = g.to(tl.bfloat16); gl = (g - gh.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(tl.trans(x), gh) + tl.dot(tl.trans(x), gl)
        else:
            acc += tl.dot(tl.trans(x), g, input_precision="tf32x3")
        k += BK
    tl.store(OUT + e * D * F2 + rd[:, None] * F2 + rf[None, :], acc,
             mask=(rd[:, None] < D) & (rf[None, :] < F2))


@triton.jit
def _drw(X, DL, OUT, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
         BF: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         BK: tl.constexpr):
    rd = tl.program_id(0) * BM + tl.arange(0, BM)
    re = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, T, BK):
        rr = k + tl.arange(0, BK)
        x = tl.load(X + rr[:, None] * D + rd[None, :],
                    mask=(rr[:, None] < T) & (rd[None, :] < D), other=0.0)
        dl = tl.load(DL + rr[:, None] * E + re[None, :],
                     mask=(rr[:, None] < T) & (re[None, :] < E), other=0.0)
        if BF:
            dh = dl.to(tl.bfloat16); dlow = (dl - dh.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(tl.trans(x), dh) + tl.dot(tl.trans(x), dlow)
        else:
            acc += tl.dot(tl.trans(x), dl, input_precision="ieee")
    tl.store(OUT + rd[:, None] * E + re[None, :], acc,
             mask=(rd[:, None] < D) & (re[None, :] < E))


def _meta(sel, order, E, A, row):
    n = sel.numel()
    off = torch.empty(E + 1, device=sel.device, dtype=torch.int32)
    _offsets[(E,)](sel, order, off, N=n, E=E, A=A)
    cap = triton.cdiv(n, row) + E
    te = torch.empty(cap, device=sel.device, dtype=torch.int32)
    ts = torch.empty_like(te)
    _tiles[(1,)](off, te, ts, E=E, CAP=cap, BM=row, num_warps=1)
    inv = torch.empty_like(order)
    _inverse[(triton.cdiv(n, 256),)](order, inv, N=n, BLOCK=256)
    return off, te, ts, inv, cap


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rw, rb, w1, w2, A):
        T, D = x.shape
        E = rw.shape[1]
        F = w2.shape[1]
        A = int(A)
        bf = x.dtype == torch.bfloat16
        s = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _router[(triton.cdiv(T, 32), triton.cdiv(E, 32))](x, rw, s, T=T, D=D, E=E,
            BM=32, BN=32, BK=32, num_warps=4, num_stages=3)
        keys = torch.empty((T, E), device=x.device, dtype=torch.int64)
        _keys[(triton.cdiv(T * E, 256),)](s, rb, keys, T=T, E=E, BLOCK=256)
        sel = torch.topk(keys, A, dim=1, largest=True, sorted=True).indices
        weights = torch.empty((T, A), device=x.device, dtype=torch.float32)
        ba = triton.next_power_of_2(A)
        _route_weights[(T,)](s, sel, weights, T=T, E=E, A=A, BA=ba)
        order = torch.argsort(sel.reshape(-1), stable=False)
        off, te, ts, inv, cap = _meta(sel, order, E, A, ROW)
        n = T * A
        gu = torch.empty((n, 2 * F), device=x.device, dtype=torch.float32)
        _expert_in[(cap, triton.cdiv(2 * F, 64))](x, w1, order, off, te, ts, gu,
            N=n, D=D, F2=2 * F, A=A, BF=bf, BM=ROW, BN=64, BK=32,
            num_warps=4, num_stages=3)
        act = torch.empty((n, F), device=x.device, dtype=torch.float32)
        _activate[(triton.cdiv(n * F, 256),)](gu, act, N=n, F=F, BLOCK=256)
        z = torch.empty((n, D), device=x.device, dtype=torch.float32)
        _expert_out[(cap, triton.cdiv(D, 64))](act, w2, off, te, ts, z,
            N=n, F=F, D=D, BF=bf, BM=ROW, BN=64, BK=32,
            num_warps=4, num_stages=3)
        y = torch.empty_like(x)
        _combine[(T, triton.cdiv(D, 256))](z, weights, inv, y,
            T=T, D=D, A=A, BD=256, num_warps=8)
        ctx.save_for_backward(x, rw, w1, w2, s, sel, weights, order, off, te, ts, inv, gu, act, z)
        ctx.shape = (T, D, E, F, A, cap, bf)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, rw, w1, w2, s, sel, weights, order, off, te, ts, inv, gu, act, z = ctx.saved_tensors
        T, D, E, F, A, cap, bf = ctx.shape
        n = T * A
        dz = torch.empty((n, D), device=x.device, dtype=torch.float32)
        _dz[(n, triton.cdiv(D, 256))](dy, weights, order, dz, N=n, D=D, A=A, BD=256)
        dw = torch.empty(n, device=x.device, dtype=torch.float32)
        bd = triton.next_power_of_2(D)
        _pair_dot[(n,)](dy, z, order, dw, N=n, D=D, A=A, BD=bd, num_warps=8)
        dl = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _route_grad[(T,)](s, sel, weights, inv, dw, dl, T=T, E=E, A=A,
            BE=triton.next_power_of_2(E), BA=triton.next_power_of_2(A))
        da = torch.empty((n, F), device=x.device, dtype=torch.float32)
        _dact[(cap, triton.cdiv(F, 64))](dz, w2, off, te, ts, da, N=n, F=F, D=D,
            BF=bf, BM=ROW, BN=64, BK=32, num_warps=4, num_stages=3)
        dgu = torch.empty_like(gu)
        _dgu[(triton.cdiv(n * F, 256),)](gu, da, dgu, N=n, F=F, BLOCK=256)
        dxr = torch.empty((n, D), device=x.device, dtype=torch.float32)
        _dx_rows[(cap, triton.cdiv(D, 64))](dgu, w1, off, te, ts, dxr, N=n, D=D,
            F2=2 * F, BF=bf, BM=ROW, BN=64, BK=32, num_warps=4, num_stages=3)
        dx = torch.empty_like(x)
        _dx_combine[(T, triton.cdiv(D, 256))](dxr, dl, rw, inv, dx,
            T=T, D=D, E=E, A=A, BF=bf, BD=256, BK=32, num_warps=8)
        gw2 = torch.empty_like(w2)
        _dw2[(E, triton.cdiv(F, 64), triton.cdiv(D, 64))](act, dz, off, gw2,
            N=n, E=E, F=F, D=D, BF=bf, BM=64, BN=64, BK=32,
            num_warps=4, num_stages=3)
        gw1 = torch.empty_like(w1)
        _dw1[(E, triton.cdiv(D, 64), triton.cdiv(2 * F, 64))](x, dgu, order, off, gw1,
            N=n, E=E, D=D, F2=2 * F, A=A, BF=bf, BM=64, BN=64, BK=32,
            num_warps=4, num_stages=3)
        grw = torch.empty_like(rw)
        _drw[(triton.cdiv(D, 32), triton.cdiv(E, 32))](x, dl, grw,
            T=T, D=D, E=E, BF=bf, BM=32, BN=32, BK=32,
            num_warps=4, num_stages=3)
        return dx, grw, None, gw1, gw2, None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
