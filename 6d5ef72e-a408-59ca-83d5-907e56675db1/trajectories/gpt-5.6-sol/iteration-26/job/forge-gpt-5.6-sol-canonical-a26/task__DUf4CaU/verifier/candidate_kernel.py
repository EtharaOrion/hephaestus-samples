import torch
import triton
import triton.language as tl

ROW = 64
COL = 64
KB = 32


@triton.jit
def _router(X, RW, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
            BM: tl.constexpr = 32, BN: tl.constexpr = 32, BK: tl.constexpr = 32):
    im = tl.program_id(0) * BM + tl.arange(0, BM)
    ie = tl.program_id(1) * BN + tl.arange(0, BN)
    z = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        ik = k + tl.arange(0, BK)
        x = tl.load(X + im[:, None] * D + ik[None, :], mask=(im[:, None] < T) & (ik[None, :] < D), other=0.0)
        w = tl.load(RW + ik[:, None] * E + ie[None, :], mask=(ik[:, None] < D) & (ie[None, :] < E), other=0.0)
        z += tl.dot(x.to(tl.float32), w.to(tl.float32), input_precision="ieee")
    tl.store(S + im[:, None] * E + ie[None, :], tl.sigmoid(z), mask=(im[:, None] < T) & (ie[None, :] < E))


@triton.jit
def _keys(S, B, K, n: tl.constexpr, E: tl.constexpr, BS: tl.constexpr = 256):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    e = q % E
    m = q < n
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
def _offsets(FLAT, OFF, N: tl.constexpr, E: tl.constexpr):
    e = tl.program_id(0)
    lo = 0
    hi = N
    for _ in range(32):
        mid = (lo + hi) // 2
        v = tl.load(FLAT + mid, mask=mid < N, other=E)
        take = v < e
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(take, hi, mid)
    tl.store(OFF + e, lo)


@triton.jit
def _prefix(OFF, TP, E: tl.constexpr, R: tl.constexpr):
    total = 0
    tl.store(TP, 0)
    for e in range(0, E):
        total += tl.cdiv(tl.load(OFF + e + 1) - tl.load(OFF + e), R)
        tl.store(TP + e + 1, total)


@triton.jit
def _tile_map(TP, OFF, TE, TS, E: tl.constexpr, R: tl.constexpr):
    p = tl.program_id(0)
    nt = tl.load(TP + E)
    lo = 0
    hi = E
    for _ in range(10):
        mid = (lo + hi) // 2
        v = tl.load(TP + mid, mask=mid <= E, other=nt + 1)
        take = v <= p
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(take, hi, mid)
    e = lo - 1
    valid = (p < nt) & (e >= 0)
    start = tl.load(OFF + e, mask=valid, other=0) + (p - tl.load(TP + e, mask=valid, other=0)) * R
    tl.store(TE + p, tl.where(valid, e, -1))
    tl.store(TS + p, start)


@triton.jit
def _expert_in(X, W1, ORDER, OFF, TE, TS, GU, D: tl.constexpr, F2: tl.constexpr, A: tl.constexpr,
               BF: tl.constexpr, R: tl.constexpr, C: tl.constexpr, KBLK: tl.constexpr):
    p = tl.program_id(0)
    e = tl.load(TE + p)
    start = tl.load(TS + p, mask=e >= 0, other=0)
    ir = tl.arange(0, R)
    ic = tl.program_id(1) * C + tl.arange(0, C)
    pos = start + ir
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    q = tl.load(ORDER + pos, mask=(e >= 0) & (pos < end), other=0)
    tok = q // A
    z = tl.zeros((R, C), tl.float32)
    for k in range(0, D, KBLK):
        ik = k + tl.arange(0, KBLK)
        x = tl.load(X + tok[:, None] * D + ik[None, :], mask=(pos[:, None] < end) & (ik[None, :] < D), other=0.0)
        w = tl.load(W1 + e * D * F2 + ik[:, None] * F2 + ic[None, :], mask=(e >= 0) & (ik[:, None] < D) & (ic[None, :] < F2), other=0.0)
        if BF:
            z += tl.dot(x, w)
        else:
            z += tl.dot(x.to(tl.float32), w.to(tl.float32), input_precision="tf32x3")
    tl.store(GU + pos[:, None] * F2 + ic[None, :], z, mask=(e >= 0) & (pos[:, None] < end) & (ic[None, :] < F2))


@triton.jit
def _activate(GU, ACT, N: tl.constexpr, F: tl.constexpr, BS: tl.constexpr = 256):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    m = q < N * F
    r = q // F
    c = q % F
    g = tl.load(GU + r * (2 * F) + c, mask=m, other=0.0)
    u = tl.load(GU + r * (2 * F) + F + c, mask=m, other=0.0)
    tl.store(ACT + q, g * tl.sigmoid(g) * u, mask=m)


@triton.jit
def _segment(A, B, ORDER, OFF, TE, TS, OUT, K: tl.constexpr, NOUT: tl.constexpr,
             TRANS: tl.constexpr, GATHER: tl.constexpr, OUTQ: tl.constexpr,
             SPLIT: tl.constexpr, BF: tl.constexpr, R: tl.constexpr,
             C: tl.constexpr, KBLK: tl.constexpr):
    p = tl.program_id(0)
    e = tl.load(TE + p)
    start = tl.load(TS + p, mask=e >= 0, other=0)
    ir = tl.arange(0, R)
    ic = tl.program_id(1) * C + tl.arange(0, C)
    pos = start + ir
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    q = tl.load(ORDER + pos, mask=(e >= 0) & (pos < end), other=0)
    z = tl.zeros((R, C), tl.float32)
    for k in range(0, K, KBLK):
        ik = k + tl.arange(0, KBLK)
        row = q if GATHER else pos
        a = tl.load(A + row[:, None] * K + ik[None, :], mask=(pos[:, None] < end) & (ik[None, :] < K), other=0.0)
        if TRANS:
            b = tl.load(B + e * NOUT * K + ic[None, :] * K + ik[:, None], mask=(e >= 0) & (ic[None, :] < NOUT) & (ik[:, None] < K), other=0.0)
        else:
            b = tl.load(B + e * K * NOUT + ik[:, None] * NOUT + ic[None, :], mask=(e >= 0) & (ik[:, None] < K) & (ic[None, :] < NOUT), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            z += tl.dot(ah, b)
            if SPLIT:
                z += tl.dot((a - ah).to(tl.bfloat16), b)
        else:
            z += tl.dot(a, b.to(tl.float32), input_precision="tf32x3")
    outrow = q if OUTQ else pos
    tl.store(OUT + outrow[:, None] * NOUT + ic[None, :], z, mask=(e >= 0) & (pos[:, None] < end) & (ic[None, :] < NOUT))


@triton.jit
def _combine(Z, W, Y, D: tl.constexpr, A: tl.constexpr, BD: tl.constexpr = 256):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    z = tl.zeros((BD,), tl.float32)
    for a in range(0, A):
        z += tl.load(W + t * A + a) * tl.load(Z + (t * A + a) * D + d, mask=d < D, other=0.0)
    tl.store(Y + t * D + d, z, mask=d < D)


@triton.jit
def _dz(DY, W, Z, DZ, PD, D: tl.constexpr, A: tl.constexpr, BD: tl.constexpr = 256):
    q = tl.program_id(0)
    d = tl.arange(0, BD)
    t = q // A
    dot = tl.zeros((BD,), tl.float32)
    ww = tl.load(W + q)
    for k in range(0, D, BD):
        j = k + d
        dy = tl.load(DY + t * D + j, mask=j < D, other=0.0).to(tl.float32)
        zz = tl.load(Z + q * D + j, mask=j < D, other=0.0)
        dot += dy * zz
        tl.store(DZ + q * D + j, dy * ww, mask=j < D)
    tl.store(PD + q, tl.sum(dot, axis=0))


@triton.jit
def _dgu(GU, DA, DGU, n: tl.constexpr, F: tl.constexpr, BS: tl.constexpr = 256):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    m = q < n
    r = q // F
    c = q % F
    g = tl.load(GU + r * (2 * F) + c, mask=m, other=0.0)
    u = tl.load(GU + r * (2 * F) + F + c, mask=m, other=0.0)
    da = tl.load(DA + q, mask=m, other=0.0)
    s = tl.sigmoid(g)
    silu = g * s
    tl.store(DGU + r * (2 * F) + c, da * u * (s + silu * (1.0 - s)), mask=m)
    tl.store(DGU + r * (2 * F) + F + c, da * silu, mask=m)


@triton.jit
def _dw1(X, G, ORDER, OFF, OUT, D: tl.constexpr, F2: tl.constexpr, ASEL: tl.constexpr,
         BF: tl.constexpr, BM: tl.constexpr = 128, BN: tl.constexpr = 64, BK: tl.constexpr = 32):
    e = tl.program_id(0)
    im = tl.program_id(1) * BM + tl.arange(0, BM)
    jn = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    z = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        ik = k + tl.arange(0, BK)
        q = tl.load(ORDER + ik, mask=ik < end, other=0)
        tok = q // ASEL
        x = tl.load(X + im[:, None] + tok[None, :] * D, mask=(im[:, None] < D) & (ik[None, :] < end), other=0.0)
        g = tl.load(G + ik[:, None] * F2 + jn[None, :], mask=(ik[:, None] < end) & (jn[None, :] < F2), other=0.0)
        if BF:
            xh = x.to(tl.bfloat16)
            xl = (x - xh).to(tl.bfloat16)
            gh = g.to(tl.bfloat16)
            gl = (g - gh).to(tl.bfloat16)
            z += tl.dot(xh, gh) + tl.dot(xh, gl) + tl.dot(xl, gh)
        else:
            z += tl.dot(x.to(tl.float32), g, input_precision="tf32x3")
    tl.store(OUT + e * D * F2 + im[:, None] * F2 + jn[None, :], z, mask=(im[:, None] < D) & (jn[None, :] < F2))


@triton.jit
def _dw2(ACT, DZ, ORDER, OFF, OUT, F: tl.constexpr, D: tl.constexpr,
         BF: tl.constexpr, BM: tl.constexpr = 128, BN: tl.constexpr = 64, BK: tl.constexpr = 32):
    e = tl.program_id(0)
    im = tl.program_id(1) * BM + tl.arange(0, BM)
    jn = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    z = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        ik = k + tl.arange(0, BK)
        q = tl.load(ORDER + ik, mask=ik < end, other=0)
        a = tl.load(ACT + im[:, None] + ik[None, :] * F, mask=(im[:, None] < F) & (ik[None, :] < end), other=0.0)
        b = tl.load(DZ + q[:, None] * D + jn[None, :], mask=(ik[:, None] < end) & (jn[None, :] < D), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            al = (a - ah).to(tl.bfloat16)
            bh = b.to(tl.bfloat16)
            bl = (b - bh).to(tl.bfloat16)
            z += tl.dot(ah, bh) + tl.dot(ah, bl) + tl.dot(al, bh)
        else:
            z += tl.dot(a, b, input_precision="tf32x3")
    tl.store(OUT + e * F * D + im[:, None] * D + jn[None, :], z, mask=(im[:, None] < F) & (jn[None, :] < D))


@triton.jit
def _route_ds(S, SEL, W, PD, DS, E: tl.constexpr, A: tl.constexpr, BA: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BA)
    m = a < A
    e = tl.load(SEL + t * A + a, mask=m, other=0)
    s = tl.load(S + t * E + e, mask=m, other=0.0)
    w = tl.load(W + t * A + a, mask=m, other=0.0)
    p = tl.load(PD + t * A + a, mask=m, other=0.0)
    tl.store(DS + t * A + a, (p - tl.sum(w * p, axis=0)) / tl.sum(s, axis=0) * s * (1.0 - s), mask=m)


@triton.jit
def _drw(X, DS, SEL, OUT, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
         A: tl.constexpr, BF: tl.constexpr, BM: tl.constexpr = 32,
         BN: tl.constexpr = 32, BK: tl.constexpr = 32):
    im = tl.program_id(0) * BM + tl.arange(0, BM)
    je = tl.program_id(1) * BN + tl.arange(0, BN)
    z = tl.zeros((BM, BN), tl.float32)
    for k in range(0, T, BK):
        it = k + tl.arange(0, BK)
        x = tl.load(X + im[:, None] + it[None, :] * D, mask=(im[:, None] < D) & (it[None, :] < T), other=0.0)
        v = tl.zeros((BK, BN), tl.float32)
        for a in range(0, A):
            ee = tl.load(SEL + it * A + a, mask=it < T, other=-1)
            ds = tl.load(DS + it * A + a, mask=it < T, other=0.0)
            v += tl.where(ee[:, None] == je[None, :], ds[:, None], 0.0)
        if BF:
            vh = v.to(tl.bfloat16)
            vl = (v - vh).to(tl.bfloat16)
            z += tl.dot(x, vh) + tl.dot(x, vl)
        else:
            z += tl.dot(x.to(tl.float32), v, input_precision="ieee")
    tl.store(OUT + im[:, None] * E + je[None, :], z, mask=(im[:, None] < D) & (je[None, :] < E))


@triton.jit
def _dx_combine(DXE, DS, SEL, RW, OUT, D: tl.constexpr, E: tl.constexpr,
                A: tl.constexpr, BD: tl.constexpr = 256):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    z = tl.zeros((BD,), tl.float32)
    for a in range(0, A):
        z += tl.load(DXE + (t * A + a) * D + d, mask=d < D, other=0.0)
        e = tl.load(SEL + t * A + a)
        z += tl.load(DS + t * A + a) * tl.load(RW + d * E + e, mask=d < D, other=0.0)
    tl.store(OUT + t * D + d, z, mask=d < D)


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
        _router[(triton.cdiv(T, 32), triton.cdiv(E, 32))](x, rw, s, T=T, D=D, E=E)
        keys = torch.empty((T, E), device=x.device, dtype=torch.int64)
        _keys[(triton.cdiv(T * E, 256),)](s, rb, keys, n=T * E, E=E)
        sel = torch.topk(keys, A, dim=1, largest=True, sorted=True).indices
        weight = torch.empty((T, A), device=x.device, dtype=torch.float32)
        ba = triton.next_power_of_2(A)
        _weights[(T,)](s, sel, weight, E=E, A=A, BA=ba)
        flat = sel.reshape(-1)
        order = torch.argsort(flat)
        sorted_e = flat.index_select(0, order)
        off = torch.empty(E + 1, device=x.device, dtype=torch.int32)
        _offsets[(E + 1,)](sorted_e, off, N=N, E=E)
        tp = torch.empty(E + 1, device=x.device, dtype=torch.int32)
        _prefix[(1,)](off, tp, E=E, R=ROW, num_warps=1)
        cap = triton.cdiv(N, ROW) + E
        te = torch.empty(cap, device=x.device, dtype=torch.int32)
        ts = torch.empty(cap, device=x.device, dtype=torch.int32)
        _tile_map[(cap,)](tp, off, te, ts, E=E, R=ROW)
        gu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _expert_in[(cap, triton.cdiv(2 * F, COL))](x, w1, order, off, te, ts, gu, D=D, F2=2 * F, A=A, BF=bf, R=ROW, C=COL, KBLK=KB, num_warps=4, num_stages=3)
        act = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _activate[(triton.cdiv(N * F, 256),)](gu, act, N=N, F=F)
        z = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _segment[(cap, triton.cdiv(D, COL))](act, w2, order, off, te, ts, z, K=F, NOUT=D, TRANS=False, GATHER=False, OUTQ=True, SPLIT=True, BF=bf, R=ROW, C=COL, KBLK=KB, num_warps=4, num_stages=3)
        y = torch.empty((T, D), device=x.device, dtype=x.dtype)
        _combine[(T, triton.cdiv(D, 256))](z, weight, y, D=D, A=A)
        ctx.save_for_backward(x, rw, w1, w2, s, sel, weight, order, off, te, ts, gu, act, z)
        ctx.meta = (A, cap, bf)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, rw, w1, w2, s, sel, weight, order, off, te, ts, gu, act, z = ctx.saved_tensors
        A, cap, bf = ctx.meta
        T, D = x.shape
        E = rw.shape[1]
        F = w2.shape[1]
        N = T * A
        dz = torch.empty((N, D), device=x.device, dtype=torch.float32)
        pd = torch.empty(N, device=x.device, dtype=torch.float32)
        _dz[(N,)](dy, weight, z, dz, pd, D=D, A=A)
        da = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _segment[(cap, triton.cdiv(F, COL))](dz, w2, order, off, te, ts, da, K=D, NOUT=F, TRANS=True, GATHER=True, OUTQ=False, SPLIT=True, BF=bf, R=ROW, C=COL, KBLK=KB, num_warps=4, num_stages=3)
        dgu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _dgu[(triton.cdiv(N * F, 256),)](gu, da, dgu, n=N * F, F=F)
        dxe = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _segment[(cap, triton.cdiv(D, COL))](dgu, w1, order, off, te, ts, dxe, K=2 * F, NOUT=D, TRANS=True, GATHER=False, OUTQ=True, SPLIT=False, BF=bf, R=ROW, C=COL, KBLK=KB, num_warps=4, num_stages=3)
        gw1 = torch.empty_like(w1)
        _dw1[(E, triton.cdiv(D, 128), triton.cdiv(2 * F, 64))](x, dgu, order, off, gw1, D=D, F2=2 * F, ASEL=A, BF=bf, num_warps=4, num_stages=3)
        gw2 = torch.empty_like(w2)
        _dw2[(E, triton.cdiv(F, 128), triton.cdiv(D, 64))](act, dz, order, off, gw2, F=F, D=D, BF=bf, num_warps=4, num_stages=3)
        ds = torch.empty((T, A), device=x.device, dtype=torch.float32)
        _route_ds[(T,)](s, sel, weight, pd, ds, E=E, A=A, BA=triton.next_power_of_2(A))
        grw = torch.empty_like(rw)
        _drw[(triton.cdiv(D, 32), triton.cdiv(E, 32))](x, ds, sel, grw, T=T, D=D, E=E, A=A, BF=bf, num_warps=4, num_stages=3)
        gx = torch.empty_like(x)
        _dx_combine[(T, triton.cdiv(D, 256))](dxe, ds, sel, rw, gx, D=D, E=E, A=A)
        return gx, grw, None, gw1, gw2, None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
