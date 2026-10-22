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
    for k in range(0, D, BK):
        ik = k + tl.arange(0, BK)
        a = tl.load(X + im[:, None] * D + ik[None, :],
                    mask=(im[:, None] < T) & (ik[None, :] < D), other=0.0)
        b = tl.load(RW + ik[:, None] * E + ie[None, :],
                    mask=(ik[:, None] < D) & (ie[None, :] < E), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(S + im[:, None] * E + ie[None, :], tl.sigmoid(acc),
             mask=(im[:, None] < T) & (ie[None, :] < E))


@triton.jit
def _offsets(SE, OFF, N: tl.constexpr, E: tl.constexpr):
    e = tl.program_id(0)
    lo = 0
    hi = N
    while lo < hi:
        mid = (lo + hi) // 2
        v = tl.load(SE + mid)
        go = v < e
        lo = tl.where(go, mid + 1, lo)
        hi = tl.where(go, hi, mid)
    tl.store(OFF + e, lo)


@triton.jit
def _inverse(ORDER, INVERSE, N: tl.constexpr, BLOCK: tl.constexpr):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    p = tl.load(ORDER + q, mask=q < N, other=0)
    tl.store(INVERSE + p, q, mask=q < N)


@triton.jit
def _tiles(OFF, TE, TS, E: tl.constexpr, CAP: tl.constexpr,
           BR: tl.constexpr):
    for j in range(0, CAP):
        tl.store(TE + j, -1)
    out = 0
    for e in range(0, E):
        begin = tl.load(OFF + e)
        end = tl.load(OFF + e + 1)
        p = begin
        while p < end:
            tl.store(TE + out, e)
            tl.store(TS + out, p)
            out += 1
            p += BR


@triton.jit
def _expert_in(X, W, ORDER, OFF, TE, TS, GU, T: tl.constexpr, D: tl.constexpr,
               N: tl.constexpr, O: tl.constexpr, A: tl.constexpr,
               BR: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
               BF: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TE + tile)
    q = tl.load(TS + tile)
    rm = q + tl.arange(0, BR)
    rn = tl.program_id(1) * BC + tl.arange(0, BC)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    valid = (e >= 0) & (rm < end)
    pair = tl.load(ORDER + rm, mask=valid, other=0)
    tok = pair // A
    acc = tl.zeros((BR, BC), tl.float32)
    for k in range(0, D, BK):
        rk = k + tl.arange(0, BK)
        x = tl.load(X + tok[:, None] * D + rk[None, :],
                    mask=valid[:, None] & (rk[None, :] < D), other=0.0)
        w = tl.load(W + e * D * O + rk[:, None] * O + rn[None, :],
                    mask=(e >= 0) & (rk[:, None] < D) & (rn[None, :] < O),
                    other=0.0)
        if BF:
            acc += tl.dot(x, w)
        else:
            acc += tl.dot(x, w, input_precision="tf32x3")
    tl.store(GU + rm[:, None] * O + rn[None, :], acc,
             mask=valid[:, None] & (rn[None, :] < O))


@triton.jit
def _activate(GU, ACT, N: tl.constexpr, F: tl.constexpr,
              BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N * F
    q = i // F
    f = i % F
    g = tl.load(GU + q * (2 * F) + f, mask=m, other=0.0)
    u = tl.load(GU + q * (2 * F) + F + f, mask=m, other=0.0)
    tl.store(ACT + i, g * tl.sigmoid(g) * u, mask=m)


@triton.jit
def _expert_out(ACT, W, OFF, TE, TS, Z, T: tl.constexpr, F: tl.constexpr,
                D: tl.constexpr, N: tl.constexpr, BR: tl.constexpr,
                BC: tl.constexpr, BK: tl.constexpr, BF: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TE + tile)
    rm = tl.load(TS + tile) + tl.arange(0, BR)
    rn = tl.program_id(1) * BC + tl.arange(0, BC)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    valid = (e >= 0) & (rm < end)
    acc = tl.zeros((BR, BC), tl.float32)
    for k in range(0, F, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(ACT + rm[:, None] * F + rk[None, :],
                    mask=valid[:, None] & (rk[None, :] < F), other=0.0)
        w = tl.load(W + e * F * D + rk[:, None] * D + rn[None, :],
                    mask=(e >= 0) & (rk[:, None] < F) & (rn[None, :] < D),
                    other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            al = (a - ah).to(tl.bfloat16)
            acc += tl.dot(ah, w) + tl.dot(al, w)
        else:
            acc += tl.dot(a, w, input_precision="tf32x3")
    tl.store(Z + rm[:, None] * D + rn[None, :], acc,
             mask=valid[:, None] & (rn[None, :] < D))


@triton.jit
def _combine(S, SEL, ORDER, Z, WGT, Y, T: tl.constexpr, D: tl.constexpr,
             E: tl.constexpr, A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    den = 0.0
    for a in range(0, A):
        e = tl.load(SEL + t * A + a)
        den += tl.load(S + t * E + e)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(0, A):
        p = t * A + a
        e = tl.load(SEL + p)
        w = tl.load(S + t * E + e) / den
        if tl.program_id(1) == 0:
            tl.store(WGT + p, w)
        q = tl.load(ORDER + p)
        z = tl.load(Z + q * D + d, mask=d < D, other=0.0)
        acc += w * z
    tl.store(Y + t * D + d, acc, mask=d < D)


@triton.jit
def _dz_route(DY, Z, WGT, ORDER, DZ, C, N: tl.constexpr, D: tl.constexpr,
              A: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr):
    q = tl.program_id(0) * BR + tl.arange(0, BR)
    pair = tl.load(ORDER + q, mask=q < N, other=0)
    tok = pair // A
    w = tl.load(WGT + pair, mask=q < N, other=0.0)
    dot = tl.zeros((BR,), tl.float32)
    for k in range(0, D, BD):
        d = k + tl.arange(0, BD)
        dy = tl.load(DY + tok[:, None] * D + d[None, :],
                     mask=(q[:, None] < N) & (d[None, :] < D), other=0.0)
        z = tl.load(Z + q[:, None] * D + d[None, :],
                    mask=(q[:, None] < N) & (d[None, :] < D), other=0.0)
        tl.store(DZ + q[:, None] * D + d[None, :], dy * w[:, None],
                 mask=(q[:, None] < N) & (d[None, :] < D))
        dot += tl.sum(dy * z, axis=1)
    tl.store(C + pair, dot, mask=q < N)


@triton.jit
def _expert_da(DZ, W, GU, OFF, TE, TS, DGU, F: tl.constexpr,
               D: tl.constexpr, N: tl.constexpr, BR: tl.constexpr,
               BC: tl.constexpr, BK: tl.constexpr, BF: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TE + tile)
    rm = tl.load(TS + tile) + tl.arange(0, BR)
    rf = tl.program_id(1) * BC + tl.arange(0, BC)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    valid = (e >= 0) & (rm < end)
    acc = tl.zeros((BR, BC), tl.float32)
    for k in range(0, D, BK):
        rd = k + tl.arange(0, BK)
        dz = tl.load(DZ + rm[:, None] * D + rd[None, :],
                     mask=valid[:, None] & (rd[None, :] < D), other=0.0)
        w = tl.load(W + e * F * D + rf[None, :] * D + rd[:, None],
                    mask=(e >= 0) & (rf[None, :] < F) & (rd[:, None] < D),
                    other=0.0)
        if BF:
            acc += tl.dot(dz.to(tl.bfloat16), w)
        else:
            acc += tl.dot(dz, w, input_precision="tf32x3")
    mask = valid[:, None] & (rf[None, :] < F)
    g = tl.load(GU + rm[:, None] * (2 * F) + rf[None, :],
                mask=mask, other=0.0)
    u = tl.load(GU + rm[:, None] * (2 * F) + F + rf[None, :],
                mask=mask, other=0.0)
    sig = tl.sigmoid(g)
    silu = g * sig
    tl.store(DGU + rm[:, None] * (2 * F) + rf[None, :],
             acc * u * (sig + silu * (1.0 - sig)), mask=mask)
    tl.store(DGU + rm[:, None] * (2 * F) + F + rf[None, :],
             acc * silu, mask=mask)


@triton.jit
def _expert_dx(DGU, W, OFF, TE, TS, ORDER, DXP, D: tl.constexpr,
               O: tl.constexpr, N: tl.constexpr, A: tl.constexpr,
               BR: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, BF: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TE + tile)
    rm = tl.load(TS + tile) + tl.arange(0, BR)
    rd = tl.program_id(1) * BC + tl.arange(0, BC)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    valid = (e >= 0) & (rm < end)
    acc = tl.zeros((BR, BC), tl.float32)
    for k in range(0, O, BK):
        ro = k + tl.arange(0, BK)
        dg = tl.load(DGU + rm[:, None] * O + ro[None, :],
                     mask=valid[:, None] & (ro[None, :] < O), other=0.0)
        w = tl.load(W + e * D * O + rd[None, :] * O + ro[:, None],
                    mask=(e >= 0) & (rd[None, :] < D) & (ro[:, None] < O),
                    other=0.0)
        if BF:
            acc += tl.dot(dg.to(tl.bfloat16), w)
        else:
            acc += tl.dot(dg, w, input_precision="tf32x3")
    pair = tl.load(ORDER + rm, mask=valid, other=0)
    tl.store(DXP + pair[:, None] * D + rd[None, :], acc,
             mask=valid[:, None] & (rd[None, :] < D))


@triton.jit
def _route_ds(S, SEL, WGT, C, DS, T: tl.constexpr, E: tl.constexpr,
              A: tl.constexpr):
    t = tl.program_id(0)
    den = 0.0
    mean = 0.0
    for a in range(0, A):
        p = t * A + a
        e = tl.load(SEL + p)
        den += tl.load(S + t * E + e)
        mean += tl.load(WGT + p) * tl.load(C + p)
    for a in range(0, A):
        p = t * A + a
        e = tl.load(SEL + p)
        s = tl.load(S + t * E + e)
        ds = (tl.load(C + p) - mean) / den
        tl.store(DS + p, ds * s * (1.0 - s))


@triton.jit
def _combine_dx(DXP, DS, SEL, RW, DX, T: tl.constexpr, D: tl.constexpr,
                E: tl.constexpr, A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(0, A):
        p = t * A + a
        e = tl.load(SEL + p)
        acc += tl.load(DXP + p * D + d, mask=d < D, other=0.0)
        acc += tl.load(DS + p) * tl.load(RW + d * E + e,
                                        mask=d < D, other=0.0)
    tl.store(DX + t * D + d, acc, mask=d < D)


@triton.jit
def _dw2(ACT, DZ, OFF, DW, F: tl.constexpr, D: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
         BF: tl.constexpr):
    e = tl.program_id(0)
    rf = tl.program_id(1) * BM + tl.arange(0, BM)
    rd = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        q = k + tl.arange(0, BK)
        a = tl.load(ACT + q[None, :] * F + rf[:, None],
                    mask=(q[None, :] < end) & (rf[:, None] < F), other=0.0)
        b = tl.load(DZ + q[:, None] * D + rd[None, :],
                    mask=(q[:, None] < end) & (rd[None, :] < D), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            bh = b.to(tl.bfloat16)
            al = (a - ah).to(tl.bfloat16)
            bl = (b - bh).to(tl.bfloat16)
            acc += tl.dot(ah, bh) + tl.dot(al, bh) + tl.dot(ah, bl)
        else:
            acc += tl.dot(a, b, input_precision="tf32x3")
    tl.store(DW + e * F * D + rf[:, None] * D + rd[None, :], acc,
             mask=(rf[:, None] < F) & (rd[None, :] < D))


@triton.jit
def _dw1(X, DGU, ORDER, OFF, DW, D: tl.constexpr, O: tl.constexpr,
         A: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         BK: tl.constexpr, BF: tl.constexpr):
    e = tl.program_id(0)
    rd = tl.program_id(1) * BM + tl.arange(0, BM)
    ro = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        q = k + tl.arange(0, BK)
        pair = tl.load(ORDER + q, mask=q < end, other=0)
        tok = pair // A
        x = tl.load(X + rd[:, None] + tok[None, :] * D,
                    mask=(rd[:, None] < D) & (q[None, :] < end), other=0.0)
        dg = tl.load(DGU + q[:, None] * O + ro[None, :],
                     mask=(q[:, None] < end) & (ro[None, :] < O), other=0.0)
        if BF:
            gh = dg.to(tl.bfloat16)
            gl = (dg - gh).to(tl.bfloat16)
            acc += tl.dot(x, gh) + tl.dot(x, gl)
        else:
            acc += tl.dot(x, dg, input_precision="tf32x3")
    tl.store(DW + e * D * O + rd[:, None] * O + ro[None, :], acc,
             mask=(rd[:, None] < D) & (ro[None, :] < O))


@triton.jit
def _drw(X, DS, SEL, DRW, T: tl.constexpr, D: tl.constexpr,
         E: tl.constexpr, A: tl.constexpr, BM: tl.constexpr,
         BN: tl.constexpr, BK: tl.constexpr):
    rd = tl.program_id(0) * BM + tl.arange(0, BM)
    re = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, T, BK):
        rt = k + tl.arange(0, BK)
        x = tl.load(X + rd[:, None] + rt[None, :] * D,
                    mask=(rd[:, None] < D) & (rt[None, :] < T),
                    other=0.0).to(tl.float32)
        dense = tl.zeros((BK, BN), tl.float32)
        for a in range(0, A):
            se = tl.load(SEL + rt * A + a, mask=rt < T, other=-1)
            v = tl.load(DS + rt * A + a, mask=rt < T, other=0.0)
            dense += tl.where(se[:, None] == re[None, :], v[:, None], 0.0)
        acc += tl.dot(x, dense, input_precision="ieee")
    tl.store(DRW + rd[:, None] * E + re[None, :], acc,
             mask=(rd[:, None] < D) & (re[None, :] < E))


def _select(s, bias, a):
    e = s.shape[1]
    b = s + bias
    b = torch.where(b == 0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    idx = torch.arange(e, device=s.device, dtype=torch.int64)
    return torch.topk((u << 10) | (e - 1 - idx), a, dim=1,
                      largest=True, sorted=True).indices


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, a):
        a = int(a)
        bf = x.dtype == torch.bfloat16
        xf, rw = x.float(), router_w.float()
        xe = x if bf else xf
        w1e = w1 if bf else w1.float()
        w2e = w2 if bf else w2.float()
        t, d = x.shape
        e = router_w.shape[1]
        f = w2.shape[1]
        n = t * a
        s = torch.empty((t, e), device=x.device, dtype=torch.float32)
        _router[(triton.cdiv(t, 32), triton.cdiv(e, 32))](
            xf, rw, s, T=t, D=d, E=e, BM=32, BN=32, BK=32)
        with torch.no_grad():
            sel = _select(s, router_b, a)
            flat = sel.reshape(-1)
            order = torch.argsort(flat, stable=True)
            se = flat.index_select(0, order).contiguous()
        inverse = torch.empty_like(order)
        _inverse[(triton.cdiv(n, 256),)](order, inverse, N=n, BLOCK=256)
        off = torch.empty(e + 1, device=x.device, dtype=torch.int32)
        _offsets[(e + 1,)](se, off, N=n, E=e + 1, num_warps=1)
        cap = triton.cdiv(n, ROW) + e
        te = torch.empty(cap, device=x.device, dtype=torch.int32)
        ts = torch.empty_like(te)
        _tiles[(1,)](off, te, ts, E=e, CAP=cap, BR=ROW, num_warps=1)
        gu = torch.empty((n, 2 * f), device=x.device, dtype=torch.float32)
        act = torch.empty((n, f), device=x.device, dtype=torch.float32)
        z = torch.empty((n, d), device=x.device, dtype=torch.float32)
        _expert_in[(cap, triton.cdiv(2 * f, COL))](
            xe, w1e, order, off, te, ts, gu, T=t, D=d, N=n, O=2 * f, A=a,
            BR=ROW, BC=COL, BK=KB, BF=bf, num_warps=4, num_stages=3)
        _activate[(triton.cdiv(n * f, 256),)](gu, act, N=n, F=f, BLOCK=256)
        _expert_out[(cap, triton.cdiv(d, COL))](
            act, w2e, off, te, ts, z, T=t, F=f, D=d, N=n, BR=ROW, BC=COL,
            BK=KB, BF=bf, num_warps=4, num_stages=3)
        wgt = torch.empty(n, device=x.device, dtype=torch.float32)
        y = torch.empty((t, d), device=x.device, dtype=torch.float32)
        _combine[(t, triton.cdiv(d, 256))](
            s, sel, inverse, z, wgt, y, T=t, D=d, E=e, A=a, BD=256)
        ctx.save_for_backward(xe, rw, w1e, w2e, s, sel, order, off, te, ts,
                              gu, act, z, wgt)
        ctx.shape = (t, d, e, f, a, n, cap, bf)
        ctx.dtypes = (x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        (x, rw, w1, w2, s, sel, order, off, te, ts, gu, act, z,
         wgt) = ctx.saved_tensors
        t, d, e, f, a, n, cap, bf = ctx.shape
        dy = dy.float().contiguous()
        dz = torch.empty((n, d), device=dy.device, dtype=torch.float32)
        c = torch.empty(n, device=dy.device, dtype=torch.float32)
        _dz_route[(triton.cdiv(n, ROW),)](
            dy, z, wgt, order, dz, c, N=n, D=d, A=a, BR=ROW, BD=64)
        dgu = torch.empty((n, 2 * f), device=dy.device, dtype=torch.float32)
        _expert_da[(cap, triton.cdiv(f, COL))](
            dz, w2, gu, off, te, ts, dgu, F=f, D=d, N=n, BR=ROW,
            BC=COL, BK=KB, BF=bf, num_warps=4, num_stages=3)
        dxp = torch.empty((n, d), device=dy.device, dtype=torch.float32)
        _expert_dx[(cap, triton.cdiv(d, COL))](
            dgu, w1, off, te, ts, order, dxp, D=d, O=2 * f, N=n, A=a,
            BR=ROW, BC=COL, BK=KB, BF=bf, num_warps=4, num_stages=3)
        ds = torch.empty(n, device=dy.device, dtype=torch.float32)
        _route_ds[(t,)](s, sel, wgt, c, ds, T=t, E=e, A=a)
        dx = torch.empty((t, d), device=dy.device, dtype=torch.float32)
        _combine_dx[(t, triton.cdiv(d, 256))](
            dxp, ds, sel, rw, dx, T=t, D=d, E=e, A=a, BD=256)
        dw2 = torch.empty_like(w2)
        _dw2[(e, triton.cdiv(f, 128), triton.cdiv(d, 64))](
            act, dz, off, dw2, F=f, D=d, BM=128, BN=64, BK=32,
            BF=bf, num_warps=4, num_stages=3)
        dw1 = torch.empty_like(w1)
        _dw1[(e, triton.cdiv(d, 128), triton.cdiv(2 * f, 64))](
            x, dgu, order, off, dw1, D=d, O=2 * f, A=a, BM=128, BN=64,
            BK=32, BF=bf, num_warps=4, num_stages=3)
        drw = torch.empty_like(rw)
        _drw[(triton.cdiv(d, 32), triton.cdiv(e, 32))](
            x, ds, sel, drw, T=t, D=d, E=e, A=a, BM=32, BN=32, BK=32,
            num_warps=4, num_stages=3)
        tx, tr, t1, t2 = ctx.dtypes
        return dx.to(tx), drw.to(tr), None, dw1.to(t1), dw2.to(t2), None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
