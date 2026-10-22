import torch
import triton
import triton.language as tl

ROW = 64
KBLK = 32


@triton.jit
def _dot(a, b, BF: tl.constexpr, SPLIT_A: tl.constexpr, SPLIT_B: tl.constexpr):
    if not BF:
        return tl.dot(a, b, input_precision="ieee")
    ah = a.to(tl.bfloat16)
    bh = b.to(tl.bfloat16)
    out = tl.dot(ah, bh)
    if SPLIT_A:
        al = (a - ah).to(tl.bfloat16)
        out += tl.dot(al, bh)
    if SPLIT_B:
        bl = (b - bh).to(tl.bfloat16)
        out += tl.dot(ah, bl)
    return out


@triton.jit
def _router(X, RW, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    im = tl.program_id(0) * BM + tl.arange(0, BM)
    ie = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        ik = k + tl.arange(0, BK)
        x = tl.load(X + im[:, None] * D + ik[None, :],
                    mask=(im[:, None] < T) & (ik[None, :] < D), other=0.0).to(tl.float32)
        w = tl.load(RW + ik[:, None] * E + ie[None, :],
                    mask=(ik[:, None] < D) & (ie[None, :] < E), other=0.0).to(tl.float32)
        acc += tl.dot(x, w, input_precision="ieee")
    tl.store(S + im[:, None] * E + ie[None, :], tl.sigmoid(acc),
             mask=(im[:, None] < T) & (ie[None, :] < E))


@triton.jit
def _pair_weights(S, SEL, W, T: tl.constexpr, E: tl.constexpr,
                  A: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BLOCK)
    m = a < A
    e = tl.load(SEL + t * A + a, mask=m, other=0)
    v = tl.load(S + t * E + e, mask=m, other=0.0)
    z = tl.sum(v, axis=0)
    tl.store(W + t * A + a, v / z, mask=m)


@triton.jit
def _offsets(SEL, ORDER, OFF, N: tl.constexpr, E: tl.constexpr):
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
    if e == E:
        tl.store(OFF + E, N)


@triton.jit
def _fill_tiles(TILE_E, CAP: tl.constexpr, BLOCK: tl.constexpr):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(TILE_E + q, -1, mask=q < CAP)


@triton.jit
def _tiles(OFF, TILE_E, TILE_R, E: tl.constexpr, BR: tl.constexpr):
    e = 0
    out = 0
    while e < E:
        begin = tl.load(OFF + e)
        end = tl.load(OFF + e + 1)
        row = begin
        while row < end:
            tl.store(TILE_E + out, e)
            tl.store(TILE_R + out, row)
            out += 1
            row += BR
        e += 1


@triton.jit
def _expert_in(X, W1, ORDER, OFF, TILE_E, TILE_R, GU,
               N: tl.constexpr, D: tl.constexpr, F2: tl.constexpr,
               A: tl.constexpr, CAP: tl.constexpr, BF: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile)
    base = tl.load(TILE_R + tile)
    ir = base + tl.arange(0, BM)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    valid = (tile < CAP) & (e >= 0) & (ir < end)
    pair = tl.load(ORDER + ir, mask=valid, other=0)
    tok = pair // A
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        ik = k + tl.arange(0, BK)
        x = tl.load(X + tok[:, None] * D + ik[None, :],
                    mask=valid[:, None] & (ik[None, :] < D), other=0.0)
        w = tl.load(W1 + e * D * F2 + ik[:, None] * F2 + j[None, :],
                    mask=(e >= 0) & (ik[:, None] < D) & (j[None, :] < F2), other=0.0)
        acc += _dot(x, w, BF, False, False)
    tl.store(GU + ir[:, None] * F2 + j[None, :], acc,
             mask=valid[:, None] & (j[None, :] < F2))


@triton.jit
def _activate(GU, ACT, N: tl.constexpr, F: tl.constexpr,
              BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = p < N * F
    r = p // F
    f = p % F
    g = tl.load(GU + r * (2 * F) + f, mask=m, other=0.0)
    u = tl.load(GU + r * (2 * F) + F + f, mask=m, other=0.0)
    tl.store(ACT + p, g * tl.sigmoid(g) * u, mask=m)


@triton.jit
def _expert_out(ACT, W2, OFF, TILE_E, TILE_R, YP,
                N: tl.constexpr, F: tl.constexpr, D: tl.constexpr,
                CAP: tl.constexpr, BF: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile)
    base = tl.load(TILE_R + tile)
    ir = base + tl.arange(0, BM)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    valid = (tile < CAP) & (e >= 0) & (ir < end)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, F, BK):
        ik = k + tl.arange(0, BK)
        a = tl.load(ACT + ir[:, None] * F + ik[None, :],
                    mask=valid[:, None] & (ik[None, :] < F), other=0.0)
        w = tl.load(W2 + e * F * D + ik[:, None] * D + j[None, :],
                    mask=(e >= 0) & (ik[:, None] < F) & (j[None, :] < D), other=0.0)
        acc += _dot(a, w, BF, True, False)
    tl.store(YP + ir[:, None] * D + j[None, :], acc,
             mask=valid[:, None] & (j[None, :] < D))


@triton.jit
def _combine(YP, W, INV, Y, T: tl.constexpr, D: tl.constexpr,
             A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    j = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(A):
        p = t * A + a
        r = tl.load(INV + p)
        v = tl.load(YP + r * D + j, mask=j < D, other=0.0)
        w = tl.load(W + p)
        acc += w * v
    tl.store(Y + t * D + j, acc, mask=j < D)


@triton.jit
def _inverse(ORDER, INV, N: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = r < N
    p = tl.load(ORDER + r, mask=m, other=0)
    tl.store(INV + p, r, mask=m)


@triton.jit
def _dz(DY, W, YP, INV, DYP, DW, T: tl.constexpr, D: tl.constexpr,
        A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    a = tl.program_id(1)
    p = t * A + a
    r = tl.load(INV + p)
    j = tl.arange(0, BD)
    total = tl.zeros((), tl.float32)
    for k in range(0, D, BD):
        jj = k + j
        dy = tl.load(DY + t * D + jj, mask=jj < D, other=0.0)
        yp = tl.load(YP + r * D + jj, mask=jj < D, other=0.0)
        total += tl.sum(dy * yp, axis=0)
        tl.store(DYP + r * D + jj, dy * tl.load(W + p), mask=jj < D)
    tl.store(DW + p, total)


@triton.jit
def _dact(DYP, W2, OFF, TILE_E, TILE_R, DA,
          N: tl.constexpr, F: tl.constexpr, D: tl.constexpr,
          CAP: tl.constexpr, BF: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile)
    base = tl.load(TILE_R + tile)
    ir = base + tl.arange(0, BM)
    f = tl.program_id(1) * BN + tl.arange(0, BN)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    valid = (tile < CAP) & (e >= 0) & (ir < end)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        j = k + tl.arange(0, BK)
        dy = tl.load(DYP + ir[:, None] * D + j[None, :],
                     mask=valid[:, None] & (j[None, :] < D), other=0.0)
        w = tl.load(W2 + e * F * D + f[None, :] * D + j[:, None],
                    mask=(e >= 0) & (f[None, :] < F) & (j[:, None] < D), other=0.0)
        acc += _dot(dy, w, BF, True, False)
    tl.store(DA + ir[:, None] * F + f[None, :], acc,
             mask=valid[:, None] & (f[None, :] < F))


@triton.jit
def _dgu(GU, DA, DGU, N: tl.constexpr, F: tl.constexpr,
         BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = p < N * F
    r = p // F
    f = p % F
    g = tl.load(GU + r * (2 * F) + f, mask=m, other=0.0)
    u = tl.load(GU + r * (2 * F) + F + f, mask=m, other=0.0)
    da = tl.load(DA + p, mask=m, other=0.0)
    s = tl.sigmoid(g)
    silu = g * s
    tl.store(DGU + r * (2 * F) + f, da * u * (s + silu * (1.0 - s)), mask=m)
    tl.store(DGU + r * (2 * F) + F + f, da * silu, mask=m)


@triton.jit
def _dx_rows(DGU, W1, OFF, TILE_E, TILE_R, DXR,
             N: tl.constexpr, D: tl.constexpr, F2: tl.constexpr,
             CAP: tl.constexpr, BF: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile)
    base = tl.load(TILE_R + tile)
    ir = base + tl.arange(0, BM)
    d = tl.program_id(1) * BN + tl.arange(0, BN)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    valid = (tile < CAP) & (e >= 0) & (ir < end)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, F2, BK):
        f = k + tl.arange(0, BK)
        g = tl.load(DGU + ir[:, None] * F2 + f[None, :],
                    mask=valid[:, None] & (f[None, :] < F2), other=0.0)
        w = tl.load(W1 + e * D * F2 + d[None, :] * F2 + f[:, None],
                    mask=(e >= 0) & (d[None, :] < D) & (f[:, None] < F2), other=0.0)
        acc += _dot(g, w, BF, True, False)
    tl.store(DXR + ir[:, None] * D + d[None, :], acc,
             mask=valid[:, None] & (d[None, :] < D))


@triton.jit
def _dw2(ACT, DYP, OFF, OUT, E: tl.constexpr, F: tl.constexpr,
         D: tl.constexpr, BF: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(0)
    f = tl.program_id(1) * BM + tl.arange(0, BM)
    d = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for r0 in range(begin, end, BK):
        r = r0 + tl.arange(0, BK)
        a = tl.load(ACT + r[None, :] * F + f[:, None],
                    mask=(r[None, :] < end) & (f[:, None] < F), other=0.0)
        g = tl.load(DYP + r[:, None] * D + d[None, :],
                    mask=(r[:, None] < end) & (d[None, :] < D), other=0.0)
        acc += _dot(a, g, BF, True, True)
    tl.store(OUT + e * F * D + f[:, None] * D + d[None, :], acc,
             mask=(f[:, None] < F) & (d[None, :] < D))


@triton.jit
def _dw1(X, DGU, ORDER, OFF, OUT, E: tl.constexpr, D: tl.constexpr,
         F2: tl.constexpr, A: tl.constexpr, BF: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(0)
    d = tl.program_id(1) * BM + tl.arange(0, BM)
    f = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for r0 in range(begin, end, BK):
        r = r0 + tl.arange(0, BK)
        pair = tl.load(ORDER + r, mask=r < end, other=0)
        tok = pair // A
        x = tl.load(X + d[:, None] + tok[None, :] * D,
                    mask=(d[:, None] < D) & (r[None, :] < end), other=0.0)
        g = tl.load(DGU + r[:, None] * F2 + f[None, :],
                    mask=(r[:, None] < end) & (f[None, :] < F2), other=0.0)
        acc += _dot(x, g, BF, False, True)
    tl.store(OUT + e * D * F2 + d[:, None] * F2 + f[None, :], acc,
             mask=(d[:, None] < D) & (f[None, :] < F2))


@triton.jit
def _route_grad(S, SEL, W, DW, DZ, T: tl.constexpr, E: tl.constexpr,
                A: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BLOCK)
    m = a < A
    e = tl.load(SEL + t * A + a, mask=m, other=0)
    s = tl.load(S + t * E + e, mask=m, other=0.0)
    w = tl.load(W + t * A + a, mask=m, other=0.0)
    dw = tl.load(DW + t * A + a, mask=m, other=0.0)
    mean = tl.sum(w * dw, axis=0)
    den = tl.sum(s, axis=0)
    ds = (dw - mean) / den
    tl.store(DZ + t * A + a, ds * s * (1.0 - s), mask=m)


@triton.jit
def _drw(X, SEL, DZ, OUT, T: tl.constexpr, D: tl.constexpr,
         E: tl.constexpr, A: tl.constexpr, BF: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    d = tl.program_id(0) * BM + tl.arange(0, BM)
    e = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for t0 in range(0, T, BK):
        t = t0 + tl.arange(0, BK)
        x = tl.load(X + d[:, None] + t[None, :] * D,
                    mask=(d[:, None] < D) & (t[None, :] < T), other=0.0)
        z = tl.zeros((BK, BN), tl.float32)
        for a in range(A):
            se = tl.load(SEL + t * A + a, mask=t < T, other=-1)
            dz = tl.load(DZ + t * A + a, mask=t < T, other=0.0)
            z += tl.where(se[:, None] == e[None, :], dz[:, None], 0.0)
        acc += _dot(x, z, BF, False, True)
    tl.store(OUT + d[:, None] * E + e[None, :], acc,
             mask=(d[:, None] < D) & (e[None, :] < E))


@triton.jit
def _combine_dx(DXR, INV, SEL, DZ, RW, DX,
                T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(A):
        p = t * A + a
        r = tl.load(INV + p)
        e = tl.load(SEL + p)
        acc += tl.load(DXR + r * D + d, mask=d < D, other=0.0)
        z = tl.load(DZ + p)
        acc += z * tl.load(RW + d * E + e, mask=d < D, other=0.0)
    tl.store(DX + t * D + d, acc, mask=d < D)


def _top(s, bias, A):
    E = s.shape[1]
    b = s + bias
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    idx = torch.arange(E, device=s.device, dtype=torch.int64)
    return torch.topk((u << 10) | (E - 1 - idx), A, dim=1,
                      largest=True, sorted=True).indices


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rw, rb, w1, w2, A):
        A = int(A)
        T, D = x.shape
        E = rw.shape[1]
        F = w2.shape[1]
        F2 = 2 * F
        N = T * A
        cap = triton.cdiv(N, ROW) + E
        bf = x.dtype == torch.bfloat16
        s = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _router[(triton.cdiv(T, 32), triton.cdiv(E, 32))](x, rw, s,
            T=T, D=D, E=E, BM=32, BN=32, BK=32, num_warps=4, num_stages=3)
        with torch.no_grad():
            sel = _top(s, rb.float(), A).contiguous()
            flat = sel.reshape(-1)
            order = torch.argsort(flat)
        weights = torch.empty((T, A), device=x.device, dtype=torch.float32)
        _pair_weights[(T,)](s, sel, weights, T=T, E=E, A=A,
                            BLOCK=triton.next_power_of_2(A), num_warps=1)
        off = torch.empty(E + 1, device=x.device, dtype=torch.int32)
        _offsets[(E + 1,)](flat, order, off, N=N, E=E, num_warps=1)
        tile_e = torch.empty(cap, device=x.device, dtype=torch.int32)
        tile_r = torch.empty(cap, device=x.device, dtype=torch.int32)
        _fill_tiles[(triton.cdiv(cap, 256),)](tile_e, CAP=cap, BLOCK=256)
        _tiles[(1,)](off, tile_e, tile_r, E=E, BR=ROW, num_warps=1)
        inv = torch.empty(N, device=x.device, dtype=torch.int64)
        _inverse[(triton.cdiv(N, 256),)](order, inv, N=N, BLOCK=256)
        gu = torch.empty((N, F2), device=x.device, dtype=torch.float32)
        _expert_in[(cap, triton.cdiv(F2, 64))](x, w1, order, off, tile_e,
            tile_r, gu, N=N, D=D, F2=F2, A=A, CAP=cap, BF=bf,
            BM=ROW, BN=64, BK=KBLK, num_warps=4, num_stages=3)
        act = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _activate[(triton.cdiv(N * F, 256),)](gu, act, N=N, F=F, BLOCK=256)
        yp = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _expert_out[(cap, triton.cdiv(D, 64))](act, w2, off, tile_e, tile_r, yp,
            N=N, F=F, D=D, CAP=cap, BF=bf, BM=ROW, BN=64, BK=KBLK,
            num_warps=4, num_stages=3)
        y = torch.empty((T, D), device=x.device, dtype=torch.float32)
        _combine[(T, triton.cdiv(D, 256))](yp, weights, inv, y,
            T=T, D=D, A=A, BD=256, num_warps=8)
        ctx.save_for_backward(x, rw, w1, w2, s, sel, weights, order,
                              off, tile_e, tile_r, inv, gu, act, yp)
        ctx.meta = (T, D, E, F, A, N, cap, bf)
        ctx.dtypes = (x.dtype, rw.dtype, w1.dtype, w2.dtype)
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        (x, rw, w1, w2, s, sel, weights, order, off, tile_e, tile_r,
         inv, gu, act, yp) = ctx.saved_tensors
        T, D, E, F, A, N, cap, bf = ctx.meta
        F2 = 2 * F
        dy = dy.float().contiguous()
        dyp = torch.empty_like(yp)
        dw = torch.empty((T, A), device=dy.device, dtype=torch.float32)
        _dz[(T, A)](dy, weights, yp, inv, dyp, dw,
                    T=T, D=D, A=A, BD=256, num_warps=8)
        da = torch.empty_like(act)
        _dact[(cap, triton.cdiv(F, 64))](dyp, w2, off, tile_e, tile_r, da,
            N=N, F=F, D=D, CAP=cap, BF=bf, BM=ROW, BN=64, BK=KBLK,
            num_warps=4, num_stages=3)
        dgu = torch.empty_like(gu)
        _dgu[(triton.cdiv(N * F, 256),)](gu, da, dgu, N=N, F=F, BLOCK=256)
        dxr = torch.empty((N, D), device=dy.device, dtype=torch.float32)
        _dx_rows[(cap, triton.cdiv(D, 64))](dgu, w1, off, tile_e, tile_r, dxr,
            N=N, D=D, F2=F2, CAP=cap, BF=bf, BM=ROW, BN=64, BK=KBLK,
            num_warps=4, num_stages=3)
        gw2 = torch.empty_like(w2)
        _dw2[(E, triton.cdiv(F, 64), triton.cdiv(D, 64))](act, dyp, off,
            gw2, E=E, F=F, D=D, BF=bf, BM=64, BN=64, BK=32,
            num_warps=4, num_stages=3)
        gw1 = torch.empty_like(w1)
        _dw1[(E, triton.cdiv(D, 64), triton.cdiv(F2, 64))](x, dgu, order,
            off, gw1, E=E, D=D, F2=F2, A=A, BF=bf, BM=64, BN=64, BK=32,
            num_warps=4, num_stages=3)
        dz = torch.empty((T, A), device=dy.device, dtype=torch.float32)
        _route_grad[(T,)](s, sel, weights, dw, dz, T=T, E=E, A=A,
                          BLOCK=triton.next_power_of_2(A), num_warps=1)
        grw = torch.empty_like(rw)
        _drw[(triton.cdiv(D, 32), triton.cdiv(E, 32))](x, sel, dz, grw,
            T=T, D=D, E=E, A=A, BF=bf, BM=32, BN=32, BK=32,
            num_warps=4, num_stages=3)
        gx = torch.empty_like(x)
        _combine_dx[(T, triton.cdiv(D, 256))](dxr, inv, sel, dz, rw, gx,
            T=T, D=D, E=E, A=A, BD=256, num_warps=8)
        dx_t, drw_t, dw1_t, dw2_t = ctx.dtypes
        return (gx.to(dx_t), grw.to(drw_t), None, gw1.to(dw1_t),
                gw2.to(dw2_t), None)


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
