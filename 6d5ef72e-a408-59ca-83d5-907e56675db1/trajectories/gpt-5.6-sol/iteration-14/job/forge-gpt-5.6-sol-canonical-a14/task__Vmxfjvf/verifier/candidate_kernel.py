import torch
import triton
import triton.language as tl

ROW = 64
KBLK = 32


@triton.jit
def _dot(a, b, bf: tl.constexpr, split_a: tl.constexpr = False,
         split_b: tl.constexpr = False):
    if bf:
        ah = a.to(tl.bfloat16)
        bh = b.to(tl.bfloat16)
        out = tl.dot(ah, bh)
        if split_a:
            out += tl.dot((a - ah).to(tl.bfloat16), bh)
        if split_b:
            out += tl.dot(ah, (b - bh).to(tl.bfloat16))
        return out
    return tl.dot(a, b, input_precision="ieee")


@triton.jit
def _router(x, rw, scores, T: tl.constexpr, D: tl.constexpr,
            E: tl.constexpr, BD: tl.constexpr, BE: tl.constexpr):
    rt = tl.program_id(0) * 32 + tl.arange(0, 32)
    re = tl.program_id(1) * BE + tl.arange(0, BE)
    acc = tl.zeros((32, BE), tl.float32)
    for k in range(0, D, BD):
        rk = k + tl.arange(0, BD)
        a = tl.load(x + rt[:, None] * D + rk[None, :],
                    mask=(rt[:, None] < T) & (rk[None, :] < D), other=0.).to(tl.float32)
        b = tl.load(rw + rk[:, None] * E + re[None, :],
                    mask=(rk[:, None] < D) & (re[None, :] < E), other=0.).to(tl.float32)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(scores + rt[:, None] * E + re[None, :], tl.sigmoid(acc),
             mask=(rt[:, None] < T) & (re[None, :] < E))


@triton.jit
def _offsets(sel, perm, offsets, N: tl.constexpr, E: tl.constexpr,
             A: tl.constexpr):
    e = tl.program_id(0)
    lo = tl.zeros((), tl.int32)
    hi = N
    for _ in range(32):
        mid = (lo + hi) // 2
        p = tl.load(perm + mid, mask=mid < N, other=N - 1)
        v = tl.load(sel + p, mask=mid < N, other=E)
        take = (mid < N) & (v < e)
        lo = tl.where((lo < hi) & take, mid + 1, lo)
        hi = tl.where((lo < hi) & ~take, mid, hi)
    tl.store(offsets + e, lo)


@triton.jit
def _fill_i32(x, n: tl.constexpr, value: tl.constexpr):
    r = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(x + r, value, mask=r < n)


@triton.jit
def _tiles(offsets, tile_e, tile_s, E: tl.constexpr, CAP: tl.constexpr,
           BR: tl.constexpr):
    cursor = 0
    for e in range(E):
        p = tl.load(offsets + e)
        end = tl.load(offsets + e + 1)
        while p < end:
            tl.store(tile_e + cursor, e)
            tl.store(tile_s + cursor, p)
            cursor += 1
            p += BR


@triton.jit
def _inverse(perm, inv, N: tl.constexpr):
    r = tl.program_id(0) * 256 + tl.arange(0, 256)
    m = r < N
    p = tl.load(perm + r, mask=m, other=0)
    tl.store(inv + p, r, mask=m)


@triton.jit
def _expert_in(x, w1, perm, offsets, tile_e, tile_s, gu, act,
               T: tl.constexpr, D: tl.constexpr, F: tl.constexpr,
               A: tl.constexpr, N: tl.constexpr, BF: tl.constexpr,
               BN: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(tile_e + tile)
    start = tl.load(tile_s + tile)
    end = tl.load(offsets + e + 1, mask=e >= 0, other=0)
    rr = tl.arange(0, BR)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    pos = start + rr
    pair = tl.load(perm + pos, mask=(e >= 0) & (pos < end), other=0)
    tok = pair // A
    accg = tl.zeros((BR, BN), tl.float32)
    accu = tl.zeros((BR, BN), tl.float32)
    for k in range(0, D, BK):
        rk = k + tl.arange(0, BK)
        xv = tl.load(x + tok[:, None] * D + rk[None, :],
                     mask=(e >= 0) & (pos[:, None] < end) & (rk[None, :] < D), other=0.).to(tl.float32)
        wg = tl.load(w1 + e * D * (2 * F) + rk[:, None] * (2 * F) + rn[None, :],
                     mask=(e >= 0) & (rk[:, None] < D) & (rn[None, :] < F), other=0.).to(tl.float32)
        wu = tl.load(w1 + e * D * (2 * F) + rk[:, None] * (2 * F) + F + rn[None, :],
                     mask=(e >= 0) & (rk[:, None] < D) & (rn[None, :] < F), other=0.).to(tl.float32)
        accg += _dot(xv, wg, BF)
        accu += _dot(xv, wu, BF)
    mask = (e >= 0) & (pos[:, None] < end) & (rn[None, :] < F)
    base = pos[:, None] * (2 * F) + rn[None, :]
    tl.store(gu + base, accg, mask=mask)
    tl.store(gu + base + F, accu, mask=mask)
    tl.store(act + pos[:, None] * F + rn[None, :], accg * tl.sigmoid(accg) * accu, mask=mask)


@triton.jit
def _expert_out(act, w2, offsets, tile_e, tile_s, fe,
                F: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
                BF: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                BR: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(tile_e + tile)
    start = tl.load(tile_s + tile)
    end = tl.load(offsets + e + 1, mask=e >= 0, other=0)
    rr = tl.arange(0, BR)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    pos = start + rr
    acc = tl.zeros((BR, BN), tl.float32)
    for k in range(0, F, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(act + pos[:, None] * F + rk[None, :],
                    mask=(e >= 0) & (pos[:, None] < end) & (rk[None, :] < F), other=0.)
        b = tl.load(w2 + e * F * D + rk[:, None] * D + rn[None, :],
                    mask=(e >= 0) & (rk[:, None] < F) & (rn[None, :] < D), other=0.).to(tl.float32)
        acc += _dot(a, b, BF, True, False)
    tl.store(fe + pos[:, None] * D + rn[None, :], acc,
             mask=(e >= 0) & (pos[:, None] < end) & (rn[None, :] < D))


@triton.jit
def _combine(scores, sel, inv, fe, weights, y,
             T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
             A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    den = tl.zeros((), tl.float32)
    for a in range(A):
        e = tl.load(sel + t * A + a)
        den += tl.load(scores + t * E + e)
    out = tl.zeros((BD,), tl.float32)
    for a in range(A):
        pair = t * A + a
        e = tl.load(sel + pair)
        w = tl.load(scores + t * E + e) / den
        pos = tl.load(inv + pair)
        f = tl.load(fe + pos * D + rd, mask=rd < D, other=0.)
        out += w * f
        tl.store(weights + pair, w)
    tl.store(y + t * D + rd, out, mask=rd < D)


@triton.jit
def _dz(dy, weights, perm, dz, D: tl.constexpr, A: tl.constexpr,
        N: tl.constexpr, BD: tl.constexpr):
    pos = tl.program_id(0)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    pair = tl.load(perm + pos)
    t = pair // A
    w = tl.load(weights + pair)
    v = tl.load(dy + t * D + rd, mask=rd < D, other=0.).to(tl.float32)
    tl.store(dz + pos * D + rd, w * v, mask=rd < D)


@triton.jit
def _dact(dz, w2, offsets, tile_e, tile_s, da,
          D: tl.constexpr, F: tl.constexpr, N: tl.constexpr,
          BF: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          BR: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(tile_e + tile)
    start = tl.load(tile_s + tile)
    end = tl.load(offsets + e + 1, mask=e >= 0, other=0)
    rr = tl.arange(0, BR)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    pos = start + rr
    acc = tl.zeros((BR, BN), tl.float32)
    for k in range(0, D, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(dz + pos[:, None] * D + rk[None, :],
                    mask=(e >= 0) & (pos[:, None] < end) & (rk[None, :] < D), other=0.)
        b = tl.load(w2 + e * F * D + rn[None, :] * D + rk[:, None],
                    mask=(e >= 0) & (rn[None, :] < F) & (rk[:, None] < D), other=0.).to(tl.float32)
        acc += _dot(a, b, BF)
    tl.store(da + pos[:, None] * F + rn[None, :], acc,
             mask=(e >= 0) & (pos[:, None] < end) & (rn[None, :] < F))


@triton.jit
def _dgu(gu, da, out, n: tl.constexpr, F: tl.constexpr):
    o = tl.program_id(0) * 256 + tl.arange(0, 256)
    m = o < n
    row = o // F
    col = o % F
    g = tl.load(gu + row * (2 * F) + col, mask=m, other=0.)
    u = tl.load(gu + row * (2 * F) + F + col, mask=m, other=0.)
    v = tl.load(da + o, mask=m, other=0.)
    s = tl.sigmoid(g)
    sg = g * s
    tl.store(out + row * (2 * F) + col, v * u * (s + sg * (1 - s)), mask=m)
    tl.store(out + row * (2 * F) + F + col, v * sg, mask=m)


@triton.jit
def _dx_rows(dgu, w1, offsets, tile_e, tile_s, dxr,
             F: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
             BF: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             BR: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(tile_e + tile)
    start = tl.load(tile_s + tile)
    end = tl.load(offsets + e + 1, mask=e >= 0, other=0)
    rr = tl.arange(0, BR)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    pos = start + rr
    acc = tl.zeros((BR, BN), tl.float32)
    for k in range(0, 2 * F, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(dgu + pos[:, None] * (2 * F) + rk[None, :],
                    mask=(e >= 0) & (pos[:, None] < end) & (rk[None, :] < 2 * F), other=0.)
        b = tl.load(w1 + e * D * (2 * F) + rn[None, :] * (2 * F) + rk[:, None],
                    mask=(e >= 0) & (rn[None, :] < D) & (rk[:, None] < 2 * F), other=0.).to(tl.float32)
        acc += _dot(a, b, BF)
    tl.store(dxr + pos[:, None] * D + rn[None, :], acc,
             mask=(e >= 0) & (pos[:, None] < end) & (rn[None, :] < D))


@triton.jit
def _dw2(act, dz, offsets, out, F: tl.constexpr, D: tl.constexpr,
         E: tl.constexpr, N: tl.constexpr, BF: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(0)
    rf = tl.program_id(1) * BM + tl.arange(0, BM)
    rd = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(offsets + e)
    end = tl.load(offsets + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(act + rk[None, :] * F + rf[:, None],
                    mask=(rk[None, :] < end) & (rf[:, None] < F), other=0.)
        b = tl.load(dz + rk[:, None] * D + rd[None, :],
                    mask=(rk[:, None] < end) & (rd[None, :] < D), other=0.)
        acc += _dot(a, b, BF, True, True)
    tl.store(out + e * F * D + rf[:, None] * D + rd[None, :], acc,
             mask=(rf[:, None] < F) & (rd[None, :] < D))


@triton.jit
def _dw1(x, dgu, perm, offsets, out, D: tl.constexpr, F: tl.constexpr,
         E: tl.constexpr, A: tl.constexpr, N: tl.constexpr,
         BF: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         BK: tl.constexpr):
    e = tl.program_id(0)
    rd = tl.program_id(1) * BM + tl.arange(0, BM)
    rf = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(offsets + e)
    end = tl.load(offsets + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        rk = k + tl.arange(0, BK)
        pair = tl.load(perm + rk, mask=rk < end, other=0)
        tok = pair // A
        a = tl.load(x + rd[:, None] + tok[None, :] * D,
                    mask=(rd[:, None] < D) & (rk[None, :] < end), other=0.).to(tl.float32)
        b = tl.load(dgu + rk[:, None] * (2 * F) + rf[None, :],
                    mask=(rk[:, None] < end) & (rf[None, :] < 2 * F), other=0.)
        acc += _dot(a, b, BF, False, True)
    tl.store(out + e * D * (2 * F) + rd[:, None] * (2 * F) + rf[None, :], acc,
             mask=(rd[:, None] < D) & (rf[None, :] < 2 * F))


@triton.jit
def _route_grad(dy, y, fe, scores, sel, inv, route,
                T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    a = tl.program_id(1)
    rd = tl.arange(0, BD)
    e = tl.load(sel + t * A + a)
    pos = tl.load(inv + t * A + a)
    dot = tl.zeros((), tl.float32)
    for k in range(0, D, BD):
        r = k + rd
        g = tl.load(dy + t * D + r, mask=r < D, other=0.).to(tl.float32)
        f = tl.load(fe + pos * D + r, mask=r < D, other=0.)
        yy = tl.load(y + t * D + r, mask=r < D, other=0.)
        dot += tl.sum(g * (f - yy))
    den = tl.zeros((), tl.float32)
    for j in range(A):
        ej = tl.load(sel + t * A + j)
        den += tl.load(scores + t * E + ej)
    s = tl.load(scores + t * E + e)
    tl.store(route + t * A + a, dot * s * (1 - s) / den)


@triton.jit
def _drw(x, route, sel, out, T: tl.constexpr, D: tl.constexpr,
         E: tl.constexpr, A: tl.constexpr, BF: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rd = tl.program_id(0) * BM + tl.arange(0, BM)
    re = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, T, BK):
        rt = k + tl.arange(0, BK)
        xv = tl.load(x + rd[:, None] + rt[None, :] * D,
                     mask=(rd[:, None] < D) & (rt[None, :] < T), other=0.).to(tl.float32)
        rv = tl.zeros((BK, BN), tl.float32)
        for a in range(A):
            ee = tl.load(sel + rt[:, None] * A + a,
                         mask=rt[:, None] < T, other=-1)
            v = tl.load(route + rt * A + a, mask=rt < T, other=0.)
            rv += tl.where(ee == re[None, :], v[:, None], 0.)
        acc += _dot(xv, rv, BF, False, True)
    tl.store(out + rd[:, None] * E + re[None, :], acc,
             mask=(rd[:, None] < D) & (re[None, :] < E))


@triton.jit
def _combine_dx(dxr, route, sel, inv, rw, out,
                T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(A):
        pair = t * A + a
        pos = tl.load(inv + pair)
        e = tl.load(sel + pair)
        acc += tl.load(dxr + pos * D + rd, mask=rd < D, other=0.)
        r = tl.load(route + pair)
        acc += r * tl.load(rw + rd * E + e, mask=rd < D, other=0.).to(tl.float32)
    tl.store(out + t * D + rd, acc, mask=rd < D)


def _select(scores, bias, A):
    E = scores.shape[1]
    biased = scores + bias
    biased = torch.where(biased == 0, torch.zeros_like(biased), biased).contiguous()
    bits = biased.view(torch.int32).to(torch.int64)
    ordered = torch.where(bits >= 0, bits ^ 0x80000000, (~bits) & 0xffffffff)
    idx = torch.arange(E, device=scores.device, dtype=torch.int64)
    keys = (ordered << 10) | (E - 1 - idx)
    return torch.topk(keys, A, dim=1, largest=True, sorted=True).indices.to(torch.int32)


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rw, bias, w1, w2, A):
        A = int(A)
        T, D = x.shape
        E = rw.shape[1]
        F = w2.shape[1]
        N = T * A
        bf = x.dtype == torch.bfloat16
        scores = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _router[(triton.cdiv(T, 32), triton.cdiv(E, 32))](x, rw, scores, T, D, E, BD=32, BE=32)
        with torch.no_grad():
            sel = _select(scores, bias.float(), A).contiguous()
            flat = sel.flatten()
            perm = torch.argsort(flat, stable=True).to(torch.int32)
        offsets = torch.empty(E + 1, device=x.device, dtype=torch.int32)
        _offsets[(E + 1,)](sel, perm, offsets, N, E + 1, A)
        cap = triton.cdiv(N, ROW) + E
        tile_e = torch.empty(cap, device=x.device, dtype=torch.int32)
        tile_s = torch.empty_like(tile_e)
        _fill_i32[(triton.cdiv(cap, 256),)](tile_e, cap, -1)
        _fill_i32[(triton.cdiv(cap, 256),)](tile_s, cap, 0)
        _tiles[(1,)](offsets, tile_e, tile_s, E, cap, BR=ROW, num_warps=1)
        inv = torch.empty(N, device=x.device, dtype=torch.int32)
        _inverse[(triton.cdiv(N, 256),)](perm, inv, N)
        gu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        act = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _expert_in[(cap, triton.cdiv(F, 64))](x, w1, perm, offsets, tile_e, tile_s, gu, act,
            T, D, F, A, N, BF=bf, BN=64, BK=KBLK, BR=ROW, num_warps=4, num_stages=3)
        fe = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _expert_out[(cap, triton.cdiv(D, 64))](act, w2, offsets, tile_e, tile_s, fe,
            F, D, N, BF=bf, BN=64, BK=KBLK, BR=ROW, num_warps=4, num_stages=3)
        weights = torch.empty(N, device=x.device, dtype=torch.float32)
        y = torch.empty((T, D), device=x.device, dtype=torch.float32)
        _combine[(T, triton.cdiv(D, 256))](scores, sel, inv, fe, weights, y,
                                           T, D, E, A, BD=256)
        ctx.save_for_backward(x, rw, w1, w2, scores, sel, perm, offsets, tile_e,
                              tile_s, inv, gu, act, fe, weights, y)
        ctx.dims = (T, D, E, F, A, N, cap, bf)
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        (x, rw, w1, w2, scores, sel, perm, offsets, tile_e, tile_s, inv,
         gu, act, fe, weights, y) = ctx.saved_tensors
        T, D, E, F, A, N, cap, bf = ctx.dims
        dy = dy.contiguous()
        dz = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _dz[(N, triton.cdiv(D, 256))](dy, weights, perm, dz, D, A, N, BD=256)
        da = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _dact[(cap, triton.cdiv(F, 64))](dz, w2, offsets, tile_e, tile_s, da,
            D, F, N, BF=bf, BN=64, BK=KBLK, BR=ROW, num_warps=4, num_stages=3)
        dgu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _dgu[(triton.cdiv(N * F, 256),)](gu, da, dgu, N * F, F)
        dxr = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _dx_rows[(cap, triton.cdiv(D, 64))](dgu, w1, offsets, tile_e, tile_s, dxr,
            F, D, N, BF=bf, BN=64, BK=KBLK, BR=ROW, num_warps=4, num_stages=3)
        dw2 = torch.empty_like(w2, dtype=torch.float32)
        _dw2[(E, triton.cdiv(F, 64), triton.cdiv(D, 64))](act, dz, offsets, dw2,
            F, D, E, N, BF=bf, BM=64, BN=64, BK=KBLK, num_warps=4, num_stages=3)
        dw1 = torch.empty_like(w1, dtype=torch.float32)
        _dw1[(E, triton.cdiv(D, 64), triton.cdiv(2 * F, 64))](x, dgu, perm, offsets, dw1,
            D, F, E, A, N, BF=bf, BM=64, BN=64, BK=KBLK, num_warps=4, num_stages=3)
        route = torch.empty(N, device=x.device, dtype=torch.float32)
        _route_grad[(T, A)](dy, y, fe, scores, sel, inv, route,
                            T, D, E, A, BD=256)
        drw = torch.empty_like(rw, dtype=torch.float32)
        _drw[(triton.cdiv(D, 32), triton.cdiv(E, 32))](x, route, sel, drw,
            T, D, E, A, BF=False, BM=32, BN=32, BK=32, num_warps=4, num_stages=3)
        dx = torch.empty_like(x, dtype=torch.float32)
        _combine_dx[(T, triton.cdiv(D, 256))](dxr, route, sel, inv, rw, dx,
                                               T, D, E, A, BD=256)
        return dx.to(x.dtype), drw.to(rw.dtype), None, dw1.to(w1.dtype), dw2.to(w2.dtype), None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
