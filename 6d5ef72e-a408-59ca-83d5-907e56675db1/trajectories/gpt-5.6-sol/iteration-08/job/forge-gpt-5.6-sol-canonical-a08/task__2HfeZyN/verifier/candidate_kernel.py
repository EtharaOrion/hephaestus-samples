import torch
import triton
import triton.language as tl

ROW_BLOCK = 64
K_BLOCK = 32


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
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="ieee")
    tl.store(S + im[:, None] * E + ie[None, :], tl.sigmoid(acc),
             mask=(im[:, None] < T) & (ie[None, :] < E))


@triton.jit
def _offsets(SORTED_E, OFF, N: tl.constexpr, E: tl.constexpr):
    target = tl.program_id(0)
    lo = 0
    hi = N
    for _ in range(32):
        mid = (lo + hi) // 2
        value = tl.load(SORTED_E + mid, mask=mid < N, other=E)
        take_right = value < target
        lo = tl.where((lo < hi) & take_right, mid + 1, lo)
        hi = tl.where((lo < hi) & ~take_right, mid, hi)
    tl.store(OFF + target, lo)


@triton.jit
def _metadata(OFF, TILE_E, TILE_ROW, E: tl.constexpr, CAP: tl.constexpr,
              BM: tl.constexpr):
    cursor = 0
    for e in tl.range(0, E):
        begin = tl.load(OFF + e)
        end = tl.load(OFF + e + 1)
        count = tl.cdiv(end - begin, BM)
        for j in tl.range(0, count):
            tl.store(TILE_E + cursor, e)
            tl.store(TILE_ROW + cursor, begin + j * BM)
            cursor += 1
    for j in tl.range(cursor, CAP):
        tl.store(TILE_E + j, -1)
        tl.store(TILE_ROW + j, 0)


@triton.jit
def _inverse(ORDER, INV, N: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair = tl.load(ORDER + pos, mask=pos < N, other=0)
    tl.store(INV + pair, pos, mask=pos < N)


@triton.jit
def _weights(S, SEL, W, DEN, T: tl.constexpr, E: tl.constexpr,
             A: tl.constexpr, BA: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BA)
    expert = tl.load(SEL + t * A + a, mask=a < A, other=0)
    raw = tl.load(S + t * E + expert, mask=a < A, other=0.0)
    den = tl.sum(raw, axis=0)
    tl.store(W + t * A + a, raw / den, mask=a < A)
    tl.store(DEN + t, den)


@triton.jit
def _expert_in(X, W1, ORDER, TILE_E, TILE_ROW, OFF, GU,
               N: tl.constexpr, D: tl.constexpr, F2: tl.constexpr,
               A: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr, BF: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile)
    start = tl.load(TILE_ROW + tile)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    rows = start + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    pair = tl.load(ORDER + rows, mask=(e >= 0) & (rows < end), other=0)
    tok = pair // A
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        kk = k + tl.arange(0, BK)
        x = tl.load(X + tok[:, None] * D + kk[None, :],
                    mask=(e >= 0) & (rows[:, None] < end) & (kk[None, :] < D), other=0.0)
        w = tl.load(W1 + e * D * F2 + kk[:, None] * F2 + cols[None, :],
                    mask=(e >= 0) & (kk[:, None] < D) & (cols[None, :] < F2), other=0.0)
        if BF:
            acc += tl.dot(x.to(tl.bfloat16), w.to(tl.bfloat16))
        else:
            acc += tl.dot(x.to(tl.float32), w.to(tl.float32), input_precision="tf32x3")
    tl.store(GU + rows[:, None] * F2 + cols[None, :], acc,
             mask=(e >= 0) & (rows[:, None] < end) & (cols[None, :] < F2))


@triton.jit
def _activate(GU, ACT, N: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = idx // F
    col = idx - row * F
    mask = idx < N * F
    g = tl.load(GU + row * (2 * F) + col, mask=mask, other=0.0)
    u = tl.load(GU + row * (2 * F) + F + col, mask=mask, other=0.0)
    tl.store(ACT + idx, g * tl.sigmoid(g) * u, mask=mask)


@triton.jit
def _expert_out(ACT, W2, TILE_E, TILE_ROW, OFF, YP,
                F: tl.constexpr, D: tl.constexpr, BM: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr, BF: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile)
    start = tl.load(TILE_ROW + tile)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    rows = start + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, F, BK):
        kk = k + tl.arange(0, BK)
        a = tl.load(ACT + rows[:, None] * F + kk[None, :],
                    mask=(e >= 0) & (rows[:, None] < end) & (kk[None, :] < F), other=0.0)
        w = tl.load(W2 + e * F * D + kk[:, None] * D + cols[None, :],
                    mask=(e >= 0) & (kk[:, None] < F) & (cols[None, :] < D), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            al = (a - ah.to(tl.float32)).to(tl.bfloat16)
            wb = w.to(tl.bfloat16)
            acc += tl.dot(ah, wb) + tl.dot(al, wb)
        else:
            acc += tl.dot(a, w.to(tl.float32), input_precision="tf32x3")
    tl.store(YP + rows[:, None] * D + cols[None, :], acc,
             mask=(e >= 0) & (rows[:, None] < end) & (cols[None, :] < D))


@triton.jit
def _combine(YP, W, INV, Y, T: tl.constexpr, D: tl.constexpr,
             A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(0, A):
        pair = t * A + a
        pos = tl.load(INV + pair)
        weight = tl.load(W + pair)
        value = tl.load(YP + pos * D + d, mask=d < D, other=0.0)
        acc += weight * value
    tl.store(Y + t * D + d, acc, mask=d < D)


@triton.jit
def _pair_dot(DY, YP, INV, Q, N: tl.constexpr, D: tl.constexpr,
              A: tl.constexpr, BD: tl.constexpr):
    pair = tl.program_id(0)
    d = tl.arange(0, BD)
    pos = tl.load(INV + pair)
    tok = pair // A
    acc = tl.sum(tl.load(DY + tok * D + d, mask=d < D, other=0.0).to(tl.float32) *
                 tl.load(YP + pos * D + d, mask=d < D, other=0.0), axis=0)
    tl.store(Q + pair, acc)


@triton.jit
def _route_grad(S, SEL, W, DEN, Q, DS, T: tl.constexpr, E: tl.constexpr,
                A: tl.constexpr, BA: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BA)
    mask = a < A
    pair = t * A + a
    expert = tl.load(SEL + pair, mask=mask, other=0)
    raw = tl.load(S + t * E + expert, mask=mask, other=0.0)
    weight = tl.load(W + pair, mask=mask, other=0.0)
    q = tl.load(Q + pair, mask=mask, other=0.0)
    center = tl.sum(weight * q, axis=0)
    ds = (q - center) / tl.load(DEN + t)
    tl.store(DS + pair, ds * raw * (1.0 - raw), mask=mask)


@triton.jit
def _dactivate(DY, W, ORDER, TILE_E, TILE_ROW, OFF, W2, GU, DA,
               D: tl.constexpr, F: tl.constexpr, A: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               BF: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile)
    start = tl.load(TILE_ROW + tile)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    rows = start + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    pair = tl.load(ORDER + rows, mask=(e >= 0) & (rows < end), other=0)
    tok = pair // A
    route_w = tl.load(W + pair, mask=(e >= 0) & (rows < end), other=0.0)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        kk = k + tl.arange(0, BK)
        dy = tl.load(DY + tok[:, None] * D + kk[None, :],
                     mask=(e >= 0) & (rows[:, None] < end) & (kk[None, :] < D), other=0.0)
        w = tl.load(W2 + e * F * D + cols[:, None] * D + kk[None, :],
                    mask=(e >= 0) & (cols[:, None] < F) & (kk[None, :] < D), other=0.0)
        a = dy.to(tl.float32) * route_w[:, None]
        if BF:
            acc += tl.dot(a.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
        else:
            acc += tl.dot(a, tl.trans(w.to(tl.float32)), input_precision="tf32x3")
    g = tl.load(GU + rows[:, None] * (2 * F) + cols[None, :],
                mask=(e >= 0) & (rows[:, None] < end) & (cols[None, :] < F), other=0.0)
    u = tl.load(GU + rows[:, None] * (2 * F) + F + cols[None, :],
                mask=(e >= 0) & (rows[:, None] < end) & (cols[None, :] < F), other=0.0)
    sig = tl.sigmoid(g)
    silu = g * sig
    tl.store(DA + rows[:, None] * (2 * F) + cols[None, :],
             acc * u * (sig + silu * (1.0 - sig)),
             mask=(e >= 0) & (rows[:, None] < end) & (cols[None, :] < F))
    tl.store(DA + rows[:, None] * (2 * F) + F + cols[None, :], acc * silu,
             mask=(e >= 0) & (rows[:, None] < end) & (cols[None, :] < F))


@triton.jit
def _dx_rows(DA, W1, TILE_E, TILE_ROW, OFF, DXR,
             D: tl.constexpr, F2: tl.constexpr, BM: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr, BF: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile)
    start = tl.load(TILE_ROW + tile)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    rows = start + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, F2, BK):
        kk = k + tl.arange(0, BK)
        da = tl.load(DA + rows[:, None] * F2 + kk[None, :],
                     mask=(e >= 0) & (rows[:, None] < end) & (kk[None, :] < F2), other=0.0)
        w = tl.load(W1 + e * D * F2 + cols[:, None] * F2 + kk[None, :],
                    mask=(e >= 0) & (cols[:, None] < D) & (kk[None, :] < F2), other=0.0)
        if BF:
            acc += tl.dot(da.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
        else:
            acc += tl.dot(da, tl.trans(w.to(tl.float32)), input_precision="tf32x3")
    tl.store(DXR + rows[:, None] * D + cols[None, :], acc,
             mask=(e >= 0) & (rows[:, None] < end) & (cols[None, :] < D))


@triton.jit
def _combine_dx(DXR, INV, DS, SEL, RW, DX, T: tl.constexpr,
                D: tl.constexpr, E: tl.constexpr, A: tl.constexpr,
                BD: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(0, A):
        pair = t * A + a
        pos = tl.load(INV + pair)
        e = tl.load(SEL + pair)
        acc += tl.load(DXR + pos * D + d, mask=d < D, other=0.0)
        acc += tl.load(DS + pair) * tl.load(RW + d * E + e, mask=d < D, other=0.0)
    tl.store(DX + t * D + d, acc, mask=d < D)


@triton.jit
def _dw2(ACT, DY, W, ORDER, OFF, OUT, F: tl.constexpr, D: tl.constexpr,
         A: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         BK: tl.constexpr, BF: tl.constexpr):
    e = tl.program_id(0)
    ff = tl.program_id(1) * BM + tl.arange(0, BM)
    dd = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        pos = k + tl.arange(0, BK)
        pair = tl.load(ORDER + pos, mask=pos < end, other=0)
        tok = pair // A
        aa = tl.load(ACT + pos[None, :] * F + ff[:, None],
                     mask=(pos[None, :] < end) & (ff[:, None] < F), other=0.0).to(tl.float32)
        bb = tl.load(DY + tok[:, None] * D + dd[None, :],
                     mask=(pos[:, None] < end) & (dd[None, :] < D), other=0.0).to(tl.float32)
        bb *= tl.load(W + pair, mask=pos < end, other=0.0)[:, None]
        if BF:
            ah = aa.to(tl.bfloat16)
            al = (aa - ah.to(tl.float32)).to(tl.bfloat16)
            bh = bb.to(tl.bfloat16)
            bl = (bb - bh.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(ah, bh) + tl.dot(al, bh) + tl.dot(ah, bl)
        else:
            acc += tl.dot(aa, bb, input_precision="tf32x3")
    tl.store(OUT + e * F * D + ff[:, None] * D + dd[None, :], acc,
             mask=(ff[:, None] < F) & (dd[None, :] < D))


@triton.jit
def _dw1(X, DA, ORDER, OFF, OUT, D: tl.constexpr, F2: tl.constexpr,
         A: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         BK: tl.constexpr, BF: tl.constexpr):
    e = tl.program_id(0)
    dd = tl.program_id(1) * BM + tl.arange(0, BM)
    ff = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        pos = k + tl.arange(0, BK)
        pair = tl.load(ORDER + pos, mask=pos < end, other=0)
        tok = pair // A
        xx = tl.load(X + tok[None, :] * D + dd[:, None],
                     mask=(pos[None, :] < end) & (dd[:, None] < D), other=0.0).to(tl.float32)
        gg = tl.load(DA + pos[:, None] * F2 + ff[None, :],
                     mask=(pos[:, None] < end) & (ff[None, :] < F2), other=0.0).to(tl.float32)
        if BF:
            gh = gg.to(tl.bfloat16)
            gl = (gg - gh.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(xx.to(tl.bfloat16), gh) + tl.dot(xx.to(tl.bfloat16), gl)
        else:
            acc += tl.dot(xx, gg, input_precision="tf32x3")
    tl.store(OUT + e * D * F2 + dd[:, None] * F2 + ff[None, :], acc,
             mask=(dd[:, None] < D) & (ff[None, :] < F2))


@triton.jit
def _drw(X, DS, ORDER, OFF, OUT, D: tl.constexpr, E: tl.constexpr,
         A: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
         BF: tl.constexpr):
    e = tl.program_id(0)
    dd = tl.program_id(1) * BM + tl.arange(0, BM)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM,), tl.float32)
    for k in range(begin, end, BK):
        pos = k + tl.arange(0, BK)
        pair = tl.load(ORDER + pos, mask=pos < end, other=0)
        tok = pair // A
        xx = tl.load(X + tok[None, :] * D + dd[:, None],
                     mask=(pos[None, :] < end) & (dd[:, None] < D), other=0.0).to(tl.float32)
        gg = tl.load(DS + pair, mask=pos < end, other=0.0).to(tl.float32)
        if BF:
            gh = gg.to(tl.bfloat16)
            gl = (gg - gh.to(tl.float32)).to(tl.bfloat16)
            tiled_h = tl.broadcast_to(gh[:, None], (BK, 16))
            tiled_l = tl.broadcast_to(gl[:, None], (BK, 16))
            dots = tl.dot(xx.to(tl.bfloat16), tiled_h)
            dots += tl.dot(xx.to(tl.bfloat16), tiled_l)
            acc += tl.sum(dots, axis=1) * 0.0625
        else:
            acc += tl.sum(xx * gg[None, :], axis=1)
    tl.store(OUT + dd * E + e, acc, mask=dd < D)


def _select(scores, bias, A):
    E = scores.shape[1]
    biased = scores + bias
    biased = torch.where(biased == 0, torch.zeros_like(biased), biased).contiguous()
    bits = biased.view(torch.int32).to(torch.int64)
    ordered = torch.where(bits >= 0, bits ^ 0x80000000, (~bits) & 0xFFFFFFFF)
    expert = torch.arange(E, device=scores.device, dtype=torch.int64)
    key = (ordered << 10) | (E - 1 - expert)
    return torch.topk(key, A, dim=1, largest=True, sorted=True).indices


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        T, D = x.shape
        E = router_w.shape[1]
        F = w2.shape[1]
        A = int(A)
        N = T * A
        bf = x.dtype == torch.bfloat16
        scores = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _router[(triton.cdiv(T, 32), triton.cdiv(E, 32))](
            x, router_w, scores, T=T, D=D, E=E, BM=32, BN=32, BK=32)
        with torch.no_grad():
            sel = _select(scores, router_b, A).contiguous()
            sorted_e, order = torch.sort(sel.reshape(-1), stable=True)
        offsets = torch.empty(E + 1, device=x.device, dtype=torch.int32)
        _offsets[(E + 1,)](sorted_e, offsets, N=N, E=E)
        cap = triton.cdiv(N, ROW_BLOCK) + E
        tile_e = torch.empty(cap, device=x.device, dtype=torch.int32)
        tile_row = torch.empty_like(tile_e)
        _metadata[(1,)](offsets, tile_e, tile_row, E=E, CAP=cap, BM=ROW_BLOCK,
                        num_warps=1)
        inv = torch.empty(N, device=x.device, dtype=torch.int64)
        _inverse[(triton.cdiv(N, 256),)](order, inv, N=N, BLOCK=256)
        weights = torch.empty(N, device=x.device, dtype=torch.float32)
        den = torch.empty(T, device=x.device, dtype=torch.float32)
        ba = triton.next_power_of_2(A)
        _weights[(T,)](scores, sel, weights, den, T=T, E=E, A=A, BA=ba)
        gu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _expert_in[(cap, triton.cdiv(2 * F, 64))](
            x, w1, order, tile_e, tile_row, offsets, gu, N=N, D=D, F2=2 * F,
            A=A, BM=ROW_BLOCK, BN=64, BK=K_BLOCK, BF=bf)
        act = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _activate[(triton.cdiv(N * F, 256),)](gu, act, N=N, F=F, BLOCK=256)
        yp = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _expert_out[(cap, triton.cdiv(D, 64))](
            act, w2, tile_e, tile_row, offsets, yp, F=F, D=D,
            BM=ROW_BLOCK, BN=64, BK=K_BLOCK, BF=bf)
        y = torch.empty_like(x)
        _combine[(T, triton.cdiv(D, 256))](yp, weights, inv, y, T=T, D=D,
                                           A=A, BD=256)
        ctx.save_for_backward(x, router_w, w1, w2, scores, sel, order, offsets,
                              tile_e, tile_row, inv, weights, den, gu, act, yp)
        ctx.shape = (T, D, E, F, A, cap, bf)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, w1, w2, scores, sel, order, offsets, tile_e, tile_row,
         inv, weights, den, gu, act, yp) = ctx.saved_tensors
        T, D, E, F, A, cap, bf = ctx.shape
        N = T * A
        dy = dy.contiguous()
        q = torch.empty(N, device=x.device, dtype=torch.float32)
        bd = triton.next_power_of_2(D)
        _pair_dot[(N,)](dy, yp, inv, q, N=N, D=D, A=A, BD=bd)
        ds = torch.empty(N, device=x.device, dtype=torch.float32)
        ba = triton.next_power_of_2(A)
        _route_grad[(T,)](scores, sel, weights, den, q, ds,
                          T=T, E=E, A=A, BA=ba)
        da = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _dactivate[(cap, triton.cdiv(F, 64))](
            dy, weights, order, tile_e, tile_row, offsets, w2, gu, da,
            D=D, F=F, A=A, BM=ROW_BLOCK, BN=64, BK=K_BLOCK, BF=bf)
        dxr = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _dx_rows[(cap, triton.cdiv(D, 64))](
            da, w1, tile_e, tile_row, offsets, dxr, D=D, F2=2 * F,
            BM=ROW_BLOCK, BN=64, BK=K_BLOCK, BF=bf)
        dx = torch.empty_like(x)
        _combine_dx[(T, triton.cdiv(D, 256))](
            dxr, inv, ds, sel, router_w, dx, T=T, D=D, E=E, A=A, BD=256)
        dw2 = torch.empty_like(w2)
        _dw2[(E, triton.cdiv(F, 64), triton.cdiv(D, 64))](
            act, dy, weights, order, offsets, dw2, F=F, D=D, A=A,
            BM=64, BN=64, BK=32, BF=bf, num_warps=4, num_stages=3)
        dw1 = torch.empty_like(w1)
        _dw1[(E, triton.cdiv(D, 64), triton.cdiv(2 * F, 64))](
            x, da, order, offsets, dw1, D=D, F2=2 * F, A=A,
            BM=64, BN=64, BK=32, BF=bf, num_warps=4, num_stages=3)
        drw = torch.empty_like(router_w)
        _drw[(E, triton.cdiv(D, 32))](x, ds, order, offsets, drw,
                                      D=D, E=E, A=A, BM=32, BK=32, BF=bf)
        return dx, drw, None, dw1, dw2, None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
