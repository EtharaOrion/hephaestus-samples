import torch
import triton
import triton.language as tl


@triton.jit
def _tile_position(pid, offsets, tile_offsets, E: tl.constexpr, BE: tl.constexpr,
                   BM: tl.constexpr):
    ei = tl.arange(0, BE)
    starts = tl.load(tile_offsets + ei, mask=ei <= E, other=0x7fffffff)
    active = pid < tl.load(tile_offsets + E)
    e = tl.sum((starts <= pid) & (ei < E), axis=0) - 1
    begin = tl.load(offsets + e)
    row = begin + (pid - tl.load(tile_offsets + e)) * BM
    end = tl.load(offsets + e + 1)
    return e, row, end, active


@triton.jit
def _router_kernel(X, RW, S, T, D: tl.constexpr, E: tl.constexpr,
                   BD: tl.constexpr, BE: tl.constexpr):
    rt = tl.program_id(0) * BD + tl.arange(0, BD)
    re = tl.program_id(1) * BE + tl.arange(0, BE)
    acc = tl.zeros((BD, BE), tl.float32)
    for k in range(0, D, 32):
        rk = k + tl.arange(0, 32)
        xv = tl.load(X + rt[:, None] * D + rk[None, :],
                     mask=(rt[:, None] < T) & (rk[None, :] < D), other=0.0)
        rv = tl.load(RW + rk[:, None] * E + re[None, :],
                     mask=(rk[:, None] < D) & (re[None, :] < E), other=0.0)
        acc += tl.dot(xv, rv, input_precision="ieee")
    tl.store(S + rt[:, None] * E + re[None, :], tl.sigmoid(acc),
             mask=(rt[:, None] < T) & (re[None, :] < E))


@triton.jit
def _weights_kernel(S, SEL, W, T, E: tl.constexpr, A: tl.constexpr,
                    BA: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BA)
    expert = tl.load(SEL + t * A + a, mask=a < A, other=0)
    score = tl.load(S + t * E + expert, mask=a < A, other=0.0)
    denom = tl.sum(score, axis=0)
    tl.store(W + t * A + a, score / denom, mask=a < A)


@triton.jit
def _dispatch_kernel(ORDER, SEL, W, TOK, WS, INV, COUNTS, R,
                     A: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = r < R
    p = tl.load(ORDER + r, mask=mask, other=0)
    e = tl.load(SEL + p, mask=mask, other=0)
    tl.store(TOK + r, p // A, mask=mask)
    tl.store(WS + r, tl.load(W + p, mask=mask, other=0.0), mask=mask)
    tl.store(INV + p, r, mask=mask)
    tl.atomic_add(COUNTS + e, 1, mask=mask)


@triton.jit
def _prefix_kernel(COUNTS, OFFSETS, TILE_OFFSETS, E: tl.constexpr,
                   BM: tl.constexpr):
    count_sum = 0
    tile_sum = 0
    tl.store(OFFSETS, 0)
    tl.store(TILE_OFFSETS, 0)
    for e in tl.static_range(0, E):
        n = tl.load(COUNTS + e)
        count_sum += n
        tile_sum += (n + BM - 1) // BM
        tl.store(OFFSETS + e + 1, count_sum)
        tl.store(TILE_OFFSETS + e + 1, tile_sum)


@triton.jit
def _expert_in_kernel(X, W1, TOK, OFFSETS, TILE_OFFSETS, G, U,
                      R, D: tl.constexpr, F: tl.constexpr, E: tl.constexpr,
                      BE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      BK: tl.constexpr):
    pid = tl.program_id(0)
    e, row0, end, active = _tile_position(pid, OFFSETS, TILE_OFFSETS, E, BE, BM)
    rr = row0 + tl.arange(0, BM)
    rf = tl.program_id(1) * BN + tl.arange(0, BN)
    tok = tl.load(TOK + rr, mask=active & (rr < end) & (rr < R), other=0)
    ag = tl.zeros((BM, BN), tl.float32)
    au = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        rk = k + tl.arange(0, BK)
        xv = tl.load(X + tok[:, None] * D + rk[None, :],
                     mask=active & (rr[:, None] < end) & (rk[None, :] < D), other=0.0)
        wg = tl.load(W1 + e * D * (2 * F) + rk[:, None] * (2 * F) + rf[None, :],
                     mask=active & (rk[:, None] < D) & (rf[None, :] < F), other=0.0)
        wu = tl.load(W1 + e * D * (2 * F) + rk[:, None] * (2 * F) + F + rf[None, :],
                     mask=active & (rk[:, None] < D) & (rf[None, :] < F), other=0.0)
        ag += tl.dot(xv, wg, input_precision="tf32x3")
        au += tl.dot(xv, wu, input_precision="tf32x3")
    mask = active & (rr[:, None] < end) & (rf[None, :] < F)
    tl.store(G + rr[:, None] * F + rf[None, :], ag, mask=mask)
    tl.store(U + rr[:, None] * F + rf[None, :], au, mask=mask)


@triton.jit
def _expert_out_kernel(G, U, W2, OFFSETS, TILE_OFFSETS, YP,
                       R, F: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                       BE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                       BK: tl.constexpr):
    pid = tl.program_id(0)
    e, row0, end, active = _tile_position(pid, OFFSETS, TILE_OFFSETS, E, BE, BM)
    rr = row0 + tl.arange(0, BM)
    rd = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, F, BK):
        rk = k + tl.arange(0, BK)
        gv = tl.load(G + rr[:, None] * F + rk[None, :],
                     mask=active & (rr[:, None] < end) & (rk[None, :] < F), other=0.0)
        uv = tl.load(U + rr[:, None] * F + rk[None, :],
                     mask=active & (rr[:, None] < end) & (rk[None, :] < F), other=0.0)
        act = gv * tl.sigmoid(gv) * uv
        wv = tl.load(W2 + e * F * D + rk[:, None] * D + rd[None, :],
                     mask=active & (rk[:, None] < F) & (rd[None, :] < D), other=0.0)
        acc += tl.dot(act, wv, input_precision="tf32x3")
    tl.store(YP + rr[:, None] * D + rd[None, :], acc,
             mask=active & (rr[:, None] < end) & (rd[None, :] < D))


@triton.jit
def _combine_kernel(YP, W, INV, Y, T, D: tl.constexpr, A: tl.constexpr,
                    BD: tl.constexpr):
    t = tl.program_id(0)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for a in tl.static_range(0, A):
        r = tl.load(INV + t * A + a)
        wa = tl.load(W + t * A + a)
        acc += wa * tl.load(YP + r * D + rd, mask=rd < D, other=0.0)
    tl.store(Y + t * D + rd, acc, mask=rd < D)


@triton.jit
def _dweight_kernel(DY, YP, INV, DW, T, D: tl.constexpr, A: tl.constexpr,
                    BD: tl.constexpr):
    p = tl.program_id(0)
    t = p // A
    a = p - t * A
    r = tl.load(INV + p)
    acc = tl.zeros((BD,), tl.float32)
    for k in range(0, D, BD):
        rd = k + tl.arange(0, BD)
        dv = tl.load(DY + t * D + rd, mask=rd < D, other=0.0)
        yv = tl.load(YP + r * D + rd, mask=rd < D, other=0.0)
        acc += dv * yv
    tl.store(DW + p, tl.sum(acc, axis=0))


@triton.jit
def _router_local_bwd_kernel(S, SEL, W, DW, DL, T, E: tl.constexpr,
                             A: tl.constexpr, BA: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BA)
    mask = a < A
    w = tl.load(W + t * A + a, mask=mask, other=0.0)
    dw = tl.load(DW + t * A + a, mask=mask, other=0.0)
    weighted = tl.sum(w * dw, axis=0)
    expert = tl.load(SEL + t * A + a, mask=mask, other=0)
    s = tl.load(S + t * E + expert, mask=mask, other=1.0)
    denom = tl.sum(tl.where(mask, s, 0.0), axis=0)
    dl = (dw - weighted) / denom * s * (1.0 - s)
    tl.store(DL + t * A + a, dl, mask=mask)


@triton.jit
def _dact_kernel(DY, WS, W2, TOK, OFFSETS, TILE_OFFSETS, DA,
                 R, F: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                 BE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                 BK: tl.constexpr):
    pid = tl.program_id(0)
    e, row0, end, active = _tile_position(pid, OFFSETS, TILE_OFFSETS, E, BE, BM)
    rr = row0 + tl.arange(0, BM)
    rf = tl.program_id(1) * BN + tl.arange(0, BN)
    tok = tl.load(TOK + rr, mask=active & (rr < end) & (rr < R), other=0)
    ws = tl.load(WS + rr, mask=active & (rr < end), other=0.0)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        rd = k + tl.arange(0, BK)
        dy = tl.load(DY + tok[:, None] * D + rd[None, :],
                     mask=active & (rr[:, None] < end) & (rd[None, :] < D), other=0.0)
        w = tl.load(W2 + e * F * D + rf[:, None] * D + rd[None, :],
                    mask=active & (rf[:, None] < F) & (rd[None, :] < D), other=0.0)
        acc += tl.dot(dy, tl.trans(w), input_precision="tf32x3")
    tl.store(DA + rr[:, None] * F + rf[None, :], acc * ws[:, None],
             mask=active & (rr[:, None] < end) & (rf[None, :] < F))


@triton.jit
def _activation_bwd_kernel(G, U, DA, DG, DU, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    g = tl.load(G + i, mask=mask, other=0.0)
    u = tl.load(U + i, mask=mask, other=0.0)
    da = tl.load(DA + i, mask=mask, other=0.0)
    s = tl.sigmoid(g)
    silu = g * s
    tl.store(DG + i, da * u * (s + silu * (1.0 - s)), mask=mask)
    tl.store(DU + i, da * silu, mask=mask)


@triton.jit
def _dw2_kernel(G, U, DY, WS, TOK, OFFSETS, DW2,
                F: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(0)
    rf = tl.program_id(1) * BM + tl.arange(0, BM)
    rd = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFFSETS + e)
    end = tl.load(OFFSETS + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        rr = k + tl.arange(0, BK)
        tok = tl.load(TOK + rr, mask=rr < end, other=0)
        g = tl.load(G + rr[None, :] * F + rf[:, None],
                    mask=(rr[None, :] < end) & (rf[:, None] < F), other=0.0)
        u = tl.load(U + rr[None, :] * F + rf[:, None],
                    mask=(rr[None, :] < end) & (rf[:, None] < F), other=0.0)
        act = g * tl.sigmoid(g) * u
        dy = tl.load(DY + tok[:, None] * D + rd[None, :],
                     mask=(rr[:, None] < end) & (rd[None, :] < D), other=0.0)
        ws = tl.load(WS + rr, mask=rr < end, other=0.0)
        acc += tl.dot(act, dy * ws[:, None], input_precision="tf32x3")
    tl.store(DW2 + e * F * D + rf[:, None] * D + rd[None, :], acc,
             mask=(rf[:, None] < F) & (rd[None, :] < D))


@triton.jit
def _dw1_kernel(X, DG, DU, TOK, OFFSETS, DW1,
                D: tl.constexpr, F: tl.constexpr, E: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(0)
    rd = tl.program_id(1) * BM + tl.arange(0, BM)
    rc = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFFSETS + e)
    end = tl.load(OFFSETS + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        rr = k + tl.arange(0, BK)
        tok = tl.load(TOK + rr, mask=rr < end, other=0)
        xv = tl.load(X + rd[:, None] + tok[None, :] * D,
                     mask=(rd[:, None] < D) & (rr[None, :] < end), other=0.0)
        f = rc - tl.where(rc < F, 0, F)
        grad = tl.where(rc[None, :] < F,
                        tl.load(DG + rr[:, None] * F + f[None, :],
                                mask=(rr[:, None] < end) & (f[None, :] < F), other=0.0),
                        tl.load(DU + rr[:, None] * F + f[None, :],
                                mask=(rr[:, None] < end) & (f[None, :] < F), other=0.0))
        acc += tl.dot(xv, grad, input_precision="tf32x3")
    tl.store(DW1 + e * D * (2 * F) + rd[:, None] * (2 * F) + rc[None, :], acc,
             mask=(rd[:, None] < D) & (rc[None, :] < 2 * F))


@triton.jit
def _dx_pair_kernel(DG, DU, W1, OFFSETS, TILE_OFFSETS, DXP,
                    R, D: tl.constexpr, F: tl.constexpr, E: tl.constexpr,
                    BE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                    BK: tl.constexpr):
    pid = tl.program_id(0)
    e, row0, end, active = _tile_position(pid, OFFSETS, TILE_OFFSETS, E, BE, BM)
    rr = row0 + tl.arange(0, BM)
    rd = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, F, BK):
        rf = k + tl.arange(0, BK)
        dg = tl.load(DG + rr[:, None] * F + rf[None, :],
                     mask=active & (rr[:, None] < end) & (rf[None, :] < F), other=0.0)
        du = tl.load(DU + rr[:, None] * F + rf[None, :],
                     mask=active & (rr[:, None] < end) & (rf[None, :] < F), other=0.0)
        wg = tl.load(W1 + e * D * (2 * F) + rd[:, None] * (2 * F) + rf[None, :],
                     mask=active & (rd[:, None] < D) & (rf[None, :] < F), other=0.0)
        wu = tl.load(W1 + e * D * (2 * F) + rd[:, None] * (2 * F) + F + rf[None, :],
                     mask=active & (rd[:, None] < D) & (rf[None, :] < F), other=0.0)
        acc += tl.dot(dg, tl.trans(wg), input_precision="tf32x3")
        acc += tl.dot(du, tl.trans(wu), input_precision="tf32x3")
    tl.store(DXP + rr[:, None] * D + rd[None, :], acc,
             mask=active & (rr[:, None] < end) & (rd[None, :] < D))


@triton.jit
def _drw_kernel(X, DL, ORDER, TOK, OFFSETS, DRW,
                D: tl.constexpr, E: tl.constexpr, A: tl.constexpr,
                BD: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(0)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    begin = tl.load(OFFSETS + e)
    end = tl.load(OFFSETS + e + 1)
    acc = tl.zeros((BD,), tl.float32)
    for k in range(begin, end, BK):
        rr = k + tl.arange(0, BK)
        tok = tl.load(TOK + rr, mask=rr < end, other=0)
        xv = tl.load(X + tok[None, :] * D + rd[:, None],
                     mask=(rr[None, :] < end) & (rd[:, None] < D), other=0.0)
        p = tl.load(ORDER + rr, mask=rr < end, other=0)
        dl = tl.load(DL + p, mask=rr < end, other=0.0)
        acc += tl.sum(xv * dl[None, :], axis=1)
    tl.store(DRW + rd * E + e, acc, mask=rd < D)

@triton.jit
def _dx_combine_kernel(DXP, DL, INV, SEL, RW, DX, T, D: tl.constexpr, E: tl.constexpr, A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for a in tl.static_range(0, A):
        p = t * A + a
        r = tl.load(INV + p)
        e = tl.load(SEL + p)
        acc += tl.load(DXP + r * D + rd, mask=rd < D, other=0.0)
        acc += tl.load(DL + p) * tl.load(RW + rd * E + e, mask=rd < D, other=0.0)
    tl.store(DX + t * D + rd, acc, mask=rd < D)


def _select(scores, bias, A):
    E = scores.shape[1]
    biased = scores + bias
    biased = torch.where(biased == 0.0, torch.zeros_like(biased), biased).contiguous()
    bits = biased.view(torch.int32).to(torch.int64)
    ordered = torch.where(bits >= 0, bits ^ 0x80000000, (~bits) & 0xFFFFFFFF)
    expert = torch.arange(E, device=scores.device, dtype=torch.int64)
    return torch.topk((ordered << 10) | (E - 1 - expert), A, dim=1, largest=True, sorted=True).indices


class _FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        T, D = x.shape
        E, F = router_w.shape[1], w1.shape[2] // 2
        R = T * A
        xf, rwf, w1f, w2f = x.float(), router_w.float(), w1.float(), w2.float()
        scores = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _router_kernel[(triton.cdiv(T, 32), triton.cdiv(E, 32))](xf, rwf, scores, T, D=D, E=E, BD=32, BE=32, num_warps=4)
        with torch.no_grad():
            sel = _select(scores, router_b, A).contiguous()
            order = torch.argsort(sel.reshape(-1), stable=True)
        weights = torch.empty((T, A), device=x.device, dtype=torch.float32)
        _weights_kernel[(T,)](scores, sel, weights, T, E=E, A=A, BA=triton.next_power_of_2(A), num_warps=4)
        tok = torch.empty(R, device=x.device, dtype=torch.int64)
        ws = torch.empty(R, device=x.device, dtype=torch.float32)
        inv = torch.empty(R, device=x.device, dtype=torch.int64)
        counts = torch.zeros(E, device=x.device, dtype=torch.int32)
        _dispatch_kernel[(triton.cdiv(R, 256),)](order, sel, weights, tok, ws, inv, counts, R, A=A, BLOCK=256)
        offsets = torch.empty(E + 1, device=x.device, dtype=torch.int32)
        tile_offsets = torch.empty(E + 1, device=x.device, dtype=torch.int32)
        _prefix_kernel[(1,)](counts, offsets, tile_offsets, E=E, BM=64, num_warps=1)
        g = torch.empty((R, F), device=x.device, dtype=torch.float32)
        u = torch.empty_like(g)
        max_tiles, be = triton.cdiv(R, 64) + E, triton.next_power_of_2(E + 1)
        _expert_in_kernel[(max_tiles, triton.cdiv(F, 64))](xf, w1f, tok, offsets, tile_offsets, g, u, R, D=D, F=F, E=E, BE=be, BM=64, BN=64, BK=32, num_warps=8, num_stages=3)
        yp = torch.empty((R, D), device=x.device, dtype=torch.float32)
        _expert_out_kernel[(max_tiles, triton.cdiv(D, 64))](g, u, w2f, offsets, tile_offsets, yp, R, F=F, D=D, E=E, BE=be, BM=64, BN=64, BK=32, num_warps=8, num_stages=3)
        y = torch.empty((T, D), device=x.device, dtype=torch.float32)
        _combine_kernel[(T, triton.cdiv(D, 256))](yp, weights, inv, y, T, D=D, A=A, BD=256, num_warps=8)
        ctx.save_for_backward(xf, rwf, w1f, w2f, scores, sel, weights, order, tok, ws, inv, offsets, tile_offsets, g, u, yp)
        ctx.meta = T, D, E, F, A, x.dtype, router_w.dtype, w1.dtype, w2.dtype
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        x, rw, w1, w2, scores, sel, weights, order, tok, ws, inv, offsets, tile_offsets, g, u, yp = ctx.saved_tensors
        T, D, E, F, A, xd, rwd, w1d, w2d = ctx.meta
        R = T * A
        dy = dy.float().contiguous()
        be, max_tiles = triton.next_power_of_2(E + 1), triton.cdiv(R, 64) + E
        dw = torch.empty(R, device=dy.device, dtype=torch.float32)
        _dweight_kernel[(R,)](dy, yp, inv, dw, T, D=D, A=A, BD=256, num_warps=8)
        dl = torch.empty(R, device=dy.device, dtype=torch.float32)
        _router_local_bwd_kernel[(T,)](scores, sel, weights, dw, dl, T, E=E, A=A, BA=triton.next_power_of_2(A), num_warps=4)
        da = torch.empty((R, F), device=dy.device, dtype=torch.float32)
        _dact_kernel[(max_tiles, triton.cdiv(F, 64))](dy, ws, w2, tok, offsets, tile_offsets, da, R, F=F, D=D, E=E, BE=be, BM=64, BN=64, BK=32, num_warps=8, num_stages=3)
        dg, du = torch.empty_like(da), torch.empty_like(da)
        _activation_bwd_kernel[(triton.cdiv(R * F, 256),)](g, u, da, dg, du, R * F, BLOCK=256)
        dw2 = torch.empty_like(w2)
        _dw2_kernel[(E, triton.cdiv(F, 64), triton.cdiv(D, 64))](g, u, dy, ws, tok, offsets, dw2, F=F, D=D, E=E, BM=64, BN=64, BK=32, num_warps=8, num_stages=3)
        dw1 = torch.empty_like(w1)
        _dw1_kernel[(E, triton.cdiv(D, 64), triton.cdiv(2 * F, 64))](x, dg, du, tok, offsets, dw1, D=D, F=F, E=E, BM=64, BN=64, BK=32, num_warps=8, num_stages=3)
        dxp = torch.empty((R, D), device=dy.device, dtype=torch.float32)
        _dx_pair_kernel[(max_tiles, triton.cdiv(D, 64))](dg, du, w1, offsets, tile_offsets, dxp, R, D=D, F=F, E=E, BE=be, BM=64, BN=64, BK=32, num_warps=8, num_stages=3)
        dx = torch.empty((T, D), device=dy.device, dtype=torch.float32)
        _dx_combine_kernel[(T, triton.cdiv(D, 256))](dxp, dl, inv, sel, rw, dx, T, D=D, E=E, A=A, BD=256, num_warps=8)
        drw = torch.empty_like(rw)
        _drw_kernel[(E, triton.cdiv(D, 128))](x, dl, order, tok, offsets, drw, D=D, E=E, A=A, BD=128, BK=128, num_warps=8)
        return dx.to(xd), drw.to(rwd), None, dw1.to(w1d), dw2.to(w2d), None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _FusedMoe.apply(x, router_w, router_b, w1, w2, A)
