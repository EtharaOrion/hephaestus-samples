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
                    mask=(im[:, None] < T) & (ik[None, :] < D), other=0.0)
        w = tl.load(RW + ik[:, None] * E + ie[None, :],
                    mask=(ik[:, None] < D) & (ie[None, :] < E), other=0.0)
        acc += tl.dot(x.to(tl.float32), w.to(tl.float32), input_precision="ieee")
    tl.store(S + im[:, None] * E + ie[None, :], tl.sigmoid(acc),
             mask=(im[:, None] < T) & (ie[None, :] < E))


@triton.jit
def _keys(S, BIAS, KEYS, T: tl.constexpr, E: tl.constexpr, BS: tl.constexpr):
    p = tl.program_id(0) * BS + tl.arange(0, BS)
    m = p < T * E
    e = p % E
    b = tl.load(S + p, mask=m, other=0.0) + tl.load(BIAS + e, mask=m, other=0.0)
    b = tl.where(b == 0.0, 0.0, b)
    ib = b.to(tl.int32, bitcast=True)
    ii = ib.to(tl.int64)
    u = tl.where(ib >= 0, ii ^ 0x80000000, (~ii) & 0xFFFFFFFF)
    tl.store(KEYS + p, (u << 10) | (E - 1 - e), mask=m)


@triton.jit
def _weights(S, SEL, W, T: tl.constexpr, E: tl.constexpr,
             A: tl.constexpr, BA: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BA)
    m = a < A
    e = tl.load(SEL + t * A + a, mask=m, other=0)
    q = tl.load(S + t * E + e, mask=m, other=0.0)
    den = tl.sum(q, axis=0)
    tl.store(W + t * A + a, q / den, mask=m)


@triton.jit
def _offsets(SEL, ORDER, OFF, N: tl.constexpr, E: tl.constexpr, A: tl.constexpr):
    e = tl.program_id(0)
    lo = 0
    hi = N
    while lo < hi:
        mid = (lo + hi) // 2
        j = tl.load(ORDER + mid)
        v = tl.load(SEL + j)
        take = v < e
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(take, hi, mid)
    tl.store(OFF + e, lo)


@triton.jit
def _prefix(OFF, PRE, E: tl.constexpr, B: tl.constexpr, R: tl.constexpr):
    i = tl.arange(0, B)
    m = i < E
    lo = tl.load(OFF + i, mask=m, other=0)
    hi = tl.load(OFF + i + 1, mask=m, other=0)
    n = tl.where(m, (hi - lo + R - 1) // R, 0)
    cs = tl.cumsum(n, axis=0)
    tl.store(PRE + 0, 0)
    tl.store(PRE + i + 1, cs, mask=m)


@triton.jit
def _inverse(ORDER, INV, N: tl.constexpr, BS: tl.constexpr):
    p = tl.program_id(0) * BS + tl.arange(0, BS)
    m = p < N
    j = tl.load(ORDER + p, mask=m, other=0)
    tl.store(INV + j, p, mask=m)


@triton.jit
def _segment_location(PRE, OFF, pid, E: tl.constexpr, R: tl.constexpr):
    lo = 0
    hi = E
    while lo < hi:
        mid = (lo + hi) // 2
        v = tl.load(PRE + mid + 1)
        take = v <= pid
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(take, hi, mid)
    e = lo
    first = tl.load(PRE + tl.minimum(e, E))
    begin = tl.load(OFF + tl.minimum(e, E))
    return e, begin + (pid - first) * R


@triton.jit
def _expert_in(X, W1, SEL, ORDER, OFF, PRE, GU,
               N: tl.constexpr, D: tl.constexpr, F2: tl.constexpr,
               E: tl.constexpr, A: tl.constexpr, BF: tl.constexpr,
               R: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    e, start = _segment_location(PRE, OFF, pid, E, R)
    ir = tl.arange(0, R)
    ic = tl.program_id(1) * C + tl.arange(0, C)
    end = tl.load(OFF + tl.minimum(e + 1, E))
    pos = start + ir
    valid = (e < E) & (pos < end) & (pos < N)
    j = tl.load(ORDER + pos, mask=valid, other=0)
    tok = j // A
    acc = tl.zeros((R, C), tl.float32)
    for k0 in range(0, D, K):
        ik = k0 + tl.arange(0, K)
        x = tl.load(X + tok[:, None] * D + ik[None, :],
                    mask=valid[:, None] & (ik[None, :] < D), other=0.0)
        w = tl.load(W1 + e * D * F2 + ik[:, None] * F2 + ic[None, :],
                    mask=(e < E) & (ik[:, None] < D) & (ic[None, :] < F2), other=0.0)
        if BF:
            acc += tl.dot(x, w)
        else:
            acc += tl.dot(x.to(tl.float32), w.to(tl.float32), input_precision="tf32x3")
    tl.store(GU + pos[:, None] * F2 + ic[None, :], acc,
             mask=valid[:, None] & (ic[None, :] < F2))


@triton.jit
def _activate(GU, ACT, N: tl.constexpr, F: tl.constexpr, BS: tl.constexpr):
    p = tl.program_id(0) * BS + tl.arange(0, BS)
    m = p < N * F
    row = p // F
    f = p % F
    g = tl.load(GU + row * (2 * F) + f, mask=m, other=0.0)
    u = tl.load(GU + row * (2 * F) + F + f, mask=m, other=0.0)
    tl.store(ACT + p, g * tl.sigmoid(g) * u, mask=m)



@triton.jit
def _expert_out(ACT, W2, OFF, PRE, YP,
                N: tl.constexpr, F: tl.constexpr, D: tl.constexpr,
                E: tl.constexpr, BF: tl.constexpr,
                R: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    e, start = _segment_location(PRE, OFF, pid, E, R)
    ir = tl.arange(0, R)
    od = tl.program_id(1) * C + tl.arange(0, C)
    end = tl.load(OFF + tl.minimum(e + 1, E))
    pos = start + ir
    valid = (e < E) & (pos < end) & (pos < N)
    acc = tl.zeros((R, C), tl.float32)
    for k0 in range(0, F, K):
        k = k0 + tl.arange(0, K)
        act = tl.load(ACT + pos[:, None] * F + k[None, :],
                      mask=valid[:, None] & (k[None, :] < F), other=0.0)
        w = tl.load(W2 + e * F * D + k[:, None] * D + od[None, :],
                    mask=(e < E) & (k[:, None] < F) & (od[None, :] < D), other=0.0)
        if BF:
            ah = act.to(tl.bfloat16)
            al = (act - ah).to(tl.bfloat16)
            acc += tl.dot(ah, w) + tl.dot(al, w)
        else:
            acc += tl.dot(act, w.to(tl.float32), input_precision="tf32x3")
    tl.store(YP + pos[:, None] * D + od[None, :], acc,
             mask=valid[:, None] & (od[None, :] < D))


@triton.jit
def _combine(YP, INV, W, Y, T: tl.constexpr, D: tl.constexpr,
             A: tl.constexpr, C: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * C + tl.arange(0, C)
    acc = tl.zeros((C,), tl.float32)
    for a in range(0, A):
        p = tl.load(INV + t * A + a)
        wa = tl.load(W + t * A + a)
        v = tl.load(YP + p * D + d, mask=d < D, other=0.0)
        acc += wa * v
    tl.store(Y + t * D + d, acc, mask=d < D)


@triton.jit
def _dz_route(DY, YP, INV, W, DZ, GPAIR,
              T: tl.constexpr, D: tl.constexpr, A: tl.constexpr,
              C: tl.constexpr):
    j = tl.program_id(0)
    t = j // A
    p = tl.load(INV + j)
    d = tl.program_id(1) * C + tl.arange(0, C)
    dy = tl.load(DY + t * D + d, mask=d < D, other=0.0).to(tl.float32)
    yp = tl.load(YP + p * D + d, mask=d < D, other=0.0)
    wa = tl.load(W + j)
    tl.store(DZ + p * D + d, dy * wa, mask=d < D)
    part = tl.sum(dy * yp, axis=0)
    if tl.program_id(1) == 0:
        total = part
        for q in range(C, D, C):
            dd = q + tl.arange(0, C)
            yy = tl.load(DY + t * D + dd, mask=dd < D, other=0.0).to(tl.float32)
            ff = tl.load(YP + p * D + dd, mask=dd < D, other=0.0)
            total += tl.sum(yy * ff, axis=0)
        tl.store(GPAIR + j, total)


@triton.jit
def _dgu(DZ, GU, W2, OFF, PRE, DGU,
         N: tl.constexpr, F: tl.constexpr, D: tl.constexpr,
         E: tl.constexpr, BF: tl.constexpr,
         R: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    e, start = _segment_location(PRE, OFF, pid, E, R)
    ir = tl.arange(0, R)
    f = tl.program_id(1) * C + tl.arange(0, C)
    end = tl.load(OFF + tl.minimum(e + 1, E))
    pos = start + ir
    valid = (e < E) & (pos < end) & (pos < N)
    da = tl.zeros((R, C), tl.float32)
    for k0 in range(0, D, K):
        k = k0 + tl.arange(0, K)
        z = tl.load(DZ + pos[:, None] * D + k[None, :],
                    mask=valid[:, None] & (k[None, :] < D), other=0.0)
        w = tl.load(W2 + e * F * D + f[:, None] * D + k[None, :],
                    mask=(e < E) & (f[:, None] < F) & (k[None, :] < D), other=0.0)
        if BF:
            da += tl.dot(z.to(tl.bfloat16), tl.trans(w))
        else:
            da += tl.dot(z, tl.trans(w.to(tl.float32)), input_precision="tf32x3")
    g = tl.load(GU + pos[:, None] * (2 * F) + f[None, :],
                mask=valid[:, None] & (f[None, :] < F), other=0.0)
    u = tl.load(GU + pos[:, None] * (2 * F) + F + f[None, :],
                mask=valid[:, None] & (f[None, :] < F), other=0.0)
    sig = tl.sigmoid(g)
    silu = g * sig
    dg = da * u * (sig + silu * (1.0 - sig))
    du = da * silu
    tl.store(DGU + pos[:, None] * (2 * F) + f[None, :], dg,
             mask=valid[:, None] & (f[None, :] < F))
    tl.store(DGU + pos[:, None] * (2 * F) + F + f[None, :], du,
             mask=valid[:, None] & (f[None, :] < F))


@triton.jit
def _dx_rows(DGU, W1, OFF, PRE, DXR,
             N: tl.constexpr, F2: tl.constexpr, D: tl.constexpr,
             E: tl.constexpr, BF: tl.constexpr,
             R: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    e, start = _segment_location(PRE, OFF, pid, E, R)
    ir = tl.arange(0, R)
    d = tl.program_id(1) * C + tl.arange(0, C)
    end = tl.load(OFF + tl.minimum(e + 1, E))
    pos = start + ir
    valid = (e < E) & (pos < end) & (pos < N)
    acc = tl.zeros((R, C), tl.float32)
    for k0 in range(0, F2, K):
        k = k0 + tl.arange(0, K)
        g = tl.load(DGU + pos[:, None] * F2 + k[None, :],
                    mask=valid[:, None] & (k[None, :] < F2), other=0.0)
        w = tl.load(W1 + e * D * F2 + d[:, None] * F2 + k[None, :],
                    mask=(e < E) & (d[:, None] < D) & (k[None, :] < F2), other=0.0)
        if BF:
            acc += tl.dot(g.to(tl.bfloat16), tl.trans(w))
        else:
            acc += tl.dot(g, tl.trans(w.to(tl.float32)), input_precision="tf32x3")
    tl.store(DXR + pos[:, None] * D + d[None, :], acc,
             mask=valid[:, None] & (d[None, :] < D))


@triton.jit
def _dw2(ACT, DZ, OFF, GW2,
         N: tl.constexpr, F: tl.constexpr, D: tl.constexpr,
         E: tl.constexpr, BF: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, K: tl.constexpr):
    e = tl.program_id(0)
    f = tl.program_id(1) * BM + tl.arange(0, BM)
    d = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for p0 in range(begin, end, K):
        p = p0 + tl.arange(0, K)
        a = tl.load(ACT + p[:, None] * F + f[None, :],
                    mask=(p[:, None] < end) & (f[None, :] < F), other=0.0)
        z = tl.load(DZ + p[:, None] * D + d[None, :],
                    mask=(p[:, None] < end) & (d[None, :] < D), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16); al = (a - ah).to(tl.bfloat16)
            zh = z.to(tl.bfloat16); zl = (z - zh).to(tl.bfloat16)
            acc += tl.dot(tl.trans(ah), zh) + tl.dot(tl.trans(al), zh) + tl.dot(tl.trans(ah), zl)
        else:
            acc += tl.dot(tl.trans(a), z, input_precision="ieee")
    tl.store(GW2 + e * F * D + f[:, None] * D + d[None, :], acc,
             mask=(f[:, None] < F) & (d[None, :] < D))


@triton.jit
def _dw1(X, DGU, ORDER, OFF, GW1,
         N: tl.constexpr, D: tl.constexpr, F2: tl.constexpr,
         E: tl.constexpr, A: tl.constexpr, BF: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, K: tl.constexpr):
    e = tl.program_id(0)
    d = tl.program_id(1) * BM + tl.arange(0, BM)
    f = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for p0 in range(begin, end, K):
        p = p0 + tl.arange(0, K)
        j = tl.load(ORDER + p, mask=p < end, other=0)
        tok = j // A
        x = tl.load(X + tok[:, None] * D + d[None, :],
                    mask=(p[:, None] < end) & (d[None, :] < D), other=0.0)
        g = tl.load(DGU + p[:, None] * F2 + f[None, :],
                    mask=(p[:, None] < end) & (f[None, :] < F2), other=0.0)
        if BF:
            gh = g.to(tl.bfloat16); gl = (g - gh).to(tl.bfloat16)
            acc += tl.dot(tl.trans(x), gh) + tl.dot(tl.trans(x), gl)
        else:
            acc += tl.dot(tl.trans(x.to(tl.float32)), g, input_precision="ieee")
    tl.store(GW1 + e * D * F2 + d[:, None] * F2 + f[None, :], acc,
             mask=(d[:, None] < D) & (f[None, :] < F2))


@triton.jit
def _route_ds(S, SEL, W, GPAIR, DS,
              T: tl.constexpr, E: tl.constexpr, A: tl.constexpr,
              BA: tl.constexpr, BE: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BA)
    ma = a < A
    q = tl.load(S + t * E + tl.load(SEL + t * A + a, mask=ma, other=0), mask=ma, other=0.0)
    wa = tl.load(W + t * A + a, mask=ma, other=0.0)
    ga = tl.load(GPAIR + t * A + a, mask=ma, other=0.0)
    mean = tl.sum(wa * ga, axis=0)
    den = tl.sum(q, axis=0)
    ee = tl.arange(0, BE)
    out = tl.zeros((BE,), tl.float32)
    for i in range(0, A):
        ei = tl.load(SEL + t * A + i)
        qi = tl.load(S + t * E + ei)
        gi = tl.load(GPAIR + t * A + i)
        di = (gi - mean) / den
        out += tl.where(ee == ei, di * qi * (1.0 - qi), 0.0)
    tl.store(DS + t * E + ee, out, mask=ee < E)


@triton.jit
def _drw(X, DS, GRW, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
         BF: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, K: tl.constexpr):
    d = tl.program_id(0) * BM + tl.arange(0, BM)
    e = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for t0 in range(0, T, K):
        t = t0 + tl.arange(0, K)
        x = tl.load(X + t[:, None] * D + d[None, :],
                    mask=(t[:, None] < T) & (d[None, :] < D), other=0.0)
        s = tl.load(DS + t[:, None] * E + e[None, :],
                    mask=(t[:, None] < T) & (e[None, :] < E), other=0.0)
        if BF:
            sh = s.to(tl.bfloat16); sl = (s - sh).to(tl.bfloat16)
            acc += tl.dot(tl.trans(x), sh) + tl.dot(tl.trans(x), sl)
        else:
            acc += tl.dot(tl.trans(x.to(tl.float32)), s, input_precision="ieee")
    tl.store(GRW + d[:, None] * E + e[None, :], acc,
             mask=(d[:, None] < D) & (e[None, :] < E))


@triton.jit
def _dx_combine(DXR, INV, DS, RW, DX,
                T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                A: tl.constexpr, BF: tl.constexpr,
                C: tl.constexpr, K: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * C + tl.arange(0, C)
    acc = tl.zeros((C,), tl.float32)
    for a in range(0, A):
        p = tl.load(INV + t * A + a)
        acc += tl.load(DXR + p * D + d, mask=d < D, other=0.0)
    for e0 in range(0, E, K):
        e = e0 + tl.arange(0, K)
        s = tl.load(DS + t * E + e, mask=e < E, other=0.0)
        w = tl.load(RW + d[:, None] * E + e[None, :],
                    mask=(d[:, None] < D) & (e[None, :] < E), other=0.0)
        if BF:
            sh = s.to(tl.bfloat16); sl = (s - sh).to(tl.bfloat16)
            acc += tl.sum(w * (sh + sl)[None, :], axis=1)
        else:
            acc += tl.sum(w.to(tl.float32) * s[None, :], axis=1)
    tl.store(DX + t * D + d, acc, mask=d < D)


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        T, D = x.shape
        E = router_w.shape[1]
        F = w2.shape[1]
        N = T * A
        bf = x.dtype == torch.bfloat16
        s = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _router[(triton.cdiv(T, 32), triton.cdiv(E, 32))](x, router_w, s, T, D, E, 32, 32, 32)
        keys = torch.empty((T, E), device=x.device, dtype=torch.int64)
        _keys[(triton.cdiv(T * E, 256),)](s, router_b, keys, T, E, 256)
        sel = torch.topk(keys, A, dim=1, largest=True, sorted=True).indices.contiguous()
        w = torch.empty((T, A), device=x.device, dtype=torch.float32)
        _weights[(T,)](s, sel, w, T, E, A, triton.next_power_of_2(A))
        flat = sel.reshape(-1)
        order = torch.argsort(flat, stable=True)
        off = torch.empty(E + 1, device=x.device, dtype=torch.int32)
        _offsets[(E + 1,)](sel, order, off, N, E, A, num_warps=1)
        pre = torch.empty(E + 1, device=x.device, dtype=torch.int32)
        _prefix[(1,)](off, pre, E, triton.next_power_of_2(E), ROW, num_warps=1)
        inv = torch.empty(N, device=x.device, dtype=torch.int64)
        _inverse[(triton.cdiv(N, 256),)](order, inv, N, 256)
        cap = triton.cdiv(N, ROW) + E
        gu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _expert_in[(cap, triton.cdiv(2 * F, COL))](x, w1, sel, order, off, pre, gu,
            N, D, 2 * F, E, A, bf, ROW, COL, KB, num_warps=4, num_stages=3)
        act = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _activate[(triton.cdiv(N * F, 256),)](gu, act, N, F, 256)
        yp = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _expert_out[(cap, triton.cdiv(D, COL))](act, w2, off, pre, yp,
            N, F, D, E, bf, ROW, COL, KB, num_warps=4, num_stages=3)
        y = torch.empty((T, D), device=x.device, dtype=x.dtype)
        _combine[(T, triton.cdiv(D, 256))](yp, inv, w, y, T, D, A, 256)
        ctx.save_for_backward(x, router_w, w1, w2, s, sel, w, order, off, pre, inv, gu, act, yp)
        ctx.dims = (T, D, E, F, A, bf)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, rw, w1, w2, s, sel, w, order, off, pre, inv, gu, act, yp = ctx.saved_tensors
        T, D, E, F, A, bf = ctx.dims
        N = T * A
        cap = triton.cdiv(N, ROW) + E
        dy = dy.contiguous()
        dz = torch.empty((N, D), device=x.device, dtype=torch.float32)
        gp = torch.empty(N, device=x.device, dtype=torch.float32)
        _dz_route[(N, triton.cdiv(D, 256))](dy, yp, inv, w, dz, gp, T, D, A, 256)
        dgu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _dgu[(cap, triton.cdiv(F, COL))](dz, gu, w2, off, pre, dgu,
            N, F, D, E, bf, ROW, COL, KB, num_warps=4, num_stages=3)
        dxr = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _dx_rows[(cap, triton.cdiv(D, COL))](dgu, w1, off, pre, dxr,
            N, 2 * F, D, E, bf, ROW, COL, KB, num_warps=4, num_stages=3)
        gw2 = torch.empty_like(w2)
        _dw2[(E, triton.cdiv(F, 128), triton.cdiv(D, 64))](act, dz, off, gw2,
            N, F, D, E, bf, 128, 64, KB, num_warps=4, num_stages=3)
        gw1 = torch.empty_like(w1)
        _dw1[(E, triton.cdiv(D, 128), triton.cdiv(2 * F, 64))](x, dgu, order, off, gw1,
            N, D, 2 * F, E, A, bf, 128, 64, KB, num_warps=4, num_stages=3)
        ds = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _route_ds[(T,)](s, sel, w, gp, ds, T, E, A,
                        triton.next_power_of_2(A), triton.next_power_of_2(E))
        grw = torch.empty_like(rw)
        _drw[(triton.cdiv(D, 64), triton.cdiv(E, 32))](x, ds, grw,
            T, D, E, bf, 64, 32, KB, num_warps=4, num_stages=3)
        dx = torch.empty_like(x)
        _dx_combine[(T, triton.cdiv(D, 256))](dxr, inv, ds, rw, dx,
            T, D, E, A, bf, 256, KB)
        return dx, grw, None, gw1, gw2, None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
