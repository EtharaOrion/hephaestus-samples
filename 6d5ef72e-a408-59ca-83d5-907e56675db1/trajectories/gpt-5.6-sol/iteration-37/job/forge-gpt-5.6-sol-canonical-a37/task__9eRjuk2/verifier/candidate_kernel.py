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
    for k0 in range(0, D, BK):
        ik = k0 + tl.arange(0, BK)
        x = tl.load(X + im[:, None] * D + ik[None, :], mask=(im[:, None] < T) & (ik[None, :] < D), other=0.0).to(tl.float32)
        w = tl.load(RW + ik[:, None] * E + ie[None, :], mask=(ik[:, None] < D) & (ie[None, :] < E), other=0.0).to(tl.float32)
        z += tl.dot(x, w, input_precision="ieee")
    tl.store(S + im[:, None] * E + ie[None, :], tl.sigmoid(z), mask=(im[:, None] < T) & (ie[None, :] < E))


@triton.jit
def _route_weights(S, SEL, W, T: tl.constexpr, E: tl.constexpr, A: tl.constexpr, B: tl.constexpr = 256):
    t = tl.program_id(0)
    a = tl.arange(0, B)
    m = a < A
    e = tl.load(SEL + t * A + a, mask=m, other=0)
    q = tl.load(S + t * E + e, mask=m, other=0.0)
    tl.store(W + t * A + a, q / tl.sum(q, axis=0), mask=m)


@triton.jit
def _offsets(SEL, ORDER, OFF, N: tl.constexpr, E: tl.constexpr):
    e = tl.program_id(0)
    lo = 0
    hi = N
    for _ in range(24):
        mid = (lo + hi) // 2
        p = tl.load(ORDER + mid, mask=mid < N, other=N - 1)
        v = tl.load(SEL + p, mask=mid < N, other=E)
        take = (mid < N) & (v < e)
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(take, hi, mid)
    tl.store(OFF + e, lo)
    if e == E - 1:
        tl.store(OFF + E, N)


@triton.jit
def _prefix(OFF, PRE, E: tl.constexpr, R: tl.constexpr):
    total = 0
    tl.store(PRE, 0)
    for e in range(0, E):
        b = tl.load(OFF + e)
        en = tl.load(OFF + e + 1)
        total += tl.cdiv(en - b, R)
        tl.store(PRE + e + 1, total)


@triton.jit
def _location(pid, PRE, OFF, E: tl.constexpr, R: tl.constexpr):
    lo = 0
    hi = E
    for _ in range(8):
        mid = (lo + hi) // 2
        v = tl.load(PRE + mid + 1, mask=mid < E, other=0x7fffffff)
        take = v <= pid
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(take, hi, mid)
    valid = lo < E
    e = tl.where(valid, lo, E - 1)
    first = tl.load(PRE + e)
    end = tl.load(OFF + e + 1)
    start = tl.where(valid, tl.load(OFF + e) + (pid - first) * R, end)
    return e, start, end


@triton.jit
def _expert_in(X, W1, ORDER, OFF, PRE, GU, D: tl.constexpr, F2: tl.constexpr,
               E: tl.constexpr, A: tl.constexpr, R: tl.constexpr,
               C: tl.constexpr, K: tl.constexpr, BF: tl.constexpr):
    pid = tl.program_id(0)
    e, start, end = _location(pid, PRE, OFF, E, R)
    rr = start + tl.arange(0, R)
    cc = tl.program_id(1) * C + tl.arange(0, C)
    p = tl.load(ORDER + rr, mask=rr < end, other=0)
    t = p // A
    z = tl.zeros((R, C), tl.float32)
    for k0 in range(0, D, K):
        kk = k0 + tl.arange(0, K)
        x = tl.load(X + t[:, None] * D + kk[None, :], mask=(rr[:, None] < end) & (kk[None, :] < D), other=0.0).to(tl.float32)
        w = tl.load(W1 + e * D * F2 + kk[:, None] * F2 + cc[None, :], mask=(kk[:, None] < D) & (cc[None, :] < F2), other=0.0).to(tl.float32)
        if BF:
            z += tl.dot(x.to(tl.bfloat16), w.to(tl.bfloat16))
        else:
            z += tl.dot(x, w, input_precision="tf32x3")
    tl.store(GU + p[:, None] * F2 + cc[None, :], z, mask=(rr[:, None] < end) & (cc[None, :] < F2))


@triton.jit
def _activate(GU, ACT, N: tl.constexpr, F: tl.constexpr, B: tl.constexpr = 256):
    o = tl.program_id(0) * B + tl.arange(0, B)
    m = o < N * F
    p = o // F
    f = o - p * F
    g = tl.load(GU + p * (2 * F) + f, mask=m, other=0.0)
    u = tl.load(GU + p * (2 * F) + F + f, mask=m, other=0.0)
    tl.store(ACT + o, g * tl.sigmoid(g) * u, mask=m)


@triton.jit
def _segment(AIN, WB, ORDER, OFF, PRE, OUT, KIN: tl.constexpr, NOUT: tl.constexpr,
             E: tl.constexpr, R: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
             TRANS_B: tl.constexpr, BF: tl.constexpr):
    pid = tl.program_id(0)
    e, start, end = _location(pid, PRE, OFF, E, R)
    rr = start + tl.arange(0, R)
    cc = tl.program_id(1) * C + tl.arange(0, C)
    p = tl.load(ORDER + rr, mask=rr < end, other=0)
    z = tl.zeros((R, C), tl.float32)
    for k0 in range(0, KIN, K):
        kk = k0 + tl.arange(0, K)
        a = tl.load(AIN + p[:, None] * KIN + kk[None, :], mask=(rr[:, None] < end) & (kk[None, :] < KIN), other=0.0).to(tl.float32)
        if TRANS_B:
            b = tl.load(WB + e * NOUT * KIN + cc[None, :] * KIN + kk[:, None], mask=(cc[None, :] < NOUT) & (kk[:, None] < KIN), other=0.0).to(tl.float32)
        else:
            b = tl.load(WB + e * KIN * NOUT + kk[:, None] * NOUT + cc[None, :], mask=(kk[:, None] < KIN) & (cc[None, :] < NOUT), other=0.0).to(tl.float32)
        if BF:
            if TRANS_B:
                z += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
            else:
                ah = a.to(tl.bfloat16)
                al = (a - ah).to(tl.bfloat16)
                bh = b.to(tl.bfloat16)
                z += tl.dot(ah, bh) + tl.dot(al, bh)
        else:
            z += tl.dot(a, b, input_precision="tf32x3")
    tl.store(OUT + p[:, None] * NOUT + cc[None, :], z, mask=(rr[:, None] < end) & (cc[None, :] < NOUT))


@triton.jit
def _combine(Z, W, Y, T: tl.constexpr, D: tl.constexpr, A: tl.constexpr, B: tl.constexpr = 256):
    t = tl.program_id(0)
    d = tl.program_id(1) * B + tl.arange(0, B)
    z = tl.zeros((B,), tl.float32)
    for a in range(0, A):
        z += tl.load(W + t * A + a) * tl.load(Z + (t * A + a) * D + d, mask=d < D, other=0.0)
    tl.store(Y + t * D + d, z, mask=d < D)


@triton.jit
def _dz_pair(DY, Z, W, DZ, DOT, D: tl.constexpr, A: tl.constexpr, B: tl.constexpr = 256):
    p = tl.program_id(0)
    d = tl.arange(0, B)
    t = p // A
    q = tl.load(W + p)
    acc = 0.0
    for d0 in range(0, D, B):
        dd = d0 + d
        dy = tl.load(DY + t * D + dd, mask=dd < D, other=0.0).to(tl.float32)
        zv = tl.load(Z + p * D + dd, mask=dd < D, other=0.0)
        tl.store(DZ + p * D + dd, q * dy, mask=dd < D)
        acc += tl.sum(dy * zv, axis=0)
    tl.store(DOT + p, acc)


@triton.jit
def _route_grad(S, SEL, W, DOT, DL, E: tl.constexpr, A: tl.constexpr, B: tl.constexpr = 256):
    t = tl.program_id(0)
    a = tl.arange(0, B)
    m = a < A
    p = t * A + a
    e = tl.load(SEL + p, mask=m, other=0)
    s = tl.load(S + t * E + e, mask=m, other=0.0)
    w = tl.load(W + p, mask=m, other=0.0)
    q = tl.load(DOT + p, mask=m, other=0.0)
    mean = tl.sum(w * q, axis=0)
    ds = (q - mean) / tl.sum(s, axis=0)
    tl.store(DL + p, ds * s * (1.0 - s), mask=m)


@triton.jit
def _dgu(GU, DACT, DGU, N: tl.constexpr, F: tl.constexpr, B: tl.constexpr = 256):
    o = tl.program_id(0) * B + tl.arange(0, B)
    m = o < N * F
    p = o // F
    f = o - p * F
    g = tl.load(GU + p * (2 * F) + f, mask=m, other=0.0)
    u = tl.load(GU + p * (2 * F) + F + f, mask=m, other=0.0)
    da = tl.load(DACT + o, mask=m, other=0.0)
    s = tl.sigmoid(g)
    sg = g * s
    tl.store(DGU + p * (2 * F) + f, da * u * (s + sg * (1.0 - s)), mask=m)
    tl.store(DGU + p * (2 * F) + F + f, da * sg, mask=m)


@triton.jit
def _dw(AIN, BGRAD, ORDER, OFF, OUT, KIN: tl.constexpr, NOUT: tl.constexpr,
        E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        A_IS_X: tl.constexpr, ASEL: tl.constexpr, BF: tl.constexpr):
    e = tl.program_id(0)
    mm = tl.program_id(1) * BM + tl.arange(0, BM)
    nn = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    z = tl.zeros((BM, BN), tl.float32)
    for r0 in range(begin, end, BK):
        rr = r0 + tl.arange(0, BK)
        p = tl.load(ORDER + rr, mask=rr < end, other=0)
        row = p // ASEL if A_IS_X else p
        a = tl.load(AIN + row[None, :] * KIN + mm[:, None], mask=(rr[None, :] < end) & (mm[:, None] < KIN), other=0.0).to(tl.float32)
        b = tl.load(BGRAD + p[:, None] * NOUT + nn[None, :], mask=(rr[:, None] < end) & (nn[None, :] < NOUT), other=0.0).to(tl.float32)
        if BF:
            ah = a.to(tl.bfloat16)
            bh = b.to(tl.bfloat16)
            bl = (b - bh).to(tl.bfloat16)
            z += tl.dot(ah, bh) + tl.dot(ah, bl)
            if not A_IS_X:
                z += tl.dot((a - ah).to(tl.bfloat16), bh)
        else:
            z += tl.dot(a, b, input_precision="tf32x3")
    tl.store(OUT + e * KIN * NOUT + mm[:, None] * NOUT + nn[None, :], z, mask=(mm[:, None] < KIN) & (nn[None, :] < NOUT))


@triton.jit
def _drw(X, DL, ORDER, OFF, DRW, D: tl.constexpr, E: tl.constexpr,
         A: tl.constexpr, BD: tl.constexpr = 64, BK: tl.constexpr = 32):
    e = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    z = tl.zeros((BD,), tl.float32)
    for r0 in range(begin, end, BK):
        rr = r0 + tl.arange(0, BK)
        p = tl.load(ORDER + rr, mask=rr < end, other=0)
        t = p // A
        x = tl.load(X + t[None, :] * D + d[:, None], mask=(rr[None, :] < end) & (d[:, None] < D), other=0.0).to(tl.float32)
        q = tl.load(DL + p, mask=rr < end, other=0.0)
        z += tl.sum(x * q[None, :], axis=1)
    tl.store(DRW + d * E + e, z, mask=d < D)


@triton.jit
def _sum_dx(DXP, DL, SEL, RW, DX, D: tl.constexpr, E: tl.constexpr,
            A: tl.constexpr, B: tl.constexpr = 256):
    t = tl.program_id(0)
    d = tl.program_id(1) * B + tl.arange(0, B)
    z = tl.zeros((B,), tl.float32)
    for a in range(0, A):
        p = t * A + a
        e = tl.load(SEL + p)
        v = tl.load(DXP + p * D + d, mask=d < D, other=0.0)
        q = tl.load(DL + p)
        rw = tl.load(RW + d * E + e, mask=d < D, other=0.0).to(tl.float32)
        z += v + q * rw
    tl.store(DX + t * D + d, z, mask=d < D)


def _selection(s, bias, A):
    e = s.shape[1]
    b = torch.where(s + bias == 0.0, torch.zeros_like(s), s + bias).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    idx = torch.arange(e, device=s.device, dtype=torch.int64)
    return torch.topk((u << 10) | (e - 1 - idx), A, dim=1, largest=True, sorted=True).indices


def _metadata(sel, E):
    n = sel.numel()
    flat = sel.reshape(-1)
    order = torch.argsort(flat)
    off = torch.empty(E + 1, device=sel.device, dtype=torch.int32)
    _offsets[(E,)](flat, order, off, N=n, E=E, num_warps=1)
    pre = torch.empty(E + 1, device=sel.device, dtype=torch.int32)
    _prefix[(1,)](off, pre, E=E, R=ROW, num_warps=1)
    return order, off, pre, triton.cdiv(n, ROW) + E


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rw, rb, w1, w2, A):
        A = int(A)
        T, D = x.shape
        E = rw.shape[1]
        F = w2.shape[1]
        N = T * A
        bf = x.dtype == torch.bfloat16
        s = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _router[(triton.cdiv(T, 32), triton.cdiv(E, 32))](x, rw, s, T=T, D=D, E=E)
        with torch.no_grad():
            sel = _selection(s, rb, A)
            order, off, pre, cap = _metadata(sel, E)
        weights = torch.empty(N, device=x.device, dtype=torch.float32)
        _route_weights[(T,)](s, sel, weights, T=T, E=E, A=A)
        gu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _expert_in[(cap, triton.cdiv(2 * F, COL))](x, w1, order, off, pre, gu, D=D, F2=2 * F, E=E, A=A, R=ROW, C=COL, K=KB, BF=bf, num_warps=4, num_stages=3)
        act = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _activate[(triton.cdiv(N * F, 256),)](gu, act, N=N, F=F)
        z = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _segment[(cap, triton.cdiv(D, COL))](act, w2, order, off, pre, z, KIN=F, NOUT=D, E=E, R=ROW, C=COL, K=KB, TRANS_B=False, BF=bf, num_warps=4, num_stages=3)
        y = torch.empty((T, D), device=x.device, dtype=x.dtype)
        _combine[(T, triton.cdiv(D, 256))](z, weights, y, T=T, D=D, A=A)
        ctx.save_for_backward(x, rw, w1, w2, s, sel, order, off, pre, weights, gu, act, z)
        ctx.shape = T, D, E, F, A, cap
        return y

    @staticmethod
    def backward(ctx, dy):
        x, rw, w1, w2, s, sel, order, off, pre, weights, gu, act, z = ctx.saved_tensors
        T, D, E, F, A, cap = ctx.shape
        N = T * A
        bf = x.dtype == torch.bfloat16
        dy = dy.contiguous()
        dz = torch.empty((N, D), device=x.device, dtype=torch.float32)
        dot = torch.empty(N, device=x.device, dtype=torch.float32)
        _dz_pair[(N,)](dy, z, weights, dz, dot, D=D, A=A)
        dl = torch.empty(N, device=x.device, dtype=torch.float32)
        _route_grad[(T,)](s, sel, weights, dot, dl, E=E, A=A)
        dact = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _segment[(cap, triton.cdiv(F, COL))](dz, w2, order, off, pre, dact, KIN=D, NOUT=F, E=E, R=ROW, C=COL, K=KB, TRANS_B=True, BF=bf, num_warps=4, num_stages=3)
        dgu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _dgu[(triton.cdiv(N * F, 256),)](gu, dact, dgu, N=N, F=F)
        dxp = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _segment[(cap, triton.cdiv(D, COL))](dgu, w1, order, off, pre, dxp, KIN=2 * F, NOUT=D, E=E, R=ROW, C=COL, K=KB, TRANS_B=True, BF=bf, num_warps=4, num_stages=3)
        gw2 = torch.empty_like(w2)
        _dw[(E, triton.cdiv(F, 128), triton.cdiv(D, 64))](act, dz, order, off, gw2, KIN=F, NOUT=D, E=E, BM=128, BN=64, BK=KB, A_IS_X=False, ASEL=A, BF=bf, num_warps=4, num_stages=3)
        gw1 = torch.empty_like(w1)
        _dw[(E, triton.cdiv(D, 128), triton.cdiv(2 * F, 64))](x, dgu, order, off, gw1, KIN=D, NOUT=2 * F, E=E, BM=128, BN=64, BK=KB, A_IS_X=True, ASEL=A, BF=bf, num_warps=4, num_stages=3)
        grw = torch.empty_like(rw)
        _drw[(E, triton.cdiv(D, 64))](x, dl, order, off, grw, D=D, E=E, A=A)
        gx = torch.empty_like(x)
        _sum_dx[(T, triton.cdiv(D, 256))](dxp, dl, sel, rw, gx, D=D, E=E, A=A)
        return gx, grw, None, gw1, gw2, None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
