import torch
import triton
import triton.language as tl

ROW = 64
COL = 64
KB = 32


@triton.jit
def _router(X, RW, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    z = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * D + rk[None, :],
                    mask=(rm[:, None] < T) & (rk[None, :] < D), other=0.0)
        w = tl.load(RW + rk[:, None] * E + rn[None, :],
                    mask=(rk[:, None] < D) & (rn[None, :] < E), other=0.0)
        z += tl.dot(x.to(tl.float32), w.to(tl.float32), input_precision="ieee")
    tl.store(S + rm[:, None] * E + rn[None, :], tl.sigmoid(z),
             mask=(rm[:, None] < T) & (rn[None, :] < E))


@triton.jit
def _offsets(FLAT, ORDER, OFF, N: tl.constexpr, E: tl.constexpr):
    e = tl.program_id(0)
    lo = 0
    hi = N
    for _ in range(24):
        mid = (lo + hi) // 2
        q = tl.load(ORDER + mid, mask=mid < N, other=0)
        v = tl.load(FLAT + q, mask=mid < N, other=E)
        take = v < e
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(take, hi, mid)
    tl.store(OFF + e, lo)


@triton.jit
def _prefix(OFF, PRE, E: tl.constexpr, R: tl.constexpr):
    total = 0
    tl.store(PRE, 0)
    for e in range(0, E):
        a = tl.load(OFF + e)
        b = tl.load(OFF + e + 1)
        total += tl.cdiv(b - a, R)
        tl.store(PRE + e + 1, total)


@triton.jit
def _inverse(ORDER, INV, N: tl.constexpr, B: tl.constexpr):
    p = tl.program_id(0) * B + tl.arange(0, B)
    m = p < N
    q = tl.load(ORDER + p, mask=m)
    tl.store(INV + q, p, mask=m)


@triton.jit
def _location(PRE, OFF, pid, E: tl.constexpr, R: tl.constexpr):
    lo = 0
    hi = E
    for _ in range(9):
        mid = (lo + hi) // 2
        v = tl.load(PRE + mid + 1, mask=mid < E, other=0x7fffffff)
        left = pid < v
        hi = tl.where(left, mid, hi)
        lo = tl.where(left, lo, mid + 1)
    e = lo
    valid = (e < E) & (pid < tl.load(PRE + E))
    start = tl.load(OFF + e, mask=valid, other=0) + (pid - tl.load(PRE + e, mask=valid, other=0)) * R
    end = tl.load(OFF + e + 1, mask=valid, other=0)
    return e, start, end, valid


@triton.jit
def _expert_in(X, W1, ORDER, OFF, PRE, GU, N: tl.constexpr, D: tl.constexpr,
               F2: tl.constexpr, A: tl.constexpr, E: tl.constexpr,
               BF: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
               K: tl.constexpr):
    pid = tl.program_id(0)
    e, start, end, valid = _location(PRE, OFF, pid, E, R)
    rr = start + tl.arange(0, R)
    cc = tl.program_id(1) * C + tl.arange(0, C)
    q = tl.load(ORDER + rr, mask=valid & (rr < end), other=0)
    tok = q // A
    z = tl.zeros((R, C), tl.float32)
    for k0 in range(0, D, K):
        kk = k0 + tl.arange(0, K)
        x = tl.load(X + tok[:, None] * D + kk[None, :],
                    mask=valid & (rr[:, None] < end) & (kk[None, :] < D), other=0.0)
        w = tl.load(W1 + e * D * F2 + kk[:, None] * F2 + cc[None, :],
                    mask=valid & (kk[:, None] < D) & (cc[None, :] < F2), other=0.0)
        if BF:
            z += tl.dot(x, w)
        else:
            z += tl.dot(x, w, input_precision="tf32x3")
    tl.store(GU + rr[:, None] * F2 + cc[None, :], z,
             mask=valid & (rr[:, None] < end) & (cc[None, :] < F2))


@triton.jit
def _activate(GU, ACT, N: tl.constexpr, F: tl.constexpr, B: tl.constexpr):
    o = tl.program_id(0) * B + tl.arange(0, B)
    m = o < N * F
    r = o // F
    c = o - r * F
    g = tl.load(GU + r * (2 * F) + c, mask=m, other=0.0)
    u = tl.load(GU + r * (2 * F) + F + c, mask=m, other=0.0)
    tl.store(ACT + o, g * tl.sigmoid(g) * u, mask=m)


@triton.jit
def _segment(AIN, W, OFF, PRE, OUT, N: tl.constexpr, KSIZE: tl.constexpr,
             O: tl.constexpr, E: tl.constexpr, BF: tl.constexpr,
             SPLIT: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
             K: tl.constexpr):
    pid = tl.program_id(0)
    e, start, end, valid = _location(PRE, OFF, pid, E, R)
    rr = start + tl.arange(0, R)
    cc = tl.program_id(1) * C + tl.arange(0, C)
    z = tl.zeros((R, C), tl.float32)
    for k0 in range(0, KSIZE, K):
        kk = k0 + tl.arange(0, K)
        a = tl.load(AIN + rr[:, None] * KSIZE + kk[None, :],
                    mask=valid & (rr[:, None] < end) & (kk[None, :] < KSIZE), other=0.0)
        w = tl.load(W + e * KSIZE * O + kk[:, None] * O + cc[None, :],
                    mask=valid & (kk[:, None] < KSIZE) & (cc[None, :] < O), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            if SPLIT:
                al = (a - ah).to(tl.bfloat16)
                z += tl.dot(ah, w) + tl.dot(al, w)
            else:
                z += tl.dot(ah, w)
        else:
            z += tl.dot(a, w, input_precision="tf32x3")
    tl.store(OUT + rr[:, None] * O + cc[None, :], z,
             mask=valid & (rr[:, None] < end) & (cc[None, :] < O))


@triton.jit
def _combine(S, SEL, INV, FE, Y, T: tl.constexpr, D: tl.constexpr,
             E: tl.constexpr, A: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1) * B + tl.arange(0, B)
    den = 0.0
    for a in range(0, A):
        e = tl.load(SEL + t * A + a)
        den += tl.load(S + t * E + e)
    z = tl.zeros((B,), tl.float32)
    for a in range(0, A):
        e = tl.load(SEL + t * A + a)
        s = tl.load(S + t * E + e)
        p = tl.load(INV + t * A + a)
        f = tl.load(FE + p * D + c, mask=c < D, other=0.0)
        z += (s / den) * f
    tl.store(Y + t * D + c, z, mask=c < D)


@triton.jit
def _dfe(DY, S, FLAT, ORDER, DFE, N: tl.constexpr, D: tl.constexpr,
         E: tl.constexpr, A: tl.constexpr, B: tl.constexpr):
    p = tl.program_id(0)
    c = tl.program_id(1) * B + tl.arange(0, B)
    q = tl.load(ORDER + p)
    t = q // A
    den = 0.0
    for a in range(0, A):
        ee = tl.load(FLAT + t * A + a)
        den += tl.load(S + t * E + ee)
    e = tl.load(FLAT + q)
    w = tl.load(S + t * E + e) / den
    dy = tl.load(DY + t * D + c, mask=c < D, other=0.0)
    tl.store(DFE + p * D + c, dy * w, mask=c < D)


@triton.jit
def _segment_bt(AIN, W, OFF, PRE, OUT, N: tl.constexpr, KSIZE: tl.constexpr,
                O: tl.constexpr, E: tl.constexpr, BF: tl.constexpr,
                R: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    e, start, end, valid = _location(PRE, OFF, pid, E, R)
    rr = start + tl.arange(0, R)
    cc = tl.program_id(1) * C + tl.arange(0, C)
    z = tl.zeros((R, C), tl.float32)
    for k0 in range(0, KSIZE, K):
        kk = k0 + tl.arange(0, K)
        a = tl.load(AIN + rr[:, None] * KSIZE + kk[None, :],
                    mask=valid & (rr[:, None] < end) & (kk[None, :] < KSIZE), other=0.0)
        w = tl.load(W + e * O * KSIZE + cc[None, :] * KSIZE + kk[:, None],
                    mask=valid & (cc[None, :] < O) & (kk[:, None] < KSIZE), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            al = (a - ah).to(tl.bfloat16)
            z += tl.dot(ah, w) + tl.dot(al, w)
        else:
            z += tl.dot(a, w, input_precision="tf32x3")
    tl.store(OUT + rr[:, None] * O + cc[None, :], z,
             mask=valid & (rr[:, None] < end) & (cc[None, :] < O))


@triton.jit
def _dactivate(GU, DA, DGU, N: tl.constexpr, F: tl.constexpr, B: tl.constexpr):
    o = tl.program_id(0) * B + tl.arange(0, B)
    m = o < N * F
    r = o // F
    c = o - r * F
    g = tl.load(GU + r * 2 * F + c, mask=m, other=0.0)
    u = tl.load(GU + r * 2 * F + F + c, mask=m, other=0.0)
    da = tl.load(DA + o, mask=m, other=0.0)
    s = tl.sigmoid(g)
    silu = g * s
    tl.store(DGU + r * 2 * F + c, da * u * (s + silu * (1.0 - s)), mask=m)
    tl.store(DGU + r * 2 * F + F + c, da * silu, mask=m)


@triton.jit
def _dw(AIN, BIN, ORDER, OFF, OUT, KSIZE: tl.constexpr, O: tl.constexpr,
        E: tl.constexpr, A: tl.constexpr, GATHER_A: tl.constexpr,
        A_BF_EXACT: tl.constexpr, BF: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(0)
    rm = tl.program_id(1) * BM + tl.arange(0, BM)
    rn = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    z = tl.zeros((BM, BN), tl.float32)
    for p0 in range(begin, end, BK):
        p = p0 + tl.arange(0, BK)
        if GATHER_A:
            q = tl.load(ORDER + p, mask=p < end, other=0)
            ar = q // A
        else:
            ar = p
        a = tl.load(AIN + ar[None, :] * KSIZE + rm[:, None],
                    mask=(p[None, :] < end) & (rm[:, None] < KSIZE), other=0.0)
        b = tl.load(BIN + p[:, None] * O + rn[None, :],
                    mask=(p[:, None] < end) & (rn[None, :] < O), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            bh = b.to(tl.bfloat16)
            al = (a - ah).to(tl.bfloat16)
            bl = (b - bh).to(tl.bfloat16)
            z += tl.dot(ah, bh) + tl.dot(ah, bl) + tl.dot(al, bh)
        else:
            z += tl.dot(a, b, input_precision="tf32x3")
    tl.store(OUT + e * KSIZE * O + rm[:, None] * O + rn[None, :], z,
             mask=(rm[:, None] < KSIZE) & (rn[None, :] < O))


@triton.jit
def _route(DY, FE, S, SEL, INV, DL, T: tl.constexpr, D: tl.constexpr,
           E: tl.constexpr, A: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0)
    lanes = tl.arange(0, B)
    den = 0.0
    mean = 0.0
    for a in range(0, A):
        e = tl.load(SEL + t * A + a)
        s = tl.load(S + t * E + e)
        p = tl.load(INV + t * A + a)
        q = tl.sum(tl.load(DY + t * D + lanes, mask=lanes < D, other=0.0) *
                   tl.load(FE + p * D + lanes, mask=lanes < D, other=0.0))
        den += s
        mean += s * q
    mean /= den
    for a in range(0, A):
        e = tl.load(SEL + t * A + a)
        s = tl.load(S + t * E + e)
        p = tl.load(INV + t * A + a)
        q = tl.sum(tl.load(DY + t * D + lanes, mask=lanes < D, other=0.0) *
                   tl.load(FE + p * D + lanes, mask=lanes < D, other=0.0))
        tl.store(DL + t * A + a, ((q - mean) / den) * s * (1.0 - s))


@triton.jit
def _dx_rows(DGU, W1, OFF, PRE, DXP, N: tl.constexpr, D: tl.constexpr,
             F2: tl.constexpr, E: tl.constexpr, BF: tl.constexpr,
             R: tl.constexpr, C: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    e, start, end, valid = _location(PRE, OFF, pid, E, R)
    rr = start + tl.arange(0, R)
    cc = tl.program_id(1) * C + tl.arange(0, C)
    z = tl.zeros((R, C), tl.float32)
    for k0 in range(0, F2, K):
        kk = k0 + tl.arange(0, K)
        a = tl.load(DGU + rr[:, None] * F2 + kk[None, :],
                    mask=valid & (rr[:, None] < end) & (kk[None, :] < F2), other=0.0)
        w = tl.load(W1 + e * D * F2 + cc[None, :] * F2 + kk[:, None],
                    mask=valid & (cc[None, :] < D) & (kk[:, None] < F2), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            al = (a - ah).to(tl.bfloat16)
            z += tl.dot(ah, w) + tl.dot(al, w)
        else:
            z += tl.dot(a, w, input_precision="tf32x3")
    tl.store(DXP + rr[:, None] * D + cc[None, :], z,
             mask=valid & (rr[:, None] < end) & (cc[None, :] < D))


@triton.jit
def _sum_dx(DXP, INV, DL, RW, DX, T: tl.constexpr, D: tl.constexpr,
            E: tl.constexpr, A: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1) * B + tl.arange(0, B)
    z = tl.zeros((B,), tl.float32)
    for a in range(0, A):
        p = tl.load(INV + t * A + a)
        e = tl.load(INV + 0)  # overwritten below without extra metadata
        z += tl.load(DXP + p * D + c, mask=c < D, other=0.0)
    tl.store(DX + t * D + c, z, mask=c < D)


@triton.jit
def _sum_dx_route(DXP, INV, SEL, DL, RW, DX, T: tl.constexpr, D: tl.constexpr,
                  E: tl.constexpr, A: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1) * B + tl.arange(0, B)
    z = tl.zeros((B,), tl.float32)
    for a in range(0, A):
        p = tl.load(INV + t * A + a)
        e = tl.load(SEL + t * A + a)
        z += tl.load(DXP + p * D + c, mask=c < D, other=0.0)
        z += tl.load(DL + t * A + a) * tl.load(RW + c * E + e, mask=c < D, other=0.0)
    tl.store(DX + t * D + c, z, mask=c < D)


@triton.jit
def _drw(X, DL, SEL, OUT, T: tl.constexpr, D: tl.constexpr,
         E: tl.constexpr, A: tl.constexpr, BM: tl.constexpr,
         BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    z = tl.zeros((BM, BN), tl.float32)
    for t0 in range(0, T, BK):
        tt = t0 + tl.arange(0, BK)
        x = tl.load(X + tt[None, :] * D + rm[:, None],
                    mask=(tt[None, :] < T) & (rm[:, None] < D), other=0.0)
        for a in range(0, A):
            ee = tl.load(SEL + tt[:, None] * A + a,
                         mask=(tt[:, None] < T) & (rn[None, :] < E), other=-1)
            dl = tl.load(DL + tt[:, None] * A + a,
                         mask=(tt[:, None] < T) & (rn[None, :] < E), other=0.0)
            b = tl.where(ee == rn[None, :], dl, 0.0)
            z += tl.dot(x.to(tl.float32), b.to(tl.float32), input_precision="ieee")
    tl.store(OUT + rm[:, None] * E + rn[None, :], z,
             mask=(rm[:, None] < D) & (rn[None, :] < E))


def _selection(s, bias, a):
    e = s.shape[1]
    b = s + bias
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xffffffff)
    idx = torch.arange(e, device=s.device, dtype=torch.int64)
    return torch.topk((u << 10) | (e - 1 - idx), a, dim=1,
                      largest=True, sorted=True).indices


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rw, rb, w1, w2, a):
        a = int(a)
        t, d = x.shape
        e = rw.shape[1]
        f = w1.shape[2] // 2
        n = t * a
        bf = x.dtype == torch.bfloat16
        s = torch.empty((t, e), device=x.device, dtype=torch.float32)
        _router[(triton.cdiv(t, 32), triton.cdiv(e, 32))](x, rw, s, T=t, D=d, E=e,
                                                          BM=32, BN=32, BK=32)
        with torch.no_grad():
            sel = _selection(s, rb, a).contiguous()
            flat = sel.reshape(-1)
            order = torch.argsort(flat)
        off = torch.empty(e + 1, device=x.device, dtype=torch.int32)
        pre = torch.empty(e + 1, device=x.device, dtype=torch.int32)
        inv = torch.empty(n, device=x.device, dtype=torch.int64)
        _offsets[(e + 1,)](flat, order, off, N=n, E=e, num_warps=1)
        _prefix[(1,)](off, pre, E=e, R=ROW, num_warps=1)
        _inverse[(triton.cdiv(n, 256),)](order, inv, N=n, B=256)
        cap = triton.cdiv(n, ROW) + e
        gu = torch.empty((n, 2 * f), device=x.device, dtype=torch.float32)
        act = torch.empty((n, f), device=x.device, dtype=torch.float32)
        fe = torch.empty((n, d), device=x.device, dtype=torch.float32)
        _expert_in[(cap, triton.cdiv(2 * f, COL))](x, w1, order, off, pre, gu,
            N=n, D=d, F2=2 * f, A=a, E=e, BF=bf, R=ROW, C=COL, K=KB,
            num_warps=4, num_stages=3)
        _activate[(triton.cdiv(n * f, 256),)](gu, act, N=n, F=f, B=256)
        _segment[(cap, triton.cdiv(d, COL))](act, w2, off, pre, fe,
            N=n, KSIZE=f, O=d, E=e, BF=bf, SPLIT=bf, R=ROW, C=COL, K=KB,
            num_warps=4, num_stages=3)
        y = torch.empty((t, d), device=x.device, dtype=x.dtype)
        _combine[(t, triton.cdiv(d, 256))](s, sel, inv, fe, y,
                                            T=t, D=d, E=e, A=a, B=256)
        ctx.save_for_backward(x, rw, w1, w2, s, sel, flat, order, off, pre, inv, gu, act, fe)
        ctx.shape = (t, d, e, f, a, cap, bf)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, rw, w1, w2, s, sel, flat, order, off, pre, inv, gu, act, fe = ctx.saved_tensors
        t, d, e, f, a, cap, bf = ctx.shape
        n = t * a
        dy = dy.contiguous()
        dfe = torch.empty((n, d), device=x.device, dtype=torch.float32)
        da = torch.empty((n, f), device=x.device, dtype=torch.float32)
        dgu = torch.empty((n, 2 * f), device=x.device, dtype=torch.float32)
        dxp = torch.empty((n, d), device=x.device, dtype=torch.float32)
        dl = torch.empty((t, a), device=x.device, dtype=torch.float32)
        _dfe[(n, triton.cdiv(d, 256))](dy, s, flat, order, dfe,
                                        N=n, D=d, E=e, A=a, B=256)
        _segment_bt[(cap, triton.cdiv(f, COL))](dfe, w2, off, pre, da,
            N=n, KSIZE=d, O=f, E=e, BF=bf, R=ROW, C=COL, K=KB,
            num_warps=4, num_stages=3)
        _dactivate[(triton.cdiv(n * f, 256),)](gu, da, dgu, N=n, F=f, B=256)
        gw2 = torch.empty_like(w2)
        gw1 = torch.empty_like(w1)
        _dw[(e, triton.cdiv(f, 128), triton.cdiv(d, 64))](act, dfe, order, off, gw2,
            KSIZE=f, O=d, E=e, A=a, GATHER_A=False, A_BF_EXACT=False,
            BF=bf, BM=128, BN=64, BK=32, num_warps=4, num_stages=3)
        _dw[(e, triton.cdiv(d, 128), triton.cdiv(2 * f, 64))](x, dgu, order, off, gw1,
            KSIZE=d, O=2 * f, E=e, A=a, GATHER_A=True, A_BF_EXACT=bf,
            BF=bf, BM=128, BN=64, BK=32, num_warps=4, num_stages=3)
        _route[(t,)](dy, fe, s, sel, inv, dl, T=t, D=d, E=e, A=a,
                     B=triton.next_power_of_2(d), num_warps=8)
        _dx_rows[(cap, triton.cdiv(d, COL))](dgu, w1, off, pre, dxp,
            N=n, D=d, F2=2 * f, E=e, BF=bf, R=ROW, C=COL, K=KB,
            num_warps=4, num_stages=3)
        dx = torch.empty_like(x)
        _sum_dx_route[(t, triton.cdiv(d, 256))](dxp, inv, sel, dl, rw, dx,
                                                 T=t, D=d, E=e, A=a, B=256)
        grw = torch.empty_like(rw)
        _drw[(triton.cdiv(d, 32), triton.cdiv(e, 32))](x, dl, sel, grw,
            T=t, D=d, E=e, A=a, BM=32, BN=32, BK=32, num_warps=4)
        return dx, grw, None, gw1, gw2, None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
