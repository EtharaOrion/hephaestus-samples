import torch
import triton
import triton.language as tl


@triton.jit
def _router(X, RW, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        rk = k + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * D + rk[None, :],
                    mask=(rm[:, None] < T) & (rk[None, :] < D), other=0.0).to(tl.float32)
        w = tl.load(RW + rk[:, None] * E + rn[None, :],
                    mask=(rk[:, None] < D) & (rn[None, :] < E), other=0.0).to(tl.float32)
        acc += tl.dot(x, w, input_precision="ieee")
    tl.store(S + rm[:, None] * E + rn[None, :], tl.sigmoid(acc),
             mask=(rm[:, None] < T) & (rn[None, :] < E))


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
    if e == E - 1:
        tl.store(OFF + E, N)


@triton.jit
def _tiles(OFF, TE, TS, E: tl.constexpr, MT: tl.constexpr,
           BM: tl.constexpr):
    if tl.program_id(0) == 0:
        p = 0
        for e in range(E):
            begin = tl.load(OFF + e)
            end = tl.load(OFF + e + 1)
            q = begin
            while q < end:
                tl.store(TE + p, e)
                tl.store(TS + p, q)
                p += 1
                q += BM
        while p < MT:
            tl.store(TE + p, -1)
            tl.store(TS + p, 0)
            p += 1


@triton.jit
def _expert_in(X, W, PERM, OFF, TE, TS, GU,
               T: tl.constexpr, D: tl.constexpr, F2: tl.constexpr,
               A: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr, BF: tl.constexpr):
    tm = tl.program_id(0)
    e = tl.load(TE + tm)
    start = tl.load(TS + tm)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    rm = start + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    pair = tl.load(PERM + rm, mask=(e >= 0) & (rm < end), other=0)
    tok = pair // A
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        rk = k + tl.arange(0, BK)
        x = tl.load(X + tok[:, None] * D + rk[None, :],
                    mask=(e >= 0) & (rm[:, None] < end) & (rk[None, :] < D), other=0.0)
        w = tl.load(W + e * D * F2 + rk[:, None] * F2 + rn[None, :],
                    mask=(e >= 0) & (rk[:, None] < D) & (rn[None, :] < F2), other=0.0)
        if BF:
            acc += tl.dot(x, w)
        else:
            acc += tl.dot(x.to(tl.float32), w.to(tl.float32), input_precision="tf32x3")
    tl.store(GU + rm[:, None] * F2 + rn[None, :], acc,
             mask=(e >= 0) & (rm[:, None] < end) & (rn[None, :] < F2))


@triton.jit
def _activate(GU, Z, N: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N * F
    r = o // F
    c = o % F
    g = tl.load(GU + r * (2 * F) + c, mask=m, other=0.0)
    u = tl.load(GU + r * (2 * F) + F + c, mask=m, other=0.0)
    tl.store(Z + o, g * tl.sigmoid(g) * u, mask=m)


@triton.jit
def _expert_out(Z, W, OFF, TE, TS, FE, F: tl.constexpr, D: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                BF: tl.constexpr):
    tm = tl.program_id(0)
    e = tl.load(TE + tm)
    start = tl.load(TS + tm)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    rm = start + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, F, BK):
        rk = k + tl.arange(0, BK)
        z = tl.load(Z + rm[:, None] * F + rk[None, :],
                    mask=(e >= 0) & (rm[:, None] < end) & (rk[None, :] < F), other=0.0)
        w = tl.load(W + e * F * D + rk[:, None] * D + rn[None, :],
                    mask=(e >= 0) & (rk[:, None] < F) & (rn[None, :] < D), other=0.0)
        if BF:
            zh = z.to(tl.bfloat16)
            zl = (z - zh.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(zh, w) + tl.dot(zl, w)
        else:
            acc += tl.dot(z, w.to(tl.float32), input_precision="tf32x3")
    tl.store(FE + rm[:, None] * D + rn[None, :], acc,
             mask=(e >= 0) & (rm[:, None] < end) & (rn[None, :] < D))


@triton.jit
def _inverse(PERM, INV, N: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    p = tl.load(PERM + r, mask=r < N, other=0)
    tl.store(INV + p, r, mask=r < N)


@triton.jit
def _combine(S, SEL, INV, FE, Y, T: tl.constexpr, E: tl.constexpr,
             D: tl.constexpr, A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    den = 0.0
    for a in range(A):
        e = tl.load(SEL + t * A + a)
        den += tl.load(S + t * E + e)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(A):
        e = tl.load(SEL + t * A + a)
        q = tl.load(S + t * E + e) / den
        r = tl.load(INV + t * A + a)
        f = tl.load(FE + r * D + rd, mask=rd < D, other=0.0)
        acc += q * f
    tl.store(Y + t * D + rd, acc, mask=rd < D)


@triton.jit
def _pair_dot(DY, S, SEL, INV, FE, DOT, DFE, T: tl.constexpr,
              E: tl.constexpr, D: tl.constexpr, A: tl.constexpr,
              BD: tl.constexpr):
    t = tl.program_id(0)
    a = tl.program_id(1)
    rd = tl.arange(0, BD)
    r = tl.load(INV + t * A + a)
    dy = tl.load(DY + t * D + rd, mask=rd < D, other=0.0).to(tl.float32)
    dot = tl.sum(dy * tl.load(FE + r * D + rd, mask=rd < D, other=0.0), axis=0)
    den = 0.0
    for j in range(A):
        e = tl.load(SEL + t * A + j)
        den += tl.load(S + t * E + e)
    e = tl.load(SEL + t * A + a)
    q = tl.load(S + t * E + e) / den
    tl.store(DOT + t * A + a, dot)
    tl.store(DFE + r * D + rd, q * dy, mask=rd < D)


@triton.jit
def _route_grad(S, SEL, DOT, DL, T: tl.constexpr, E: tl.constexpr,
                A: tl.constexpr):
    t = tl.program_id(0)
    a = tl.program_id(1)
    den = 0.0
    center = 0.0
    for j in range(A):
        e = tl.load(SEL + t * A + j)
        ss = tl.load(S + t * E + e)
        den += ss
        center += ss * tl.load(DOT + t * A + j)
    e = tl.load(SEL + t * A + a)
    ss = tl.load(S + t * E + e)
    dot = tl.load(DOT + t * A + a)
    tl.store(DL + t * A + a, (dot - center / den) / den * ss * (1.0 - ss))


@triton.jit
def _dfe(DY, S, SEL, PERM, DFE, N: tl.constexpr, T: tl.constexpr,
         E: tl.constexpr, D: tl.constexpr, A: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    pair = tl.load(PERM + r)
    t = pair // A
    den = 0.0
    for a in range(A):
        e = tl.load(SEL + t * A + a)
        den += tl.load(S + t * E + e)
    e = tl.load(SEL + pair)
    q = tl.load(S + t * E + e) / den
    dy = tl.load(DY + t * D + rd, mask=rd < D, other=0.0).to(tl.float32)
    tl.store(DFE + r * D + rd, q * dy, mask=rd < D)


@triton.jit
def _dz(DFE, W, OFF, TE, TS, DZ, F: tl.constexpr, D: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BF: tl.constexpr):
    tm = tl.program_id(0)
    e = tl.load(TE + tm)
    start = tl.load(TS + tm)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    rm = start + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(DFE + rm[:, None] * D + rk[None, :],
                    mask=(e >= 0) & (rm[:, None] < end) & (rk[None, :] < D), other=0.0)
        w = tl.load(W + e * F * D + rn[None, :] * D + rk[:, None],
                    mask=(e >= 0) & (rn[None, :] < F) & (rk[:, None] < D), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            al = (a - ah.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(ah, w) + tl.dot(al, w)
        else:
            acc += tl.dot(a, w.to(tl.float32), input_precision="tf32x3")
    tl.store(DZ + rm[:, None] * F + rn[None, :], acc,
             mask=(e >= 0) & (rm[:, None] < end) & (rn[None, :] < F))


@triton.jit
def _dactivate(GU, DZ, DGU, N: tl.constexpr, F: tl.constexpr,
               BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N * F
    r = o // F
    c = o % F
    g = tl.load(GU + r * (2 * F) + c, mask=m, other=0.0)
    u = tl.load(GU + r * (2 * F) + F + c, mask=m, other=0.0)
    dz = tl.load(DZ + o, mask=m, other=0.0)
    sg = tl.sigmoid(g)
    silu = g * sg
    tl.store(DGU + r * (2 * F) + c, dz * u * (sg + silu * (1.0 - sg)), mask=m)
    tl.store(DGU + r * (2 * F) + F + c, dz * silu, mask=m)


@triton.jit
def _dx_rows(DGU, W, OFF, TE, TS, DXR, D: tl.constexpr, F2: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             BF: tl.constexpr):
    tm = tl.program_id(0)
    e = tl.load(TE + tm)
    start = tl.load(TS + tm)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    rm = start + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, F2, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(DGU + rm[:, None] * F2 + rk[None, :],
                    mask=(e >= 0) & (rm[:, None] < end) & (rk[None, :] < F2), other=0.0)
        w = tl.load(W + e * D * F2 + rn[None, :] * F2 + rk[:, None],
                    mask=(e >= 0) & (rn[None, :] < D) & (rk[:, None] < F2), other=0.0)
        if BF:
            acc += tl.dot(a.to(tl.bfloat16), w)
        else:
            acc += tl.dot(a, w.to(tl.float32), input_precision="tf32x3")
    tl.store(DXR + rm[:, None] * D + rn[None, :], acc,
             mask=(e >= 0) & (rm[:, None] < end) & (rn[None, :] < D))


@triton.jit
def _combine_dx(DXR, INV, DL, SEL, RW, DX, T: tl.constexpr, E: tl.constexpr,
                D: tl.constexpr, A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(A):
        r = tl.load(INV + t * A + a)
        acc += tl.load(DXR + r * D + rd, mask=rd < D, other=0.0)
        e = tl.load(SEL + t * A + a)
        dl = tl.load(DL + t * A + a)
        acc += dl * tl.load(RW + rd * E + e, mask=rd < D, other=0.0).to(tl.float32)
    tl.store(DX + t * D + rd, acc, mask=rd < D)


@triton.jit
def _dw2(Z, DFE, OFF, DW, E: tl.constexpr, F: tl.constexpr, D: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BF: tl.constexpr):
    e = tl.program_id(0)
    rf = tl.program_id(1) * BM + tl.arange(0, BM)
    rd = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(Z + rk[None, :] * F + rf[:, None],
                    mask=(rk[None, :] < end) & (rf[:, None] < F), other=0.0)
        b = tl.load(DFE + rk[:, None] * D + rd[None, :],
                    mask=(rk[:, None] < end) & (rd[None, :] < D), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16); al = (a - ah.to(tl.float32)).to(tl.bfloat16)
            bh = b.to(tl.bfloat16); bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(ah, bh) + tl.dot(al, bh) + tl.dot(ah, bl)
        else:
            acc += tl.dot(a, b, input_precision="tf32x3")
    tl.store(DW + e * F * D + rf[:, None] * D + rd[None, :], acc,
             mask=(rf[:, None] < F) & (rd[None, :] < D))


@triton.jit
def _dw1(X, DGU, PERM, OFF, DW, E: tl.constexpr, D: tl.constexpr,
         F2: tl.constexpr, A: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         BK: tl.constexpr, BF: tl.constexpr):
    e = tl.program_id(0)
    rd = tl.program_id(1) * BM + tl.arange(0, BM)
    rf = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        rk = k + tl.arange(0, BK)
        pair = tl.load(PERM + rk, mask=rk < end, other=0)
        tok = pair // A
        x = tl.load(X + rd[:, None] + tok[None, :] * D,
                    mask=(rd[:, None] < D) & (rk[None, :] < end), other=0.0)
        g = tl.load(DGU + rk[:, None] * F2 + rf[None, :],
                    mask=(rk[:, None] < end) & (rf[None, :] < F2), other=0.0)
        if BF:
            gh = g.to(tl.bfloat16); gl = (g - gh.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(x, gh) + tl.dot(x, gl)
        else:
            acc += tl.dot(x.to(tl.float32), g, input_precision="tf32x3")
    tl.store(DW + e * D * F2 + rd[:, None] * F2 + rf[None, :], acc,
             mask=(rd[:, None] < D) & (rf[None, :] < F2))


@triton.jit
def _drw(X, DL, SEL, DRW, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
         A: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         BK: tl.constexpr, BF: tl.constexpr):
    rd = tl.program_id(0) * BM + tl.arange(0, BM)
    re = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, T, BK):
        rt = k + tl.arange(0, BK)
        x = tl.load(X + rd[:, None] + rt[None, :] * D,
                    mask=(rd[:, None] < D) & (rt[None, :] < T), other=0.0)
        vals = tl.zeros((BK, BN), tl.float32)
        for a in range(A):
            ee = tl.load(SEL + rt[:, None] * A + a,
                         mask=rt[:, None] < T, other=-1)
            v = tl.load(DL + rt[:, None] * A + a,
                        mask=rt[:, None] < T, other=0.0)
            vals += tl.where(ee == re[None, :], v, 0.0)
        acc += tl.dot(x.to(tl.float32), vals, input_precision="ieee")
    tl.store(DRW + rd[:, None] * E + re[None, :], acc,
             mask=(rd[:, None] < D) & (re[None, :] < E))


def _selection(s, b, A):
    E = s.shape[1]
    v = s + b
    v = torch.where(v == 0.0, torch.zeros_like(v), v).contiguous()
    bits = v.view(torch.int32).to(torch.int64)
    ordered = torch.where(bits >= 0, bits ^ 0x80000000, (~bits) & 0xFFFFFFFF)
    idx = torch.arange(E, device=s.device, dtype=torch.int64)
    keys = (ordered << 10) | (E - 1 - idx)
    return torch.topk(keys, A, dim=1, largest=True, sorted=True).indices


def _metadata(sel, E, BM):
    flat = sel.flatten()
    se, perm = torch.sort(flat)
    N = flat.numel()
    off = torch.empty(E + 1, device=sel.device, dtype=torch.int64)
    _offsets[(E,)](se, off, N=N, E=E)
    mt = triton.cdiv(N, BM) + E
    te = torch.empty(mt, device=sel.device, dtype=torch.int32)
    ts = torch.empty(mt, device=sel.device, dtype=torch.int64)
    _tiles[(1,)](off, te, ts, E=E, MT=mt, BM=BM, num_warps=1)
    inv = torch.empty_like(perm)
    _inverse[(triton.cdiv(N, 256),)](perm, inv, N=N, BLOCK=256)
    return perm, off, te, ts, inv, mt


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
        with torch.no_grad():
            sel = _selection(s, rb, A)
            perm, off, te, ts, inv, mt = _metadata(sel, E, 64)
        N = T * A
        gu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _expert_in[(mt, triton.cdiv(2 * F, 64))](x, w1, perm, off, te, ts, gu,
            T=T, D=D, F2=2 * F, A=A, BM=64, BN=64, BK=32, BF=bf,
            num_warps=4, num_stages=3)
        z = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _activate[(triton.cdiv(N * F, 256),)](gu, z, N=N, F=F, BLOCK=256)
        fe = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _expert_out[(mt, triton.cdiv(D, 64))](z, w2, off, te, ts, fe,
            F=F, D=D, BM=64, BN=64, BK=32, BF=bf, num_warps=4, num_stages=3)
        y = torch.empty((T, D), device=x.device, dtype=x.dtype)
        _combine[(T, triton.cdiv(D, 256))](s, sel, inv, fe, y, T=T, E=E, D=D,
            A=A, BD=256, num_warps=8)
        ctx.save_for_backward(x, rw, w1, w2, s, sel, perm, off, te, ts, inv, gu, z, fe)
        ctx.dims = (T, D, E, F, A, mt, bf)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, rw, w1, w2, s, sel, perm, off, te, ts, inv, gu, z, fe = ctx.saved_tensors
        T, D, E, F, A, mt, bf = ctx.dims
        N = T * A
        dy = dy.contiguous()
        dots = torch.empty((T, A), device=x.device, dtype=torch.float32)
        dfe = torch.empty((N, D), device=x.device, dtype=torch.float32)
        route_bd = triton.next_power_of_2(D)
        _pair_dot[(T, A)](dy, s, sel, inv, fe, dots, dfe, T=T, E=E, D=D,
            A=A, BD=route_bd, num_warps=8)
        dl = torch.empty((T, A), device=x.device, dtype=torch.float32)
        _route_grad[(T, A)](s, sel, dots, dl, T=T, E=E, A=A, num_warps=1)
        dz = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _dz[(mt, triton.cdiv(F, 64))](dfe, w2, off, te, ts, dz, F=F, D=D,
            BM=64, BN=64, BK=32, BF=bf, num_warps=4, num_stages=3)
        dgu = torch.empty_like(gu)
        _dactivate[(triton.cdiv(N * F, 256),)](gu, dz, dgu, N=N, F=F, BLOCK=256)
        dxr = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _dx_rows[(mt, triton.cdiv(D, 128))](dgu, w1, off, te, ts, dxr, D=D,
            F2=2 * F, BM=64, BN=128, BK=32, BF=bf, num_warps=8, num_stages=3)
        dx = torch.empty_like(x)
        _combine_dx[(T, triton.cdiv(D, 256))](dxr, inv, dl, sel, rw, dx,
            T=T, E=E, D=D, A=A, BD=256, num_warps=8)
        dw2 = torch.empty_like(w2)
        _dw2[(E, triton.cdiv(F, 64), triton.cdiv(D, 64))](z, dfe, off, dw2,
            E=E, F=F, D=D, BM=64, BN=64, BK=32, BF=bf,
            num_warps=4, num_stages=3)
        dw1 = torch.empty_like(w1)
        _dw1[(E, triton.cdiv(D, 64), triton.cdiv(2 * F, 64))](x, dgu, perm, off,
            dw1, E=E, D=D, F2=2 * F, A=A, BM=64, BN=64, BK=32, BF=bf,
            num_warps=4, num_stages=3)
        drw = torch.empty_like(rw)
        _drw[(triton.cdiv(D, 32), triton.cdiv(E, 32))](x, dl, sel, drw,
            T=T, D=D, E=E, A=A, BM=32, BN=32, BK=32, BF=bf,
            num_warps=4, num_stages=3)
        return dx, drw, None, dw1, dw2, None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
