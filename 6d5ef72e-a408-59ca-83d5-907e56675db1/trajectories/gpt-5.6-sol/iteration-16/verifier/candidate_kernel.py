import torch
import triton
import triton.language as tl

ROW = 64


@triton.jit
def _router(X, W, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
            BM: tl.constexpr = 32, BN: tl.constexpr = 32, BK: tl.constexpr = 32):
    im = tl.program_id(0) * BM + tl.arange(0, BM)
    ie = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        ik = k + tl.arange(0, BK)
        x = tl.load(X + im[:, None] * D + ik[None, :], mask=(im[:, None] < T) & (ik[None, :] < D), other=0.0)
        w = tl.load(W + ik[:, None] * E + ie[None, :], mask=(ik[:, None] < D) & (ie[None, :] < E), other=0.0)
        acc += tl.dot(x, w, input_precision="ieee")
    tl.store(S + im[:, None] * E + ie[None, :], tl.sigmoid(acc), mask=(im[:, None] < T) & (ie[None, :] < E))


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


@triton.jit
def _tiles(OFF, TE, TS, E: tl.constexpr, CAP: tl.constexpr, BR: tl.constexpr):
    q = 0
    e = 0
    while e < E:
        r = tl.load(OFF + e)
        z = tl.load(OFF + e + 1)
        while r < z:
            tl.store(TE + q, e)
            tl.store(TS + q, r)
            q += 1
            r += BR
        e += 1
    while q < CAP:
        tl.store(TE + q, -1)
        tl.store(TS + q, 0)
        q += 1


@triton.jit
def _inverse(ORDER, INV, N: tl.constexpr, B: tl.constexpr = 256):
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < N
    p = tl.load(ORDER + i, mask=m, other=0)
    tl.store(INV + p, i, mask=m)


@triton.jit
def _expert_in(X, W1, ORDER, OFF, TE, TS, GU, N: tl.constexpr, A: tl.constexpr,
               D: tl.constexpr, F2: tl.constexpr, BF: tl.constexpr,
               BR: tl.constexpr = 64, BN: tl.constexpr = 64, BK: tl.constexpr = 32):
    q = tl.program_id(0)
    e = tl.load(TE + q)
    start = tl.load(TS + q)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    ir = start + tl.arange(0, BR)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    p = tl.load(ORDER + ir, mask=(e >= 0) & (ir < end), other=0)
    tok = p // A
    acc = tl.zeros((BR, BN), tl.float32)
    for k in range(0, D, BK):
        ik = k + tl.arange(0, BK)
        x = tl.load(X + tok[:, None] * D + ik[None, :], mask=(e >= 0) & (ir[:, None] < end) & (ik[None, :] < D), other=0.0)
        w = tl.load(W1 + e * D * F2 + ik[:, None] * F2 + j[None, :], mask=(e >= 0) & (ik[:, None] < D) & (j[None, :] < F2), other=0.0)
        if BF:
            acc += tl.dot(x, w)
        else:
            acc += tl.dot(x, w, input_precision="tf32x3")
    tl.store(GU + ir[:, None] * F2 + j[None, :], acc, mask=(e >= 0) & (ir[:, None] < end) & (j[None, :] < F2))


@triton.jit
def _activate(GU, ACT, N: tl.constexpr, F: tl.constexpr, B: tl.constexpr = 256):
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < N * F
    r = i // F
    j = i - r * F
    g = tl.load(GU + r * (2 * F) + j, mask=m, other=0.0)
    u = tl.load(GU + r * (2 * F) + F + j, mask=m, other=0.0)
    tl.store(ACT + i, g * tl.sigmoid(g) * u, mask=m)


@triton.jit
def _expert_out(ACT, W2, OFF, TE, TS, FE, N: tl.constexpr, F: tl.constexpr,
                D: tl.constexpr, BF: tl.constexpr, BR: tl.constexpr = 64,
                BN: tl.constexpr = 64, BK: tl.constexpr = 32):
    q = tl.program_id(0)
    e = tl.load(TE + q)
    start = tl.load(TS + q)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    ir = start + tl.arange(0, BR)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BR, BN), tl.float32)
    for k in range(0, F, BK):
        ik = k + tl.arange(0, BK)
        a = tl.load(ACT + ir[:, None] * F + ik[None, :], mask=(e >= 0) & (ir[:, None] < end) & (ik[None, :] < F), other=0.0)
        w = tl.load(W2 + e * F * D + ik[:, None] * D + j[None, :], mask=(e >= 0) & (ik[:, None] < F) & (j[None, :] < D), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            al = (a - ah).to(tl.bfloat16)
            acc += tl.dot(ah, w) + tl.dot(al, w)
        else:
            acc += tl.dot(a, w, input_precision="tf32x3")
    tl.store(FE + ir[:, None] * D + j[None, :], acc, mask=(e >= 0) & (ir[:, None] < end) & (j[None, :] < D))


@triton.jit
def _combine(S, SEL, INV, FE, Y, T: tl.constexpr, E: tl.constexpr,
             A: tl.constexpr, D: tl.constexpr, BD: tl.constexpr = 128):
    t = tl.program_id(0)
    j = tl.program_id(1) * BD + tl.arange(0, BD)
    den = 0.0
    for a in range(A):
        e = tl.load(SEL + t * A + a)
        den += tl.load(S + t * E + e)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(A):
        e = tl.load(SEL + t * A + a)
        r = tl.load(INV + t * A + a)
        w = tl.load(S + t * E + e) / den
        f = tl.load(FE + r * D + j, mask=j < D, other=0.0)
        acc += w * f
    tl.store(Y + t * D + j, acc, mask=j < D)


@triton.jit
def _dz(DY, S, SEL, INV, DZ, T: tl.constexpr, E: tl.constexpr,
        A: tl.constexpr, D: tl.constexpr, BD: tl.constexpr = 128):
    t = tl.program_id(0)
    j = tl.program_id(1) * BD + tl.arange(0, BD)
    dy = tl.load(DY + t * D + j, mask=j < D, other=0.0)
    den = 0.0
    for a in range(A):
        e = tl.load(SEL + t * A + a)
        den += tl.load(S + t * E + e)
    for a in range(A):
        e = tl.load(SEL + t * A + a)
        r = tl.load(INV + t * A + a)
        w = tl.load(S + t * E + e) / den
        tl.store(DZ + r * D + j, dy * w, mask=j < D)


@triton.jit
def _pair_q(DY, FE, INV, Q, T: tl.constexpr, A: tl.constexpr,
            D: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    a = tl.program_id(1)
    j = tl.arange(0, BD)
    r = tl.load(INV + t * A + a)
    acc = tl.zeros((BD,), tl.float32)
    for k in range(0, D, BD):
        jj = k + j
        dy = tl.load(DY + t * D + jj, mask=jj < D, other=0.0)
        f = tl.load(FE + r * D + jj, mask=jj < D, other=0.0)
        acc += dy * f
    tl.store(Q + t * A + a, tl.sum(acc, axis=0))


@triton.jit
def _dact(DZ, W2, OFF, TE, TS, DA, N: tl.constexpr, F: tl.constexpr, D: tl.constexpr,
          BF: tl.constexpr, BR: tl.constexpr = 64, BN: tl.constexpr = 64,
          BK: tl.constexpr = 32):
    q = tl.program_id(0)
    e = tl.load(TE + q)
    start = tl.load(TS + q)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    ir = start + tl.arange(0, BR)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BR, BN), tl.float32)
    for k in range(0, D, BK):
        ik = k + tl.arange(0, BK)
        z = tl.load(DZ + ir[:, None] * D + ik[None, :], mask=(e >= 0) & (ir[:, None] < end) & (ik[None, :] < D), other=0.0)
        w = tl.load(W2 + e * F * D + j[None, :] * D + ik[:, None], mask=(e >= 0) & (j[None, :] < F) & (ik[:, None] < D), other=0.0)
        if BF:
            acc += tl.dot(z.to(tl.bfloat16), w)
        else:
            acc += tl.dot(z, w, input_precision="tf32x3")
    tl.store(DA + ir[:, None] * F + j[None, :], acc, mask=(e >= 0) & (ir[:, None] < end) & (j[None, :] < F))


@triton.jit
def _dgu(GU, DA, DGU, N: tl.constexpr, F: tl.constexpr, B: tl.constexpr = 256):
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < N * F
    r = i // F
    j = i - r * F
    g = tl.load(GU + r * (2 * F) + j, mask=m, other=0.0)
    u = tl.load(GU + r * (2 * F) + F + j, mask=m, other=0.0)
    da = tl.load(DA + i, mask=m, other=0.0)
    sig = tl.sigmoid(g)
    sg = g * sig
    tl.store(DGU + r * (2 * F) + j, da * u * (sig + sg * (1.0 - sig)), mask=m)
    tl.store(DGU + r * (2 * F) + F + j, da * sg, mask=m)


@triton.jit
def _dx_rows(DGU, W1, OFF, TE, TS, DXR, N: tl.constexpr, D: tl.constexpr,
             F2: tl.constexpr, BF: tl.constexpr, BR: tl.constexpr = 64,
             BN: tl.constexpr = 64, BK: tl.constexpr = 32):
    q = tl.program_id(0)
    e = tl.load(TE + q)
    start = tl.load(TS + q)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    ir = start + tl.arange(0, BR)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BR, BN), tl.float32)
    for k in range(0, F2, BK):
        ik = k + tl.arange(0, BK)
        g = tl.load(DGU + ir[:, None] * F2 + ik[None, :], mask=(e >= 0) & (ir[:, None] < end) & (ik[None, :] < F2), other=0.0)
        w = tl.load(W1 + e * D * F2 + j[None, :] * F2 + ik[:, None], mask=(e >= 0) & (j[None, :] < D) & (ik[:, None] < F2), other=0.0)
        if BF:
            acc += tl.dot(g.to(tl.bfloat16), w)
        else:
            acc += tl.dot(g, w, input_precision="tf32x3")
    tl.store(DXR + ir[:, None] * D + j[None, :], acc, mask=(e >= 0) & (ir[:, None] < end) & (j[None, :] < D))


@triton.jit
def _route_grad(S, SEL, Q, DL, T: tl.constexpr, E: tl.constexpr,
                A: tl.constexpr, BE: tl.constexpr):
    t = tl.program_id(0)
    ie = tl.arange(0, BE)
    den = 0.0
    qnum = 0.0
    for a in range(A):
        e = tl.load(SEL + t * A + a)
        sv = tl.load(S + t * E + e)
        q = tl.load(Q + t * A + a)
        den += sv
        qnum += sv * q
    val = tl.zeros((BE,), tl.float32)
    for a in range(A):
        e = tl.load(SEL + t * A + a)
        sv = tl.load(S + t * E + e)
        q = tl.load(Q + t * A + a)
        ds = (q - qnum / den) / den
        val = tl.where(ie == e, ds * sv * (1.0 - sv), val)
    tl.store(DL + t * E + ie, val, mask=ie < E)


@triton.jit
def _combine_dx(DXR, INV, DL, RW, DX, T: tl.constexpr, A: tl.constexpr,
                E: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
                BE: tl.constexpr):
    t = tl.program_id(0)
    j = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(A):
        r = tl.load(INV + t * A + a)
        acc += tl.load(DXR + r * D + j, mask=j < D, other=0.0)
    ie = tl.arange(0, BE)
    dl = tl.load(DL + t * E + ie, mask=ie < E, other=0.0)
    rw = tl.load(RW + j[:, None] * E + ie[None, :], mask=(j[:, None] < D) & (ie[None, :] < E), other=0.0)
    acc += tl.sum(rw * dl[None, :], axis=1)
    tl.store(DX + t * D + j, acc, mask=j < D)


@triton.jit
def _dw2(ACT, DZ, OFF, DW, E: tl.constexpr, F: tl.constexpr,
         D: tl.constexpr, BF: tl.constexpr, BM: tl.constexpr = 64,
         BN: tl.constexpr = 64, BK: tl.constexpr = 32):
    e = tl.program_id(0)
    i = tl.program_id(1) * BM + tl.arange(0, BM)
    j = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        r = k + tl.arange(0, BK)
        a = tl.load(ACT + r[None, :] * F + i[:, None], mask=(r[None, :] < end) & (i[:, None] < F), other=0.0).to(tl.float32)
        z = tl.load(DZ + r[:, None] * D + j[None, :], mask=(r[:, None] < end) & (j[None, :] < D), other=0.0).to(tl.float32)
        if BF:
            ah = a.to(tl.bfloat16); al = (a - ah).to(tl.bfloat16)
            zh = z.to(tl.bfloat16); zl = (z - zh).to(tl.bfloat16)
            acc += tl.dot(ah, zh) + tl.dot(al, zh) + tl.dot(ah, zl)
        else:
            acc += tl.dot(a, z, input_precision="tf32x3")
    tl.store(DW + e * F * D + i[:, None] * D + j[None, :], acc, mask=(i[:, None] < F) & (j[None, :] < D))


@triton.jit
def _dw1(X, DGU, ORDER, OFF, DW, E: tl.constexpr, A: tl.constexpr,
         D: tl.constexpr, F2: tl.constexpr, BF: tl.constexpr,
         BM: tl.constexpr = 64, BN: tl.constexpr = 64, BK: tl.constexpr = 32):
    e = tl.program_id(0)
    i = tl.program_id(1) * BM + tl.arange(0, BM)
    j = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e)
    end = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        r = k + tl.arange(0, BK)
        p = tl.load(ORDER + r, mask=r < end, other=0)
        tok = p // A
        x = tl.load(X + i[:, None] + tok[None, :] * D, mask=(i[:, None] < D) & (r[None, :] < end), other=0.0).to(tl.float32)
        g = tl.load(DGU + r[:, None] * F2 + j[None, :], mask=(r[:, None] < end) & (j[None, :] < F2), other=0.0).to(tl.float32)
        if BF:
            gh = g.to(tl.bfloat16); gl = (g - gh).to(tl.bfloat16)
            acc += tl.dot(x.to(tl.bfloat16), gh) + tl.dot(x.to(tl.bfloat16), gl)
        else:
            acc += tl.dot(x, g, input_precision="tf32x3")
    tl.store(DW + e * D * F2 + i[:, None] * F2 + j[None, :], acc, mask=(i[:, None] < D) & (j[None, :] < F2))


@triton.jit
def _drw(X, DL, DW, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
         BF: tl.constexpr, BM: tl.constexpr = 64, BN: tl.constexpr = 32,
         BK: tl.constexpr = 32):
    i = tl.program_id(0) * BM + tl.arange(0, BM)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, T, BK):
        r = k + tl.arange(0, BK)
        x = tl.load(X + i[:, None] + r[None, :] * D, mask=(i[:, None] < D) & (r[None, :] < T), other=0.0).to(tl.float32)
        g = tl.load(DL + r[:, None] * E + j[None, :], mask=(r[:, None] < T) & (j[None, :] < E), other=0.0).to(tl.float32)
        acc += tl.dot(x, g, input_precision="ieee")
    tl.store(DW + i[:, None] * E + j[None, :], acc, mask=(i[:, None] < D) & (j[None, :] < E))


def _select(s, bias, A):
    E = s.shape[1]
    b = s + bias
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    idx = torch.arange(E, device=s.device, dtype=torch.int64)
    return torch.topk((u << 10) | (E - 1 - idx), A, dim=1, largest=True, sorted=True).indices


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
            sel = _select(s, rb, A)
            flat = sel.reshape(-1)
            order = torch.argsort(flat, stable=True)
        off = torch.empty(E + 1, device=x.device, dtype=torch.int32)
        _offsets[(E + 1,)](flat, order, off, N=N, E=E, num_warps=1)
        cap = triton.cdiv(N, ROW) + E
        te = torch.empty(cap, device=x.device, dtype=torch.int32)
        ts = torch.empty_like(te)
        _tiles[(1,)](off, te, ts, E=E, CAP=cap, BR=ROW, num_warps=1)
        inv = torch.empty(N, device=x.device, dtype=torch.int64)
        _inverse[(triton.cdiv(N, 256),)](order, inv, N=N)
        gu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _expert_in[(cap, triton.cdiv(2 * F, 64))](x, w1, order, off, te, ts, gu, N=N, A=A, D=D, F2=2 * F, BF=bf)
        act = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _activate[(triton.cdiv(N * F, 256),)](gu, act, N=N, F=F)
        fe = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _expert_out[(cap, triton.cdiv(D, 64))](act, w2, off, te, ts, fe, N=N, F=F, D=D, BF=bf)
        yf = torch.empty((T, D), device=x.device, dtype=torch.float32)
        _combine[(T, triton.cdiv(D, 128))](s, sel, inv, fe, yf, T=T, E=E, A=A, D=D)
        ctx.save_for_backward(x, rw, w1, w2, s, sel, order, inv, off, te, ts, gu, act, fe)
        ctx.dims = (T, D, E, F, A, N, cap, bf)
        return yf.to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        x, rw, w1, w2, s, sel, order, inv, off, te, ts, gu, act, fe = ctx.saved_tensors
        T, D, E, F, A, N, cap, bf = ctx.dims
        dy = dy.contiguous()
        dz = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _dz[(T, triton.cdiv(D, 128))](dy, s, sel, inv, dz, T=T, E=E, A=A, D=D)
        q = torch.empty((T, A), device=x.device, dtype=torch.float32)
        bq = triton.next_power_of_2(D)
        _pair_q[(T, A)](dy, fe, inv, q, T=T, A=A, D=D, BD=bq)
        da = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _dact[(cap, triton.cdiv(F, 64))](dz, w2, off, te, ts, da, N=N, F=F, D=D, BF=bf)
        dgu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _dgu[(triton.cdiv(N * F, 256),)](gu, da, dgu, N=N, F=F)
        dxr = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _dx_rows[(cap, triton.cdiv(D, 64))](dgu, w1, off, te, ts, dxr, N=N, D=D, F2=2 * F, BF=bf)
        dl = torch.empty((T, E), device=x.device, dtype=torch.float32)
        be = triton.next_power_of_2(E)
        _route_grad[(T,)](s, sel, q, dl, T=T, E=E, A=A, BE=be)
        dx = torch.empty((T, D), device=x.device, dtype=torch.float32)
        _combine_dx[(T, triton.cdiv(D, 64))](dxr, inv, dl, rw, dx, T=T, A=A, E=E, D=D, BD=64, BE=be)
        dw2 = torch.empty(w2.shape, device=x.device, dtype=torch.float32)
        _dw2[(E, triton.cdiv(F, 64), triton.cdiv(D, 64))](act, dz, off, dw2, E=E, F=F, D=D, BF=bf)
        dw1 = torch.empty(w1.shape, device=x.device, dtype=torch.float32)
        _dw1[(E, triton.cdiv(D, 64), triton.cdiv(2 * F, 64))](x, dgu, order, off, dw1, E=E, A=A, D=D, F2=2 * F, BF=bf)
        drw = torch.empty(rw.shape, device=x.device, dtype=torch.float32)
        _drw[(triton.cdiv(D, 64), triton.cdiv(E, 32))](x, dl, drw, T=T, D=D, E=E, BF=bf)
        return dx.to(x.dtype), drw.to(rw.dtype), None, dw1.to(w1.dtype), dw2.to(w2.dtype), None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
