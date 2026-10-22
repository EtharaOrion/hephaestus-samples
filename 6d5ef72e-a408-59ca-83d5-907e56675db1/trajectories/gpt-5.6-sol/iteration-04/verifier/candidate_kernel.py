import torch
import triton
import triton.language as tl


@triton.jit
def _router(X, RW, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    im = tl.program_id(0) * BM + tl.arange(0, BM)
    ie = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        ik = k + tl.arange(0, BK)
        x = tl.load(X + im[:, None].to(tl.int64) * D + ik[None, :],
                    mask=(im[:, None] < T) & (ik[None, :] < D), other=0.0)
        r = tl.load(RW + ik[:, None].to(tl.int64) * E + ie[None, :],
                    mask=(ik[:, None] < D) & (ie[None, :] < E), other=0.0)
        acc += tl.dot(x.to(tl.float32), r.to(tl.float32), input_precision="ieee")
    tl.store(S + im[:, None].to(tl.int64) * E + ie[None, :], tl.sigmoid(acc),
             mask=(im[:, None] < T) & (ie[None, :] < E))


@triton.jit
def _route_weights(S, SEL, W, T: tl.constexpr, E: tl.constexpr,
                   A: tl.constexpr, BA: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BA)
    m = a < A
    e = tl.load(SEL + t * A + a, mask=m, other=0)
    v = tl.load(S + t.to(tl.int64) * E + e, mask=m, other=0.0)
    den = tl.sum(v, axis=0)
    tl.store(W + t * A + a, v / den, mask=m)


@triton.jit
def _lower_bounds(EXP, OFF, N: tl.constexpr, E: tl.constexpr):
    e = tl.program_id(0)
    lo = 0
    hi = N
    for _ in range(32):
        mid = (lo + hi) // 2
        v = tl.load(EXP + mid, mask=mid < N, other=E)
        take = v < e
        lo = tl.where((lo < hi) & take, mid + 1, lo)
        hi = tl.where((lo < hi) & ~take, mid, hi)
    tl.store(OFF + e, lo)


@triton.jit
def _tile_table(OFF, TE, TR, MT: tl.constexpr, E: tl.constexpr,
                BM: tl.constexpr):
    p = tl.program_id(0)
    q = 0
    ce = -1
    cr = 0
    for e in range(E):
        b = tl.load(OFF + e)
        z = tl.load(OFF + e + 1)
        nt = tl.cdiv(z - b, BM)
        hit = (p >= q) & (p < q + nt)
        ce = tl.where(hit, e, ce)
        cr = tl.where(hit, b + (p - q) * BM, cr)
        q += nt
    tl.store(TE + p, ce)
    tl.store(TR + p, cr)


@triton.jit
def _expert_in(X, W1, TOK, OFF, TE, TR, GU,
               N: tl.constexpr, D: tl.constexpr, F2: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               BF: tl.constexpr):
    p = tl.program_id(0)
    e = tl.load(TE + p)
    row = tl.load(TR + p) + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    tok = tl.load(TOK + row, mask=(e >= 0) & (row < end), other=0)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        kk = k + tl.arange(0, BK)
        x = tl.load(X + tok[:, None].to(tl.int64) * D + kk[None, :],
                    mask=(e >= 0) & (row[:, None] < end) & (kk[None, :] < D), other=0.0)
        woff = e.to(tl.int64) * D * F2 + kk[:, None].to(tl.int64) * F2 + col[None, :]
        w = tl.load(W1 + woff,
                    mask=(e >= 0) & (kk[:, None] < D) & (col[None, :] < F2), other=0.0)
        if BF:
            acc += tl.dot(x, w)
        else:
            acc += tl.dot(x, w, input_precision="tf32x3")
    tl.store(GU + row[:, None].to(tl.int64) * F2 + col[None, :], acc,
             mask=(e >= 0) & (row[:, None] < end) & (col[None, :] < F2))


@triton.jit
def _activation(GU, ACT, N: tl.constexpr, F: tl.constexpr, B: tl.constexpr):
    o = tl.program_id(0) * B + tl.arange(0, B)
    m = o < N * F
    r = o // F
    c = o % F
    g = tl.load(GU + r.to(tl.int64) * (2 * F) + c, mask=m, other=0.0)
    u = tl.load(GU + r.to(tl.int64) * (2 * F) + F + c, mask=m, other=0.0)
    tl.store(ACT + o, g * tl.sigmoid(g) * u, mask=m)


@triton.jit
def _expert_out(ACT, W2, OFF, TE, TR, FE,
                N: tl.constexpr, F: tl.constexpr, D: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    p = tl.program_id(0)
    e = tl.load(TE + p)
    row = tl.load(TR + p) + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, F, BK):
        kk = k + tl.arange(0, BK)
        a = tl.load(ACT + row[:, None].to(tl.int64) * F + kk[None, :],
                    mask=(e >= 0) & (row[:, None] < end) & (kk[None, :] < F), other=0.0)
        wo = e.to(tl.int64) * F * D + kk[:, None].to(tl.int64) * D + col[None, :]
        w = tl.load(W2 + wo,
                    mask=(e >= 0) & (kk[:, None] < F) & (col[None, :] < D), other=0.0)
        acc += tl.dot(a, w.to(tl.float32), input_precision="tf32x3")
    tl.store(FE + row[:, None].to(tl.int64) * D + col[None, :], acc,
             mask=(e >= 0) & (row[:, None] < end) & (col[None, :] < D))


@triton.jit
def _inverse(ORDER, INV, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    j = tl.load(ORDER + i, mask=i < N, other=0)
    tl.store(INV + j, i, mask=i < N)


@triton.jit
def _combine(FE, W, INV, Y, T: tl.constexpr, D: tl.constexpr,
             A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(A):
        j = t * A + a
        r = tl.load(INV + j)
        w = tl.load(W + j)
        f = tl.load(FE + r.to(tl.int64) * D + d, mask=d < D, other=0.0)
        acc += w * f
    tl.store(Y + t.to(tl.int64) * D + d, acc, mask=d < D)


@triton.jit
def _route_grad(DY, FE, S, SEL, W, INV, DL,
                T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                A: tl.constexpr, BA: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BA)
    ma = a < A
    j = t * A + a
    r = tl.load(INV + j, mask=ma, other=0)
    ww = tl.load(W + j, mask=ma, other=0.0)
    dot = tl.zeros((BA,), tl.float32)
    for k in range(0, D, BD):
        d = k + tl.arange(0, BD)
        dy = tl.load(DY + t.to(tl.int64) * D + d, mask=d < D, other=0.0)
        f = tl.load(FE + r[:, None].to(tl.int64) * D + d[None, :],
                    mask=ma[:, None] & (d[None, :] < D), other=0.0)
        dot += tl.sum(f * dy[None, :], axis=1)
    center = tl.sum(dot * ww, axis=0)
    e = tl.load(SEL + j, mask=ma, other=0)
    ss = tl.load(S + t.to(tl.int64) * E + e, mask=ma, other=0.0)
    den = tl.sum(ss, axis=0)
    dl = (dot - center) / den * ss * (1.0 - ss)
    tl.store(DL + j, dl, mask=ma)


@triton.jit
def _dact(DY, W, ORDER, W2, OFF, TE, TR, DA,
          N: tl.constexpr, F: tl.constexpr, D: tl.constexpr, A: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    p = tl.program_id(0)
    e = tl.load(TE + p)
    row = tl.load(TR + p) + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    j = tl.load(ORDER + row, mask=(e >= 0) & (row < end), other=0)
    tok = j // A
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        d = k + tl.arange(0, BK)
        dy = tl.load(DY + tok[:, None].to(tl.int64) * D + d[None, :],
                     mask=(e >= 0) & (row[:, None] < end) & (d[None, :] < D), other=0.0)
        ww = tl.load(W + j, mask=(e >= 0) & (row < end), other=0.0)
        wo = e.to(tl.int64) * F * D + col[:, None].to(tl.int64) * D + d[None, :]
        w2 = tl.load(W2 + wo,
                     mask=(e >= 0) & (col[:, None] < F) & (d[None, :] < D), other=0.0)
        acc += tl.dot((dy * ww[:, None]).to(tl.float32), tl.trans(w2.to(tl.float32)), input_precision="tf32x3")
    tl.store(DA + row[:, None].to(tl.int64) * F + col[None, :], acc,
             mask=(e >= 0) & (row[:, None] < end) & (col[None, :] < F))


@triton.jit
def _dgu(GU, DA, DGU, N: tl.constexpr, F: tl.constexpr, B: tl.constexpr):
    o = tl.program_id(0) * B + tl.arange(0, B)
    m = o < N * F
    r, c = o // F, o % F
    g = tl.load(GU + r.to(tl.int64) * (2 * F) + c, mask=m, other=0.0)
    u = tl.load(GU + r.to(tl.int64) * (2 * F) + F + c, mask=m, other=0.0)
    da = tl.load(DA + o, mask=m, other=0.0)
    q = tl.sigmoid(g)
    tl.store(DGU + r.to(tl.int64) * (2 * F) + c, da * u * (q + g * q * (1.0 - q)), mask=m)
    tl.store(DGU + r.to(tl.int64) * (2 * F) + F + c, da * g * q, mask=m)


@triton.jit
def _dx(DGU, W1, DL, SEL, INV, RW, DX, T: tl.constexpr, D: tl.constexpr,
        E: tl.constexpr, F2: tl.constexpr, A: tl.constexpr,
        BD: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for a in range(A):
        j = t * A + a
        r, e = tl.load(INV + j), tl.load(SEL + j)
        for k in range(0, F2, BK):
            f = k + tl.arange(0, BK)
            z = tl.load(DGU + r.to(tl.int64) * F2 + f, mask=f < F2, other=0.0)
            wo = e.to(tl.int64) * D * F2 + d[:, None].to(tl.int64) * F2 + f[None, :]
            v = tl.load(W1 + wo, mask=(d[:, None] < D) & (f[None, :] < F2), other=0.0)
            acc += tl.sum(v * z[None, :], axis=1)
        acc += tl.load(DL + j) * tl.load(RW + d.to(tl.int64) * E + e, mask=d < D, other=0.0)
    tl.store(DX + t.to(tl.int64) * D + d, acc, mask=d < D)

@triton.jit
def _dw2(ACT, DY, W, ORDER, OFF, DW, E: tl.constexpr, F: tl.constexpr,
         D: tl.constexpr, A: tl.constexpr, BM: tl.constexpr,
         BN: tl.constexpr, BK: tl.constexpr, BF: tl.constexpr):
    e = tl.program_id(0)
    f = tl.program_id(1) * BM + tl.arange(0, BM)
    d = tl.program_id(2) * BN + tl.arange(0, BN)
    begin, end = tl.load(OFF + e), tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        r = k + tl.arange(0, BK)
        j = tl.load(ORDER + r, mask=r < end, other=0)
        tok = j // A
        aa = tl.load(ACT + r[:, None].to(tl.int64) * F + f[None, :], mask=(r[:, None] < end) & (f[None, :] < F), other=0.0)
        z = tl.load(DY + tok[:, None].to(tl.int64) * D + d[None, :], mask=(r[:, None] < end) & (d[None, :] < D), other=0.0)
        z *= tl.load(W + j, mask=r < end, other=0.0)[:, None]
        if BF:
            ah = aa.to(tl.bfloat16); al = (aa - ah.to(tl.float32)).to(tl.bfloat16)
            zh = z.to(tl.bfloat16); zl = (z - zh.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(tl.trans(ah), zh) + tl.dot(tl.trans(al), zh) + tl.dot(tl.trans(ah), zl)
        else:
            acc += tl.dot(tl.trans(aa), z, input_precision="tf32x3")
    off = e.to(tl.int64) * F * D + f[:, None].to(tl.int64) * D + d[None, :]
    tl.store(DW + off, acc, mask=(f[:, None] < F) & (d[None, :] < D))


@triton.jit
def _dw1(X, DGU, TOK, OFF, DW, E: tl.constexpr, D: tl.constexpr,
         F2: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         BK: tl.constexpr, BF: tl.constexpr):
    e = tl.program_id(0)
    d = tl.program_id(1) * BM + tl.arange(0, BM)
    f = tl.program_id(2) * BN + tl.arange(0, BN)
    begin, end = tl.load(OFF + e), tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        r = k + tl.arange(0, BK)
        tok = tl.load(TOK + r, mask=r < end, other=0)
        x = tl.load(X + tok[:, None].to(tl.int64) * D + d[None, :], mask=(r[:, None] < end) & (d[None, :] < D), other=0.0)
        z = tl.load(DGU + r[:, None].to(tl.int64) * F2 + f[None, :], mask=(r[:, None] < end) & (f[None, :] < F2), other=0.0)
        if BF:
            zh = z.to(tl.bfloat16); zl = (z - zh.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(tl.trans(x), zh) + tl.dot(tl.trans(x), zl)
        else:
            acc += tl.dot(tl.trans(x), z, input_precision="tf32x3")
    off = e.to(tl.int64) * D * F2 + d[:, None].to(tl.int64) * F2 + f[None, :]
    tl.store(DW + off, acc, mask=(d[:, None] < D) & (f[None, :] < F2))


@triton.jit
def _dense_dl(DL, SEL, FULL, T: tl.constexpr, E: tl.constexpr, A: tl.constexpr, BE: tl.constexpr):
    t = tl.program_id(0)
    e = tl.program_id(1) * BE + tl.arange(0, BE)
    out = tl.zeros((BE,), tl.float32)
    for a in range(A):
        j = t * A + a
        out = tl.where(e == tl.load(SEL + j), tl.load(DL + j), out)
    tl.store(FULL + t.to(tl.int64) * E + e, out, mask=e < E)


@triton.jit
def _drw(X, FULL, DRW, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BF: tl.constexpr):
    d = tl.program_id(0) * BM + tl.arange(0, BM)
    e = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, T, BK):
        t = k + tl.arange(0, BK)
        x = tl.load(X + t[:, None].to(tl.int64) * D + d[None, :], mask=(t[:, None] < T) & (d[None, :] < D), other=0.0)
        z = tl.load(FULL + t[:, None].to(tl.int64) * E + e[None, :], mask=(t[:, None] < T) & (e[None, :] < E), other=0.0)
        if BF:
            zh = z.to(tl.bfloat16); zl = (z - zh.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(tl.trans(x), zh) + tl.dot(tl.trans(x), zl)
        else:
            acc += tl.dot(tl.trans(x), z, input_precision="ieee")
    tl.store(DRW + d[:, None].to(tl.int64) * E + e[None, :], acc, mask=(d[:, None] < D) & (e[None, :] < E))


def _select(s, b, A):
    E = s.shape[1]
    v = torch.where(s + b == 0, torch.zeros_like(s), s + b).contiguous()
    ib = v.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xffffffff)
    e = torch.arange(E, device=s.device, dtype=torch.int64)
    return torch.topk((u << 10) | (E - 1 - e), A, dim=1, largest=True, sorted=True).indices


def _metadata(sel, A, E, bm):
    flat = sel.reshape(-1); order = torch.argsort(flat); exp = flat.index_select(0, order)
    n = flat.numel(); tok = torch.div(order, A, rounding_mode="floor").to(torch.int32)
    off = torch.empty(E + 1, device=sel.device, dtype=torch.int32)
    _lower_bounds[(E,)](exp, off, N=n, E=E, num_warps=1); off[E] = n
    mt = triton.cdiv(n, bm) + E
    te = torch.empty(mt, device=sel.device, dtype=torch.int32); tr = torch.empty_like(te)
    _tile_table[(mt,)](off, te, tr, MT=mt, E=E, BM=bm, num_warps=1)
    inv = torch.empty(n, device=sel.device, dtype=torch.int64)
    _inverse[(triton.cdiv(n, 256),)](order, inv, N=n, B=256)
    return order, tok, off, te, tr, inv


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rw, rb, w1, w2, A):
        T, D = x.shape; E = rw.shape[1]; F = w2.shape[1]; A = int(A); N = T * A
        bf = x.dtype == torch.bfloat16
        s = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _router[(triton.cdiv(T, 32), triton.cdiv(E, 32))](x, rw, s, T=T, D=D, E=E, BM=32, BN=32, BK=32, num_warps=4)
        with torch.no_grad(): sel = _select(s, rb, A)
        weights = torch.empty((T, A), device=x.device, dtype=torch.float32)
        _route_weights[(T,)](s, sel, weights, T=T, E=E, A=A, BA=triton.next_power_of_2(A), num_warps=1)
        order, tok, off, te, tr, inv = _metadata(sel, A, E, 64); mt = te.numel()
        gu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _expert_in[(mt, triton.cdiv(2 * F, 64))](x, w1, tok, off, te, tr, gu, N=N, D=D, F2=2*F, BM=64, BN=64, BK=32, BF=bf, num_warps=4, num_stages=3)
        act = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _activation[(triton.cdiv(N * F, 256),)](gu, act, N=N, F=F, B=256)
        fe = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _expert_out[(mt, triton.cdiv(D, 64))](act, w2, off, te, tr, fe, N=N, F=F, D=D, BM=64, BN=64, BK=32, num_warps=4, num_stages=3)
        y = torch.empty((T, D), device=x.device, dtype=x.dtype)
        _combine[(T, triton.cdiv(D, 256))](fe, weights, inv, y, T=T, D=D, A=A, BD=256)
        ctx.save_for_backward(x, rw, w1, w2, s, sel, weights, order, tok, off, te, tr, inv, gu, act, fe)
        ctx.shape = T, D, E, F, A, bf
        return y

    @staticmethod
    def backward(ctx, dy):
        x, rw, w1, w2, s, sel, weights, order, tok, off, te, tr, inv, gu, act, fe = ctx.saved_tensors
        T, D, E, F, A, bf = ctx.shape; N = T * A; mt = te.numel()
        dl = torch.empty((T, A), device=x.device, dtype=torch.float32)
        _route_grad[(T,)](dy, fe, s, sel, weights, inv, dl, T=T, D=D, E=E, A=A, BA=triton.next_power_of_2(A), BD=128, num_warps=4)
        da = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _dact[(mt, triton.cdiv(F, 64))](dy, weights, order, w2, off, te, tr, da, N=N, F=F, D=D, A=A, BM=64, BN=64, BK=32, num_warps=4, num_stages=3)
        dgu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _dgu[(triton.cdiv(N * F, 256),)](gu, da, dgu, N=N, F=F, B=256)
        dx = torch.empty_like(x)
        _dx[(T, triton.cdiv(D, 128))](dgu, w1, dl, sel, inv, rw, dx, T=T, D=D, E=E, F2=2*F, A=A, BD=128, BK=32, num_warps=4)
        dw2 = torch.empty_like(w2)
        _dw2[(E, triton.cdiv(F, 64), triton.cdiv(D, 64))](act, dy, weights, order, off, dw2, E=E, F=F, D=D, A=A, BM=64, BN=64, BK=32, BF=bf, num_warps=4, num_stages=3)
        dw1 = torch.empty_like(w1)
        _dw1[(E, triton.cdiv(D, 64), triton.cdiv(2*F, 64))](x, dgu, tok, off, dw1, E=E, D=D, F2=2*F, BM=64, BN=64, BK=32, BF=bf, num_warps=4, num_stages=3)
        full = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _dense_dl[(T, triton.cdiv(E, 64))](dl, sel, full, T=T, E=E, A=A, BE=64)
        drw = torch.empty_like(rw)
        _drw[(triton.cdiv(D, 32), triton.cdiv(E, 32))](x, full, drw, T=T, D=D, E=E, BM=32, BN=32, BK=32, BF=bf, num_warps=4, num_stages=3)
        return dx, drw, None, dw1, dw2, None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
