import torch
import triton
import triton.language as tl


@triton.jit
def _mm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        sam, sak, sbk, sbn, scm, scn, IEEE: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak,
                    (rm[:, None] < M) & (rk[None, :] < K), 0.0)
        b = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn,
                    (rk[:, None] < K) & (rn[None, :] < N), 0.0)
        if IEEE:
            acc += tl.dot(a, b, input_precision="ieee")
        else:
            acc += tl.dot(a, b, input_precision="tf32x3")
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc,
             (rm[:, None] < M) & (rn[None, :] < N))


def _mm_launch(a, b, ieee):
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    _mm[(triton.cdiv(m, 64), triton.cdiv(n, 64))](
        a, b, c, M=m, N=n, K=k, sam=a.stride(0), sak=a.stride(1),
        sbk=b.stride(0), sbn=b.stride(1), scm=c.stride(0), scn=c.stride(1),
        IEEE=ieee, BM=64, BN=64, BK=32, num_warps=4, num_stages=3)
    return c


class _Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, ieee):
        ctx.save_for_backward(a, b)
        ctx.ieee = ieee
        return _mm_launch(a, b, ieee)

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        dc = dc.contiguous()
        return _mm_launch(dc, b.t(), ctx.ieee), _mm_launch(a.t(), dc, True), None


@triton.jit
def _dispatch_keys(SEL, KEYS, N: tl.constexpr, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    e = tl.load(SEL + q, q < N, 0).to(tl.int64)
    tl.store(KEYS + q, e * (N + 1) + q, q < N)


@triton.jit
def _gather(X, ORDER, SEL, XG, ES, INV, N: tl.constexpr, D: tl.constexpr,
            A: tl.constexpr, BD: tl.constexpr):
    r, pd = tl.program_id(0), tl.program_id(1)
    d = pd * BD + tl.arange(0, BD)
    q = tl.load(ORDER + r).to(tl.int64)
    t = q // A
    tl.store(XG + r * D + d, tl.load(X + t * D + d, d < D, 0.0), d < D)
    if pd == 0:
        tl.store(ES + r, tl.load(SEL + q))
        tl.store(INV + q, r)


@triton.jit
def _offsets(ES, OFF, N: tl.constexpr):
    e = tl.program_id(0)
    lo = 0
    hi = N
    while lo < hi:
        mid = (lo + hi) // 2
        v = tl.load(ES + mid)
        take = v < e
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(take, hi, mid)
    tl.store(OFF + e, lo)


@triton.jit
def _tile_map(OFF, TE, TS, E: tl.constexpr, CAP: tl.constexpr,
              BM: tl.constexpr):
    e = 0
    p = 0
    while e < E:
        lo = tl.load(OFF + e)
        hi = tl.load(OFF + e + 1)
        s = lo
        while s < hi:
            tl.store(TE + p, e)
            tl.store(TS + p, s)
            p += 1
            s += BM
        e += 1
    while p < CAP:
        tl.store(TE + p, -1)
        tl.store(TS + p, 0)
        p += 1


@triton.jit
def _seg_fwd(A, B, C, OFF, TE, TS,
             N: tl.constexpr, K: tl.constexpr, O: tl.constexpr,
             sa0, sa1, sb0, sb1, sb2, sc0, sc1,
             TRANS_B: tl.constexpr, SPLIT: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pt, pn = tl.program_id(0), tl.program_id(1)
    e = tl.load(TE + pt)
    start = tl.load(TS + pt)
    rm = start + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    hi = tl.load(OFF + e + 1, e >= 0, 0)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * sa0 + rk[None, :] * sa1,
                    (e >= 0) & (rm[:, None] < hi) & (rk[None, :] < K), 0.0)
        if TRANS_B:
            bp = B + e * sb0 + rn[None, :] * sb1 + rk[:, None] * sb2
        else:
            bp = B + e * sb0 + rk[:, None] * sb1 + rn[None, :] * sb2
        b = tl.load(bp, (e >= 0) & (rk[:, None] < K) & (rn[None, :] < O), 0.0)
        if SPLIT:
            ah = a.to(tl.bfloat16)
            al = (a - ah).to(tl.bfloat16)
            bh = b.to(tl.bfloat16)
            bl = (b - bh).to(tl.bfloat16)
            acc += tl.dot(ah, bh) + tl.dot(ah, bl) + tl.dot(al, bh)
        else:
            acc += tl.dot(a, b, input_precision="tf32x3")
    tl.store(C + rm[:, None] * sc0 + rn[None, :] * sc1, acc,
             (e >= 0) & (rm[:, None] < hi) & (rn[None, :] < O))


@triton.jit
def _seg_dw(A, DC, DB, OFF,
            K: tl.constexpr, O: tl.constexpr,
            sa0, sa1, sd0, sd1, sb0, sb1, sb2,
            BF: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
            BK: tl.constexpr):
    e, pk, po = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rk = pk * BM + tl.arange(0, BM)
    ro = po * BN + tl.arange(0, BN)
    lo = tl.load(OFF + e)
    hi = tl.load(OFF + e + 1)
    acc = tl.zeros((BM, BN), tl.float32)
    s = lo
    while s < hi:
        rr = s + tl.arange(0, BK)
        a = tl.load(A + rr[None, :] * sa0 + rk[:, None] * sa1,
                    (rr[None, :] < hi) & (rk[:, None] < K), 0.0)
        d = tl.load(DC + rr[:, None] * sd0 + ro[None, :] * sd1,
                    (rr[:, None] < hi) & (ro[None, :] < O), 0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            al = (a - ah).to(tl.bfloat16)
            dh = d.to(tl.bfloat16)
            dl = (d - dh).to(tl.bfloat16)
            acc += tl.dot(ah, dh) + tl.dot(ah, dl) + tl.dot(al, dh)
        else:
            acc += tl.dot(a, d, input_precision="ieee")
        s += BK
    tl.store(DB + e * sb0 + rk[:, None] * sb1 + ro[None, :] * sb2, acc,
             (rk[:, None] < K) & (ro[None, :] < O))


class _SegmentMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, off, te, ts, bf):
        n, k = a.shape
        e, _, o = b.shape
        c = torch.empty((n, o), device=a.device, dtype=torch.float32)
        cap = te.numel()
        _seg_fwd[(cap, triton.cdiv(o, 64))](
            a, b, c, off, te, ts, N=n, K=k, O=o,
            sa0=a.stride(0), sa1=a.stride(1), sb0=b.stride(0),
            sb1=b.stride(1), sb2=b.stride(2), sc0=c.stride(0), sc1=c.stride(1),
            TRANS_B=False, SPLIT=bf, BM=64, BN=64, BK=32,
            num_warps=4, num_stages=3)
        ctx.save_for_backward(a, b, off, te, ts)
        ctx.bf = bf
        return c

    @staticmethod
    def backward(ctx, dc):
        a, b, off, te, ts = ctx.saved_tensors
        dc = dc.contiguous()
        n, k = a.shape
        e, _, o = b.shape
        da = torch.empty_like(a)
        cap = te.numel()
        _seg_fwd[(cap, triton.cdiv(k, 64))](
            dc, b, da, off, te, ts, N=n, K=o, O=k,
            sa0=dc.stride(0), sa1=dc.stride(1), sb0=b.stride(0),
            sb1=b.stride(1), sb2=b.stride(2), sc0=da.stride(0), sc1=da.stride(1),
            TRANS_B=True, SPLIT=ctx.bf, BM=64, BN=64, BK=32,
            num_warps=4, num_stages=3)
        db = torch.empty_like(b)
        _seg_dw[(e, triton.cdiv(k, 128), triton.cdiv(o, 64))](
            a, dc, db, off, K=k, O=o,
            sa0=a.stride(0), sa1=a.stride(1), sd0=dc.stride(0), sd1=dc.stride(1),
            sb0=db.stride(0), sb1=db.stride(1), sb2=db.stride(2),
            BF=ctx.bf, BM=128, BN=64, BK=32, num_warps=4, num_stages=3)
        return da, db, None, None, None, None


@triton.jit
def _act_fwd(GU, OUT, N: tl.constexpr, F: tl.constexpr, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    m = q < N * F
    r, f = q // F, q % F
    g = tl.load(GU + r * (2 * F) + f, m, 0.0)
    u = tl.load(GU + r * (2 * F) + F + f, m, 0.0)
    tl.store(OUT + q, g * tl.sigmoid(g) * u, m)


@triton.jit
def _act_bwd(GU, DO, DGU, N: tl.constexpr, F: tl.constexpr, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    m = q < N * F
    r, f = q // F, q % F
    g = tl.load(GU + r * (2 * F) + f, m, 0.0)
    u = tl.load(GU + r * (2 * F) + F + f, m, 0.0)
    d = tl.load(DO + q, m, 0.0)
    s = tl.sigmoid(g)
    tl.store(DGU + r * (2 * F) + f, d * u * (s + g * s * (1.0 - s)), m)
    tl.store(DGU + r * (2 * F) + F + f, d * g * s, m)


class _Activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gu):
        n, f2 = gu.shape
        f = f2 // 2
        out = torch.empty((n, f), device=gu.device, dtype=torch.float32)
        _act_fwd[(triton.cdiv(n * f, 256),)](gu, out, N=n, F=f, BS=256)
        ctx.save_for_backward(gu)
        return out

    @staticmethod
    def backward(ctx, do):
        (gu,) = ctx.saved_tensors
        do = do.contiguous()
        n, f2 = gu.shape
        f = f2 // 2
        dgu = torch.empty_like(gu)
        _act_bwd[(triton.cdiv(n * f, 256),)](gu, do, dgu, N=n, F=f, BS=256)
        return dgu


@triton.jit
def _gather_bwd(DXG, INV, DX, T: tl.constexpr, D: tl.constexpr,
                A: tl.constexpr, BD: tl.constexpr):
    t, pd = tl.program_id(0), tl.program_id(1)
    d = pd * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    a = 0
    while a < A:
        r = tl.load(INV + t * A + a)
        acc += tl.load(DXG + r * D + d, d < D, 0.0)
        a += 1
    tl.store(DX + t * D + d, acc, d < D)


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, order, sel, inv, es, A):
        n = order.numel()
        d = x.shape[1]
        xg = torch.empty((n, d), device=x.device, dtype=torch.float32)
        _gather[(n, triton.cdiv(d, 256))](x, order, sel, xg, es, inv,
                                          N=n, D=d, A=A, BD=256)
        ctx.save_for_backward(inv)
        ctx.shape = x.shape
        ctx.A = A
        return xg

    @staticmethod
    def backward(ctx, dxg):
        (inv,) = ctx.saved_tensors
        t, d = ctx.shape
        dx = torch.empty((t, d), device=dxg.device, dtype=torch.float32)
        _gather_bwd[(t, triton.cdiv(d, 256))](dxg.contiguous(), inv, dx,
                                              T=t, D=d, A=ctx.A, BD=256)
        return dx, None, None, None, None, None


@triton.jit
def _combine(YP, W, INV, Y, T: tl.constexpr, D: tl.constexpr,
             A: tl.constexpr, BD: tl.constexpr):
    t, pd = tl.program_id(0), tl.program_id(1)
    d = pd * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    a = 0
    while a < A:
        r = tl.load(INV + t * A + a)
        z = tl.load(YP + r * D + d, d < D, 0.0)
        w = tl.load(W + t * A + a)
        acc += z * w
        a += 1
    tl.store(Y + t * D + d, acc, d < D)


@triton.jit
def _combine_dyp(DY, W, ORDER, DYP, N: tl.constexpr, D: tl.constexpr,
                 A: tl.constexpr, BD: tl.constexpr):
    r, pd = tl.program_id(0), tl.program_id(1)
    d = pd * BD + tl.arange(0, BD)
    q = tl.load(ORDER + r).to(tl.int64)
    t = q // A
    w = tl.load(W + q)
    v = tl.load(DY + t * D + d, d < D, 0.0)
    tl.store(DYP + r * D + d, v * w, d < D)


@triton.jit
def _combine_dw(DY, YP, INV, DW, D: tl.constexpr,
                A: tl.constexpr, BD: tl.constexpr):
    q = tl.program_id(0)
    t = q // A
    r = tl.load(INV + q)
    acc = 0.0
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        dy = tl.load(DY + t * D + d, d < D, 0.0)
        y = tl.load(YP + r * D + d, d < D, 0.0)
        acc += tl.sum(dy * y)
    tl.store(DW + q, acc)


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, yp, w, inv, order):
        t, a = w.shape
        d = yp.shape[1]
        y = torch.empty((t, d), device=yp.device, dtype=torch.float32)
        _combine[(t, triton.cdiv(d, 256))](yp, w, inv, y, T=t, D=d, A=a, BD=256)
        ctx.save_for_backward(yp, w, inv, order)
        return y

    @staticmethod
    def backward(ctx, dy):
        yp, w, inv, order = ctx.saved_tensors
        dy = dy.contiguous()
        n, d = yp.shape
        a = w.shape[1]
        dyp = torch.empty_like(yp)
        dw = torch.empty_like(w)
        _combine_dyp[(n, triton.cdiv(d, 256))](dy, w, order, dyp,
                                               N=n, D=d, A=a, BD=256)
        _combine_dw[(n,)](dy, yp, inv, dw, D=d, A=a, BD=256, num_warps=4)
        return dyp, dw, None, None


def _selection(s, bias, A):
    e = s.shape[1]
    b = s + bias
    b = torch.where(b == 0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    ei = torch.arange(e, device=s.device, dtype=torch.int64)
    return torch.topk((u << 10) | (e - 1 - ei), A, dim=1,
                      largest=True, sorted=True).indices


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rw, rb, w1, w2, A):
        A = int(A)
        bf = x.dtype == torch.bfloat16
        with torch.enable_grad():
            xf = x.detach().float().requires_grad_(True)
            rf = rw.detach().float().requires_grad_(True)
            b1 = w1.detach().float().requires_grad_(True)
            b2 = w2.detach().float().requires_grad_(True)
            t = xf.shape[0]
            e = rf.shape[1]
            s = torch.sigmoid(_Matmul.apply(xf, rf, True))
            with torch.no_grad():
                sel = _selection(s.detach(), rb, A).contiguous()
                n = t * A
                keys = torch.empty(n, device=x.device, dtype=torch.int64)
                _dispatch_keys[(triton.cdiv(n, 256),)](sel, keys, N=n, BS=256)
                order = torch.argsort(keys)
                inv = torch.empty(n, device=x.device, dtype=torch.int64)
                es = torch.empty(n, device=x.device, dtype=torch.int64)
            ss = torch.gather(s, 1, sel)
            weight = ss / ss.sum(1, keepdim=True)
            xg = _Gather.apply(xf, order, sel, inv, es, A)
            off = torch.empty(e + 1, device=x.device, dtype=torch.int32)
            _offsets[(e + 1,)](es, off, N=n)
            cap = triton.cdiv(n, 64) + e
            te = torch.empty(cap, device=x.device, dtype=torch.int32)
            ts = torch.empty(cap, device=x.device, dtype=torch.int32)
            _tile_map[(1,)](off, te, ts, E=e, CAP=cap, BM=64, num_warps=1)
            gu = _SegmentMM.apply(xg, b1, off, te, ts, bf)
            act = _Activation.apply(gu)
            yp = _SegmentMM.apply(act, b2, off, te, ts, bf)
            y = _Combine.apply(yp, weight, inv, order)
        ctx.save_for_backward(xf, rf, b1, b2, y)
        ctx.dtypes = x.dtype, rw.dtype, w1.dtype, w2.dtype
        return y.detach().to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, rf, b1, b2, y = ctx.saved_tensors
        grads = torch.autograd.grad(y, (xf, rf, b1, b2), dy.float())
        gx, gr, g1, g2 = (g.to(dt) for g, dt in zip(grads, ctx.dtypes))
        return gx, gr, None, g1, g2, None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
