import torch
import triton
import triton.language as tl

R, C, K = 64, 64, 32


@triton.jit
def _router(x, rw, s, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
            BM: tl.constexpr = 32, BN: tl.constexpr = 32, BK: tl.constexpr = 32):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    z = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(x + m[:, None] * D + k[None, :], mask=(m[:, None] < T) & (k[None, :] < D), other=0.0).to(tl.float32)
        b = tl.load(rw + k[:, None] * E + n[None, :], mask=(k[:, None] < D) & (n[None, :] < E), other=0.0).to(tl.float32)
        z += tl.dot(a, b, input_precision="ieee")
    tl.store(s + m[:, None] * E + n[None, :], tl.sigmoid(z), mask=(m[:, None] < T) & (n[None, :] < E))


@triton.jit
def _keys(s, bias, keys, size: tl.constexpr, E: tl.constexpr, B: tl.constexpr = 256):
    i = tl.program_id(0) * B + tl.arange(0, B)
    mask = i < size
    e = i % E
    v = tl.load(s + i, mask=mask, other=0.0) + tl.load(bias + e, mask=mask, other=0.0)
    v = tl.where(v == 0.0, 0.0, v)
    bits = v.to(tl.int32, bitcast=True)
    wide = bits.to(tl.int64)
    ordered = tl.where(bits >= 0, wide ^ 0x80000000, (~wide) & 0xffffffff)
    tl.store(keys + i, (ordered << 10) | (E - 1 - e).to(tl.int64), mask=mask)


@triton.jit
def _weights(s, sel, w, T: tl.constexpr, E: tl.constexpr, A: tl.constexpr,
             B: tl.constexpr = 256):
    t = tl.program_id(0) * B + tl.arange(0, B)
    mask = t < T
    den = tl.zeros((B,), tl.float32)
    for a in range(A):
        e = tl.load(sel + t * A + a, mask=mask, other=0)
        den += tl.load(s + t * E + e, mask=mask, other=0.0)
    for a in range(A):
        e = tl.load(sel + t * A + a, mask=mask, other=0)
        raw = tl.load(s + t * E + e, mask=mask, other=0.0)
        tl.store(w + t * A + a, raw / den, mask=mask)


@triton.jit
def _offsets(sel, order, off, N: tl.constexpr, E: tl.constexpr):
    e = tl.program_id(0)
    lo, hi = 0, N
    for _ in range(24):
        mid = (lo + hi) // 2
        q = tl.load(order + tl.minimum(mid, N - 1))
        v = tl.load(sel + q)
        left = (mid < N) & (v < e)
        lo = tl.where(left, mid + 1, lo)
        hi = tl.where(left, hi, mid)
    tl.store(off + e, lo)


@triton.jit
def _prefix(off, pre, E: tl.constexpr, BR: tl.constexpr):
    total = 0
    for e in range(E):
        tl.store(pre + e, total)
        total += tl.cdiv(tl.load(off + e + 1) - tl.load(off + e), BR)
    tl.store(pre + E, total)


@triton.jit
def _inverse(order, inv, N: tl.constexpr, B: tl.constexpr = 256):
    p = tl.program_id(0) * B + tl.arange(0, B)
    mask = p < N
    q = tl.load(order + p, mask=mask, other=0)
    tl.store(inv + q, p, mask=mask)


@triton.jit
def _location(pid, off, pre, E: tl.constexpr, BR: tl.constexpr):
    lo, hi = 0, E
    for _ in range(10):
        mid = (lo + hi) // 2
        left = tl.load(pre + tl.minimum(mid + 1, E)) <= pid
        lo = tl.where(left, mid + 1, lo)
        hi = tl.where(left, hi, mid)
    e = lo
    begin = tl.load(off + e) + (pid - tl.load(pre + e)) * BR
    return e, begin, tl.load(off + e + 1)


@triton.jit
def _expert_in(x, w1, order, off, pre, out, D: tl.constexpr, O: tl.constexpr,
               E: tl.constexpr, A: tl.constexpr, BF: tl.constexpr,
               BM: tl.constexpr = R, BN: tl.constexpr = C, BK: tl.constexpr = K):
    pid = tl.program_id(0)
    if pid >= tl.load(pre + E):
        return
    e, begin, end = _location(pid, off, pre, E, BM)
    p = begin + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    q = tl.load(order + p, mask=p < end, other=0)
    t = q // A
    z = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(x + t[:, None] * D + k[None, :], mask=(p[:, None] < end) & (k[None, :] < D), other=0.0)
        b = tl.load(w1 + e * D * O + k[:, None] * O + n[None, :], mask=(k[:, None] < D) & (n[None, :] < O), other=0.0)
        if BF:
            z += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
        else:
            z += tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="tf32x3")
    tl.store(out + p[:, None] * O + n[None, :], z, mask=(p[:, None] < end) & (n[None, :] < O))


@triton.jit
def _gather_x(x, order, off, pre, out, D: tl.constexpr, E: tl.constexpr,
              A: tl.constexpr, BM: tl.constexpr = R, BN: tl.constexpr = 256):
    pid = tl.program_id(0)
    if pid >= tl.load(pre + E):
        return
    _, begin, end = _location(pid, off, pre, E, BM)
    p = begin + tl.arange(0, BM)
    d = tl.program_id(1) * BN + tl.arange(0, BN)
    q = tl.load(order + p, mask=p < end, other=0)
    t = q // A
    v = tl.load(x + t[:, None] * D + d[None, :], mask=(p[:, None] < end) & (d[None, :] < D), other=0.0)
    tl.store(out + p[:, None] * D + d[None, :], v, mask=(p[:, None] < end) & (d[None, :] < D))


@triton.jit
def _activate(gu, act, N: tl.constexpr, F: tl.constexpr, B: tl.constexpr = 256):
    i = tl.program_id(0) * B + tl.arange(0, B)
    mask = i < N * F
    p, f = i // F, i % F
    g = tl.load(gu + p * 2 * F + f, mask=mask, other=0.0)
    u = tl.load(gu + p * 2 * F + F + f, mask=mask, other=0.0)
    tl.store(act + i, g * tl.sigmoid(g) * u, mask=mask)


@triton.jit
def _segment(a0, b0, off, pre, out, K0: tl.constexpr, N0: tl.constexpr,
             E: tl.constexpr, BF: tl.constexpr, BM: tl.constexpr = R,
             BN: tl.constexpr = C, BK: tl.constexpr = K):
    pid = tl.program_id(0)
    if pid >= tl.load(pre + E):
        return
    e, begin, end = _location(pid, off, pre, E, BM)
    m = begin + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    z = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K0, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(a0 + m[:, None] * K0 + k[None, :], mask=(m[:, None] < end) & (k[None, :] < K0), other=0.0).to(tl.float32)
        b = tl.load(b0 + e * K0 * N0 + k[:, None] * N0 + n[None, :], mask=(k[:, None] < K0) & (n[None, :] < N0), other=0.0).to(tl.float32)
        if BF:
            ah = a.to(tl.bfloat16)
            z += tl.dot(ah, b.to(tl.bfloat16))
            z += tl.dot((a - ah).to(tl.bfloat16), b.to(tl.bfloat16))
        else:
            z += tl.dot(a, b, input_precision="tf32x3")
    tl.store(out + m[:, None] * N0 + n[None, :], z, mask=(m[:, None] < end) & (n[None, :] < N0))


@triton.jit
def _segment_bt(a0, b0, off, pre, out, K0: tl.constexpr, N0: tl.constexpr,
                E: tl.constexpr, BF: tl.constexpr, BM: tl.constexpr = R,
                BN: tl.constexpr = C, BK: tl.constexpr = K):
    pid = tl.program_id(0)
    if pid >= tl.load(pre + E):
        return
    e, begin, end = _location(pid, off, pre, E, BM)
    m = begin + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    z = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K0, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(a0 + m[:, None] * K0 + k[None, :], mask=(m[:, None] < end) & (k[None, :] < K0), other=0.0).to(tl.float32)
        b = tl.load(b0 + e * N0 * K0 + n[None, :] * K0 + k[:, None], mask=(n[None, :] < N0) & (k[:, None] < K0), other=0.0).to(tl.float32)
        if BF:
            ah = a.to(tl.bfloat16)
            z += tl.dot(ah, b.to(tl.bfloat16))
            z += tl.dot((a - ah).to(tl.bfloat16), b.to(tl.bfloat16))
        else:
            z += tl.dot(a, b, input_precision="tf32x3")
    tl.store(out + m[:, None] * N0 + n[None, :], z, mask=(m[:, None] < end) & (n[None, :] < N0))


@triton.jit
def _combine(z0, w, inv, y, T: tl.constexpr, D: tl.constexpr, A: tl.constexpr,
             BD: tl.constexpr = 256):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    out = tl.zeros((BD,), tl.float32)
    for a in range(A):
        q = t * A + a
        p = tl.load(inv + q)
        out += tl.load(w + q) * tl.load(z0 + p * D + d, mask=d < D, other=0.0)
    tl.store(y + t * D + d, out, mask=d < D)


@triton.jit
def _dz(dy, w, order, dz, N: tl.constexpr, D: tl.constexpr, A: tl.constexpr,
        B: tl.constexpr = 256):
    i = tl.program_id(0) * B + tl.arange(0, B)
    mask = i < N * D
    p, d = i // D, i % D
    q = tl.load(order + p, mask=mask, other=0)
    t = q // A
    out = tl.load(w + q, mask=mask, other=0.0) * tl.load(dy + t * D + d, mask=mask, other=0.0).to(tl.float32)
    tl.store(dz + i, out, mask=mask)


@triton.jit
def _dgu(gu, da, dgu, N: tl.constexpr, F: tl.constexpr, B: tl.constexpr = 256):
    i = tl.program_id(0) * B + tl.arange(0, B)
    mask = i < N * F
    p, f = i // F, i % F
    g = tl.load(gu + p * 2 * F + f, mask=mask, other=0.0)
    u = tl.load(gu + p * 2 * F + F + f, mask=mask, other=0.0)
    v = tl.load(da + i, mask=mask, other=0.0)
    sig = tl.sigmoid(g)
    tl.store(dgu + p * 2 * F + f, v * u * (sig + g * sig * (1.0 - sig)), mask=mask)
    tl.store(dgu + p * 2 * F + F + f, v * g * sig, mask=mask)


@triton.jit
def _dw(a0, b0, off, out, K0: tl.constexpr, N0: tl.constexpr, E: tl.constexpr,
        BF: tl.constexpr, A_EXACT: tl.constexpr, BM: tl.constexpr = 128,
        BN: tl.constexpr = 64, BK: tl.constexpr = 32):
    e = tl.program_id(0)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    begin, end = tl.load(off + e), tl.load(off + e + 1)
    z = tl.zeros((BM, BN), tl.float32)
    for p0 in range(begin, end, BK):
        p = p0 + tl.arange(0, BK)
        a = tl.load(a0 + p[None, :] * K0 + m[:, None], mask=(p[None, :] < end) & (m[:, None] < K0), other=0.0).to(tl.float32)
        b = tl.load(b0 + p[:, None] * N0 + n[None, :], mask=(p[:, None] < end) & (n[None, :] < N0), other=0.0).to(tl.float32)
        if BF:
            ah, bh = a.to(tl.bfloat16), b.to(tl.bfloat16)
            z += tl.dot(ah, bh)
            if not A_EXACT:
                z += tl.dot((a - ah).to(tl.bfloat16), bh)
            z += tl.dot(ah, (b - bh).to(tl.bfloat16))
        else:
            z += tl.dot(a, b, input_precision="tf32x3")
    tl.store(out + e * K0 * N0 + m[:, None] * N0 + n[None, :], z, mask=(m[:, None] < K0) & (n[None, :] < N0))


@triton.jit
def _pair_dot(z0, dy, order, dots, N: tl.constexpr, D: tl.constexpr,
              A: tl.constexpr, BK: tl.constexpr = 256):
    p = tl.program_id(0)
    q = tl.load(order + p)
    t = q // A
    acc = 0.0
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        acc += tl.sum(tl.load(z0 + p * D + k, mask=k < D, other=0.0) * tl.load(dy + t * D + k, mask=k < D, other=0.0).to(tl.float32), axis=0)
    tl.store(dots + q, acc)


@triton.jit
def _route(s, sel, w, dots, ds, T: tl.constexpr, E: tl.constexpr,
           A: tl.constexpr, BE: tl.constexpr = 64):
    t = tl.program_id(0)
    eout = tl.program_id(1) * BE + tl.arange(0, BE)
    den, mean = 0.0, 0.0
    for a in range(A):
        e = tl.load(sel + t * A + a)
        den += tl.load(s + t * E + e)
        mean += tl.load(w + t * A + a) * tl.load(dots + t * A + a)
    out = tl.zeros((BE,), tl.float32)
    for a in range(A):
        e = tl.load(sel + t * A + a)
        raw = tl.load(s + t * E + e)
        v = (tl.load(dots + t * A + a) - mean) / den * raw * (1.0 - raw)
        out += tl.where(eout == e, v, 0.0)
    tl.store(ds + t * E + eout, out, mask=eout < E)


@triton.jit
def _dense(a0, b0, out, M: tl.constexpr, N: tl.constexpr, K0: tl.constexpr,
           TA: tl.constexpr, TB: tl.constexpr, BM: tl.constexpr = 64,
           BN: tl.constexpr = 64, BK: tl.constexpr = 32):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    z = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K0, BK):
        k = k0 + tl.arange(0, BK)
        if TA:
            a = tl.load(a0 + k[None, :] * M + m[:, None], mask=(m[:, None] < M) & (k[None, :] < K0), other=0.0).to(tl.float32)
        else:
            a = tl.load(a0 + m[:, None] * K0 + k[None, :], mask=(m[:, None] < M) & (k[None, :] < K0), other=0.0).to(tl.float32)
        if TB:
            b = tl.load(b0 + n[None, :] * K0 + k[:, None], mask=(k[:, None] < K0) & (n[None, :] < N), other=0.0).to(tl.float32)
        else:
            b = tl.load(b0 + k[:, None] * N + n[None, :], mask=(k[:, None] < K0) & (n[None, :] < N), other=0.0).to(tl.float32)
        z += tl.dot(a, b, input_precision="ieee")
    tl.store(out + m[:, None] * N + n[None, :], z, mask=(m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _sum_dx(dxp, inv, dxr, dx, T: tl.constexpr, D: tl.constexpr,
            A: tl.constexpr, BD: tl.constexpr = 256):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    out = tl.load(dxr + t * D + d, mask=d < D, other=0.0)
    for a in range(A):
        out += tl.load(dxp + tl.load(inv + t * A + a) * D + d, mask=d < D, other=0.0)
    tl.store(dx + t * D + d, out, mask=d < D)


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rw, rb, w1, w2, A):
        T, D = x.shape
        E, F, A = rw.shape[1], w2.shape[1], int(A)
        N, cap = T * A, triton.cdiv(T * A, R) + E
        bf = x.dtype == torch.bfloat16
        s = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _router[(triton.cdiv(T, 32), triton.cdiv(E, 32))](x, rw, s, T=T, D=D, E=E)
        keys = torch.empty((T, E), device=x.device, dtype=torch.int64)
        _keys[(triton.cdiv(T * E, 256),)](s, rb, keys, size=T * E, E=E)
        sel = torch.topk(keys, A, dim=1, largest=True, sorted=True).indices
        w = torch.empty((T, A), device=x.device, dtype=torch.float32)
        _weights[(triton.cdiv(T, 256),)](s, sel, w, T=T, E=E, A=A)
        flat = sel.reshape(-1)
        order = torch.argsort(flat)
        off = torch.empty(E + 1, device=x.device, dtype=torch.int32)
        _offsets[(E + 1,)](flat, order, off, N=N, E=E)
        pre = torch.empty(E + 1, device=x.device, dtype=torch.int32)
        _prefix[(1,)](off, pre, E=E, BR=R, num_warps=1)
        inv = torch.empty(N, device=x.device, dtype=torch.int64)
        _inverse[(triton.cdiv(N, 256),)](order, inv, N=N)
        gu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32)
        _expert_in[(cap, triton.cdiv(2 * F, C))](x, w1, order, off, pre, gu, D=D, O=2 * F, E=E, A=A, BF=bf, num_warps=4, num_stages=3)
        act = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _activate[(triton.cdiv(N * F, 256),)](gu, act, N=N, F=F)
        z = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _segment[(cap, triton.cdiv(D, C))](act, w2, off, pre, z, K0=F, N0=D, E=E, BF=bf, num_warps=4, num_stages=3)
        y = torch.empty_like(x)
        _combine[(T, triton.cdiv(D, 256))](z, w, inv, y, T=T, D=D, A=A)
        ctx.save_for_backward(x, rw, w1, w2, s, sel, w, order, off, pre, inv, gu, act, z)
        ctx.dims = T, D, E, F, A, cap
        return y

    @staticmethod
    def backward(ctx, dy):
        x, rw, w1, w2, s, sel, w, order, off, pre, inv, gu, act, z = ctx.saved_tensors
        T, D, E, F, A, cap = ctx.dims
        N = T * A
        bf = x.dtype == torch.bfloat16
        dy = dy.contiguous()
        dz = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _dz[(triton.cdiv(N * D, 256),)](dy, w, order, dz, N=N, D=D, A=A)
        da = torch.empty((N, F), device=x.device, dtype=torch.float32)
        _segment_bt[(cap, triton.cdiv(F, C))](dz, w2, off, pre, da, K0=D, N0=F, E=E, BF=bf, num_warps=4, num_stages=3)
        dgu = torch.empty_like(gu)
        _dgu[(triton.cdiv(N * F, 256),)](gu, da, dgu, N=N, F=F)
        dxp = torch.empty((N, D), device=x.device, dtype=torch.float32)
        _segment_bt[(cap, triton.cdiv(D, C))](dgu, w1, off, pre, dxp, K0=2 * F, N0=D, E=E, BF=bf, num_warps=4, num_stages=3)
        dw2 = torch.empty_like(w2)
        _dw[(E, triton.cdiv(F, 128), triton.cdiv(D, 64))](act, dz, off, dw2, K0=F, N0=D, E=E, BF=bf, A_EXACT=False, num_warps=4)
        xs = torch.empty((N, D), device=x.device, dtype=x.dtype)
        _gather_x[(cap, triton.cdiv(D, 256))](x, order, off, pre, xs, D=D, E=E, A=A)
        dw1 = torch.empty_like(w1)
        _dw[(E, triton.cdiv(D, 128), triton.cdiv(2 * F, 64))](xs, dgu, off, dw1, K0=D, N0=2 * F, E=E, BF=bf, A_EXACT=bf, num_warps=4)
        dots = torch.empty(N, device=x.device, dtype=torch.float32)
        _pair_dot[(N,)](z, dy, order, dots, N=N, D=D, A=A)
        ds = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _route[(T, triton.cdiv(E, 64))](s, sel, w, dots, ds, T=T, E=E, A=A)
        drw = torch.empty_like(rw)
        _dense[(triton.cdiv(D, 64), triton.cdiv(E, 64))](x, ds, drw, M=D, N=E, K0=T, TA=True, TB=False)
        dxr = torch.empty((T, D), device=x.device, dtype=torch.float32)
        _dense[(triton.cdiv(T, 64), triton.cdiv(D, 64))](ds, rw, dxr, M=T, N=D, K0=E, TA=False, TB=True)
        dx = torch.empty_like(x)
        _sum_dx[(T, triton.cdiv(D, 256))](dxp, inv, dxr, dx, T=T, D=D, A=A)
        return dx, drw, None, dw1, dw2, None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
