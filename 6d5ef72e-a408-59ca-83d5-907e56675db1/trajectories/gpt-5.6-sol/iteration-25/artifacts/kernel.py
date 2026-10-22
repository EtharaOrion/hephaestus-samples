import torch
import triton
import triton.language as tl

ROW = 64
COL = 64
KB = 32

@triton.jit
def _router(X, RW, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    im = tl.program_id(0) * BM + tl.arange(0, BM); ie = tl.program_id(1) * BN + tl.arange(0, BN)
    z = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        ik = k + tl.arange(0, BK)
        x = tl.load(X + im[:, None] * D + ik[None, :], mask=(im[:, None] < T) & (ik[None, :] < D), other=0.).to(tl.float32)
        w = tl.load(RW + ik[:, None] * E + ie[None, :], mask=(ik[:, None] < D) & (ie[None, :] < E), other=0.).to(tl.float32)
        z += tl.dot(x, w, input_precision="ieee")
    tl.store(S + im[:, None] * E + ie[None, :], tl.sigmoid(z), mask=(im[:, None] < T) & (ie[None, :] < E))

@triton.jit
def _keys(S, B, O, n: tl.constexpr, E: tl.constexpr, BLOCK: tl.constexpr):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = q < n; e = q % E
    v = tl.load(S + q, mask=m, other=0.) + tl.load(B + e, mask=m, other=0.); v = tl.where(v == 0., 0., v)
    i = v.to(tl.int32, bitcast=True); j = i.to(tl.int64) & 0xffffffff
    u = tl.where(i >= 0, j ^ 0x80000000, (~j) & 0xffffffff)
    tl.store(O + q, (u << 10) | (E - 1 - e), mask=m)

@triton.jit
def _route_weights(S, SEL, W, T: tl.constexpr, E: tl.constexpr, A: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = t < T
    den = tl.zeros((BLOCK,), tl.float32)
    for a in tl.static_range(A):
        e = tl.load(SEL + t * A + a, mask=m, other=0); den += tl.load(S + t * E + e, mask=m, other=0.)
    for a in tl.static_range(A):
        e = tl.load(SEL + t * A + a, mask=m, other=0); s = tl.load(S + t * E + e, mask=m, other=0.)
        tl.store(W + t * A + a, s / den, mask=m)

@triton.jit
def _offsets(SEL, ORD, OFF, N: tl.constexpr, E: tl.constexpr):
    e = tl.program_id(0); lo = 0; hi = N
    while lo < hi:
        md = (lo + hi) // 2; p = tl.load(ORD + md); v = tl.load(SEL + p); take = v < e
        lo = tl.where(take, md + 1, lo); hi = tl.where(take, hi, md)
    tl.store(OFF + e, lo)

@triton.jit
def _tiles(OFF, TE, TS, CAP: tl.constexpr, E: tl.constexpr, R: tl.constexpr):
    for q0 in range(0, CAP):
        tl.store(TE + q0, -1); tl.store(TS + q0, 0)
    q = 0
    for e in range(0, E):
        p = tl.load(OFF + e); end = tl.load(OFF + e + 1)
        while p < end:
            tl.store(TE + q, e); tl.store(TS + q, p); q += 1; p += R

@triton.jit
def _inverse(ORD, INV, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = i < N; p = tl.load(ORD + i, mask=m, other=0)
    tl.store(INV + p, i, mask=m)

@triton.jit
def _expert_in(X, W1, ORD, OFF, TE, TS, GU, N: tl.constexpr, D: tl.constexpr, F2: tl.constexpr, A: tl.constexpr, R: tl.constexpr, C: tl.constexpr, K: tl.constexpr, BF: tl.constexpr):
    tile = tl.program_id(0); e = tl.load(TE + tile); start = tl.load(TS + tile)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    ir = tl.arange(0, R); ic = tl.program_id(1) * C + tl.arange(0, C); pos = start + ir; valid = (e >= 0) & (pos < end)
    pair = tl.load(ORD + pos, mask=valid, other=0); tok = pair // A; z = tl.zeros((R, C), tl.float32)
    for k in range(0, D, K):
        ik = k + tl.arange(0, K)
        x = tl.load(X + tok[:, None] * D + ik[None, :], mask=valid[:, None] & (ik[None, :] < D), other=0.)
        w = tl.load(W1 + e * D * F2 + ik[:, None] * F2 + ic[None, :], mask=(e >= 0) & (ik[:, None] < D) & (ic[None, :] < F2), other=0.)
        if BF: z += tl.dot(x, w)
        else: z += tl.dot(x.to(tl.float32), w.to(tl.float32), input_precision="tf32x3")
    tl.store(GU + pos[:, None] * F2 + ic[None, :], z, mask=valid[:, None] & (ic[None, :] < F2))

@triton.jit
def _activate(GU, AC, n: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = q < n; r = q // F; f = q % F
    g = tl.load(GU + r * (2 * F) + f, mask=m, other=0.); u = tl.load(GU + r * (2 * F) + F + f, mask=m, other=0.)
    tl.store(AC + q, g * tl.sigmoid(g) * u, mask=m)

@triton.jit
def _segment(A0, B0, OFF, TE, TS, O, N: tl.constexpr, K0: tl.constexpr, N0: tl.constexpr, R: tl.constexpr, C: tl.constexpr, K: tl.constexpr, BF: tl.constexpr):
    tile = tl.program_id(0); e = tl.load(TE + tile); start = tl.load(TS + tile)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    ir = tl.arange(0, R); ic = tl.program_id(1) * C + tl.arange(0, C); pos = start + ir; valid = (e >= 0) & (pos < end)
    z = tl.zeros((R, C), tl.float32)
    for k in range(0, K0, K):
        ik = k + tl.arange(0, K)
        a = tl.load(A0 + pos[:, None] * K0 + ik[None, :], mask=valid[:, None] & (ik[None, :] < K0), other=0.)
        b = tl.load(B0 + e * K0 * N0 + ik[:, None] * N0 + ic[None, :], mask=(e >= 0) & (ik[:, None] < K0) & (ic[None, :] < N0), other=0.)
        if BF:
            ah = a.to(tl.bfloat16)
            z += tl.dot(ah, b) + tl.dot((a - ah).to(tl.bfloat16), b)
        else: z += tl.dot(a, b.to(tl.float32), input_precision="tf32x3")
    tl.store(O + pos[:, None] * N0 + ic[None, :], z, mask=valid[:, None] & (ic[None, :] < N0))

@triton.jit
def _combine(Z, W, INV, Y, T: tl.constexpr, D: tl.constexpr, A: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0); d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK); m = d < D; z = tl.zeros((BLOCK,), tl.float32)
    for a in tl.static_range(A):
        p = t * A + a; i = tl.load(INV + p); w = tl.load(W + p); z += w * tl.load(Z + i * D + d, mask=m, other=0.)
    tl.store(Y + t * D + d, z, mask=m)

@triton.jit
def _dz(DY, W, ORD, DZ, N: tl.constexpr, D: tl.constexpr, A: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0); d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK); m = d < D
    p = tl.load(ORD + i); t = p // A; w = tl.load(W + p); v = tl.load(DY + t * D + d, mask=m, other=0.).to(tl.float32)
    tl.store(DZ + i * D + d, w * v, mask=m)

@triton.jit
def _segment_back(A0, B0, OFF, TE, TS, O, N: tl.constexpr, K0: tl.constexpr, N0: tl.constexpr, R: tl.constexpr, C: tl.constexpr, K: tl.constexpr, BF: tl.constexpr):
    tile = tl.program_id(0); e = tl.load(TE + tile); start = tl.load(TS + tile)
    end = tl.load(OFF + e + 1, mask=e >= 0, other=0)
    ir = tl.arange(0, R); jn = tl.program_id(1) * C + tl.arange(0, C); pos = start + ir; valid = (e >= 0) & (pos < end)
    z = tl.zeros((R, C), tl.float32)
    for k in range(0, K0, K):
        ik = k + tl.arange(0, K)
        a = tl.load(A0 + pos[:, None] * K0 + ik[None, :], mask=valid[:, None] & (ik[None, :] < K0), other=0.)
        b = tl.load(B0 + e * N0 * K0 + jn[None, :] * K0 + ik[:, None], mask=(e >= 0) & (jn[None, :] < N0) & (ik[:, None] < K0), other=0.)
        if BF: z += tl.dot(a.to(tl.bfloat16), b)
        else: z += tl.dot(a, b.to(tl.float32), input_precision="tf32x3")
    tl.store(O + pos[:, None] * N0 + jn[None, :], z, mask=valid[:, None] & (jn[None, :] < N0))

@triton.jit
def _dactivate(GU, DA, DGU, n: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = q < n; r = q // F; f = q % F
    g = tl.load(GU + r * (2 * F) + f, mask=m, other=0.); u = tl.load(GU + r * (2 * F) + F + f, mask=m, other=0.); da = tl.load(DA + q, mask=m, other=0.)
    s = tl.sigmoid(g); silu = g * s
    tl.store(DGU + r * (2 * F) + f, da * u * (s + silu * (1. - s)), mask=m); tl.store(DGU + r * (2 * F) + F + f, da * silu, mask=m)

@triton.jit
def _dw(A0, B0, ORD, OFF, OUT, M: tl.constexpr, NN: tl.constexpr, ASEL: tl.constexpr, GATHER: tl.constexpr, BF: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(0); im = tl.program_id(1) * BM + tl.arange(0, BM); jn = tl.program_id(2) * BN + tl.arange(0, BN)
    begin = tl.load(OFF + e); end = tl.load(OFF + e + 1); z = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        ik = k + tl.arange(0, BK); km = ik < end
        if GATHER:
            p = tl.load(ORD + ik, mask=km, other=0); row = p // ASEL
            a = tl.load(A0 + row[None, :] * M + im[:, None], mask=(im[:, None] < M) & km[None, :], other=0.)
        else: a = tl.load(A0 + ik[None, :] * M + im[:, None], mask=(im[:, None] < M) & km[None, :], other=0.)
        b = tl.load(B0 + ik[:, None] * NN + jn[None, :], mask=km[:, None] & (jn[None, :] < NN), other=0.)
        if BF:
            ah = a.to(tl.bfloat16); bh = b.to(tl.bfloat16); bl = (b - bh).to(tl.bfloat16)
            z += tl.dot(ah, bh) + tl.dot(ah, bl)
            if not GATHER: z += tl.dot((a - ah).to(tl.bfloat16), bh)
        else: z += tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="tf32x3")
    tl.store(OUT + e * M * NN + im[:, None] * NN + jn[None, :], z, mask=(im[:, None] < M) & (jn[None, :] < NN))

@triton.jit
def _pair_dot(Z, INV, DY, C, N: tl.constexpr, D: tl.constexpr,
              A: tl.constexpr, BD: tl.constexpr):
    p = tl.program_id(0)
    i = tl.load(INV + p)
    t = p // A
    acc = tl.zeros((BD,), tl.float32)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        z = tl.load(Z + i * D + d, mask=d < D, other=0.)
        dy = tl.load(DY + t * D + d, mask=d < D, other=0.).to(tl.float32)
        acc += z * dy
    tl.store(C + p, tl.sum(acc))

@triton.jit
def _route_grad(S, SEL, W, C, DS, T: tl.constexpr, E: tl.constexpr,
                A: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = t < T
    mean = tl.zeros((BLOCK,), tl.float32)
    den = tl.zeros((BLOCK,), tl.float32)
    for a in tl.static_range(A):
        p = t * A + a
        mean += tl.load(W + p, mask=m, other=0.) * tl.load(C + p, mask=m, other=0.)
        e = tl.load(SEL + p, mask=m, other=0)
        den += tl.load(S + t * E + e, mask=m, other=0.)
    for a in tl.static_range(A):
        p = t * A + a
        e = tl.load(SEL + p, mask=m, other=0)
        s = tl.load(S + t * E + e, mask=m, other=0.)
        c = tl.load(C + p, mask=m, other=0.)
        tl.store(DS + p, (c - mean) / den * s * (1. - s), mask=m)

@triton.jit
def _drw(X, SEL, DS, OUT, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr, A: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    id = tl.program_id(0) * BM + tl.arange(0, BM); ie = tl.program_id(1) * BN + tl.arange(0, BN); z = tl.zeros((BM, BN), tl.float32)
    for k in range(0, T, BK):
        it = k + tl.arange(0, BK); x = tl.load(X + it[None, :] * D + id[:, None], mask=(id[:, None] < D) & (it[None, :] < T), other=0.).to(tl.float32); q = tl.zeros((BK, BN), tl.float32)
        for a in tl.static_range(A):
            se = tl.load(SEL + it[:, None] * A + a, mask=it[:, None] < T, other=-1); ds = tl.load(DS + it[:, None] * A + a, mask=it[:, None] < T, other=0.); q += tl.where(se == ie[None, :], ds, 0.)
        z += tl.dot(x, q, input_precision="ieee")
    tl.store(OUT + id[:, None] * E + ie[None, :], z, mask=(id[:, None] < D) & (ie[None, :] < E))

@triton.jit
def _dx_combine(DXR, INV, RW, SEL, DS, OUT, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr, A: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    it = tl.program_id(0) * BM + tl.arange(0, BM); jd = tl.program_id(1) * BN + tl.arange(0, BN); z = tl.zeros((BM, BN), tl.float32)
    for a in tl.static_range(A):
        p = it * A + a; i = tl.load(INV + p, mask=it < T, other=0); z += tl.load(DXR + i[:, None] * D + jd[None, :], mask=(it[:, None] < T) & (jd[None, :] < D), other=0.)
    for k in range(0, E, BK):
        ie = k + tl.arange(0, BK); q = tl.zeros((BM, BK), tl.float32)
        for a in tl.static_range(A):
            se = tl.load(SEL + it[:, None] * A + a, mask=it[:, None] < T, other=-1); ds = tl.load(DS + it[:, None] * A + a, mask=it[:, None] < T, other=0.); q += tl.where(se == ie[None, :], ds, 0.)
        w = tl.load(RW + jd[None, :] * E + ie[:, None], mask=(jd[None, :] < D) & (ie[:, None] < E), other=0.).to(tl.float32); z += tl.dot(q, w, input_precision="ieee")
    tl.store(OUT + it[:, None] * D + jd[None, :], z, mask=(it[:, None] < T) & (jd[None, :] < D))

class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rw, rb, w1, w2, experts_per_token):
        A = int(experts_per_token); T, D = x.shape; E = rw.shape[1]; F = w2.shape[1]; N = T * A; bf = x.dtype == torch.bfloat16
        s = torch.empty((T, E), device=x.device, dtype=torch.float32); _router[(triton.cdiv(T, 32), triton.cdiv(E, 32))](x, rw, s, T, D, E, 32, 32, 32)
        keys = torch.empty((T, E), device=x.device, dtype=torch.int64); _keys[(triton.cdiv(T * E, 256),)](s, rb, keys, T * E, E, 256)
        sel = torch.topk(keys, A, dim=1, largest=True, sorted=True).indices.contiguous(); weight = torch.empty((T, A), device=x.device, dtype=torch.float32); _route_weights[(triton.cdiv(T, 256),)](s, sel, weight, T, E, A, 256)
        order = torch.argsort(sel.reshape(-1)).contiguous(); off = torch.empty(E + 1, device=x.device, dtype=torch.int32); _offsets[(E + 1,)](sel, order, off, N, E, num_warps=1)
        cap = triton.cdiv(N, ROW) + E; te = torch.empty(cap, device=x.device, dtype=torch.int32); ts = torch.empty(cap, device=x.device, dtype=torch.int32); _tiles[(1,)](off, te, ts, cap, E, ROW, num_warps=1)
        inv = torch.empty(N, device=x.device, dtype=torch.int64); _inverse[(triton.cdiv(N, 256),)](order, inv, N, 256)
        gu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32); _expert_in[(cap, triton.cdiv(2 * F, COL))](x, w1, order, off, te, ts, gu, N, D, 2 * F, A, ROW, COL, KB, bf, num_warps=4, num_stages=3)
        ac = torch.empty((N, F), device=x.device, dtype=torch.float32); _activate[(triton.cdiv(N * F, 256),)](gu, ac, N * F, F, 256)
        z = torch.empty((N, D), device=x.device, dtype=torch.float32); _segment[(cap, triton.cdiv(D, COL))](ac, w2, off, te, ts, z, N, F, D, ROW, COL, KB, bf, num_warps=4, num_stages=3)
        y = torch.empty((T, D), device=x.device, dtype=x.dtype); _combine[(T, triton.cdiv(D, 256))](z, weight, inv, y, T, D, A, 256)
        ctx.save_for_backward(x, rw, w1, w2, s, sel, weight, order, off, te, ts, inv, gu, ac, z); ctx.meta = (T, D, E, F, A, N, cap, bf); return y
    @staticmethod
    def backward(ctx, dy):
        x, rw, w1, w2, s, sel, weight, order, off, te, ts, inv, gu, ac, z = ctx.saved_tensors; T, D, E, F, A, N, cap, bf = ctx.meta; dy = dy.contiguous()
        dz = torch.empty((N, D), device=x.device, dtype=torch.float32); _dz[(N, triton.cdiv(D, 256))](dy, weight, order, dz, N, D, A, 256)
        da = torch.empty((N, F), device=x.device, dtype=torch.float32); _segment_back[(cap, triton.cdiv(F, COL))](dz, w2, off, te, ts, da, N, D, F, ROW, COL, KB, bf, num_warps=4, num_stages=3)
        dgu = torch.empty((N, 2 * F), device=x.device, dtype=torch.float32); _dactivate[(triton.cdiv(N * F, 256),)](gu, da, dgu, N * F, F, 256)
        dxr = torch.empty((N, D), device=x.device, dtype=torch.float32); _segment_back[(cap, triton.cdiv(D, COL))](dgu, w1, off, te, ts, dxr, N, 2 * F, D, ROW, COL, KB, bf, num_warps=4, num_stages=3)
        gw2 = torch.empty_like(w2); _dw[(E, triton.cdiv(F, 128), triton.cdiv(D, 64))](ac, dz, order, off, gw2, F, D, A, False, bf, 128, 64, KB, num_warps=4, num_stages=3)
        gw1 = torch.empty_like(w1); _dw[(E, triton.cdiv(D, 128), triton.cdiv(2 * F, 64))](x, dgu, order, off, gw1, D, 2 * F, A, True, bf, 128, 64, KB, num_warps=4, num_stages=3)
        pair_c = torch.empty(N, device=x.device, dtype=torch.float32); _pair_dot[(N,)](z, inv, dy, pair_c, N, D, A, 256, num_warps=4)
        ds = torch.empty((T, A), device=x.device, dtype=torch.float32); _route_grad[(triton.cdiv(T, 256),)](s, sel, weight, pair_c, ds, T, E, A, 256)
        grw = torch.empty_like(rw); _drw[(triton.cdiv(D, 32), triton.cdiv(E, 32))](x, sel, ds, grw, T, D, E, A, 32, 32, 32, num_warps=4, num_stages=3)
        gx = torch.empty_like(x); _dx_combine[(triton.cdiv(T, 16), triton.cdiv(D, 64))](dxr, inv, rw, sel, ds, gx, T, D, E, A, 16, 64, 32, num_warps=4, num_stages=3)
        return gx, grw, None, gw1, gw2, None

def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
