import torch
import triton
import triton.language as tl

R, C, KB = 64, 64, 32

@triton.jit
def _router(X, RW, S, T, D, E, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    z = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(X + m[:, None] * D + k[None, :], mask=(m[:, None] < T) & (k[None, :] < D), other=0.).to(tl.float32)
        b = tl.load(RW + k[:, None] * E + n[None, :], mask=(k[:, None] < D) & (n[None, :] < E), other=0.).to(tl.float32)
        z += tl.dot(a, b, input_precision="ieee")
    tl.store(S + m[:, None] * E + n[None, :], tl.sigmoid(z), mask=(m[:, None] < T) & (n[None, :] < E))

@triton.jit
def _keys(S, B, O, size, E, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    mask = q < size
    e = q % E
    v = tl.load(S + q, mask=mask, other=0.) + tl.load(B + e, mask=mask, other=0.)
    v = tl.where(v == 0., 0., v)
    i = v.to(tl.int32, bitcast=True)
    u = tl.where(i >= 0, i.to(tl.int64) ^ 0x80000000, (~i).to(tl.int64) & 0xffffffff)
    tl.store(O + q, (u << 10) | (E - 1 - e).to(tl.int64), mask=mask)

@triton.jit
def _weights(S, SEL, W, T, E, A, BA: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BA)
    mask = a < A
    e = tl.load(SEL + t * A + a, mask=mask, other=0).to(tl.int64)
    v = tl.load(S + t * E + e, mask=mask, other=0.)
    tl.store(W + t * A + a, v / tl.sum(v, axis=0), mask=mask)

@triton.jit
def _offsets(FLAT, ORDER, OFF, N, E, IT: tl.constexpr):
    e = tl.program_id(0)
    lo, hi = 0, N
    for _ in range(IT):
        mid = (lo + hi) // 2
        q = tl.load(ORDER + mid, mask=mid < N, other=0)
        v = tl.load(FLAT + q, mask=mid < N, other=E)
        take = (mid < N) & (v < e)
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(take, hi, mid)
    tl.store(OFF + e, lo)

@triton.jit
def _prefix(OFF, PRE, E, BR: tl.constexpr):
    z = 0
    tl.store(PRE, 0)
    for e in range(0, E):
        b, end = tl.load(OFF + e), tl.load(OFF + e + 1)
        z += (end - b + BR - 1) // BR
        tl.store(PRE + e + 1, z)

@triton.jit
def _loc(pid, OFF, PRE, E, IT: tl.constexpr, BR: tl.constexpr):
    lo, hi = 0, E
    for _ in range(IT):
        mid = (lo + hi) // 2
        v = tl.load(PRE + mid + 1, mask=mid < E, other=0x7fffffff)
        take = v <= pid
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(take, hi, mid)
    active = pid < tl.load(PRE + E)
    expert = tl.where(active, lo, E - 1)
    start = tl.load(OFF + expert) + (pid - tl.load(PRE + expert)) * BR
    start = tl.where(active, start, tl.load(OFF + E))
    return expert, start

@triton.jit
def _ein(X, W, ORDER, OFF, PRE, OUT, D, O, E, A, BF: tl.constexpr, IT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e, p0 = _loc(tl.program_id(0), OFF, PRE, E, IT, BM)
    rm = p0 + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    end = tl.load(OFF + e + 1)
    q = tl.load(ORDER + rm, mask=rm < end, other=0)
    tok = q // A
    z = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(X + tok[:, None] * D + k[None, :], mask=(rm[:, None] < end) & (k[None, :] < D), other=0.)
        b = tl.load(W + e * D * O + k[:, None] * O + rn[None, :], mask=(k[:, None] < D) & (rn[None, :] < O), other=0.)
        if BF: z += tl.dot(a, b)
        else: z += tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="tf32x3")
    tl.store(OUT + q[:, None] * O + rn[None, :], z, mask=(rm[:, None] < end) & (rn[None, :] < O))

@triton.jit
def _act(GU, OUT, N, F, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    mask = q < N * F
    r, f = q // F, q % F
    g = tl.load(GU + r * 2 * F + f, mask=mask, other=0.)
    u = tl.load(GU + r * 2 * F + F + f, mask=mask, other=0.)
    tl.store(OUT + q, g * tl.sigmoid(g) * u, mask=mask)

@triton.jit
def _seg(AIN, W, ORDER, OFF, PRE, OUT, K, N, E, BF: tl.constexpr, SPLIT: tl.constexpr, IT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e, p0 = _loc(tl.program_id(0), OFF, PRE, E, IT, BM)
    rm = p0 + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    end = tl.load(OFF + e + 1)
    q = tl.load(ORDER + rm, mask=rm < end, other=0)
    z = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(AIN + q[:, None] * K + k[None, :], mask=(rm[:, None] < end) & (k[None, :] < K), other=0.).to(tl.float32)
        b = tl.load(W + e * K * N + k[:, None] * N + rn[None, :], mask=(k[:, None] < K) & (rn[None, :] < N), other=0.)
        if BF:
            ah = a.to(tl.bfloat16)
            z += tl.dot(ah, b)
            if SPLIT: z += tl.dot((a - ah).to(tl.bfloat16), b)
        else: z += tl.dot(a, b.to(tl.float32), input_precision="tf32x3")
    tl.store(OUT + q[:, None] * N + rn[None, :], z, mask=(rm[:, None] < end) & (rn[None, :] < N))

@triton.jit
def _segbt(AIN, W, ORDER, OFF, PRE, OUT, K, N, E, BF: tl.constexpr, IT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e, p0 = _loc(tl.program_id(0), OFF, PRE, E, IT, BM)
    rm = p0 + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    end = tl.load(OFF + e + 1)
    q = tl.load(ORDER + rm, mask=rm < end, other=0)
    z = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(AIN + q[:, None] * K + k[None, :], mask=(rm[:, None] < end) & (k[None, :] < K), other=0.).to(tl.float32)
        b = tl.load(W + e * N * K + rn[None, :] * K + k[:, None], mask=(rn[None, :] < N) & (k[:, None] < K), other=0.)
        if BF:
            z += tl.dot(a.to(tl.bfloat16), b)
        else: z += tl.dot(a, b.to(tl.float32), input_precision="tf32x3")
    tl.store(OUT + q[:, None] * N + rn[None, :], z, mask=(rm[:, None] < end) & (rn[None, :] < N))

@triton.jit
def _combine(FE, W, Y, T, D, A, BD: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    z = tl.zeros((BD,), tl.float32)
    for a in range(0, A): z += tl.load(W + t * A + a) * tl.load(FE + (t * A + a) * D + d, mask=d < D, other=0.)
    tl.store(Y + t * D + d, z, mask=d < D)

@triton.jit
def _dz(DY, W, O, N, D, A, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    mask = q < N * D
    p, d = q // D, q % D
    tl.store(O + q, tl.load(DY + (p // A) * D + d, mask=mask, other=0.).to(tl.float32) * tl.load(W + p, mask=mask, other=0.), mask=mask)

@triton.jit
def _dgu(GU, DA, O, N, F, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    mask = q < N * F
    r, f = q // F, q % F
    g = tl.load(GU + r * 2 * F + f, mask=mask, other=0.)
    u = tl.load(GU + r * 2 * F + F + f, mask=mask, other=0.)
    da = tl.load(DA + q, mask=mask, other=0.)
    s, si = tl.sigmoid(g), g * tl.sigmoid(g)
    tl.store(O + r * 2 * F + f, da * u * (s + si * (1. - s)), mask=mask)
    tl.store(O + r * 2 * F + F + f, da * si, mask=mask)

@triton.jit
def _dw(PA, PB, ORDER, OFF, O, K, N, E, A, XMODE: tl.constexpr, BF: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(0)
    i = tl.program_id(1) * BM + tl.arange(0, BM)
    j = tl.program_id(2) * BN + tl.arange(0, BN)
    beg, end = tl.load(OFF + e), tl.load(OFF + e + 1)
    z = tl.zeros((BM, BN), tl.float32)
    for p0 in range(beg, end, BK):
        p = p0 + tl.arange(0, BK)
        q = tl.load(ORDER + p, mask=p < end, other=0)
        row = q // A if XMODE else q
        aa = tl.load(PA + row[None, :] * K + i[:, None], mask=(p[None, :] < end) & (i[:, None] < K), other=0.).to(tl.float32)
        bb = tl.load(PB + q[:, None] * N + j[None, :], mask=(p[:, None] < end) & (j[None, :] < N), other=0.).to(tl.float32)
        if BF:
            ah, bh = aa.to(tl.bfloat16), bb.to(tl.bfloat16)
            z += tl.dot(ah, bh) + tl.dot(ah, (bb - bh).to(tl.bfloat16))
            if not XMODE: z += tl.dot((aa - ah).to(tl.bfloat16), bh)
        else: z += tl.dot(aa, bb, input_precision="tf32x3")
    tl.store(O + e * K * N + i[:, None] * N + j[None, :], z, mask=(i[:, None] < K) & (j[None, :] < N))

@triton.jit
def _pair(DY, FE, O, N, D, A, BD: tl.constexpr):
    q = tl.program_id(0)
    lane = tl.arange(0, BD)
    z = tl.zeros((BD,), tl.float32)
    for d0 in range(0, D, BD):
        d = d0 + lane
        z += tl.load(DY + (q // A) * D + d, mask=d < D, other=0.).to(tl.float32) * tl.load(FE + q * D + d, mask=d < D, other=0.)
    tl.store(O + q, tl.sum(z, axis=0))

@triton.jit
def _clear(X, n, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS)
    tl.store(X + q, 0., mask=q < n)

@triton.jit
def _route(S, SEL, W, R0, DS, T, E, A, BA: tl.constexpr):
    t = tl.program_id(0)
    a = tl.arange(0, BA)
    mask = a < A
    e = tl.load(SEL + t * A + a, mask=mask, other=0).to(tl.int64)
    w = tl.load(W + t * A + a, mask=mask, other=0.)
    r = tl.load(R0 + t * A + a, mask=mask, other=0.)
    s = tl.load(S + t * E + e, mask=mask, other=0.)
    ds = (r - tl.sum(w * r, axis=0)) / tl.sum(s, axis=0) * s * (1. - s)
    tl.store(DS + t * E + e, ds, mask=mask)

@triton.jit
def _dense(A0, B0, O, M, K, N, TA: tl.constexpr, TB: tl.constexpr, STRICT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    z = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        ap = A0 + (k[None, :] * M + m[:, None] if TA else m[:, None] * K + k[None, :])
        bp = B0 + (n[None, :] * K + k[:, None] if TB else k[:, None] * N + n[None, :])
        a = tl.load(ap, mask=(m[:, None] < M) & (k[None, :] < K), other=0.).to(tl.float32)
        b = tl.load(bp, mask=(k[:, None] < K) & (n[None, :] < N), other=0.).to(tl.float32)
        if STRICT: z += tl.dot(a, b, input_precision="ieee")
        else: z += tl.dot(a, b, input_precision="tf32x3")
    tl.store(O + m[:, None] * N + n[None, :], z, mask=(m[:, None] < M) & (n[None, :] < N))

@triton.jit
def _sumdx(P, R0, O, T, D, A, BD: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    z = tl.load(R0 + t * D + d, mask=d < D, other=0.)
    for a in range(0, A): z += tl.load(P + (t * A + a) * D + d, mask=d < D, other=0.)
    tl.store(O + t * D + d, z, mask=d < D)

def _metadata(sel, E):
    flat = sel.reshape(-1)
    order = torch.argsort(flat)
    n = flat.numel()
    off = torch.empty(E + 1, device=sel.device, dtype=torch.int32)
    _offsets[(E + 1,)](flat, order, off, n, E, IT=max(1, n.bit_length()), num_warps=1)
    pre = torch.empty_like(off)
    _prefix[(1,)](off, pre, E, BR=R, num_warps=1)
    return order, off, pre, triton.cdiv(n, R) + E

class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rw, rb, w1, w2, A):
        A = int(A); T, D = x.shape; E = rw.shape[1]; F = w1.shape[2] // 2; N = T * A
        bf = x.dtype == torch.bfloat16
        s = torch.empty((T, E), device=x.device, dtype=torch.float32)
        _router[(triton.cdiv(T,32),triton.cdiv(E,32))](x,rw,s,T,D,E,BM=32,BN=32,BK=32)
        keys = torch.empty((T,E),device=x.device,dtype=torch.int64)
        _keys[(triton.cdiv(T*E,256),)](s,rb,keys,T*E,E,BS=256)
        sel = torch.topk(keys,A,dim=1,largest=True,sorted=True).indices
        w = torch.empty((T,A),device=x.device,dtype=torch.float32)
        _weights[(T,)](s,sel,w,T,E,A,BA=triton.next_power_of_2(A))
        order,off,pre,cap = _metadata(sel,E); it=max(1,E.bit_length())
        gu=torch.empty((N,2*F),device=x.device,dtype=torch.float32)
        _ein[(cap,triton.cdiv(2*F,C))](x,w1,order,off,pre,gu,D,2*F,E,A,BF=bf,IT=it,BM=R,BN=C,BK=KB,num_warps=4,num_stages=3)
        act=torch.empty((N,F),device=x.device,dtype=torch.float32); _act[(triton.cdiv(N*F,256),)](gu,act,N,F,BS=256)
        fe=torch.empty((N,D),device=x.device,dtype=torch.float32)
        _seg[(cap,triton.cdiv(D,C))](act,w2,order,off,pre,fe,F,D,E,BF=bf,SPLIT=bf,IT=it,BM=R,BN=C,BK=KB,num_warps=4,num_stages=3)
        y=torch.empty((T,D),device=x.device,dtype=x.dtype); _combine[(T,triton.cdiv(D,256))](fe,w,y,T,D,A,BD=256)
        ctx.save_for_backward(x,rw,w1,w2,s,sel,w,order,off,pre,gu,act,fe); ctx.p=(A,cap,it,bf)
        return y
    @staticmethod
    def backward(ctx,dy):
        x,rw,w1,w2,s,sel,w,order,off,pre,gu,act,fe=ctx.saved_tensors; A,cap,it,bf=ctx.p
        T,D=x.shape; E=rw.shape[1]; F=act.shape[1]; N=T*A; dy=dy.contiguous()
        dz=torch.empty((N,D),device=x.device,dtype=torch.float32); _dz[(triton.cdiv(N*D,256),)](dy,w,dz,N,D,A,BS=256)
        da=torch.empty((N,F),device=x.device,dtype=torch.float32); _segbt[(cap,triton.cdiv(F,C))](dz,w2,order,off,pre,da,D,F,E,BF=bf,IT=it,BM=R,BN=C,BK=KB,num_warps=4,num_stages=3)
        dgu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _dgu[(triton.cdiv(N*F,256),)](gu,da,dgu,N,F,BS=256)
        dxp=torch.empty((N,D),device=x.device,dtype=torch.float32); _segbt[(cap,triton.cdiv(D,C))](dgu,w1,order,off,pre,dxp,2*F,D,E,BF=bf,IT=it,BM=R,BN=C,BK=KB,num_warps=4,num_stages=3)
        gw2=torch.empty_like(w2); _dw[(E,triton.cdiv(F,128),triton.cdiv(D,64))](act,dz,order,off,gw2,F,D,E,A,XMODE=False,BF=bf,BM=128,BN=64,BK=32,num_warps=4,num_stages=3)
        gw1=torch.empty_like(w1); _dw[(E,triton.cdiv(D,128),triton.cdiv(2*F,64))](x,dgu,order,off,gw1,D,2*F,E,A,XMODE=True,BF=bf,BM=128,BN=64,BK=32,num_warps=4,num_stages=3)
        pr=torch.empty(N,device=x.device,dtype=torch.float32); _pair[(N,)](dy,fe,pr,N,D,A,BD=256)
        ds=torch.empty((T,E),device=x.device,dtype=torch.float32); _clear[(triton.cdiv(T*E,256),)](ds,T*E,BS=256); _route[(T,)](s,sel,w,pr,ds,T,E,A,BA=triton.next_power_of_2(A))
        grw=torch.empty_like(rw); _dense[(triton.cdiv(D,64),triton.cdiv(E,32))](x,ds,grw,D,T,E,TA=True,TB=False,STRICT=True,BM=64,BN=32,BK=32,num_warps=4,num_stages=3)
        dxr=torch.empty((T,D),device=x.device,dtype=torch.float32); _dense[(triton.cdiv(T,64),triton.cdiv(D,64))](ds,rw,dxr,T,E,D,TA=False,TB=True,STRICT=True,BM=64,BN=64,BK=32,num_warps=4,num_stages=3)
        gx=torch.empty_like(x); _sumdx[(T,triton.cdiv(D,256))](dxp,dxr,gx,T,D,A,BD=256)
        return gx,grw,None,gw1,gw2,None

def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x,router_w,router_b,w1,w2,A)
