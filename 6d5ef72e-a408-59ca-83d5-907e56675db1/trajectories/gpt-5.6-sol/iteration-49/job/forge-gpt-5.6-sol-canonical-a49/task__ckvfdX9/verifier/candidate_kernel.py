import torch
import triton
import triton.language as tl

R, C, KB = 64, 256, 32

@triton.jit
def _router(X, RW, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    im = tl.program_id(0) * BM + tl.arange(0, BM); ie = tl.program_id(1) * BN + tl.arange(0, BN)
    z = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(X + im[:, None] * D + k[None, :], mask=(im[:, None] < T) & (k[None, :] < D), other=0.0).to(tl.float32)
        w = tl.load(RW + k[:, None] * E + ie[None, :], mask=(k[:, None] < D) & (ie[None, :] < E), other=0.0).to(tl.float32)
        z += tl.dot(x, w, input_precision="ieee")
    tl.store(S + im[:, None] * E + ie[None, :], tl.sigmoid(z), mask=(im[:, None] < T) & (ie[None, :] < E))

@triton.jit
def _make_keys(S, B, KEYS, T: tl.constexpr, E: tl.constexpr, BS: tl.constexpr):
    p = tl.program_id(0) * BS + tl.arange(0, BS); m = p < T * E; e = p % E
    v = tl.load(S + p, mask=m, other=0.0) + tl.load(B + e, mask=m, other=0.0)
    v = tl.where(v == 0.0, 0.0, v)
    ib = v.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    tl.store(KEYS + p, (u << 10) | (E - 1 - e), mask=m)

@triton.jit
def _dispatch_keys(SEL, DK, N: tl.constexpr, BS: tl.constexpr):
    p = tl.program_id(0) * BS + tl.arange(0, BS); m = p < N
    e = tl.load(SEL + p, mask=m, other=0).to(tl.int64)
    tl.store(DK + p, e * (N + 1) + p, mask=m)

@triton.jit
def _weights(S, SEL, W, T: tl.constexpr, E: tl.constexpr, A: tl.constexpr, BA: tl.constexpr):
    t = tl.program_id(0); a = tl.arange(0, BA); m = a < A
    e = tl.load(SEL + t * A + a, mask=m, other=0)
    q = tl.load(S + t * E + e, mask=m, other=0.0); den = tl.sum(q, axis=0)
    tl.store(W + t * A + a, q / den, mask=m)

@triton.jit
def _inverse(ORDER, TOK, INV, SE, SEL, N: tl.constexpr, A: tl.constexpr, BS: tl.constexpr):
    q = tl.program_id(0) * BS + tl.arange(0, BS); m = q < N
    p = tl.load(ORDER + q, mask=m, other=0)
    tl.store(TOK + q, p // A, mask=m); tl.store(INV + p, q, mask=m)
    tl.store(SE + q, tl.load(SEL + p, mask=m, other=0), mask=m)

@triton.jit
def _offsets(SE, OFF, N: tl.constexpr, E: tl.constexpr, IT: tl.constexpr):
    e = tl.program_id(0); lo = 0; hi = N
    for _ in range(IT):
        md = (lo + hi) // 2; v = tl.load(SE + md, mask=md < N, other=E); take = v < e
        lo = tl.where(take, md + 1, lo); hi = tl.where(take, hi, md)
    tl.store(OFF + e, lo)

@triton.jit
def _prefix(OFF, PRE, E: tl.constexpr, ROW: tl.constexpr):
    total = 0; tl.store(PRE, 0)
    for e in range(0, E):
        total += tl.cdiv(tl.load(OFF + e + 1) - tl.load(OFF + e), ROW)
        tl.store(PRE + e + 1, total)

@triton.jit
def _loc(pid, PRE, OFF, E: tl.constexpr, IT: tl.constexpr, ROW: tl.constexpr):
    lo = 0; hi = E
    for _ in range(IT):
        md = (lo + hi) // 2; take = tl.load(PRE + md + 1) <= pid
        lo = tl.where(take, md + 1, lo); hi = tl.where(take, hi, md)
    valid = lo < E
    e = tl.minimum(lo, E - 1)
    final = tl.load(OFF + E)
    base = tl.load(OFF + e)
    start = tl.where(valid, base + (pid - tl.load(PRE + e)) * ROW, final)
    end = tl.where(valid, tl.load(OFF + e + 1), final)
    return e, start, end

@triton.jit
def _ein(X, W1, TOK, OFF, PRE, GU, D: tl.constexpr, F2: tl.constexpr, E: tl.constexpr, IT: tl.constexpr, BF: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e, p0, pend = _loc(tl.program_id(0), PRE, OFF, E, IT, BM)
    rm = p0 + tl.arange(0, BM); rn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        k = k0 + tl.arange(0, BK); tok = tl.load(TOK + rm, mask=rm < pend, other=0)
        a = tl.load(X + tok[:, None] * D + k[None, :], mask=(rm[:, None] < pend) & (k[None, :] < D), other=0.0)
        b = tl.load(W1 + e * D * F2 + k[:, None] * F2 + rn[None, :], mask=(e < E) & (k[:, None] < D) & (rn[None, :] < F2), other=0.0)
        if BF: acc += tl.dot(a, b)
        else: acc += tl.dot(a, b, input_precision="tf32x3")
    tl.store(GU + rm[:, None] * F2 + rn[None, :], acc, mask=(rm[:, None] < pend) & (rn[None, :] < F2))

@triton.jit
def _activate(GU, ACT, N: tl.constexpr, F: tl.constexpr, BS: tl.constexpr):
    p = tl.program_id(0) * BS + tl.arange(0, BS); m = p < N * F; r = p // F; f = p % F
    g = tl.load(GU + r * (2 * F) + f, mask=m, other=0.0); u = tl.load(GU + r * (2 * F) + F + f, mask=m, other=0.0)
    tl.store(ACT + p, g * tl.sigmoid(g) * u, mask=m)

@triton.jit
def _seg(A0, B, OFF, PRE, OUT, K0: tl.constexpr, N0: tl.constexpr, E: tl.constexpr, IT: tl.constexpr, BF: tl.constexpr, SPLIT: tl.constexpr, TRANS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e, p0, pend = _loc(tl.program_id(0), PRE, OFF, E, IT, BM)
    rm = p0 + tl.arange(0, BM); rn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K0, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(A0 + rm[:, None] * K0 + k[None, :], mask=(rm[:, None] < pend) & (k[None, :] < K0), other=0.0).to(tl.float32)
        if TRANS: b = tl.load(B + e * N0 * K0 + rn[None, :] * K0 + k[:, None], mask=(e < E) & (rn[None, :] < N0) & (k[:, None] < K0), other=0.0)
        else: b = tl.load(B + e * K0 * N0 + k[:, None] * N0 + rn[None, :], mask=(e < E) & (k[:, None] < K0) & (rn[None, :] < N0), other=0.0)
        if BF:
            ah = a.to(tl.bfloat16)
            acc += tl.dot(ah, b)
            if SPLIT:
                acc += tl.dot((a - ah).to(tl.bfloat16), b)
        else: acc += tl.dot(a, b, input_precision="tf32x3")
    tl.store(OUT + rm[:, None] * N0 + rn[None, :], acc, mask=(rm[:, None] < pend) & (rn[None, :] < N0))

@triton.jit
def _combine(Z, W, INV, Y, T: tl.constexpr, D: tl.constexpr, A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0); d = tl.program_id(1) * BD + tl.arange(0, BD); z = tl.zeros((BD,), tl.float32)
    for a in range(0, A):
        p = t * A + a; q = tl.load(INV + p); z += tl.load(W + p) * tl.load(Z + q * D + d, mask=d < D, other=0.0)
    tl.store(Y + t * D + d, z, mask=d < D)

@triton.jit
def _dz(DY, W, ORDER, TOK, DZ, D: tl.constexpr, BS: tl.constexpr):
    q = tl.program_id(0); d = tl.arange(0, BS); p = tl.load(ORDER + q); t = tl.load(TOK + q)
    tl.store(DZ + q * D + d, tl.load(W + p) * tl.load(DY + t * D + d, mask=d < D, other=0.0), mask=d < D)

@triton.jit
def _dgu(GU, DA, DGU, N: tl.constexpr, F: tl.constexpr, BS: tl.constexpr):
    p = tl.program_id(0) * BS + tl.arange(0, BS); m = p < N * F; r = p // F; f = p % F
    g = tl.load(GU + r * 2 * F + f, mask=m, other=0.0); u = tl.load(GU + r * 2 * F + F + f, mask=m, other=0.0); da = tl.load(DA + p, mask=m, other=0.0)
    s = tl.sigmoid(g); silu = g * s
    tl.store(DGU + r * 2 * F + f, da * u * (s + silu * (1.0 - s)), mask=m)
    tl.store(DGU + r * 2 * F + F + f, da * silu, mask=m)

@triton.jit
def _dw(A0, B0, X, TOK, OFF, OUT, K0: tl.constexpr, N0: tl.constexpr, BF: tl.constexpr, XMODE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(0); rm = tl.program_id(1) * BM + tl.arange(0, BM); rn = tl.program_id(2) * BN + tl.arange(0, BN)
    beg = tl.load(OFF + e); end = tl.load(OFF + e + 1); acc = tl.zeros((BM, BN), tl.float32)
    for p0 in range(beg, end, BK):
        p = p0 + tl.arange(0, BK)
        if XMODE:
            tok = tl.load(TOK + p, mask=p < end, other=0)
            av = tl.load(X + tok[:, None] * K0 + rm[None, :], mask=(p[:, None] < end) & (rm[None, :] < K0), other=0.0)
        else: av = tl.load(A0 + p[:, None] * K0 + rm[None, :], mask=(p[:, None] < end) & (rm[None, :] < K0), other=0.0).to(tl.float32)
        bv = tl.load(B0 + p[:, None] * N0 + rn[None, :], mask=(p[:, None] < end) & (rn[None, :] < N0), other=0.0).to(tl.float32)
        if BF:
            ah = av.to(tl.bfloat16); al = (av - ah).to(tl.bfloat16); bh = bv.to(tl.bfloat16); bl = (bv - bh).to(tl.bfloat16)
            acc += tl.dot(tl.trans(ah), bh) + tl.dot(tl.trans(al), bh) + tl.dot(tl.trans(ah), bl)
        else: acc += tl.dot(tl.trans(av), bv, input_precision="tf32x3")
    tl.store(OUT + e * K0 * N0 + rm[:, None] * N0 + rn[None, :], acc, mask=(rm[:, None] < K0) & (rn[None, :] < N0))

@triton.jit
def _pair_dot(DY, Z, INV, DW, D: tl.constexpr, A: tl.constexpr, BD: tl.constexpr):
    p = tl.program_id(0); d = tl.arange(0, BD); q = tl.load(INV + p); t = p // A
    a = tl.load(DY + t * D + d, mask=d < D, other=0.0).to(tl.float32); b = tl.load(Z + q * D + d, mask=d < D, other=0.0)
    tl.store(DW + p, tl.sum(a * b, axis=0))

@triton.jit
def _route(S, SEL, W, DW, DS, T: tl.constexpr, E: tl.constexpr, A: tl.constexpr, BA: tl.constexpr, BE: tl.constexpr):
    t = tl.program_id(0); ee = tl.arange(0, BE); tl.store(DS + t * E + ee, 0.0, mask=ee < E)
    a = tl.arange(0, BA); m = a < A; e = tl.load(SEL + t * A + a, mask=m, other=0)
    s = tl.load(S + t * E + e, mask=m, other=0.0); w = tl.load(W + t * A + a, mask=m, other=0.0); dw = tl.load(DW + t * A + a, mask=m, other=0.0)
    ds = (dw - tl.sum(w * dw, axis=0)) / tl.sum(s, axis=0) * s * (1.0 - s)
    tl.store(DS + t * E + e, ds, mask=m)

@triton.jit
def _drw(X, DS, OUT, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    d = tl.program_id(0) * BM + tl.arange(0, BM); e = tl.program_id(1) * BN + tl.arange(0, BN); acc = tl.zeros((BM, BN), tl.float32)
    for t0 in range(0, T, BK):
        t = t0 + tl.arange(0, BK); x = tl.load(X + t[:, None] * D + d[None, :], mask=(t[:, None] < T) & (d[None, :] < D), other=0.0).to(tl.float32)
        s = tl.load(DS + t[:, None] * E + e[None, :], mask=(t[:, None] < T) & (e[None, :] < E), other=0.0)
        acc += tl.dot(tl.trans(x), s, input_precision="ieee")
    tl.store(OUT + d[:, None] * E + e[None, :], acc, mask=(d[:, None] < D) & (e[None, :] < E))

@triton.jit
def _rdx(DS, RW, OUT, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0) * BM + tl.arange(0, BM); d = tl.program_id(1) * BN + tl.arange(0, BN); acc = tl.zeros((BM, BN), tl.float32)
    for e0 in range(0, E, BK):
        e = e0 + tl.arange(0, BK); s = tl.load(DS + t[:, None] * E + e[None, :], mask=(t[:, None] < T) & (e[None, :] < E), other=0.0)
        w = tl.load(RW + d[None, :] * E + e[:, None], mask=(d[None, :] < D) & (e[:, None] < E), other=0.0).to(tl.float32)
        acc += tl.dot(s, w, input_precision="ieee")
    tl.store(OUT + t[:, None] * D + d[None, :], acc, mask=(t[:, None] < T) & (d[None, :] < D))

@triton.jit
def _sum_dx(DXP, RDX, INV, DX, T: tl.constexpr, D: tl.constexpr, A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0); d = tl.program_id(1) * BD + tl.arange(0, BD); z = tl.load(RDX + t * D + d, mask=d < D, other=0.0)
    for a in range(0, A): z += tl.load(DXP + tl.load(INV + t * A + a) * D + d, mask=d < D, other=0.0)
    tl.store(DX + t * D + d, z, mask=d < D)

class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rw, rb, w1, w2, A):
        A = int(A); T, D = x.shape; E = rw.shape[1]; F = w2.shape[1]; N = T * A; bf = x.dtype == torch.bfloat16
        s = torch.empty((T, E), device=x.device, dtype=torch.float32); _router[(triton.cdiv(T,32),triton.cdiv(E,32))](x,rw,s,T=T,D=D,E=E,BM=32,BN=32,BK=32)
        keys = torch.empty((T,E),device=x.device,dtype=torch.int64); _make_keys[(triton.cdiv(T*E,256),)](s,rb,keys,T=T,E=E,BS=256)
        sel = torch.topk(keys,A,dim=1,largest=True,sorted=True).indices.contiguous(); w = torch.empty(N,device=x.device,dtype=torch.float32)
        _weights[(T,)](s,sel,w,T=T,E=E,A=A,BA=triton.next_power_of_2(A))
        dk=torch.empty(N,device=x.device,dtype=torch.int64); _dispatch_keys[(triton.cdiv(N,256),)](sel,dk,N=N,BS=256); order=torch.argsort(dk)
        tok=torch.empty(N,device=x.device,dtype=torch.int64); inv=torch.empty_like(tok); se=torch.empty_like(tok); _inverse[(triton.cdiv(N,256),)](order,tok,inv,se,sel,N=N,A=A,BS=256)
        off=torch.empty(E+1,device=x.device,dtype=torch.int32); _offsets[(E+1,)](se,off,N=N,E=E,IT=max(1,(N+1).bit_length()))
        pre=torch.empty_like(off); _prefix[(1,)](off,pre,E=E,ROW=R,num_warps=1); cap=triton.cdiv(N,R)+E; eit=max(1,E.bit_length())
        gu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _ein[(cap,triton.cdiv(2*F,C))](x,w1,tok,off,pre,gu,D=D,F2=2*F,E=E,IT=eit,BF=bf,BM=R,BN=C,BK=KB,num_warps=4,num_stages=3)
        act=torch.empty((N,F),device=x.device,dtype=torch.float32); _activate[(triton.cdiv(N*F,256),)](gu,act,N=N,F=F,BS=256)
        z=torch.empty((N,D),device=x.device,dtype=torch.float32); _seg[(cap,triton.cdiv(D,C))](act,w2,off,pre,z,K0=F,N0=D,E=E,IT=eit,BF=bf,SPLIT=True,TRANS=False,BM=R,BN=C,BK=KB,num_warps=4,num_stages=3)
        y=torch.empty((T,D),device=x.device,dtype=x.dtype); _combine[(T,triton.cdiv(D,256))](z,w,inv,y,T=T,D=D,A=A,BD=256)
        ctx.save_for_backward(x,rw,w1,w2,s,sel,w,order,tok,inv,off,pre,gu,act,z); ctx.dims=T,D,E,F,A,cap,eit,bf; return y
    @staticmethod
    def backward(ctx,dy):
        x,rw,w1,w2,s,sel,w,order,tok,inv,off,pre,gu,act,z=ctx.saved_tensors; T,D,E,F,A,cap,eit,bf=ctx.dims; N=T*A; dy=dy.contiguous()
        dz=torch.empty((N,D),device=x.device,dtype=torch.float32); _dz[(N,)](dy,w,order,tok,dz,D=D,BS=triton.next_power_of_2(D))
        gw2=torch.empty_like(w2); _dw[(E,triton.cdiv(F,64),triton.cdiv(D,128))](act,dz,x,tok,off,gw2,K0=F,N0=D,BF=bf,XMODE=False,BM=64,BN=128,BK=32,num_warps=4,num_stages=3)
        da=torch.empty((N,F),device=x.device,dtype=torch.float32); _seg[(cap,triton.cdiv(F,C))](dz,w2,off,pre,da,K0=D,N0=F,E=E,IT=eit,BF=bf,SPLIT=True,TRANS=True,BM=R,BN=C,BK=KB,num_warps=4,num_stages=3)
        dgu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _dgu[(triton.cdiv(N*F,256),)](gu,da,dgu,N=N,F=F,BS=256)
        gw1=torch.empty_like(w1); _dw[(E,triton.cdiv(D,64),triton.cdiv(2*F,128))](dgu,dgu,x,tok,off,gw1,K0=D,N0=2*F,BF=bf,XMODE=True,BM=64,BN=128,BK=32,num_warps=4,num_stages=3)
        dxp=torch.empty((N,D),device=x.device,dtype=torch.float32); _seg[(cap,triton.cdiv(D,C))](dgu,w1,off,pre,dxp,K0=2*F,N0=D,E=E,IT=eit,BF=bf,SPLIT=True,TRANS=True,BM=R,BN=C,BK=KB,num_warps=4,num_stages=3)
        dw=torch.empty(N,device=x.device,dtype=torch.float32); _pair_dot[(N,)](dy,z,inv,dw,D=D,A=A,BD=triton.next_power_of_2(D))
        ds=torch.empty((T,E),device=x.device,dtype=torch.float32); _route[(T,)](s,sel,w,dw,ds,T=T,E=E,A=A,BA=triton.next_power_of_2(A),BE=triton.next_power_of_2(E))
        grw=torch.empty_like(rw); _drw[(triton.cdiv(D,64),triton.cdiv(E,32))](x,ds,grw,T=T,D=D,E=E,BM=64,BN=32,BK=32,num_warps=4,num_stages=3)
        rdx=torch.empty((T,D),device=x.device,dtype=torch.float32); _rdx[(triton.cdiv(T,32),triton.cdiv(D,64))](ds,rw,rdx,T=T,D=D,E=E,BM=32,BN=64,BK=32,num_warps=4,num_stages=3)
        gx=torch.empty_like(x); _sum_dx[(T,triton.cdiv(D,256))](dxp,rdx,inv,gx,T=T,D=D,A=A,BD=256)
        return gx,grw,None,gw1,gw2,None

def fused_moe(x,router_w,router_b,w1,w2,A): return _Moe.apply(x,router_w,router_b,w1,w2,A)
