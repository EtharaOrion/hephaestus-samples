import torch
import triton
import triton.language as tl

ROW = 64
COL = 64
KB = 32

@triton.jit
def _router(X, W, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    im = tl.program_id(0) * BM + tl.arange(0, BM); ie = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, D, BK):
        ik = k + tl.arange(0, BK)
        x = tl.load(X + im[:, None] * D + ik[None, :], mask=(im[:, None] < T) & (ik[None, :] < D), other=0.).to(tl.float32)
        w = tl.load(W + ik[:, None] * E + ie[None, :], mask=(ik[:, None] < D) & (ie[None, :] < E), other=0.).to(tl.float32)
        acc += tl.dot(x, w, input_precision="ieee")
    tl.store(S + im[:, None] * E + ie[None, :], tl.sigmoid(acc), mask=(im[:, None] < T) & (ie[None, :] < E))

@triton.jit
def _weights(S, SEL, W, E: tl.constexpr, A: tl.constexpr, BA: tl.constexpr):
    t = tl.program_id(0); a = tl.arange(0, BA); m = a < A
    e = tl.load(SEL + t * A + a, mask=m, other=0); s = tl.load(S + t * E + e, mask=m, other=0.)
    tl.store(W + t * A + a, s / tl.sum(s, axis=0), mask=m)

@triton.jit
def _offsets(SEL, ORDER, OFF, N: tl.constexpr):
    e = tl.program_id(0); lo = 0; hi = N
    while lo < hi:
        mid = (lo + hi) // 2; p = tl.load(ORDER + mid); v = tl.load(SEL + p); take = v < e
        lo = tl.where(take, mid + 1, lo); hi = tl.where(take, hi, mid)
    tl.store(OFF + e, lo)

@triton.jit
def _tiles(OFF, TE, TS, TOTAL, E: tl.constexpr, BR: tl.constexpr, MAX: tl.constexpr):
    e = tl.program_id(0); begin = tl.load(OFF + e); end = tl.load(OFF + e + 1); base = 0
    for j in range(0, E):
        jb = tl.load(OFF + j); je = tl.load(OFF + j + 1)
        base += tl.where(j < e, tl.cdiv(je - jb, BR), 0)
    count = tl.cdiv(end - begin, BR)
    for j in range(0, MAX):
        if j < count:
            tl.store(TE + base + j, e); tl.store(TS + base + j, begin + j * BR)
    if e == E - 1: tl.store(TOTAL, base + count)

@triton.jit
def _meta(ORDER, PW, TOK, SW, INV, N: tl.constexpr, A: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B); m = i < N; p = tl.load(ORDER + i, mask=m, other=0)
    tl.store(TOK + i, p // A, mask=m); tl.store(SW + i, tl.load(PW + p, mask=m, other=0.), mask=m); tl.store(INV + p, i, mask=m)

@triton.jit
def _ein(X, W1, TOK, OFF, TE, TS, TOTAL, GU, N: tl.constexpr, D: tl.constexpr, F2: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, BF: tl.constexpr):
    tile = tl.program_id(0); active = tile < tl.load(TOTAL); e = tl.load(TE + tile, mask=active, other=0); start = tl.load(TS + tile, mask=active, other=0); end = tl.load(OFF + e + 1, mask=active, other=0)
    ir = start + tl.arange(0, BR); ic = tl.program_id(1) * BC + tl.arange(0, BC); tok = tl.load(TOK + ir, mask=active & (ir < end), other=0)
    acc = tl.zeros((BR, BC), tl.float32)
    for k in range(0, D, BK):
        ik = k + tl.arange(0, BK)
        x = tl.load(X + tok[:, None] * D + ik[None, :], mask=active & (ir[:, None] < end) & (ik[None, :] < D), other=0.)
        w = tl.load(W1 + e * D * F2 + ik[:, None] * F2 + ic[None, :], mask=active & (ik[:, None] < D) & (ic[None, :] < F2), other=0.)
        if BF: acc += tl.dot(x, w)
        else: acc += tl.dot(x.to(tl.float32), w.to(tl.float32), input_precision="tf32x3")
    tl.store(GU + ir[:, None] * F2 + ic[None, :], acc, mask=active & (ir[:, None] < end) & (ic[None, :] < F2))

@triton.jit
def _act(GU, ACT, N: tl.constexpr, F: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B); m = i < N * F; r = i // F; c = i - r * F
    g = tl.load(GU + r * 2 * F + c, mask=m, other=0.); u = tl.load(GU + r * 2 * F + F + c, mask=m, other=0.)
    tl.store(ACT + i, g * tl.sigmoid(g) * u, mask=m)

@triton.jit
def _eout(ACT, W2, OFF, TE, TS, TOTAL, YP, N: tl.constexpr, F: tl.constexpr, D: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, BF: tl.constexpr):
    tile = tl.program_id(0); active = tile < tl.load(TOTAL); e = tl.load(TE + tile, mask=active, other=0); start = tl.load(TS + tile, mask=active, other=0); end = tl.load(OFF + e + 1, mask=active, other=0)
    ir = start + tl.arange(0, BR); ic = tl.program_id(1) * BC + tl.arange(0, BC); acc = tl.zeros((BR, BC), tl.float32)
    for k in range(0, F, BK):
        ik = k + tl.arange(0, BK)
        a = tl.load(ACT + ir[:, None] * F + ik[None, :], mask=active & (ir[:, None] < end) & (ik[None, :] < F), other=0.).to(tl.float32)
        w = tl.load(W2 + e * F * D + ik[:, None] * D + ic[None, :], mask=active & (ik[:, None] < F) & (ic[None, :] < D), other=0.)
        if BF:
            ah = a.to(tl.bfloat16); al = (a - ah.to(tl.float32)).to(tl.bfloat16); acc += tl.dot(ah, w) + tl.dot(al, w)
        else: acc += tl.dot(a, w.to(tl.float32), input_precision="tf32x3")
    tl.store(YP + ir[:, None] * D + ic[None, :], acc, mask=active & (ir[:, None] < end) & (ic[None, :] < D))

@triton.jit
def _combine(YP, PW, INV, Y, D: tl.constexpr, A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0); d = tl.program_id(1) * BD + tl.arange(0, BD); acc = tl.zeros((BD,), tl.float32)
    for a in range(0, A):
        p = t * A + a; r = tl.load(INV + p); acc += tl.load(PW + p) * tl.load(YP + r * D + d, mask=d < D, other=0.)
    tl.store(Y + t * D + d, acc, mask=d < D)

@triton.jit
def _route_grad(S, SEL, PW, INV, YP, DY, DZ, D: tl.constexpr, E: tl.constexpr, A: tl.constexpr, BD: tl.constexpr, BE: tl.constexpr):
    t = tl.program_id(0); d = tl.arange(0, BD); dy = tl.load(DY + t * D + d, mask=d < D, other=0.).to(tl.float32); mean = 0.; den = 0.
    for a in range(0, A):
        p = t * A + a; e = tl.load(SEL + p); r = tl.load(INV + p); q = tl.sum(dy * tl.load(YP + r * D + d, mask=d < D, other=0.), axis=0); mean += q * tl.load(PW + p); den += tl.load(S + t * E + e)
    ie = tl.arange(0, BE); out = tl.zeros((BE,), tl.float32)
    for a in range(0, A):
        p = t * A + a; e = tl.load(SEL + p); r = tl.load(INV + p); q = tl.sum(dy * tl.load(YP + r * D + d, mask=d < D, other=0.), axis=0)
        s = tl.load(S + t * E + e); ds = (q - mean) / den; out += tl.where(ie == e, ds * s * (1. - s), 0.)
    tl.store(DZ + t * E + ie, out, mask=ie < E)

@triton.jit
def _dgu(DY, SW, TOK, W2, GU, OFF, TE, TS, TOTAL, DGU, N: tl.constexpr, D: tl.constexpr, F: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, BF: tl.constexpr):
    tile = tl.program_id(0); active = tile < tl.load(TOTAL); e = tl.load(TE + tile, mask=active, other=0); start = tl.load(TS + tile, mask=active, other=0); end = tl.load(OFF + e + 1, mask=active, other=0)
    ir = start + tl.arange(0, BR); jf = tl.program_id(1) * BC + tl.arange(0, BC); tok = tl.load(TOK + ir, mask=active & (ir < end), other=0); sw = tl.load(SW + ir, mask=active & (ir < end), other=0.).to(tl.float32)
    acc = tl.zeros((BR, BC), tl.float32)
    for k in range(0, D, BK):
        kd = k + tl.arange(0, BK); dy = tl.load(DY + tok[:, None] * D + kd[None, :], mask=active & (ir[:, None] < end) & (kd[None, :] < D), other=0.).to(tl.float32) * sw[:, None]
        w = tl.load(W2 + e * F * D + jf[:, None] * D + kd[None, :], mask=active & (jf[:, None] < F) & (kd[None, :] < D), other=0.)
        if BF:
            dh = dy.to(tl.bfloat16); dl = (dy - dh.to(tl.float32)).to(tl.bfloat16); acc += tl.dot(dh, tl.trans(w)) + tl.dot(dl, tl.trans(w))
        else: acc += tl.dot(dy, tl.trans(w.to(tl.float32)), input_precision="tf32x3")
    g = tl.load(GU + ir[:, None] * 2 * F + jf[None, :], mask=active & (ir[:, None] < end) & (jf[None, :] < F), other=0.); u = tl.load(GU + ir[:, None] * 2 * F + F + jf[None, :], mask=active & (ir[:, None] < end) & (jf[None, :] < F), other=0.); sig = tl.sigmoid(g)
    mask = active & (ir[:, None] < end) & (jf[None, :] < F)
    tl.store(DGU + ir[:, None] * 2 * F + jf[None, :], acc * u * (sig + g * sig * (1. - sig)), mask=mask); tl.store(DGU + ir[:, None] * 2 * F + F + jf[None, :], acc * g * sig, mask=mask)

@triton.jit
def _edx(DGU, W1, OFF, TE, TS, TOTAL, DXP, N: tl.constexpr, D: tl.constexpr, F2: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, BF: tl.constexpr):
    tile = tl.program_id(0); active = tile < tl.load(TOTAL); e = tl.load(TE + tile, mask=active, other=0); start = tl.load(TS + tile, mask=active, other=0); end = tl.load(OFF + e + 1, mask=active, other=0)
    ir = start + tl.arange(0, BR); jd = tl.program_id(1) * BC + tl.arange(0, BC); acc = tl.zeros((BR, BC), tl.float32)
    for k in range(0, F2, BK):
        kk = k + tl.arange(0, BK); q = tl.load(DGU + ir[:, None] * F2 + kk[None, :], mask=active & (ir[:, None] < end) & (kk[None, :] < F2), other=0.).to(tl.float32)
        w = tl.load(W1 + e * D * F2 + jd[:, None] * F2 + kk[None, :], mask=active & (jd[:, None] < D) & (kk[None, :] < F2), other=0.)
        if BF: acc += tl.dot(q.to(tl.bfloat16), tl.trans(w))
        else: acc += tl.dot(q, tl.trans(w.to(tl.float32)), input_precision="tf32x3")
    tl.store(DXP + ir[:, None] * D + jd[None, :], acc, mask=active & (ir[:, None] < end) & (jd[None, :] < D))

@triton.jit
def _cdx(DXP, INV, RDX, DX, D: tl.constexpr, A: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0); d = tl.program_id(1) * BD + tl.arange(0, BD); acc = tl.load(RDX + t * D + d, mask=d < D, other=0.).to(tl.float32)
    for a in range(0, A): acc += tl.load(DXP + tl.load(INV + t * A + a) * D + d, mask=d < D, other=0.)
    tl.store(DX + t * D + d, acc, mask=d < D)

@triton.jit
def _dw2(ACT, DY, SW, TOK, OFF, OUT, F: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BF: tl.constexpr):
    e = tl.program_id(0); fi = tl.program_id(1) * BM + tl.arange(0, BM); dj = tl.program_id(2) * BN + tl.arange(0, BN); begin = tl.load(OFF + e); end = tl.load(OFF + e + 1); acc = tl.zeros((BM, BN), tl.float32)
    for k in range(begin, end, BK):
        r = k + tl.arange(0, BK); tok = tl.load(TOK + r, mask=r < end, other=0); a = tl.load(ACT + r[:, None] * F + fi[None, :], mask=(r[:, None] < end) & (fi[None, :] < F), other=0.).to(tl.float32)
        b = tl.load(DY + tok[:, None] * D + dj[None, :], mask=(r[:, None] < end) & (dj[None, :] < D), other=0.).to(tl.float32) * tl.load(SW + r, mask=r < end, other=0.)[:, None]
        if BF:
            ah = a.to(tl.bfloat16); al = (a-ah.to(tl.float32)).to(tl.bfloat16); bh = b.to(tl.bfloat16); bl = (b-bh.to(tl.float32)).to(tl.bfloat16); acc += tl.dot(tl.trans(ah),bh)+tl.dot(tl.trans(al),bh)+tl.dot(tl.trans(ah),bl)
        else: acc += tl.dot(tl.trans(a), b, input_precision="tf32x3")
    tl.store(OUT + e*F*D + fi[:,None]*D + dj[None,:], acc, mask=(fi[:,None]<F)&(dj[None,:]<D))

@triton.jit
def _dw1(X, DGU, TOK, OFF, OUT, D: tl.constexpr, F2: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BF: tl.constexpr):
    e=tl.program_id(0); di=tl.program_id(1)*BM+tl.arange(0,BM); fj=tl.program_id(2)*BN+tl.arange(0,BN); begin=tl.load(OFF+e); end=tl.load(OFF+e+1); acc=tl.zeros((BM,BN),tl.float32)
    for k in range(begin,end,BK):
        r=k+tl.arange(0,BK); tok=tl.load(TOK+r,mask=r<end,other=0); a=tl.load(X+tok[:,None]*D+di[None,:],mask=(r[:,None]<end)&(di[None,:]<D),other=0.); b=tl.load(DGU+r[:,None]*F2+fj[None,:],mask=(r[:,None]<end)&(fj[None,:]<F2),other=0.).to(tl.float32)
        if BF:
            bh=b.to(tl.bfloat16); bl=(b-bh.to(tl.float32)).to(tl.bfloat16); acc+=tl.dot(tl.trans(a),bh)+tl.dot(tl.trans(a),bl)
        else: acc+=tl.dot(tl.trans(a.to(tl.float32)),b,input_precision="tf32x3")
    tl.store(OUT+e*D*F2+di[:,None]*F2+fj[None,:],acc,mask=(di[:,None]<D)&(fj[None,:]<F2))

@triton.jit
def _dense(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,TA:tl.constexpr,TB:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    im=tl.program_id(0)*BM+tl.arange(0,BM); jn=tl.program_id(1)*BN+tl.arange(0,BN); acc=tl.zeros((BM,BN),tl.float32)
    for k in range(0,K,BK):
        kk=k+tl.arange(0,BK)
        if TA: a=tl.load(A+kk[None,:]*M+im[:,None],mask=(im[:,None]<M)&(kk[None,:]<K),other=0.).to(tl.float32)
        else: a=tl.load(A+im[:,None]*K+kk[None,:],mask=(im[:,None]<M)&(kk[None,:]<K),other=0.).to(tl.float32)
        if TB: b=tl.load(B+jn[None,:]*K+kk[:,None],mask=(kk[:,None]<K)&(jn[None,:]<N),other=0.).to(tl.float32)
        else: b=tl.load(B+kk[:,None]*N+jn[None,:],mask=(kk[:,None]<K)&(jn[None,:]<N),other=0.).to(tl.float32)
        acc+=tl.dot(a,b,input_precision="ieee")
    tl.store(C+im[:,None]*N+jn[None,:],acc,mask=(im[:,None]<M)&(jn[None,:]<N))

def _select(s,bias,A):
    E=s.shape[1]; b=s+bias; b=torch.where(b==0,torch.zeros_like(b),b).contiguous(); ib=b.view(torch.int32).to(torch.int64); u=torch.where(ib>=0,ib^0x80000000,(~ib)&0xFFFFFFFF); idx=torch.arange(E,device=s.device,dtype=torch.int64)
    return torch.topk((u<<10)|(E-1-idx),A,dim=1,sorted=True).indices

class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,rw,rb,w1,w2,A):
        T,D=x.shape; E=rw.shape[1]; F=w2.shape[1]; A=int(A); N=T*A; bf=x.dtype==torch.bfloat16
        s=torch.empty((T,E),device=x.device,dtype=torch.float32); _router[(triton.cdiv(T,32),triton.cdiv(E,32))](x,rw,s,T=T,D=D,E=E,BM=32,BN=32,BK=32)
        with torch.no_grad(): sel=_select(s,rb,A); order=torch.argsort(sel.reshape(-1),stable=True)
        pw=torch.empty(N,device=x.device,dtype=torch.float32); _weights[(T,)](s,sel,pw,E=E,A=A,BA=triton.next_power_of_2(A),num_warps=1)
        off=torch.empty(E+1,device=x.device,dtype=torch.int32); _offsets[(E+1,)](sel,order,off,N=N,num_warps=1)
        cap=triton.cdiv(N,ROW)+E; te=torch.empty(cap,device=x.device,dtype=torch.int32); ts=torch.empty_like(te); total=torch.empty(1,device=x.device,dtype=torch.int32); _tiles[(E,)](off,te,ts,total,E=E,BR=ROW,MAX=triton.cdiv(T,ROW),num_warps=1)
        tok=torch.empty(N,device=x.device,dtype=torch.int64); sw=torch.empty(N,device=x.device,dtype=torch.float32); inv=torch.empty(N,device=x.device,dtype=torch.int64); _meta[(triton.cdiv(N,256),)](order,pw,tok,sw,inv,N=N,A=A,B=256)
        gu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _ein[(cap,triton.cdiv(2*F,COL))](x,w1,tok,off,te,ts,total,gu,N=N,D=D,F2=2*F,BR=ROW,BC=COL,BK=KB,BF=bf,num_warps=4,num_stages=3)
        act=torch.empty((N,F),device=x.device,dtype=torch.float32); _act[(triton.cdiv(N*F,256),)](gu,act,N=N,F=F,B=256)
        yp=torch.empty((N,D),device=x.device,dtype=torch.float32); _eout[(cap,triton.cdiv(D,COL))](act,w2,off,te,ts,total,yp,N=N,F=F,D=D,BR=ROW,BC=COL,BK=KB,BF=bf,num_warps=4,num_stages=3)
        y=torch.empty_like(x); _combine[(T,triton.cdiv(D,256))](yp,pw,inv,y,D=D,A=A,BD=256)
        ctx.save_for_backward(x,rw,w1,w2,s,sel,pw,off,te,ts,total,tok,sw,inv,gu,act,yp); ctx.dims=(T,D,E,F,A,N,cap,bf); return y
    @staticmethod
    def backward(ctx,dy):
        x,rw,w1,w2,s,sel,pw,off,te,ts,total,tok,sw,inv,gu,act,yp=ctx.saved_tensors; T,D,E,F,A,N,cap,bf=ctx.dims; dy=dy.contiguous()
        dz=torch.empty((T,E),device=x.device,dtype=torch.float32); _route_grad[(T,)](s,sel,pw,inv,yp,dy,dz,D=D,E=E,A=A,BD=triton.next_power_of_2(D),BE=triton.next_power_of_2(E),num_warps=8)
        rdx=torch.empty((T,D),device=x.device,dtype=torch.float32); _dense[(triton.cdiv(T,32),triton.cdiv(D,32))](dz,rw,rdx,M=T,N=D,K=E,TA=False,TB=True,BM=32,BN=32,BK=32)
        grw=torch.empty_like(rw); _dense[(triton.cdiv(D,32),triton.cdiv(E,32))](x,dz,grw,M=D,N=E,K=T,TA=True,TB=False,BM=32,BN=32,BK=32)
        dgu=torch.empty_like(gu); _dgu[(cap,triton.cdiv(F,COL))](dy,sw,tok,w2,gu,off,te,ts,total,dgu,N=N,D=D,F=F,BR=ROW,BC=COL,BK=KB,BF=bf,num_warps=4,num_stages=3)
        dxp=torch.empty((N,D),device=x.device,dtype=torch.float32); _edx[(cap,triton.cdiv(D,COL))](dgu,w1,off,te,ts,total,dxp,N=N,D=D,F2=2*F,BR=ROW,BC=COL,BK=KB,BF=bf,num_warps=4,num_stages=3)
        dx=torch.empty_like(x); _cdx[(T,triton.cdiv(D,256))](dxp,inv,rdx,dx,D=D,A=A,BD=256)
        gw2=torch.empty_like(w2); _dw2[(E,triton.cdiv(F,128),triton.cdiv(D,64))](act,dy,sw,tok,off,gw2,F=F,D=D,BM=128,BN=64,BK=32,BF=bf,num_warps=4,num_stages=3)
        gw1=torch.empty_like(w1); _dw1[(E,triton.cdiv(D,128),triton.cdiv(2*F,64))](x,dgu,tok,off,gw1,D=D,F2=2*F,BM=128,BN=64,BK=32,BF=bf,num_warps=4,num_stages=3)
        return dx,grw,None,gw1,gw2,None

def fused_moe(x,router_w,router_b,w1,w2,A): return _Moe.apply(x,router_w,router_b,w1,w2,A)
