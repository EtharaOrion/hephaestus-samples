import torch
import triton
import triton.language as tl

R, C, KBLK = 64, 64, 32

@triton.jit
def _dot(a, b, BF: tl.constexpr, SA: tl.constexpr, SB: tl.constexpr):
    if BF:
        ah, bh = a.to(tl.bfloat16), b.to(tl.bfloat16)
        z = tl.dot(ah, bh)
        if SA: z += tl.dot((a-ah).to(tl.bfloat16), bh)
        if SB: z += tl.dot(ah, (b-bh).to(tl.bfloat16))
        return z
    return tl.dot(a, b, input_precision="tf32x3")

@triton.jit
def _router(X,RW,S,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    i=tl.program_id(0)*BM+tl.arange(0,BM); j=tl.program_id(1)*BN+tl.arange(0,BN)
    z=tl.zeros((BM,BN),tl.float32)
    for k0 in range(0,D,BK):
        k=k0+tl.arange(0,BK)
        a=tl.load(X+i[:,None]*D+k[None,:],mask=(i[:,None]<T)&(k[None,:]<D),other=0.).to(tl.float32)
        b=tl.load(RW+k[:,None]*E+j[None,:],mask=(k[:,None]<D)&(j[None,:]<E),other=0.).to(tl.float32)
        z+=tl.dot(a,b,input_precision="ieee")
    tl.store(S+i[:,None]*E+j[None,:],tl.sigmoid(z),mask=(i[:,None]<T)&(j[None,:]<E))

@triton.jit
def _keys(S,B,KY,n:tl.constexpr,E:tl.constexpr,BS:tl.constexpr):
    q=tl.program_id(0)*BS+tl.arange(0,BS); e=q%E; m=q<n
    v=tl.load(S+q,mask=m,other=0.)+tl.load(B+e,mask=m,other=0.); v=tl.where(v==0.,0.,v)
    ib=v.to(tl.int32,bitcast=True).to(tl.int64); u=tl.where(ib>=0,ib^0x80000000,(~ib)&0xffffffff)
    tl.store(KY+q,(u<<10)|(E-1-e),mask=m)

@triton.jit
def _routes(S,SEL,W,FLAT,T:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BA:tl.constexpr):
    t=tl.program_id(0); a=tl.arange(0,BA); m=a<A
    e=tl.load(SEL+t*A+a,mask=m,other=0); s=tl.load(S+t*E+e,mask=m,other=0.)
    tl.store(W+t*A+a,s/tl.sum(s,axis=0),mask=m); tl.store(FLAT+t*A+a,e,mask=m)

@triton.jit
def _inverse(ORDER,INV,N:tl.constexpr,BS:tl.constexpr):
    i=tl.program_id(0)*BS+tl.arange(0,BS); m=i<N; p=tl.load(ORDER+i,mask=m,other=0); tl.store(INV+p,i,mask=m)

@triton.jit
def _offsets(FLAT,ORDER,OFF,N:tl.constexpr,E:tl.constexpr,IT:tl.constexpr):
    e=tl.program_id(0); lo=0; hi=N
    for _ in range(IT):
        md=(lo+hi)//2; mm=md<N; p=tl.load(ORDER+md,mask=mm,other=0); v=tl.load(FLAT+p,mask=mm,other=E)
        take=v<e; lo=tl.where(take,md+1,lo); hi=tl.where(take,hi,md)
    tl.store(OFF+e,lo)

@triton.jit
def _prefix(OFF,PRE,E:tl.constexpr,ROW:tl.constexpr):
    total=0; tl.store(PRE,0)
    for e in range(E):
        total+=tl.cdiv(tl.load(OFF+e+1)-tl.load(OFF+e),ROW); tl.store(PRE+e+1,total)

@triton.jit
def _loc(pid,OFF,PRE,E:tl.constexpr,ROW:tl.constexpr,IT:tl.constexpr):
    lo=0; hi=E
    for _ in range(IT):
        md=(lo+hi)//2; v=tl.load(PRE+md+1,mask=md<E,other=0x7fffffff); take=pid>=v
        lo=tl.where(take,md+1,lo); hi=tl.where(take,hi,md)
    e=lo; ok=e<E; return e,tl.load(OFF+e,mask=ok,other=0)+(pid-tl.load(PRE+e,mask=ok,other=0))*ROW,ok

@triton.jit
def _ein(X,W1,ORDER,OFF,PRE,GU,N:tl.constexpr,D:tl.constexpr,F2:tl.constexpr,A:tl.constexpr,E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr,IT:tl.constexpr):
    e,beg,ok=_loc(tl.program_id(0),OFF,PRE,E,BM,IT); end=tl.load(OFF+e+1,mask=ok,other=0)
    r=beg+tl.arange(0,BM); j=tl.program_id(1)*BN+tl.arange(0,BN); rm=ok&(r<end)
    p=tl.load(ORDER+r,mask=rm,other=0); tok=p//A; z=tl.zeros((BM,BN),tl.float32)
    for k0 in range(0,D,BK):
        k=k0+tl.arange(0,BK)
        a=tl.load(X+tok[:,None]*D+k[None,:],mask=rm[:,None]&(k[None,:]<D),other=0.).to(tl.float32)
        b=tl.load(W1+e*D*F2+k[:,None]*F2+j[None,:],mask=ok&(k[:,None]<D)&(j[None,:]<F2),other=0.).to(tl.float32)
        z+=_dot(a,b,BF,False,False)
    tl.store(GU+r[:,None]*F2+j[None,:],z,mask=rm[:,None]&(j[None,:]<F2))

@triton.jit
def _act(GU,ACT,N:tl.constexpr,F:tl.constexpr,BS:tl.constexpr):
    q=tl.program_id(0)*BS+tl.arange(0,BS); m=q<N*F; r=q//F; c=q%F
    g=tl.load(GU+r*2*F+c,mask=m,other=0.); u=tl.load(GU+r*2*F+F+c,mask=m,other=0.)
    tl.store(ACT+q,g*tl.sigmoid(g)*u,mask=m)

@triton.jit
def _seg(A0,B,OFF,PRE,OUT,N:tl.constexpr,KD:tl.constexpr,ND:tl.constexpr,E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr,SA:tl.constexpr,SB:tl.constexpr,IT:tl.constexpr):
    e,beg,ok=_loc(tl.program_id(0),OFF,PRE,E,BM,IT); end=tl.load(OFF+e+1,mask=ok,other=0)
    i=beg+tl.arange(0,BM); j=tl.program_id(1)*BN+tl.arange(0,BN); rm=ok&(i<end); z=tl.zeros((BM,BN),tl.float32)
    for k0 in range(0,KD,BK):
        k=k0+tl.arange(0,BK)
        a=tl.load(A0+i[:,None]*KD+k[None,:],mask=rm[:,None]&(k[None,:]<KD),other=0.).to(tl.float32)
        b=tl.load(B+e*KD*ND+k[:,None]*ND+j[None,:],mask=ok&(k[:,None]<KD)&(j[None,:]<ND),other=0.).to(tl.float32)
        z+=_dot(a,b,BF,SA,SB)
    tl.store(OUT+i[:,None]*ND+j[None,:],z,mask=rm[:,None]&(j[None,:]<ND))

@triton.jit
def _combine(OUT,INV,W,Y,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,BD:tl.constexpr):
    t=tl.program_id(0); d=tl.program_id(1)*BD+tl.arange(0,BD); z=tl.zeros((BD,),tl.float32)
    for a in range(A):
        p=t*A+a; r=tl.load(INV+p); z+=tl.load(W+p)*tl.load(OUT+r*D+d,mask=d<D,other=0.)
    tl.store(Y+t*D+d,z,mask=d<D)

@triton.jit
def _dz(DY,W,ORDER,Z,N:tl.constexpr,D:tl.constexpr,A:tl.constexpr,BS:tl.constexpr):
    q=tl.program_id(0)*BS+tl.arange(0,BS); m=q<N*D; r=q//D; d=q%D; p=tl.load(ORDER+r,mask=m,other=0)
    tl.store(Z+q,tl.load(DY+(p//A)*D+d,mask=m,other=0.).to(tl.float32)*tl.load(W+p,mask=m,other=0.),mask=m)

@triton.jit
def _segbt(A0,B,OFF,PRE,OUT,N:tl.constexpr,KD:tl.constexpr,ND:tl.constexpr,E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr,SA:tl.constexpr,IT:tl.constexpr):
    e,beg,ok=_loc(tl.program_id(0),OFF,PRE,E,BM,IT); end=tl.load(OFF+e+1,mask=ok,other=0)
    i=beg+tl.arange(0,BM); j=tl.program_id(1)*BN+tl.arange(0,BN); rm=ok&(i<end); z=tl.zeros((BM,BN),tl.float32)
    for k0 in range(0,KD,BK):
        k=k0+tl.arange(0,BK)
        a=tl.load(A0+i[:,None]*KD+k[None,:],mask=rm[:,None]&(k[None,:]<KD),other=0.).to(tl.float32)
        b=tl.load(B+e*ND*KD+j[None,:]*KD+k[:,None],mask=ok&(j[None,:]<ND)&(k[:,None]<KD),other=0.).to(tl.float32)
        z+=_dot(a,b,BF,SA,False)
    tl.store(OUT+i[:,None]*ND+j[None,:],z,mask=rm[:,None]&(j[None,:]<ND))

@triton.jit
def _dgu(GU,DA,DGU,N:tl.constexpr,F:tl.constexpr,BS:tl.constexpr):
    q=tl.program_id(0)*BS+tl.arange(0,BS); m=q<N*F; r=q//F; c=q%F
    g=tl.load(GU+r*2*F+c,mask=m,other=0.); u=tl.load(GU+r*2*F+F+c,mask=m,other=0.); da=tl.load(DA+q,mask=m,other=0.)
    s=tl.sigmoid(g); si=g*s; tl.store(DGU+r*2*F+c,da*u*(s+si*(1-s)),mask=m); tl.store(DGU+r*2*F+F+c,da*si,mask=m)

@triton.jit
def _dw(A0,B0,ORDER,OFF,OUT,K0:tl.constexpr,N0:tl.constexpr,ASEL:tl.constexpr,E:tl.constexpr,GX:tl.constexpr,XD:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr,SA:tl.constexpr,SB:tl.constexpr):
    e=tl.program_id(0); i=tl.program_id(1)*BM+tl.arange(0,BM); j=tl.program_id(2)*BN+tl.arange(0,BN)
    beg=tl.load(OFF+e); end=tl.load(OFF+e+1); z=tl.zeros((BM,BN),tl.float32)
    for p0 in range(beg,end,BK):
        r=p0+tl.arange(0,BK); rm=r<end
        if GX:
            p=tl.load(ORDER+r,mask=rm,other=0); a=tl.load(A0+(p//ASEL)[None,:]*XD+i[:,None],mask=(i[:,None]<K0)&rm[None,:],other=0.).to(tl.float32)
        else: a=tl.load(A0+r[None,:]*K0+i[:,None],mask=rm[None,:]&(i[:,None]<K0),other=0.).to(tl.float32)
        b=tl.load(B0+r[:,None]*N0+j[None,:],mask=rm[:,None]&(j[None,:]<N0),other=0.).to(tl.float32); z+=_dot(a,b,BF,SA,SB)
    tl.store(OUT+e*K0*N0+i[:,None]*N0+j[None,:],z,mask=(i[:,None]<K0)&(j[None,:]<N0))

@triton.jit
def _route_grad(S,SEL,W,OUT,INV,DY,DS,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BA:tl.constexpr,BD:tl.constexpr):
    t=tl.program_id(0); a=tl.arange(0,BA); am=a<A; e=tl.load(SEL+t*A+a,mask=am,other=0); r=tl.load(INV+t*A+a,mask=am,other=0)
    q=tl.zeros((BA,),tl.float32)
    for d0 in range(0,D,BD):
        d=d0+tl.arange(0,BD); dy=tl.load(DY+t*D+d,mask=d<D,other=0.).to(tl.float32)
        fe=tl.load(OUT+r[:,None]*D+d[None,:],mask=am[:,None]&(d[None,:]<D),other=0.); q+=tl.sum(fe*dy[None,:],axis=1)
    w=tl.load(W+t*A+a,mask=am,other=0.); ss=tl.load(S+t*E+e,mask=am,other=0.); den=tl.sum(ss,axis=0); mean=tl.sum(w*q,axis=0)
    for e0 in range(0,E,BA):
        ee=e0+a; tl.store(DS+t*E+ee,0.,mask=ee<E)
    tl.store(DS+t*E+e,(q-mean)/den*ss*(1-ss),mask=am)

@triton.jit
def _dense(A0,B0,OUT,M:tl.constexpr,N:tl.constexpr,KD:tl.constexpr,TA:tl.constexpr,TB:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    i=tl.program_id(0)*BM+tl.arange(0,BM); j=tl.program_id(1)*BN+tl.arange(0,BN); z=tl.zeros((BM,BN),tl.float32)
    for k0 in range(0,KD,BK):
        k=k0+tl.arange(0,BK)
        if TA: a=tl.load(A0+k[None,:]*M+i[:,None],mask=(i[:,None]<M)&(k[None,:]<KD),other=0.).to(tl.float32)
        else: a=tl.load(A0+i[:,None]*KD+k[None,:],mask=(i[:,None]<M)&(k[None,:]<KD),other=0.).to(tl.float32)
        if TB: b=tl.load(B0+j[None,:]*KD+k[:,None],mask=(k[:,None]<KD)&(j[None,:]<N),other=0.).to(tl.float32)
        else: b=tl.load(B0+k[:,None]*N+j[None,:],mask=(k[:,None]<KD)&(j[None,:]<N),other=0.).to(tl.float32)
        z+=tl.dot(a,b,input_precision="ieee")
    tl.store(OUT+i[:,None]*N+j[None,:],z,mask=(i[:,None]<M)&(j[None,:]<N))

@triton.jit
def _sumdx(DXP,INV,DXR,DX,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,BD:tl.constexpr):
    t=tl.program_id(0); d=tl.program_id(1)*BD+tl.arange(0,BD); z=tl.load(DXR+t*D+d,mask=d<D,other=0.)
    for a in range(A): z+=tl.load(DXP+tl.load(INV+t*A+a)*D+d,mask=d<D,other=0.)
    tl.store(DX+t*D+d,z,mask=d<D)

def _segment(a,b,off,pre,n,kd,nd,e,cap,bf,sa=False,sb=False):
    o=torch.empty((n,nd),device=a.device,dtype=torch.float32); it=e.bit_length()
    _seg[(cap,triton.cdiv(nd,C))](a,b,off,pre,o,n,kd,nd,e,BM=R,BN=C,BK=KBLK,BF=bf,SA=sa,SB=sb,IT=it,num_warps=4,num_stages=3); return o

class _Moe(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,rw,rb,w1,w2,A):
    T,D=x.shape; E=rw.shape[1]; F=w2.shape[1]; A=int(A); N=T*A; bf=x.dtype==torch.bfloat16; it=E.bit_length()
    s=torch.empty((T,E),device=x.device,dtype=torch.float32); _router[(triton.cdiv(T,32),triton.cdiv(E,32))](x,rw,s,T,D,E,BM=32,BN=32,BK=32,num_warps=4)
    ky=torch.empty((T,E),device=x.device,dtype=torch.int64); _keys[(triton.cdiv(T*E,256),)](s,rb,ky,T*E,E,BS=256)
    sel=torch.topk(ky,A,dim=1,largest=True,sorted=True).indices.contiguous(); w=torch.empty(N,device=x.device); flat=torch.empty(N,device=x.device,dtype=torch.int64)
    _routes[(T,)](s,sel,w,flat,T,E,A,BA=triton.next_power_of_2(A)); order=torch.argsort(flat).contiguous(); inv=torch.empty_like(order); _inverse[(triton.cdiv(N,256),)](order,inv,N,BS=256)
    off=torch.empty(E+1,device=x.device,dtype=torch.int32); _offsets[(E+1,)](flat,order,off,N,E,IT=max(1,N.bit_length())); pre=torch.empty_like(off); _prefix[(1,)](off,pre,E,ROW=R,num_warps=1)
    cap=triton.cdiv(N,R)+E; gu=torch.empty((N,2*F),device=x.device); _ein[(cap,triton.cdiv(2*F,C))](x,w1,order,off,pre,gu,N,D,2*F,A,E,BM=R,BN=C,BK=KBLK,BF=bf,IT=it,num_warps=4,num_stages=3)
    act=torch.empty((N,F),device=x.device); _act[(triton.cdiv(N*F,256),)](gu,act,N,F,BS=256); out=_segment(act,w2,off,pre,N,F,D,E,cap,bf,sa=bf)
    y=torch.empty((T,D),device=x.device); _combine[(T,triton.cdiv(D,256))](out,inv,w,y,T,D,A,BD=256)
    ctx.save_for_backward(x,rw,w1,w2,s,sel,w,order,inv,off,pre,gu,act,out); ctx.meta=T,D,E,F,A,N,cap,bf,it; return y.to(x.dtype)
 @staticmethod
 def backward(ctx,dy):
    x,rw,w1,w2,s,sel,w,order,inv,off,pre,gu,act,out=ctx.saved_tensors; T,D,E,F,A,N,cap,bf,it=ctx.meta; dy=dy.contiguous()
    z=torch.empty((N,D),device=x.device); _dz[(triton.cdiv(N*D,256),)](dy,w,order,z,N,D,A,BS=256)
    da=torch.empty((N,F),device=x.device); _segbt[(cap,triton.cdiv(F,C))](z,w2,off,pre,da,N,D,F,E,BM=R,BN=C,BK=KBLK,BF=bf,SA=bf,IT=it,num_warps=4,num_stages=3)
    dgu=torch.empty((N,2*F),device=x.device); _dgu[(triton.cdiv(N*F,256),)](gu,da,dgu,N,F,BS=256)
    dxp=torch.empty((N,D),device=x.device); _segbt[(cap,triton.cdiv(D,C))](dgu,w1,off,pre,dxp,N,2*F,D,E,BM=R,BN=C,BK=KBLK,BF=bf,SA=False,IT=it,num_warps=4,num_stages=3)
    dw2=torch.empty_like(w2); _dw[(E,triton.cdiv(F,128),triton.cdiv(D,64))](act,z,order,off,dw2,F,D,A,E,False,D,BM=128,BN=64,BK=32,BF=bf,SA=bf,SB=bf,num_warps=4,num_stages=3)
    dw1=torch.empty_like(w1); _dw[(E,triton.cdiv(D,128),triton.cdiv(2*F,64))](x,dgu,order,off,dw1,D,2*F,A,E,True,D,BM=128,BN=64,BK=32,BF=bf,SA=False,SB=bf,num_warps=4,num_stages=3)
    ds=torch.empty((T,E),device=x.device); _route_grad[(T,)](s,sel,w,out,inv,dy,ds,T,D,E,A,BA=triton.next_power_of_2(max(A,E)),BD=64,num_warps=4)
    dr=torch.empty((D,E),device=x.device); _dense[(triton.cdiv(D,32),triton.cdiv(E,32))](x,ds,dr,D,E,T,True,False,BM=32,BN=32,BK=32,num_warps=4)
    dxr=torch.empty((T,D),device=x.device); _dense[(triton.cdiv(T,32),triton.cdiv(D,32))](ds,rw,dxr,T,D,E,False,True,BM=32,BN=32,BK=32,num_warps=4)
    dx=torch.empty_like(x); _sumdx[(T,triton.cdiv(D,256))](dxp,inv,dxr,dx,T,D,A,BD=256); return dx,dr.to(rw.dtype),None,dw1,dw2,None

def fused_moe(x,router_w,router_b,w1,w2,A): return _Moe.apply(x,router_w,router_b,w1,w2,A)
