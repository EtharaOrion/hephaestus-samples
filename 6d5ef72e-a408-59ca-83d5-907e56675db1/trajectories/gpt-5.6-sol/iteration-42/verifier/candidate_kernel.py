import torch
import triton
import triton.language as tl

R, C, K = 64, 64, 32

@triton.jit
def _dot(a, b, BF: tl.constexpr, SA: tl.constexpr, SB: tl.constexpr, IEEE: tl.constexpr):
    if BF:
        ah, bh = a.to(tl.bfloat16), b.to(tl.bfloat16)
        z = tl.dot(ah, bh)
        if SA: z += tl.dot((a-ah).to(tl.bfloat16), bh)
        if SB: z += tl.dot(ah, (b-bh).to(tl.bfloat16))
        return z
    return tl.dot(a, b, input_precision="ieee" if IEEE else "tf32x3")

@triton.jit
def _router(X,RW,RB,S,KEY,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN)
    z=tl.zeros((BM,BN),tl.float32)
    for k0 in range(0,D,BK):
        k=k0+tl.arange(0,BK)
        a=tl.load(X+m[:,None]*D+k[None,:],mask=(m[:,None]<T)&(k[None,:]<D),other=0.).to(tl.float32)
        b=tl.load(RW+k[:,None]*E+n[None,:],mask=(k[:,None]<D)&(n[None,:]<E),other=0.).to(tl.float32)
        z+=tl.dot(a,b,input_precision="ieee")
    s=tl.sigmoid(z); mask=(m[:,None]<T)&(n[None,:]<E)
    tl.store(S+m[:,None]*E+n[None,:],s,mask=mask)
    v=s+tl.load(RB+n,mask=n<E,other=0.)[None,:]; v=tl.where(v==0.,0.,v)
    ib=v.to(tl.int32,bitcast=True); u=tl.where(ib>=0,ib.to(tl.int64)^0x80000000,(~ib).to(tl.int64)&0xffffffff)
    tl.store(KEY+m[:,None]*E+n[None,:],(u<<10)|(E-1-n[None,:]).to(tl.int64),mask=mask)

@triton.jit
def _weights(S,SEL,W,T:tl.constexpr,E:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0)*B+tl.arange(0,B); mask=t<T; den=tl.zeros((B,),tl.float32)
    for a in range(A):
        e=tl.load(SEL+t*A+a,mask=mask,other=0); den+=tl.load(S+t*E+e,mask=mask,other=0.)
    for a in range(A):
        e=tl.load(SEL+t*A+a,mask=mask,other=0); s=tl.load(S+t*E+e,mask=mask,other=0.)
        tl.store(W+t*A+a,s/den,mask=mask)

@triton.jit
def _offs(SEL,ORD,OFF,N:tl.constexpr,E:tl.constexpr):
    e=tl.program_id(0); lo=tl.zeros((),tl.int32); hi=tl.full((),N,tl.int32)
    for _ in range(24):
        mid=(lo+hi)//2; p=tl.load(ORD+mid,mask=mid<N,other=0); v=tl.load(SEL+p,mask=mid<N,other=E); q=(mid<N)&(v<e)
        lo=tl.where(q,mid+1,lo); hi=tl.where(q,hi,mid)
    tl.store(OFF+e,lo)

@triton.jit
def _prefix(OFF,PRE,E:tl.constexpr,BR:tl.constexpr):
    total=tl.zeros((),tl.int32); tl.store(PRE,total)
    for e in range(E):
        total+=(tl.load(OFF+e+1)-tl.load(OFF+e)+BR-1)//BR; tl.store(PRE+e+1,total)

@triton.jit
def _inv(ORD,INV,N:tl.constexpr,B:tl.constexpr):
    q=tl.program_id(0)*B+tl.arange(0,B); mask=q<N; p=tl.load(ORD+q,mask=mask,other=0); tl.store(INV+p,q,mask=mask)

@triton.jit
def _loc(pid,PRE,OFF,E:tl.constexpr,BR:tl.constexpr):
    lo=tl.zeros((),tl.int32); hi=tl.full((),E,tl.int32)
    for _ in range(11):
        mid=(lo+hi)//2; v=tl.load(PRE+mid+1,mask=mid<E,other=0x7fffffff); left=pid<v
        hi=tl.where(left,mid,hi); lo=tl.where(left,lo,mid+1)
    e=lo; start=tl.load(OFF+e)+(pid-tl.load(PRE+e))*BR
    return e,start,tl.load(OFF+e+1)

@triton.jit
def _ein(X,W1,ORD,OFF,PRE,GU,A:tl.constexpr,D:tl.constexpr,O:tl.constexpr,E:tl.constexpr,BF:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0); n=tl.program_id(1)*BN+tl.arange(0,BN); e,start,end=_loc(pid,PRE,OFF,E,BM); active=pid<tl.load(PRE+E)
    q=start+tl.arange(0,BM); p=tl.load(ORD+q,mask=active&(q<end),other=0); tok=p//A; z=tl.zeros((BM,BN),tl.float32)
    for k0 in range(0,D,BK):
        k=k0+tl.arange(0,BK)
        a=tl.load(X+tok[:,None]*D+k[None,:],mask=active&(q[:,None]<end)&(k[None,:]<D),other=0.).to(tl.float32)
        b=tl.load(W1+e*D*O+k[:,None]*O+n[None,:],mask=active&(k[:,None]<D)&(n[None,:]<O),other=0.).to(tl.float32)
        z+=_dot(a,b,BF,False,False,False)
    tl.store(GU+q[:,None]*O+n[None,:],z,mask=active&(q[:,None]<end)&(n[None,:]<O))

@triton.jit
def _act(GU,ACT,N:tl.constexpr,F:tl.constexpr,B:tl.constexpr):
    o=tl.program_id(0)*B+tl.arange(0,B); mask=o<N*F; q=o//F; j=o-q*F
    g=tl.load(GU+q*2*F+j,mask=mask,other=0.); u=tl.load(GU+q*2*F+F+j,mask=mask,other=0.)
    tl.store(ACT+o,g*tl.sigmoid(g)*u,mask=mask)

@triton.jit
def _seg(AIN,BW,OFF,PRE,OUT,KS:tl.constexpr,O:tl.constexpr,E:tl.constexpr,BF:tl.constexpr,SA:tl.constexpr,SB:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0); n=tl.program_id(1)*BN+tl.arange(0,BN); e,start,end=_loc(pid,PRE,OFF,E,BM); active=pid<tl.load(PRE+E)
    q=start+tl.arange(0,BM); z=tl.zeros((BM,BN),tl.float32)
    for k0 in range(0,KS,BK):
        k=k0+tl.arange(0,BK)
        a=tl.load(AIN+q[:,None]*KS+k[None,:],mask=active&(q[:,None]<end)&(k[None,:]<KS),other=0.).to(tl.float32)
        b=tl.load(BW+e*KS*O+k[:,None]*O+n[None,:],mask=active&(k[:,None]<KS)&(n[None,:]<O),other=0.).to(tl.float32)
        z+=_dot(a,b,BF,SA,SB,False)
    tl.store(OUT+q[:,None]*O+n[None,:],z,mask=active&(q[:,None]<end)&(n[None,:]<O))

@triton.jit
def _segbt(AIN,BW,OFF,PRE,OUT,KS:tl.constexpr,O:tl.constexpr,E:tl.constexpr,BF:tl.constexpr,SA:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0); n=tl.program_id(1)*BN+tl.arange(0,BN); e,start,end=_loc(pid,PRE,OFF,E,BM); active=pid<tl.load(PRE+E)
    q=start+tl.arange(0,BM); z=tl.zeros((BM,BN),tl.float32)
    for k0 in range(0,KS,BK):
        k=k0+tl.arange(0,BK)
        a=tl.load(AIN+q[:,None]*KS+k[None,:],mask=active&(q[:,None]<end)&(k[None,:]<KS),other=0.).to(tl.float32)
        b=tl.load(BW+e*O*KS+n[None,:]*KS+k[:,None],mask=active&(n[None,:]<O)&(k[:,None]<KS),other=0.).to(tl.float32)
        z+=_dot(a,b,BF,SA,False,False)
    tl.store(OUT+q[:,None]*O+n[None,:],z,mask=active&(q[:,None]<end)&(n[None,:]<O))

@triton.jit
def _combine(YP,W,INV,Y,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0); j=tl.program_id(1)*B+tl.arange(0,B); z=tl.zeros((B,),tl.float32)
    for a in range(A):
        p=t*A+a; q=tl.load(INV+p); z+=tl.load(W+p)*tl.load(YP+q*D+j,mask=j<D,other=0.)
    tl.store(Y+t*D+j,z,mask=j<D)

@triton.jit
def _dz(DY,W,ORD,DZ,N:tl.constexpr,A:tl.constexpr,D:tl.constexpr,B:tl.constexpr):
    o=tl.program_id(0)*B+tl.arange(0,B); mask=o<N*D; q=o//D; j=o-q*D; p=tl.load(ORD+q,mask=mask,other=0)
    tl.store(DZ+o,tl.load(W+p,mask=mask,other=0.)*tl.load(DY+(p//A)*D+j,mask=mask,other=0.),mask=mask)

@triton.jit
def _dact(GU,DA,DGU,N:tl.constexpr,F:tl.constexpr,B:tl.constexpr):
    o=tl.program_id(0)*B+tl.arange(0,B); mask=o<N*F; q=o//F; j=o-q*F
    g=tl.load(GU+q*2*F+j,mask=mask,other=0.); u=tl.load(GU+q*2*F+F+j,mask=mask,other=0.); da=tl.load(DA+o,mask=mask,other=0.); s=tl.sigmoid(g); silu=g*s
    tl.store(DGU+q*2*F+j,da*u*(s+silu*(1-s)),mask=mask); tl.store(DGU+q*2*F+F+j,da*silu,mask=mask)

@triton.jit
def _dw(AIN,BIN,ORD,OFF,OUT,ASEL:tl.constexpr,KS:tl.constexpr,O:tl.constexpr,E:tl.constexpr,GATHER:tl.constexpr,BF:tl.constexpr,SA:tl.constexpr,SB:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    e=tl.program_id(0); m=tl.program_id(1)*BM+tl.arange(0,BM); n=tl.program_id(2)*BN+tl.arange(0,BN); begin=tl.load(OFF+e); end=tl.load(OFF+e+1); z=tl.zeros((BM,BN),tl.float32)
    for q0 in range(begin,end,BK):
        q=q0+tl.arange(0,BK)
        if GATHER: row=tl.load(ORD+q,mask=q<end,other=0)//ASEL
        else: row=q
        a=tl.load(AIN+row[None,:]*KS+m[:,None],mask=(q[None,:]<end)&(m[:,None]<KS),other=0.).to(tl.float32)
        b=tl.load(BIN+q[:,None]*O+n[None,:],mask=(q[:,None]<end)&(n[None,:]<O),other=0.).to(tl.float32)
        z+=_dot(a,b,BF,SA,SB,False)
    tl.store(OUT+e*KS*O+m[:,None]*O+n[None,:],z,mask=(m[:,None]<KS)&(n[None,:]<O))

@triton.jit
def _route(DY,YP,S,SEL,INV,DS,D:tl.constexpr,E:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0); den=tl.zeros((),tl.float32); mean=tl.zeros((),tl.float32); j=tl.arange(0,B)
    for a in range(A):
        p=t*A+a; q=tl.load(INV+p); e=tl.load(SEL+p); s=tl.load(S+t*E+e); z=tl.zeros((),tl.float32)
        for d0 in range(0,D,B):
            jj=d0+j; z+=tl.sum(tl.load(DY+t*D+jj,mask=jj<D,other=0.)*tl.load(YP+q*D+jj,mask=jj<D,other=0.))
        den+=s; mean+=s*z
    mean/=den
    for a in range(A):
        p=t*A+a; q=tl.load(INV+p); e=tl.load(SEL+p); s=tl.load(S+t*E+e); z=tl.zeros((),tl.float32)
        for d0 in range(0,D,B):
            jj=d0+j; z+=tl.sum(tl.load(DY+t*D+jj,mask=jj<D,other=0.)*tl.load(YP+q*D+jj,mask=jj<D,other=0.))
        tl.store(DS+t*E+e,(z-mean)/den*s*(1-s))

@triton.jit
def _dense(AIN,BIN,OUT,M:tl.constexpr,N:tl.constexpr,KS:tl.constexpr,TA:tl.constexpr,TB:tl.constexpr,IEEE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); z=tl.zeros((BM,BN),tl.float32)
    for k0 in range(0,KS,BK):
        k=k0+tl.arange(0,BK)
        if TA: a=tl.load(AIN+k[None,:]*M+m[:,None],mask=(m[:,None]<M)&(k[None,:]<KS),other=0.).to(tl.float32)
        else: a=tl.load(AIN+m[:,None]*KS+k[None,:],mask=(m[:,None]<M)&(k[None,:]<KS),other=0.).to(tl.float32)
        if TB: b=tl.load(BIN+n[None,:]*KS+k[:,None],mask=(k[:,None]<KS)&(n[None,:]<N),other=0.).to(tl.float32)
        else: b=tl.load(BIN+k[:,None]*N+n[None,:],mask=(k[:,None]<KS)&(n[None,:]<N),other=0.).to(tl.float32)
        z+=_dot(a,b,False,False,False,IEEE)
    tl.store(OUT+m[:,None]*N+n[None,:],z,mask=(m[:,None]<M)&(n[None,:]<N))

@triton.jit
def _sumdx(DXE,INV,DXR,DX,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0); j=tl.program_id(1)*B+tl.arange(0,B); z=tl.load(DXR+t*D+j,mask=j<D,other=0.)
    for a in range(A): z+=tl.load(DXE+tl.load(INV+t*A+a)*D+j,mask=j<D,other=0.)
    tl.store(DX+t*D+j,z,mask=j<D)

def _meta(sel,E):
    n=sel.numel(); order=torch.argsort(sel.reshape(-1)); off=torch.empty(E+1,device=sel.device,dtype=torch.int32); _offs[(E+1,)](sel,order,off,N=n,E=E,num_warps=1)
    pre=torch.empty_like(off); _prefix[(1,)](off,pre,E=E,BR=R,num_warps=1); inv=torch.empty(n,device=sel.device,dtype=torch.int64); _inv[(triton.cdiv(n,256),)](order,inv,N=n,B=256)
    return order,off,pre,inv

class _Moe(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,rw,rb,w1,w2,A):
    A=int(A); T,D=x.shape; E=rw.shape[1]; F=w1.shape[2]//2; N=T*A; bf=x.dtype==torch.bfloat16
    s=torch.empty((T,E),device=x.device,dtype=torch.float32); keys=torch.empty((T,E),device=x.device,dtype=torch.int64); _router[(triton.cdiv(T,32),triton.cdiv(E,32))](x,rw,rb,s,keys,T=T,D=D,E=E,BM=32,BN=32,BK=32,num_warps=4,num_stages=3)
    sel=torch.topk(keys,A,dim=1,largest=True,sorted=True).indices; weights=torch.empty(N,device=x.device,dtype=torch.float32); _weights[(triton.cdiv(T,256),)](s,sel,weights,T=T,E=E,A=A,B=256)
    order,off,pre,inv=_meta(sel,E); cap=triton.cdiv(N,R)+E; gu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _ein[(cap,triton.cdiv(2*F,C))](x,w1,order,off,pre,gu,A=A,D=D,O=2*F,E=E,BF=bf,BM=R,BN=C,BK=K,num_warps=4,num_stages=3)
    act=torch.empty((N,F),device=x.device,dtype=torch.float32); _act[(triton.cdiv(N*F,256),)](gu,act,N=N,F=F,B=256); yp=torch.empty((N,D),device=x.device,dtype=torch.float32); _seg[(cap,triton.cdiv(D,C))](act,w2,off,pre,yp,KS=F,O=D,E=E,BF=bf,SA=bf,SB=False,BM=R,BN=C,BK=K,num_warps=4,num_stages=3)
    y=torch.empty_like(x); _combine[(T,triton.cdiv(D,256))](yp,weights,inv,y,T=T,D=D,A=A,B=256); ctx.save_for_backward(x,rw,w1,w2,s,sel,weights,order,off,pre,inv,gu,act,yp); ctx.dims=T,D,E,F,A,cap,bf
    return y
 @staticmethod
 def backward(ctx,dy):
    x,rw,w1,w2,s,sel,weights,order,off,pre,inv,gu,act,yp=ctx.saved_tensors; T,D,E,F,A,cap,bf=ctx.dims; N=T*A; dy=dy.contiguous()
    dz=torch.empty((N,D),device=x.device,dtype=torch.float32); _dz[(triton.cdiv(N*D,256),)](dy,weights,order,dz,N=N,A=A,D=D,B=256); da=torch.empty((N,F),device=x.device,dtype=torch.float32); _segbt[(cap,triton.cdiv(F,C))](dz,w2,off,pre,da,KS=D,O=F,E=E,BF=bf,SA=bf,BM=R,BN=C,BK=K,num_warps=4,num_stages=3)
    dgu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _dact[(triton.cdiv(N*F,256),)](gu,da,dgu,N=N,F=F,B=256); dxe=torch.empty((N,D),device=x.device,dtype=torch.float32); _segbt[(cap,triton.cdiv(D,C))](dgu,w1,off,pre,dxe,KS=2*F,O=D,E=E,BF=bf,SA=bf,BM=R,BN=C,BK=K,num_warps=4,num_stages=3)
    dw2=torch.empty_like(w2); _dw[(E,triton.cdiv(F,128),triton.cdiv(D,64))](act,dz,order,off,dw2,ASEL=A,KS=F,O=D,E=E,GATHER=False,BF=bf,SA=True,SB=True,BM=128,BN=64,BK=32,num_warps=4,num_stages=3)
    dw1=torch.empty_like(w1); _dw[(E,triton.cdiv(D,128),triton.cdiv(2*F,64))](x,dgu,order,off,dw1,ASEL=A,KS=D,O=2*F,E=E,GATHER=True,BF=bf,SA=False,SB=True,BM=128,BN=64,BK=32,num_warps=4,num_stages=3)
    ds=torch.zeros((T,E),device=x.device,dtype=torch.float32); _route[(T,)](dy,yp,s,sel,inv,ds,D=D,E=E,A=A,B=256,num_warps=4); drw=torch.empty_like(rw); _dense[(triton.cdiv(D,64),triton.cdiv(E,64))](x,ds,drw,M=D,N=E,KS=T,TA=True,TB=False,IEEE=True,BM=64,BN=64,BK=32,num_warps=4,num_stages=3)
    dxr=torch.empty((T,D),device=x.device,dtype=torch.float32); _dense[(triton.cdiv(T,64),triton.cdiv(D,64))](ds,rw,dxr,M=T,N=D,KS=E,TA=False,TB=True,IEEE=True,BM=64,BN=64,BK=32,num_warps=4,num_stages=3); dx=torch.empty_like(x); _sumdx[(T,triton.cdiv(D,256))](dxe,inv,dxr,dx,T=T,D=D,A=A,B=256)
    return dx,drw,None,dw1,dw2,None

def fused_moe(x,router_w,router_b,w1,w2,A):
    return _Moe.apply(x,router_w,router_b,w1,w2,A)
