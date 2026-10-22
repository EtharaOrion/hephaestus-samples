import torch
import triton
import triton.language as tl

ROW, COL, KB = 64, 64, 32

@triton.jit
def _router(X, RW, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr, BM: tl.constexpr=32, BN: tl.constexpr=32, BK: tl.constexpr=32):
    m=tl.program_id(0)*BM+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),tl.float32)
    for k0 in range(0,D,BK):
        k=k0+tl.arange(0,BK)
        a=tl.load(X+m[:,None]*D+k[None,:],mask=(m[:,None]<T)&(k[None,:]<D),other=0.).to(tl.float32)
        b=tl.load(RW+k[:,None]*E+n[None,:],mask=(k[:,None]<D)&(n[None,:]<E),other=0.).to(tl.float32)
        acc+=tl.dot(a,b,input_precision="ieee")
    tl.store(S+m[:,None]*E+n[None,:],tl.sigmoid(acc),mask=(m[:,None]<T)&(n[None,:]<E))

@triton.jit
def _keys(S,RB,K,T:tl.constexpr,E:tl.constexpr,B:tl.constexpr=256):
    q=tl.program_id(0)*B+tl.arange(0,B); mask=q<T*E; e=q%E
    v=tl.load(S+q,mask=mask,other=0.)+tl.load(RB+e,mask=mask,other=0.); v=tl.where(v==0.,0.,v)
    i=v.to(tl.int32,bitcast=True).to(tl.int64); u=tl.where(i>=0,i^0x80000000,(~i)&0xFFFFFFFF)
    tl.store(K+q,(u<<10)|(E-1-e).to(tl.int64),mask=mask)

@triton.jit
def _dkeys(SEL,K,N:tl.constexpr,B:tl.constexpr=256):
    p=tl.program_id(0)*B+tl.arange(0,B); m=p<N; e=tl.load(SEL+p,mask=m,other=0).to(tl.int64)
    tl.store(K+p,e*(N+1)+p,mask=m)

@triton.jit
def _offsets(ORDER,SEL,OFF,N:tl.constexpr,E:tl.constexpr):
    e=tl.program_id(0); lo=0; hi=N
    while lo<hi:
        mid=(lo+hi)//2; p=tl.load(ORDER+mid); v=tl.load(SEL+p); take=v<e
        lo=tl.where(take,mid+1,lo); hi=tl.where(take,hi,mid)
    tl.store(OFF+e,lo)
    if e==E-1: tl.store(OFF+E,N)

@triton.jit
def _prefix(OFF,PRE,E:tl.constexpr,R:tl.constexpr):
    z=0; tl.store(PRE,0)
    for e in range(0,E):
        z+=tl.cdiv(tl.load(OFF+e+1)-tl.load(OFF+e),R); tl.store(PRE+e+1,z)

@triton.jit
def _inverse(ORDER,INV,N:tl.constexpr,B:tl.constexpr=256):
    p=tl.program_id(0)*B+tl.arange(0,B); m=p<N; old=tl.load(ORDER+p,mask=m,other=0); tl.store(INV+old,p,mask=m)

@triton.jit
def _loc(pid,OFF,PRE,E:tl.constexpr,R:tl.constexpr):
    lo=0; hi=E
    while lo<hi:
        mid=(lo+hi)//2; take=pid>=tl.load(PRE+mid+1); lo=tl.where(take,mid+1,lo); hi=tl.where(take,hi,mid)
    e=lo; ok=e<E; pe=tl.load(PRE+e,mask=ok,other=0); begin=tl.load(OFF+e,mask=ok,other=0)+(pid-pe)*R; end=tl.load(OFF+e+1,mask=ok,other=0)
    return e,begin,end,ok

@triton.jit
def _ein(X,W,ORDER,OFF,PRE,OUT,A:tl.constexpr,D:tl.constexpr,O:tl.constexpr,E:tl.constexpr,BF:tl.constexpr,R:tl.constexpr,C:tl.constexpr,KBLK:tl.constexpr):
    pid=tl.program_id(0); j=tl.program_id(1)*C+tl.arange(0,C); e,beg,end,ok=_loc(pid,OFF,PRE,E,R)
    p=beg+tl.arange(0,R); rm=ok&(p<end); old=tl.load(ORDER+p,mask=rm,other=0); tok=old//A; acc=tl.zeros((R,C),tl.float32)
    for k0 in range(0,D,KBLK):
        k=k0+tl.arange(0,KBLK); a=tl.load(X+tok[:,None]*D+k[None,:],mask=rm[:,None]&(k[None,:]<D),other=0.)
        w=tl.load(W+e*D*O+k[:,None]*O+j[None,:],mask=ok&(k[:,None]<D)&(j[None,:]<O),other=0.)
        if BF: acc+=tl.dot(a,w)
        else: acc+=tl.dot(a.to(tl.float32),w.to(tl.float32),input_precision="tf32x3")
    tl.store(OUT+p[:,None]*O+j[None,:],acc,mask=rm[:,None]&(j[None,:]<O))

@triton.jit
def _act(GU,ACT,N:tl.constexpr,F:tl.constexpr,B:tl.constexpr=256):
    q=tl.program_id(0)*B+tl.arange(0,B); m=q<N*F; p=q//F; f=q%F
    g=tl.load(GU+p*2*F+f,mask=m,other=0.); u=tl.load(GU+p*2*F+F+f,mask=m,other=0.)
    tl.store(ACT+q,g*tl.sigmoid(g)*u,mask=m)

@triton.jit
def _seg(AIN,W,OFF,PRE,OUT,K:tl.constexpr,O:tl.constexpr,E:tl.constexpr,BF:tl.constexpr,SPLIT:tl.constexpr,R:tl.constexpr,C:tl.constexpr,KBLK:tl.constexpr):
    pid=tl.program_id(0); j=tl.program_id(1)*C+tl.arange(0,C); e,beg,end,ok=_loc(pid,OFF,PRE,E,R); p=beg+tl.arange(0,R); rm=ok&(p<end); acc=tl.zeros((R,C),tl.float32)
    for k0 in range(0,K,KBLK):
        k=k0+tl.arange(0,KBLK); a=tl.load(AIN+p[:,None]*K+k[None,:],mask=rm[:,None]&(k[None,:]<K),other=0.)
        w=tl.load(W+e*K*O+k[:,None]*O+j[None,:],mask=ok&(k[:,None]<K)&(j[None,:]<O),other=0.)
        if BF and SPLIT:
            ah=a.to(tl.bfloat16); al=(a-ah).to(tl.bfloat16); acc+=tl.dot(ah,w)+tl.dot(al,w)
        elif BF: acc+=tl.dot(a,w)
        else: acc+=tl.dot(a.to(tl.float32),w.to(tl.float32),input_precision="tf32x3")
    tl.store(OUT+p[:,None]*O+j[None,:],acc,mask=rm[:,None]&(j[None,:]<O))

@triton.jit
def _combine(Z,INV,S,SEL,Y,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,A:tl.constexpr,C:tl.constexpr=256):
    t=tl.program_id(0); d=tl.program_id(1)*C+tl.arange(0,C); den=0.
    for a in range(0,A):
        e=tl.load(SEL+t*A+a); den+=tl.load(S+t*E+e)
    acc=tl.zeros((C,),tl.float32)
    for a in range(0,A):
        old=t*A+a; p=tl.load(INV+old); e=tl.load(SEL+old); acc+=(tl.load(S+t*E+e)/den)*tl.load(Z+p*D+d,mask=d<D,other=0.)
    tl.store(Y+t*D+d,acc,mask=d<D)

@triton.jit
def _dz(DY,Z,S,SEL,INV,DZ,DL,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,A:tl.constexpr,C:tl.constexpr=256):
    t=tl.program_id(0); aa=tl.program_id(1); lane=tl.arange(0,C); old=t*A+aa; p=tl.load(INV+old); e=tl.load(SEL+old); den=0.; wq=0.; selfq=0.
    for a in range(0,A):
        o=t*A+a; pp=tl.load(INV+o); ee=tl.load(SEL+o); ss=tl.load(S+t*E+ee); den+=ss; q=tl.zeros((C,),tl.float32)
        for d0 in range(0,D,C):
            d=d0+lane; q+=tl.load(DY+t*D+d,mask=d<D,other=0.).to(tl.float32)*tl.load(Z+pp*D+d,mask=d<D,other=0.)
        qr=tl.sum(q,axis=0); wq+=ss*qr; selfq=tl.where(a==aa,qr,selfq)
    ss=tl.load(S+t*E+e)
    for d0 in range(0,D,C):
        d=d0+lane; tl.store(DZ+p*D+d,tl.load(DY+t*D+d,mask=d<D,other=0.).to(tl.float32)*(ss/den),mask=d<D)
    tl.store(DL+t*E+e,((selfq-wq/den)/den)*ss*(1.-ss))

@triton.jit
def _dact(GU,DA,DGU,N:tl.constexpr,F:tl.constexpr,B:tl.constexpr=256):
    q=tl.program_id(0)*B+tl.arange(0,B); m=q<N*F; p=q//F; f=q%F; g=tl.load(GU+p*2*F+f,mask=m,other=0.); u=tl.load(GU+p*2*F+F+f,mask=m,other=0.); da=tl.load(DA+q,mask=m,other=0.); s=tl.sigmoid(g)
    tl.store(DGU+p*2*F+f,da*u*(s+g*s*(1.-s)),mask=m); tl.store(DGU+p*2*F+F+f,da*g*s,mask=m)

@triton.jit
def _seg_t(DO,W,OFF,PRE,DI,K:tl.constexpr,O:tl.constexpr,E:tl.constexpr,BF:tl.constexpr,R:tl.constexpr,C:tl.constexpr,KBLK:tl.constexpr):
    pid=tl.program_id(0); k=tl.program_id(1)*C+tl.arange(0,C); e,beg,end,ok=_loc(pid,OFF,PRE,E,R); p=beg+tl.arange(0,R); rm=ok&(p<end); acc=tl.zeros((R,C),tl.float32)
    for o0 in range(0,O,KBLK):
        o=o0+tl.arange(0,KBLK); a=tl.load(DO+p[:,None]*O+o[None,:],mask=rm[:,None]&(o[None,:]<O),other=0.); w=tl.load(W+e*K*O+k[:,None]*O+o[None,:],mask=ok&(k[:,None]<K)&(o[None,:]<O),other=0.)
        if BF:
            ah=a.to(tl.bfloat16); al=(a-ah).to(tl.bfloat16); acc+=tl.dot(ah,tl.trans(w))+tl.dot(al,tl.trans(w))
        else: acc+=tl.dot(a,tl.trans(w),input_precision="tf32x3")
    tl.store(DI+p[:,None]*K+k[None,:],acc,mask=rm[:,None]&(k[None,:]<K))

@triton.jit
def _dw(A,B,OFF,GW,M:tl.constexpr,N:tl.constexpr,E:tl.constexpr,BF:tl.constexpr,BM:tl.constexpr=128,BN:tl.constexpr=64,BK:tl.constexpr=32):
    e=tl.program_id(0); m=tl.program_id(1)*BM+tl.arange(0,BM); n=tl.program_id(2)*BN+tl.arange(0,BN); beg=tl.load(OFF+e); end=tl.load(OFF+e+1); acc=tl.zeros((BM,BN),tl.float32)
    for p0 in range(beg,end,BK):
        p=p0+tl.arange(0,BK); a=tl.load(A+p[:,None]*M+m[None,:],mask=(p[:,None]<end)&(m[None,:]<M),other=0.); b=tl.load(B+p[:,None]*N+n[None,:],mask=(p[:,None]<end)&(n[None,:]<N),other=0.)
        if BF:
            ah=a.to(tl.bfloat16); al=(a-ah).to(tl.bfloat16); bh=b.to(tl.bfloat16); bl=(b-bh).to(tl.bfloat16); acc+=tl.dot(tl.trans(ah),bh)+tl.dot(tl.trans(al),bh)+tl.dot(tl.trans(ah),bl)
        else: acc+=tl.dot(tl.trans(a),b,input_precision="ieee")
    tl.store(GW+e*M*N+m[:,None]*N+n[None,:],acc,mask=(m[:,None]<M)&(n[None,:]<N))

@triton.jit
def _dw1(X,G,ORDER,OFF,GW,A:tl.constexpr,D:tl.constexpr,O:tl.constexpr,E:tl.constexpr,BF:tl.constexpr,BM:tl.constexpr=128,BN:tl.constexpr=64,BK:tl.constexpr=32):
    e=tl.program_id(0); m=tl.program_id(1)*BM+tl.arange(0,BM); n=tl.program_id(2)*BN+tl.arange(0,BN); beg=tl.load(OFF+e); end=tl.load(OFF+e+1); acc=tl.zeros((BM,BN),tl.float32)
    for p0 in range(beg,end,BK):
        p=p0+tl.arange(0,BK); old=tl.load(ORDER+p,mask=p<end,other=0); tok=old//A; x=tl.load(X+tok[:,None]*D+m[None,:],mask=(p[:,None]<end)&(m[None,:]<D),other=0.); g=tl.load(G+p[:,None]*O+n[None,:],mask=(p[:,None]<end)&(n[None,:]<O),other=0.)
        if BF:
            gh=g.to(tl.bfloat16); r=g-gh; gm=r.to(tl.bfloat16); gl=(r-gm).to(tl.bfloat16); acc+=tl.dot(tl.trans(x),gh)+tl.dot(tl.trans(x),gm)+tl.dot(tl.trans(x),gl)
        else: acc+=tl.dot(tl.trans(x.to(tl.float32)),g,input_precision="ieee")
    tl.store(GW+e*D*O+m[:,None]*O+n[None,:],acc,mask=(m[:,None]<D)&(n[None,:]<O))

@triton.jit
def _dxr(G,W,OFF,PRE,DX,D:tl.constexpr,O:tl.constexpr,E:tl.constexpr,BF:tl.constexpr,R:tl.constexpr,C:tl.constexpr,KBLK:tl.constexpr):
    pid=tl.program_id(0); d=tl.program_id(1)*C+tl.arange(0,C); e,beg,end,ok=_loc(pid,OFF,PRE,E,R); p=beg+tl.arange(0,R); rm=ok&(p<end); acc=tl.zeros((R,C),tl.float32)
    for j0 in range(0,O,KBLK):
        j=j0+tl.arange(0,KBLK); g=tl.load(G+p[:,None]*O+j[None,:],mask=rm[:,None]&(j[None,:]<O),other=0.); w=tl.load(W+e*D*O+d[:,None]*O+j[None,:],mask=ok&(d[:,None]<D)&(j[None,:]<O),other=0.)
        if BF:
            gh=g.to(tl.bfloat16); gl=(g-gh).to(tl.bfloat16); acc+=tl.dot(gh,tl.trans(w))+tl.dot(gl,tl.trans(w))
        else: acc+=tl.dot(g,tl.trans(w),input_precision="tf32x3")
    tl.store(DX+p[:,None]*D+d[None,:],acc,mask=rm[:,None]&(d[None,:]<D))

@triton.jit
def _dense(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BF:tl.constexpr,TA:tl.constexpr,TB:tl.constexpr,BM:tl.constexpr=64,BN:tl.constexpr=32,BK:tl.constexpr=32):
    m=tl.program_id(0)*BM+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); acc=tl.zeros((BM,BN),tl.float32)
    for k0 in range(0,K,BK):
        k=k0+tl.arange(0,BK)
        if TA: a=tl.load(A+k[:,None]*M+m[None,:],mask=(k[:,None]<K)&(m[None,:]<M),other=0.); a=tl.trans(a)
        else: a=tl.load(A+m[:,None]*K+k[None,:],mask=(m[:,None]<M)&(k[None,:]<K),other=0.)
        if TB: b=tl.load(B+n[:,None]*K+k[None,:],mask=(n[:,None]<N)&(k[None,:]<K),other=0.); b=tl.trans(b)
        else: b=tl.load(B+k[:,None]*N+n[None,:],mask=(k[:,None]<K)&(n[None,:]<N),other=0.)
        if BF:
            ah=a.to(tl.bfloat16); bh=b.to(tl.bfloat16); r=b-bh; bm=r.to(tl.bfloat16); bl=(r-bm).to(tl.bfloat16); acc+=tl.dot(ah,bh)+tl.dot(ah,bm)+tl.dot(ah,bl)
        else: acc+=tl.dot(a.to(tl.float32),b.to(tl.float32),input_precision="ieee")
    tl.store(C+m[:,None]*N+n[None,:],acc,mask=(m[:,None]<M)&(n[None,:]<N))

@triton.jit
def _sumdx(DXR,INV,RDX,DX,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,C:tl.constexpr=256):
    t=tl.program_id(0); d=tl.program_id(1)*C+tl.arange(0,C); acc=tl.load(RDX+t*D+d,mask=d<D,other=0.)
    for a in range(0,A): acc+=tl.load(DXR+tl.load(INV+t*A+a)*D+d,mask=d<D,other=0.)
    tl.store(DX+t*D+d,acc,mask=d<D)

class _Moe(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,rw,rb,w1,w2,A):
    A=int(A); T,D=x.shape; E=rw.shape[1]; F=w2.shape[1]; N=T*A; bf=x.dtype==torch.bfloat16
    s=torch.empty((T,E),device=x.device,dtype=torch.float32); _router[(triton.cdiv(T,32),triton.cdiv(E,32))](x,rw,s,T=T,D=D,E=E)
    keys=torch.empty((T,E),device=x.device,dtype=torch.int64); _keys[(triton.cdiv(T*E,256),)](s,rb,keys,T=T,E=E); sel=torch.topk(keys,A,dim=1,largest=True,sorted=True).indices.contiguous()
    dk=torch.empty(N,device=x.device,dtype=torch.int64); _dkeys[(triton.cdiv(N,256),)](sel,dk,N=N); order=torch.argsort(dk).contiguous(); off=torch.empty(E+1,device=x.device,dtype=torch.int32); pre=torch.empty_like(off); inv=torch.empty(N,device=x.device,dtype=torch.int64)
    _offsets[(E,)](order,sel,off,N=N,E=E,num_warps=1); _prefix[(1,)](off,pre,E=E,R=ROW,num_warps=1); _inverse[(triton.cdiv(N,256),)](order,inv,N=N); cap=triton.cdiv(N,ROW)+E
    gu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _ein[(cap,triton.cdiv(2*F,COL))](x,w1,order,off,pre,gu,A=A,D=D,O=2*F,E=E,BF=bf,R=ROW,C=COL,KBLK=KB,num_warps=4,num_stages=3)
    act=torch.empty((N,F),device=x.device,dtype=torch.float32); _act[(triton.cdiv(N*F,256),)](gu,act,N=N,F=F); z=torch.empty((N,D),device=x.device,dtype=torch.float32); _seg[(cap,triton.cdiv(D,COL))](act,w2,off,pre,z,K=F,O=D,E=E,BF=bf,SPLIT=bf,R=ROW,C=COL,KBLK=KB,num_warps=4,num_stages=3)
    y=torch.empty((T,D),device=x.device,dtype=x.dtype); _combine[(T,triton.cdiv(D,256))](z,inv,s,sel,y,T=T,D=D,E=E,A=A); ctx.save_for_backward(x,rw,w1,w2,s,sel,order,off,pre,inv,gu,act,z); ctx.meta=T,D,E,F,A,cap,bf; return y
 @staticmethod
 def backward(ctx,dy):
    x,rw,w1,w2,s,sel,order,off,pre,inv,gu,act,z=ctx.saved_tensors; T,D,E,F,A,cap,bf=ctx.meta; N=T*A; dy=dy.contiguous(); dz=torch.empty((N,D),device=x.device,dtype=torch.float32); dl=torch.zeros((T,E),device=x.device,dtype=torch.float32); _dz[(T,A)](dy,z,s,sel,inv,dz,dl,T=T,D=D,E=E,A=A,num_warps=4)
    da=torch.empty((N,F),device=x.device,dtype=torch.float32); _seg_t[(cap,triton.cdiv(F,COL))](dz,w2,off,pre,da,K=F,O=D,E=E,BF=bf,R=ROW,C=COL,KBLK=KB,num_warps=4,num_stages=3); dg=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _dact[(triton.cdiv(N*F,256),)](gu,da,dg,N=N,F=F)
    gw2=torch.empty_like(w2); _dw[(E,triton.cdiv(F,128),triton.cdiv(D,64))](act,dz,off,gw2,M=F,N=D,E=E,BF=bf,num_warps=4,num_stages=3); gw1=torch.empty_like(w1); _dw1[(E,triton.cdiv(D,128),triton.cdiv(2*F,64))](x,dg,order,off,gw1,A=A,D=D,O=2*F,E=E,BF=bf,num_warps=4,num_stages=3)
    dxr=torch.empty((N,D),device=x.device,dtype=torch.float32); _dxr[(cap,triton.cdiv(D,COL))](dg,w1,off,pre,dxr,D=D,O=2*F,E=E,BF=bf,R=ROW,C=COL,KBLK=KB,num_warps=4,num_stages=3); grw=torch.empty_like(rw); _dense[(triton.cdiv(D,64),triton.cdiv(E,32))](x,dl,grw,M=D,N=E,K=T,BF=bf,TA=True,TB=False,num_warps=4,num_stages=3); rdx=torch.empty((T,D),device=x.device,dtype=torch.float32); _dense[(triton.cdiv(T,64),triton.cdiv(D,32))](dl,rw,rdx,M=T,N=D,K=E,BF=bf,TA=False,TB=True,num_warps=4,num_stages=3); dx=torch.empty_like(x); _sumdx[(T,triton.cdiv(D,256))](dxr,inv,rdx,dx,T=T,D=D,A=A); return dx,grw,None,gw1,gw2,None

def fused_moe(x,router_w,router_b,w1,w2,A): return _Moe.apply(x,router_w,router_b,w1,w2,A)
