import torch
import triton
import triton.language as tl

R, C, K = tl.constexpr(64), tl.constexpr(64), tl.constexpr(32)

@triton.jit
def _router(X, W, S, T: tl.constexpr, D: tl.constexpr, E: tl.constexpr, B: tl.constexpr=32):
    t = tl.program_id(0)*B + tl.arange(0,B); e = tl.program_id(1)*B + tl.arange(0,B)
    z = tl.zeros((B,B), tl.float32)
    for k0 in range(0,D,B):
        k = k0 + tl.arange(0,B)
        x = tl.load(X+t[:,None]*D+k[None,:], mask=(t[:,None]<T)&(k[None,:]<D), other=0.).to(tl.float32)
        w = tl.load(W+k[:,None]*E+e[None,:], mask=(k[:,None]<D)&(e[None,:]<E), other=0.).to(tl.float32)
        z += tl.dot(x,w,input_precision="ieee")
    tl.store(S+t[:,None]*E+e[None,:],tl.sigmoid(z),mask=(t[:,None]<T)&(e[None,:]<E))

@triton.jit
def _keys(S,RB,O,N:tl.constexpr,E:tl.constexpr,B:tl.constexpr=256):
    p=tl.program_id(0)*B+tl.arange(0,B); m=p<N; e=p%E
    v=tl.load(S+p,mask=m,other=0.)+tl.load(RB+e,mask=m,other=0.); v=tl.where(v==0.,0.,v)
    i=v.to(tl.int32,bitcast=True).to(tl.int64); u=tl.where(i>=0,i^0x80000000,(~i)&0xffffffff)
    tl.store(O+p,(u<<10)|(E-1-e),mask=m)

@triton.jit
def _weights(S,SEL,W,T:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BA:tl.constexpr):
    t=tl.program_id(0); a=tl.arange(0,BA); m=a<A; e=tl.load(SEL+t*A+a,mask=m,other=0)
    v=tl.load(S+t*E+e,mask=m,other=0.); d=tl.sum(tl.where(m,v,0.),axis=0)
    tl.store(W+t*A+a,v/d,mask=m)

@triton.jit
def _offsets(SEL,ORD,OFF,N:tl.constexpr,E:tl.constexpr):
    target=tl.program_id(0); lo=0; hi=N
    for _ in range(24):
        mid=(lo+hi)//2; q=tl.minimum(mid,N-1); p=tl.load(ORD+q,mask=mid<N,other=0)
        e=tl.load(SEL+p,mask=mid<N,other=E); take=(mid<N)&(e<target)
        lo=tl.where(take,mid+1,lo); hi=tl.where(take,hi,mid)
    tl.store(OFF+target,lo)

@triton.jit
def _prefix(OFF,PRE,E:tl.constexpr):
    s=0; tl.store(PRE,0)
    for e in range(E):
        s+=tl.cdiv(tl.load(OFF+e+1)-tl.load(OFF+e),R); tl.store(PRE+e+1,s)

@triton.jit
def _inverse(ORD,INV,N:tl.constexpr,B:tl.constexpr=256):
    q=tl.program_id(0)*B+tl.arange(0,B); m=q<N; p=tl.load(ORD+q,mask=m,other=0); tl.store(INV+p,q,mask=m)

@triton.jit
def _loc(pid,OFF,PRE,E:tl.constexpr):
    lo=0; hi=E
    for _ in range(8):
        mid=(lo+hi)//2; v=tl.load(PRE+tl.minimum(mid+1,E)); take=(mid<E)&(v<=pid)
        lo=tl.where(take,mid+1,lo); hi=tl.where(take,hi,mid)
    valid=lo<E; e=tl.minimum(lo,E-1); b=tl.load(OFF+e)+(pid-tl.load(PRE+e))*R
    return e,b,tl.load(OFF+e+1),valid

@triton.jit
def _ein(X,W,ORD,OFF,PRE,O,A:tl.constexpr,D:tl.constexpr,NOUT:tl.constexpr,E:tl.constexpr,BF:tl.constexpr):
    e,b,end,v=_loc(tl.program_id(0),OFF,PRE,E); r=b+tl.arange(0,R); n=tl.program_id(1)*C+tl.arange(0,C); mr=v&(r<end)
    p=tl.load(ORD+r,mask=mr,other=0); t=p//A; z=tl.zeros((R,C),tl.float32)
    for k0 in range(0,D,K):
        k=k0+tl.arange(0,K); x=tl.load(X+t[:,None]*D+k[None,:],mask=mr[:,None]&(k[None,:]<D),other=0.)
        w=tl.load(W+e*D*NOUT+k[:,None]*NOUT+n[None,:],mask=(k[:,None]<D)&(n[None,:]<NOUT),other=0.)
        if BF: z+=tl.dot(x,w)
        else: z+=tl.dot(x.to(tl.float32),w.to(tl.float32),input_precision="tf32x3")
    tl.store(O+r[:,None]*NOUT+n[None,:],z,mask=mr[:,None]&(n[None,:]<NOUT))

@triton.jit
def _act(G,A,N:tl.constexpr,F:tl.constexpr,B:tl.constexpr=256):
    p=tl.program_id(0)*B+tl.arange(0,B); m=p<N; r=p//F; f=p%F
    g=tl.load(G+r*2*F+f,mask=m,other=0.); u=tl.load(G+r*2*F+F+f,mask=m,other=0.)
    tl.store(A+p,g*tl.sigmoid(g)*u,mask=m)

@triton.jit
def _seg(A,W,OFF,PRE,O,KS:tl.constexpr,NS:tl.constexpr,E:tl.constexpr,BF:tl.constexpr,SPLIT:tl.constexpr):
    e,b,end,v=_loc(tl.program_id(0),OFF,PRE,E); r=b+tl.arange(0,R); n=tl.program_id(1)*C+tl.arange(0,C); mr=v&(r<end)
    z=tl.zeros((R,C),tl.float32)
    for k0 in range(0,KS,K):
        k=k0+tl.arange(0,K); a=tl.load(A+r[:,None]*KS+k[None,:],mask=mr[:,None]&(k[None,:]<KS),other=0.).to(tl.float32)
        w=tl.load(W+e*KS*NS+k[:,None]*NS+n[None,:],mask=(k[:,None]<KS)&(n[None,:]<NS),other=0.)
        if BF:
            ah=a.to(tl.bfloat16); z+=tl.dot(ah,w)
            if SPLIT: z+=tl.dot((a-ah.to(tl.float32)).to(tl.bfloat16),w)
        else: z+=tl.dot(a,w.to(tl.float32),input_precision="tf32x3")
    tl.store(O+r[:,None]*NS+n[None,:],z,mask=mr[:,None]&(n[None,:]<NS))

@triton.jit
def _combine(FE,W,INV,Y,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,B:tl.constexpr=256):
    t=tl.program_id(0); d=tl.program_id(1)*B+tl.arange(0,B); z=tl.zeros((B,),tl.float32)
    for a in range(A):
        p=t*A+a; q=tl.load(INV+p); z+=tl.load(W+p)*tl.load(FE+q*D+d,mask=d<D,other=0.)
    tl.store(Y+t*D+d,z,mask=d<D)

@triton.jit
def _dz(DY,FE,W,ORD,DZ,DL,N:tl.constexpr,A:tl.constexpr,D:tl.constexpr,BD:tl.constexpr):
    q=tl.program_id(0); d=tl.arange(0,BD); p=tl.load(ORD+q); t=p//A
    dy=tl.load(DY+t*D+d,mask=d<D,other=0.).to(tl.float32); fe=tl.load(FE+q*D+d,mask=d<D,other=0.)
    tl.store(DZ+q*D+d,dy*tl.load(W+p),mask=d<D); tl.store(DL+p,tl.sum(dy*fe,axis=0))

@triton.jit
def _segbt(A,W,OFF,PRE,O,KS:tl.constexpr,NS:tl.constexpr,E:tl.constexpr,BF:tl.constexpr):
    e,b,end,v=_loc(tl.program_id(0),OFF,PRE,E); r=b+tl.arange(0,R); n=tl.program_id(1)*C+tl.arange(0,C); mr=v&(r<end)
    z=tl.zeros((R,C),tl.float32)
    for k0 in range(0,KS,K):
        k=k0+tl.arange(0,K); a=tl.load(A+r[:,None]*KS+k[None,:],mask=mr[:,None]&(k[None,:]<KS),other=0.).to(tl.float32)
        w=tl.load(W+e*NS*KS+n[None,:]*KS+k[:,None],mask=(n[None,:]<NS)&(k[:,None]<KS),other=0.); wt=w
        if BF:
            ah=a.to(tl.bfloat16); z+=tl.dot(ah,wt)+tl.dot((a-ah.to(tl.float32)).to(tl.bfloat16),wt)
        else: z+=tl.dot(a,wt.to(tl.float32),input_precision="tf32x3")
    tl.store(O+r[:,None]*NS+n[None,:],z,mask=mr[:,None]&(n[None,:]<NS))

@triton.jit
def _dact(G,DA,DG,N:tl.constexpr,F:tl.constexpr,B:tl.constexpr=256):
    p=tl.program_id(0)*B+tl.arange(0,B); m=p<N; r=p//F; f=p%F
    g=tl.load(G+r*2*F+f,mask=m,other=0.); u=tl.load(G+r*2*F+F+f,mask=m,other=0.); da=tl.load(DA+p,mask=m,other=0.)
    s=tl.sigmoid(g); si=g*s; tl.store(DG+r*2*F+f,da*u*(s+si*(1-s)),mask=m); tl.store(DG+r*2*F+F+f,da*si,mask=m)

@triton.jit
def _dw(AIN,G,ORD,OFF,O,KS:tl.constexpr,NS:tl.constexpr,ASEL:tl.constexpr,E:tl.constexpr,GATHER:tl.constexpr,BF:tl.constexpr,BM:tl.constexpr=128,BN:tl.constexpr=64,BK:tl.constexpr=32):
    e=tl.program_id(0); i=tl.program_id(1)*BM+tl.arange(0,BM); n=tl.program_id(2)*BN+tl.arange(0,BN)
    beg=tl.load(OFF+e); end=tl.load(OFF+e+1); z=tl.zeros((BM,BN),tl.float32)
    for q0 in range(beg,end,BK):
        q=q0+tl.arange(0,BK); mq=q<end
        if GATHER: ar=tl.load(ORD+q,mask=mq,other=0)//ASEL
        else: ar=q
        a=tl.load(AIN+ar[None,:]*KS+i[:,None],mask=mq[None,:]&(i[:,None]<KS),other=0.).to(tl.float32)
        g=tl.load(G+q[:,None]*NS+n[None,:],mask=mq[:,None]&(n[None,:]<NS),other=0.).to(tl.float32)
        if BF:
            ah=a.to(tl.bfloat16); gh=g.to(tl.bfloat16); gl=(g-gh.to(tl.float32)).to(tl.bfloat16)
            z+=tl.dot(ah,gh)+tl.dot(ah,gl)
            if not GATHER:
                z+=tl.dot((a-ah.to(tl.float32)).to(tl.bfloat16),gh)
        else: z+=tl.dot(a,g,input_precision="tf32x3")
    tl.store(O+e*KS*NS+i[:,None]*NS+n[None,:],z,mask=(i[:,None]<KS)&(n[None,:]<NS))

@triton.jit
def _route_ds(S,SEL,W,DL,DS,T:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BE:tl.constexpr,BA:tl.constexpr):
    t=tl.program_id(0); eb=tl.arange(0,BE); aa=tl.arange(0,BA); ma=aa<A
    ww=tl.load(W+t*A+aa,mask=ma,other=0.); dl=tl.load(DL+t*A+aa,mask=ma,other=0.); mean=tl.sum(ww*dl,axis=0)
    se=tl.load(SEL+t*A+aa,mask=ma,other=0); raw=tl.load(S+t*E+se,mask=ma,other=0.); den=tl.sum(tl.where(ma,raw,0.),axis=0); out=tl.zeros((BE,),tl.float32)
    for a in range(A):
        e=tl.load(SEL+t*A+a); r=tl.load(S+t*E+e); v=(tl.load(DL+t*A+a)-mean)/den*r*(1-r); out=tl.where(eb==e,v,out)
    tl.store(DS+t*E+eb,out,mask=eb<E)

@triton.jit
def _dense(A,B,O,M:tl.constexpr,KS:tl.constexpr,NS:tl.constexpr,TA:tl.constexpr,TB:tl.constexpr,BF:tl.constexpr,BM:tl.constexpr=64,BN:tl.constexpr=64,BK:tl.constexpr=32):
    i=tl.program_id(0)*BM+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); z=tl.zeros((BM,BN),tl.float32)
    for k0 in range(0,KS,BK):
        k=k0+tl.arange(0,BK)
        if TA: a=tl.load(A+k[None,:]*M+i[:,None],mask=(i[:,None]<M)&(k[None,:]<KS),other=0.).to(tl.float32)
        else: a=tl.load(A+i[:,None]*KS+k[None,:],mask=(i[:,None]<M)&(k[None,:]<KS),other=0.).to(tl.float32)
        if TB: b=tl.load(B+n[None,:]*KS+k[:,None],mask=(n[None,:]<NS)&(k[:,None]<KS),other=0.)
        else: b=tl.load(B+k[:,None]*NS+n[None,:],mask=(k[:,None]<KS)&(n[None,:]<NS),other=0.)
        if BF:
            bf=b.to(tl.float32); ah=a.to(tl.bfloat16); bh=bf.to(tl.bfloat16)
            if TA:
                z+=tl.dot(ah,bh)+tl.dot(ah,(bf-bh.to(tl.float32)).to(tl.bfloat16))
            else:
                z+=tl.dot(ah,bh)+tl.dot((a-ah.to(tl.float32)).to(tl.bfloat16),bh)
        else: z+=tl.dot(a,b.to(tl.float32),input_precision="tf32x3")
    tl.store(O+i[:,None]*NS+n[None,:],z,mask=(i[:,None]<M)&(n[None,:]<NS))

@triton.jit
def _dxsum(P,INV,RX,O,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,B:tl.constexpr=256):
    t=tl.program_id(0); d=tl.program_id(1)*B+tl.arange(0,B); z=tl.load(RX+t*D+d,mask=d<D,other=0.)
    for a in range(A): z+=tl.load(P+tl.load(INV+t*A+a)*D+d,mask=d<D,other=0.)
    tl.store(O+t*D+d,z,mask=d<D)

class _Moe(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,rw,rb,w1,w2,A):
    T,D=x.shape; E=rw.shape[1]; F=w2.shape[1]; A=int(A); N=T*A; bf=x.dtype==torch.bfloat16
    s=torch.empty((T,E),device=x.device,dtype=torch.float32); _router[(triton.cdiv(T,32),triton.cdiv(E,32))](x,rw,s,T=T,D=D,E=E)
    keys=torch.empty((T,E),device=x.device,dtype=torch.int64); _keys[(triton.cdiv(T*E,256),)](s,rb,keys,N=T*E,E=E)
    sel=torch.topk(keys,A,dim=1,largest=True,sorted=True).indices.contiguous(); weights=torch.empty((T,A),device=x.device,dtype=torch.float32)
    _weights[(T,)](s,sel,weights,T=T,E=E,A=A,BA=triton.next_power_of_2(A)); flat=sel.reshape(-1); order=torch.argsort(flat,stable=True)
    off=torch.empty(E+1,device=x.device,dtype=torch.int32); _offsets[(E+1,)](flat,order,off,N=N,E=E)
    pre=torch.empty(E+1,device=x.device,dtype=torch.int32); _prefix[(1,)](off,pre,E=E,num_warps=1)
    inv=torch.empty(N,device=x.device,dtype=torch.int64); _inverse[(triton.cdiv(N,256),)](order,inv,N=N); cap=triton.cdiv(N,R)+E
    gu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _ein[(cap,triton.cdiv(2*F,C))](x,w1,order,off,pre,gu,A=A,D=D,NOUT=2*F,E=E,BF=bf,num_warps=4,num_stages=3)
    act=torch.empty((N,F),device=x.device,dtype=torch.float32); _act[(triton.cdiv(N*F,256),)](gu,act,N=N*F,F=F)
    fe=torch.empty((N,D),device=x.device,dtype=torch.float32); _seg[(cap,triton.cdiv(D,C))](act,w2,off,pre,fe,KS=F,NS=D,E=E,BF=bf,SPLIT=bf,num_warps=4,num_stages=3)
    y=torch.empty((T,D),device=x.device,dtype=x.dtype); _combine[(T,triton.cdiv(D,256))](fe,weights,inv,y,T=T,D=D,A=A)
    ctx.save_for_backward(x,rw,w1,w2,s,sel,weights,order,off,pre,inv,gu,act,fe); ctx.m=(T,D,E,F,A,N,cap,bf); return y
 @staticmethod
 def backward(ctx,dy):
    x,rw,w1,w2,s,sel,weights,order,off,pre,inv,gu,act,fe=ctx.saved_tensors; T,D,E,F,A,N,cap,bf=ctx.m; dy=dy.contiguous()
    dz=torch.empty((N,D),device=x.device,dtype=torch.float32); dl=torch.empty(N,device=x.device,dtype=torch.float32); _dz[(N,)](dy,fe,weights,order,dz,dl,N=N,A=A,D=D,BD=triton.next_power_of_2(D),num_warps=8)
    da=torch.empty((N,F),device=x.device,dtype=torch.float32); _segbt[(cap,triton.cdiv(F,C))](dz,w2,off,pre,da,KS=D,NS=F,E=E,BF=bf,num_warps=4,num_stages=3)
    dgu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _dact[(triton.cdiv(N*F,256),)](gu,da,dgu,N=N*F,F=F)
    dxp=torch.empty((N,D),device=x.device,dtype=torch.float32); _segbt[(cap,triton.cdiv(D,C))](dgu,w1,off,pre,dxp,KS=2*F,NS=D,E=E,BF=bf,num_warps=4,num_stages=3)
    dw2=torch.empty_like(w2); _dw[(E,triton.cdiv(F,128),triton.cdiv(D,64))](act,dz,order,off,dw2,KS=F,NS=D,ASEL=A,E=E,GATHER=False,BF=bf)
    dw1=torch.empty_like(w1); _dw[(E,triton.cdiv(D,128),triton.cdiv(2*F,64))](x,dgu,order,off,dw1,KS=D,NS=2*F,ASEL=A,E=E,GATHER=True,BF=bf)
    ds=torch.empty((T,E),device=x.device,dtype=torch.float32); _route_ds[(T,)](s,sel,weights,dl,ds,T=T,E=E,A=A,BE=triton.next_power_of_2(E),BA=triton.next_power_of_2(A))
    drw=torch.empty_like(rw); _dense[(triton.cdiv(D,64),triton.cdiv(E,64))](x,ds,drw,M=D,KS=T,NS=E,TA=True,TB=False,BF=bf)
    rx=torch.empty((T,D),device=x.device,dtype=torch.float32); _dense[(triton.cdiv(T,64),triton.cdiv(D,64))](ds,rw,rx,M=T,KS=E,NS=D,TA=False,TB=True,BF=bf)
    dx=torch.empty_like(x); _dxsum[(T,triton.cdiv(D,256))](dxp,inv,rx,dx,T=T,D=D,A=A); return dx,drw,None,dw1,dw2,None

def fused_moe(x,router_w,router_b,w1,w2,A): return _Moe.apply(x,router_w,router_b,w1,w2,A)
