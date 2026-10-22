import torch
import triton
import triton.language as tl

_BT = 8
_BV = 64

@triton.jit
def _ex(x):
    return tl.exp(tl.maximum(tl.minimum(x, 60.0), -60.0))

@triton.jit
def _inv(l):
    r = tl.arange(0, 16)
    m = l
    for j in tl.static_range(1, 16):
        row = tl.sum(tl.where(r[:, None] == j, m, 0.0), axis=0)
        row += tl.sum(row[:, None] * m, axis=0) * (r < j)
        m = tl.where(r[:, None] == j, row, m)
    return m + (r[:, None] == r[None, :]).to(tl.float32)

@triton.jit
def _fwd(q,k,v,a,b,g,o,h,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    iv, z = tl.program_id(0), tl.program_id(1)
    ib, ih = z // H, z % H
    i=tl.arange(0,16); x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV)
    mx=x<K; my=y<V; S=tl.zeros((BK,BV),tl.float32)
    scale:tl.constexpr=K**-0.5
    for c in range(NC):
        t=c*8+i; mt=(i<8)&(t<T); mk=mt[:,None]&mx[None,:]; mv=mt[:,None]&my[None,:]
        ox=(ib*T*H+t[:,None]*H+ih)*K+x[None,:]
        oy=(ib*T*H+t[:,None]*H+ih)*V+y[None,:]
        tl.store(h+(((ib*NC+c)*H+ih)*K+x[:,None])*V+y[None,:],S,mask=mx[:,None]&my[None,:])
        G=tl.load(g+ox,mask=mk,other=0.).to(tl.float32); C=tl.cumsum(G,axis=0); P=C-G
        last=tl.minimum(7,tl.maximum(T-c*8-1,0)); mid=tl.minimum(4,last)
        end=tl.sum(tl.where(i[:,None]==last,C,0.),axis=0); off=tl.sum(tl.where(i[:,None]==mid,C,0.),axis=0)
        Q=tl.load(q+ox,mask=mk,other=0.).to(tl.float32)*scale*_ex(C-off[None,:])
        X=tl.load(k+ox,mask=mk,other=0.).to(tl.float32)*_ex(off[None,:]-C)
        A=tl.load(a+ox,mask=mk,other=0.).to(tl.float32)*_ex(P-off[None,:])
        B=tl.load(b+ox,mask=mk,other=0.).to(tl.float32)*_ex(off[None,:]-C)
        F=_ex(off); E=_ex(end-off); S=S*F[:,None]
        low=(i[:,None]>i[None,:])&(mt[:,None]&mt[None,:]); inc=(i[:,None]>=i[None,:])&(mt[:,None]&mt[None,:])
        L=tl.where(low,tl.dot(A,tl.trans(B),input_precision="ieee"),0.); AK=tl.where(low,tl.dot(A,tl.trans(X),input_precision="ieee"),0.)
        QK=tl.where(inc,tl.dot(Q,tl.trans(X),input_precision="ieee"),0.); QB=tl.where(inc,tl.dot(Q,tl.trans(B),input_precision="ieee"),0.)
        M=_inv(L); W=tl.load(v+oy,mask=mv,other=0.).to(tl.float32)
        R=tl.dot(M,tl.dot(A,S,input_precision="ieee")+tl.dot(AK,W,input_precision="ieee"),input_precision="ieee")
        O=tl.dot(Q,S,input_precision="ieee")+tl.dot(QK,W,input_precision="ieee")+tl.dot(QB,R,input_precision="ieee")
        tl.store(o+oy,O,mask=mv)
        S=(S+tl.dot(tl.trans(X),W,input_precision="ieee")+tl.dot(tl.trans(B),R,input_precision="ieee"))*E[:,None]

@triton.jit
def _bwd(q,k,v,a,b,g,do,h,pq,pk,pa,pb,pg,dv,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    iv,z=tl.program_id(0),tl.program_id(1); ib,ih=z//H,z%H
    i=tl.arange(0,16); x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); mx=x<K; my=y<V
    dS=tl.zeros((BK,BV),tl.float32); scale:tl.constexpr=K**-0.5
    for cc in range(NC):
        c=NC-1-cc; t=c*8+i; mt=(i<8)&(t<T); mk=mt[:,None]&mx[None,:]; mv=mt[:,None]&my[None,:]
        ox=(ib*T*H+t[:,None]*H+ih)*K+x[None,:]; oy=(ib*T*H+t[:,None]*H+ih)*V+y[None,:]
        S=tl.load(h+(((ib*NC+c)*H+ih)*K+x[:,None])*V+y[None,:],mask=mx[:,None]&my[None,:],other=0.).to(tl.float32)
        G=tl.load(g+ox,mask=mk,other=0.).to(tl.float32); C=tl.cumsum(G,axis=0); P=C-G
        last=tl.minimum(7,tl.maximum(T-c*8-1,0)); mid=tl.minimum(4,last)
        end=tl.sum(tl.where(i[:,None]==last,C,0.),axis=0); off=tl.sum(tl.where(i[:,None]==mid,C,0.),axis=0)
        fq=_ex(C-off[None,:]); fk=_ex(off[None,:]-C); fa=_ex(P-off[None,:]); F=_ex(off); E=_ex(end-off)
        Q=tl.load(q+ox,mask=mk,other=0.).to(tl.float32)*scale*fq; X=tl.load(k+ox,mask=mk,other=0.).to(tl.float32)*fk
        A=tl.load(a+ox,mask=mk,other=0.).to(tl.float32)*fa; B=tl.load(b+ox,mask=mk,other=0.).to(tl.float32)*fk
        S0=S; S=S0*F[:,None]
        low=(i[:,None]>i[None,:])&(mt[:,None]&mt[None,:]); inc=(i[:,None]>=i[None,:])&(mt[:,None]&mt[None,:])
        L=tl.where(low,tl.dot(A,tl.trans(B),input_precision="ieee"),0.); AK=tl.where(low,tl.dot(A,tl.trans(X),input_precision="ieee"),0.)
        QK=tl.where(inc,tl.dot(Q,tl.trans(X),input_precision="ieee"),0.); QB=tl.where(inc,tl.dot(Q,tl.trans(B),input_precision="ieee"),0.)
        M=_inv(L); W=tl.load(v+oy,mask=mv,other=0.).to(tl.float32)
        R=tl.dot(M,tl.dot(A,S,input_precision="ieee")+tl.dot(AK,W,input_precision="ieee"),input_precision="ieee")
        state_sum=S+tl.dot(tl.trans(X),W,input_precision="ieee")+tl.dot(tl.trans(B),R,input_precision="ieee")
        dY=dS*E[:,None]; dE=tl.sum(dS*state_sum,axis=1)
        dQ=tl.zeros((16,BK),tl.float32); dX=tl.dot(W,tl.trans(dY),input_precision="ieee"); dA=tl.zeros((16,BK),tl.float32); dB=tl.dot(R,tl.trans(dY),input_precision="ieee")
        dW=tl.dot(X,dY,input_precision="ieee"); dR=tl.dot(B,dY,input_precision="ieee"); dS0=dY
        dO=tl.load(do+oy,mask=mv,other=0.).to(tl.float32); dQ+=tl.dot(dO,tl.trans(S),input_precision="ieee"); dS0+=tl.dot(tl.trans(Q),dO,input_precision="ieee")
        dQK=tl.where(inc,tl.dot(dO,tl.trans(W),input_precision="ieee"),0.); dQB=tl.where(inc,tl.dot(dO,tl.trans(R),input_precision="ieee"),0.)
        dQ+=tl.dot(dQK,X,input_precision="ieee")+tl.dot(dQB,B,input_precision="ieee"); dX+=tl.dot(tl.trans(dQK),Q,input_precision="ieee"); dB+=tl.dot(tl.trans(dQB),Q,input_precision="ieee")
        dW+=tl.dot(tl.trans(QK),dO,input_precision="ieee"); dR+=tl.dot(tl.trans(QB),dO,input_precision="ieee")
        dU=tl.dot(tl.trans(M),dR,input_precision="ieee"); dL=tl.where(low,tl.dot(dU,tl.trans(R),input_precision="ieee"),0.); dAK=tl.where(low,tl.dot(dU,tl.trans(W),input_precision="ieee"),0.)
        dA+=tl.dot(dU,tl.trans(S),input_precision="ieee")+tl.dot(dAK,X,input_precision="ieee")+tl.dot(dL,B,input_precision="ieee")
        dS0+=tl.dot(tl.trans(A),dU,input_precision="ieee"); dW+=tl.dot(tl.trans(AK),dU,input_precision="ieee")
        dX+=tl.dot(tl.trans(dAK),A,input_precision="ieee"); dB+=tl.dot(tl.trans(dL),A,input_precision="ieee")
        dF=tl.sum(dS0*S0,axis=1); dS=dS0*F[:,None]
        dq0=dQ*fq*scale; dk0=dX*fk; da0=dA*fa; db0=dB*fk
        dc=(Q*dQ-X*dX-B*dB).to(tl.float64); dp=(A*dA).to(tl.float64)
        doff=tl.sum((-Q*dQ+X*dX+B*dB-A*dA).to(tl.float64),axis=0)+(F*dF-E*dE).to(tl.float64)
        dc+=tl.where(i[:,None]==last,(E*dE).to(tl.float64)[None,:],0.)
        dc+=tl.where(i[:,None]==mid,doff[None,:],0.)
        dg0=tl.sum(tl.where(i[:,None,None]<=i[None,:,None],dc[None,:,:],0.),axis=1)
        dg0+=tl.sum(tl.where(i[:,None,None]<i[None,:,None],dp[None,:,:],0.),axis=1)
        dg0=tl.where(t[:,None]==0,0.0,dg0)
        base=((iv*T+t[:,None])*H+ih)*K+x[None,:]
        tl.store(pq+base,dq0,mask=mk); tl.store(pk+base,dk0,mask=mk); tl.store(pa+base,da0,mask=mk); tl.store(pb+base,db0,mask=mk); tl.store(pg+base,dg0,mask=mk); tl.store(dv+oy,dW,mask=mv)

@triton.jit
def _reduce(pq,pk,pa,pb,pg,dq,dk,da,db,dg,N:tl.constexpr,NV:tl.constexpr,BS:tl.constexpr):
    x=tl.program_id(0)*BS+tl.arange(0,BS); m=x<N
    sq=tl.zeros((BS,),tl.float32); sk=tl.zeros((BS,),tl.float32); sa=tl.zeros((BS,),tl.float32); sb=tl.zeros((BS,),tl.float32); sg=tl.zeros((BS,),tl.float64)
    for j in range(NV):
        z=j*N+x; sq+=tl.load(pq+z,mask=m,other=0.); sk+=tl.load(pk+z,mask=m,other=0.); sa+=tl.load(pa+z,mask=m,other=0.); sb+=tl.load(pb+z,mask=m,other=0.); sg+=tl.load(pg+z,mask=m,other=0.)
    tl.store(dq+x,sq,mask=m); tl.store(dk+x,sk,mask=m); tl.store(da+x,sa,mask=m); tl.store(db+x,sb,mask=m); tl.store(dg+x,sg,mask=m)

class _Dplr(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,a,b,g):
    B,T,H,K=q.shape; V=v.shape[-1]; nc=triton.cdiv(T,_BT); nv=triton.cdiv(V,_BV); bk=max(16,triton.next_power_of_2(K))
    o=torch.empty_like(v); h=torch.empty((B,nc,H,K,V),device=q.device,dtype=torch.float32)
    _fwd[(nv,B*H)](q,k,v,a,b,g,o,h,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=_BV,num_warps=8,num_stages=2); ctx.save_for_backward(q,k,v,a,b,g,h); return o
 @staticmethod
 def backward(ctx,do):
    q,k,v,a,b,g,h=ctx.saved_tensors; do=do.contiguous(); B,T,H,K=q.shape; V=v.shape[-1]; nc=triton.cdiv(T,_BT); nv=triton.cdiv(V,_BV); bk=max(16,triton.next_power_of_2(K))
    shape=(nv,B,T,H,K); parts=[torch.zeros(shape,device=q.device,dtype=torch.float32) for _ in range(4)]; parts.append(torch.zeros(shape,device=q.device,dtype=torch.float64)); dv=torch.empty_like(v)
    _bwd[(nv,B*H)](q,k,v,a,b,g,do,h,*parts,dv,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=_BV,num_warps=8,num_stages=2)
    outs=[torch.empty_like(x) for x in (q,k,a,b,g)]; n=B*T*H*K
    _reduce[(triton.cdiv(n,256),)](*parts,*outs,N=n,NV=nv,BS=256,num_warps=4); return outs[0],outs[1],dv,outs[2],outs[3],outs[4]

def chunked_dplr_delta(q,k,v,alpha,beta,gk):
    return _Dplr.apply(q,k,v,alpha,beta,gk)
