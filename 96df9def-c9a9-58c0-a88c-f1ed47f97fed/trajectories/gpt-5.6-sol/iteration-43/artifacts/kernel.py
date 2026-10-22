import torch
import triton
import triton.language as tl

_BT, _FBV, _BBV = 16, 32, 16

@triton.jit
def _inv_lower(x, eye: tl.constexpr):
    rows = tl.arange(0, 16)
    result = eye
    for i in tl.static_range(1, 16):
        source = tl.sum(tl.where((rows == i)[:, None], x, 0.0), axis=0)
        solved = tl.sum(source[:, None] * result, axis=0)
        result = tl.where((rows == i)[:, None], eye + solved[None, :], result)
    return result

@triton.jit
def _prep(q,k,a,b,g,aqk,aqb,aak,inv,T,H:tl.constexpr,K:tl.constexpr,BK:tl.constexpr,S:tl.constexpr):
    c=tl.program_id(0).to(tl.int64); bh=tl.program_id(1); ib=bh//H; ih=bh%H
    r=tl.arange(0,16); x=tl.arange(0,BK); t=c*16+r; mt=t<T; mx=x<K
    o=((ib*T+t[:,None])*H+ih)*K+x[None,:]; m=mt[:,None]&mx[None,:]
    Q=tl.load(q+o,mask=m,other=0.).to(tl.float32)*S; K0=tl.load(k+o,mask=m,other=0.).to(tl.float32)
    A=tl.load(a+o,mask=m,other=0.).to(tl.float32); B=tl.load(b+o,mask=m,other=0.).to(tl.float32)
    G=tl.load(g+o,mask=m,other=0.).to(tl.float32); Z=tl.cumsum(G,axis=0); E=Z-G
    QK=tl.zeros([16,16],tl.float32); QB=tl.zeros([16,16],tl.float32)
    AK=tl.zeros([16,16],tl.float32); AB=tl.zeros([16,16],tl.float32)
    for j in tl.static_range(0,16):
        pick=r==j; kj=tl.sum(tl.where(pick[:,None],K0,0.),axis=0); bj=tl.sum(tl.where(pick[:,None],B,0.),axis=0)
        zj=tl.sum(tl.where(pick[:,None],Z,0.),axis=0); eq=tl.exp(Z-zj[None,:]); ea=tl.exp(E-zj[None,:])
        inc=(r>=j)&mt&(c*16+j<T); strict=(r>j)&mt&(c*16+j<T); col=r[None,:]==j
        QK+=tl.where(inc[:,None]&col,tl.sum(Q*kj[None,:]*eq,axis=1)[:,None],0.)
        QB+=tl.where(inc[:,None]&col,tl.sum(Q*bj[None,:]*eq,axis=1)[:,None],0.)
        AK+=tl.where(strict[:,None]&col,tl.sum(A*kj[None,:]*ea,axis=1)[:,None],0.)
        AB+=tl.where(strict[:,None]&col,tl.sum(A*bj[None,:]*ea,axis=1)[:,None],0.)
    I=tl.where(r[:,None]==r[None,:],1.,0.); R=_inv_lower(AB,I)
    oo=((ib*T+t[:,None])*H+ih)*16+r[None,:]
    tl.store(aqk+oo,QK,mask=mt[:,None]); tl.store(aqb+oo,QB,mask=mt[:,None]); tl.store(aak+oo,AK,mask=mt[:,None]); tl.store(inv+oo,R,mask=mt[:,None])

@triton.jit
def _fwd(q,k,v,a,b,g,aqk,aqb,aak,inv,out,hbuf,T,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr,NC:tl.constexpr,S:tl.constexpr):
    iv=tl.program_id(0); bh=tl.program_id(1); ib=bh//H; ih=bh%H
    x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); mx=x<K; my=y<V; st=tl.zeros([BK,BV],tl.float32); r=tl.arange(0,16)
    for c in range(NC):
        t=c*16+r; mt=t<T; ho=((((ib*NC+c)*H+ih)*K+x[:,None])*V+y[None,:]); tl.store(hbuf+ho,st,mask=mx[:,None]&my[None,:])
        ko=((ib*T+t[:,None])*H+ih)*K+x[None,:]; vo=((ib*T+t[:,None])*H+ih)*V+y[None,:]; m=mt[:,None]
        Q=tl.load(q+ko,mask=m&mx[None,:],other=0.).to(tl.float32)*S; K0=tl.load(k+ko,mask=m&mx[None,:],other=0.).to(tl.float32)
        A=tl.load(a+ko,mask=m&mx[None,:],other=0.).to(tl.float32); B=tl.load(b+ko,mask=m&mx[None,:],other=0.).to(tl.float32)
        G=tl.load(g+ko,mask=m&mx[None,:],other=0.).to(tl.float32); V0=tl.load(v+vo,mask=m&my[None,:],other=0.).to(tl.float32)
        Z=tl.cumsum(G,axis=0); last=tl.maximum(tl.minimum(T-c*16,16),1)-1; zl=tl.sum(tl.where((r==last)[:,None],Z,0.),axis=0)
        Q*=tl.exp(Z); A*=tl.exp(Z-G); K0*=tl.exp(zl[None,:]-Z); B*=tl.exp(zl[None,:]-Z)
        ao=((ib*T+t[:,None])*H+ih)*16+r[None,:]
        QK=tl.load(aqk+ao,mask=m,other=0.); QB=tl.load(aqb+ao,mask=m,other=0.); AK=tl.load(aak+ao,mask=m,other=0.); R=tl.load(inv+ao,mask=m,other=0.)
        U=tl.dot(R,tl.dot(AK,V0,input_precision="ieee"),input_precision="ieee"); W=tl.dot(R,A,input_precision="ieee"); VN=U+tl.dot(W,st,input_precision="ieee")
        O=tl.dot(QK,V0,input_precision="ieee")+tl.dot(QB,VN,input_precision="ieee")+tl.dot(Q,st,input_precision="ieee")
        tl.store(out+vo,O,mask=m&my[None,:]); st*=tl.exp(zl)[:,None]; st+=tl.dot(tl.trans(K0),V0,input_precision="ieee")+tl.dot(tl.trans(B),VN,input_precision="ieee")

@triton.jit
def _expand(k,v,a,b,g,hbuf,states,T,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr,NC:tl.constexpr):
    iv=tl.program_id(0); c=tl.program_id(1); bh=tl.program_id(2); ib=bh//H; ih=bh%H
    x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); mx=x<K; my=y<V
    ho=((((ib*NC+c)*H+ih)*K+x[:,None])*V+y[None,:]); st=tl.load(hbuf+ho,mask=mx[:,None]&my[None,:],other=0.)
    for j in tl.static_range(0,16):
        t=c*16+j; valid=t<T; so=((((ib*T+t)*H+ih)*K+x[:,None])*V+y[None,:]); tl.store(states+so,st,mask=valid&mx[:,None]&my[None,:])
        ko=((ib*T+t)*H+ih)*K+x; vo=((ib*T+t)*H+ih)*V+y
        K0=tl.load(k+ko,mask=valid&mx,other=0.).to(tl.float32); A=tl.load(a+ko,mask=valid&mx,other=0.).to(tl.float32)
        B=tl.load(b+ko,mask=valid&mx,other=0.).to(tl.float32); G=tl.load(g+ko,mask=valid&mx,other=0.).to(tl.float32); V0=tl.load(v+vo,mask=valid&my,other=0.).to(tl.float32)
        rd=tl.sum(st*A[:,None],axis=0); st=st*tl.exp(G)[:,None]+K0[:,None]*V0[None,:]+B[:,None]*rd[None,:]

@triton.jit
def _bounds(q,a,b,g,do,dend,T,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr,NC:tl.constexpr,S:tl.constexpr):
    iv=tl.program_id(0); bh=tl.program_id(1); ib=bh//H; ih=bh%H
    x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); mx=x<K; my=y<V; ds=tl.zeros([BK,BV],tl.float32)
    for ci in range(NC):
        c=NC-1-ci; eo=((((ib*NC+c)*H+ih)*K+x[:,None])*V+y[None,:]); tl.store(dend+eo,ds,mask=mx[:,None]&my[None,:])
        for jj in tl.static_range(0,16):
            t=c*16+15-jj; valid=t<T; ko=((ib*T+t)*H+ih)*K+x; vo=((ib*T+t)*H+ih)*V+y
            Q=tl.load(q+ko,mask=valid&mx,other=0.).to(tl.float32)*S; A=tl.load(a+ko,mask=valid&mx,other=0.).to(tl.float32)
            B=tl.load(b+ko,mask=valid&mx,other=0.).to(tl.float32); G=tl.load(g+ko,mask=valid&mx,other=0.).to(tl.float32); dO=tl.load(do+vo,mask=valid&my,other=0.).to(tl.float32)
            ds+=Q[:,None]*dO[None,:]; dr=tl.sum(ds*B[:,None],axis=0); ds=ds*tl.exp(G)[:,None]+A[:,None]*dr[None,:]

@triton.jit
def _bwd(q,k,v,a,b,g,do,states,dend,dv,part,T,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr,NC:tl.constexpr,NV:tl.constexpr,BATCH:tl.constexpr,S:tl.constexpr):
    iv=tl.program_id(0); c=tl.program_id(1); bh=tl.program_id(2); ib=bh//H; ih=bh%H
    x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); mx=x<K; my=y<V; eo=((((ib*NC+c)*H+ih)*K+x[:,None])*V+y[None,:]); ds=tl.load(dend+eo,mask=mx[:,None]&my[None,:],other=0.)
    for jj in tl.static_range(0,16):
        t=c*16+15-jj; valid=t<T; ko=((ib*T+t)*H+ih)*K+x; vo=((ib*T+t)*H+ih)*V+y; so=((((ib*T+t)*H+ih)*K+x[:,None])*V+y[None,:])
        P=tl.load(states+so,mask=valid&mx[:,None]&my[None,:],other=0.); Q=tl.load(q+ko,mask=valid&mx,other=0.).to(tl.float32)*S
        K0=tl.load(k+ko,mask=valid&mx,other=0.).to(tl.float32); A=tl.load(a+ko,mask=valid&mx,other=0.).to(tl.float32); B=tl.load(b+ko,mask=valid&mx,other=0.).to(tl.float32)
        G=tl.load(g+ko,mask=valid&mx,other=0.).to(tl.float32); V0=tl.load(v+vo,mask=valid&my,other=0.).to(tl.float32); dO=tl.load(do+vo,mask=valid&my,other=0.).to(tl.float32); D=tl.exp(G); rd=tl.sum(P*A[:,None],axis=0)
        C=P*D[:,None]+K0[:,None]*V0[None,:]+B[:,None]*rd[None,:]; dq=tl.sum(C*dO[None,:],axis=1)*S; ds+=Q[:,None]*dO[None,:]
        dv0=tl.sum(ds*K0[:,None],axis=0); dk=tl.sum(ds*V0[None,:],axis=1); db=tl.sum(ds*rd[None,:],axis=1); dr=tl.sum(ds*B[:,None],axis=0); da=tl.sum(P*dr[None,:],axis=1); dg=tl.sum(ds*P*D[:,None],axis=1)
        tl.store(dv+vo,dv0,mask=valid&my); po=((iv*BATCH+ib)*T*H+t*H+ih)*K+x; stride=NV*BATCH*T*H*K; pm=valid&mx
        tl.store(part+po,dq,mask=pm); tl.store(part+stride+po,dk,mask=pm); tl.store(part+2*stride+po,da,mask=pm); tl.store(part+3*stride+po,db,mask=pm); tl.store(part+4*stride+po,dg,mask=pm)
        ds=ds*D[:,None]+A[:,None]*dr[None,:]

@triton.jit
def _reduce(p,dq,dk,da,db,dg,N:tl.constexpr,NV:tl.constexpr,BS:tl.constexpr):
    x=tl.program_id(0)*BS+tl.arange(0,BS); m=x<N; stride=N*NV
    sq=tl.zeros([BS],tl.float32); sk=tl.zeros([BS],tl.float32); sa=tl.zeros([BS],tl.float32); sb=tl.zeros([BS],tl.float32); sg=tl.zeros([BS],tl.float32)
    for i in range(NV):
        z=i*N+x; sq+=tl.load(p+z,mask=m,other=0.); sk+=tl.load(p+stride+z,mask=m,other=0.); sa+=tl.load(p+2*stride+z,mask=m,other=0.); sb+=tl.load(p+3*stride+z,mask=m,other=0.); sg+=tl.load(p+4*stride+z,mask=m,other=0.)
    tl.store(dq+x,sq,mask=m); tl.store(dk+x,sk,mask=m); tl.store(da+x,sa,mask=m); tl.store(db+x,sb,mask=m); tl.store(dg+x,sg,mask=m)

class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,a,b,g):
        B,T,H,K=q.shape; V=v.shape[-1]; nc=triton.cdiv(T,16); bk=max(16,triton.next_power_of_2(K)); nv=triton.cdiv(V,_FBV)
        mats=[torch.empty(B,T,H,16,device=q.device,dtype=torch.float32) for _ in range(4)]
        _prep[(nc,B*H)](q,k,a,b,g,*mats,T,H=H,K=K,BK=bk,S=K**-.5,num_warps=4,num_stages=2)
        out=torch.empty_like(v); h=torch.empty(B,nc,H,K,V,device=q.device,dtype=torch.float32)
        _fwd[(nv,B*H)](q,k,v,a,b,g,*mats,out,h,T,H=H,K=K,V=V,BK=bk,BV=_FBV,NC=nc,S=K**-.5,num_warps=8,num_stages=2)
        ctx.save_for_backward(q,k,v,a,b,g,h); return out
    @staticmethod
    def backward(ctx,do):
        q,k,v,a,b,g,h=ctx.saved_tensors; do=do.contiguous(); B,T,H,K=q.shape; V=v.shape[-1]; nc=triton.cdiv(T,16); bk=max(16,triton.next_power_of_2(K)); bv=_BBV; nv=triton.cdiv(V,bv)
        states=torch.empty(B,T,H,K,V,device=q.device,dtype=torch.float32); dend=torch.empty(B,nc,H,K,V,device=q.device,dtype=torch.float32)
        _expand[(nv,nc,B*H)](k,v,a,b,g,h,states,T,H=H,K=K,V=V,BK=bk,BV=bv,NC=nc,num_warps=4,num_stages=2)
        _bounds[(nv,B*H)](q,a,b,g,do,dend,T,H=H,K=K,V=V,BK=bk,BV=bv,NC=nc,S=K**-.5,num_warps=4,num_stages=2)
        dv=torch.empty_like(v); part=torch.zeros(5,nv,B,T,H,K,device=q.device,dtype=torch.float32)
        _bwd[(nv,nc,B*H)](q,k,v,a,b,g,do,states,dend,dv,part,T,H=H,K=K,V=V,BK=bk,BV=bv,NC=nc,NV=nv,S=K**-.5,BATCH=B,num_warps=4,num_stages=2)
        dq,dk,da,db=torch.empty_like(q),torch.empty_like(k),torch.empty_like(a),torch.empty_like(b); dg=torch.empty_like(g); n=B*T*H*K
        _reduce[(triton.cdiv(n,256),)](part,dq,dk,da,db,dg,N=n,NV=nv,BS=256,num_warps=4); return dq,dk,dv,da,db,dg

def chunked_dplr_delta(q,k,v,alpha,beta,gk):
    return _Dplr.apply(q,k,v,alpha,beta,gk)
