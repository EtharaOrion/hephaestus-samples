import torch
import triton
import triton.language as tl

BT, FBV, BBV = 4, 32, 16

@triton.jit
def _inv(l):
    x = tl.arange(0, 16)
    r = tl.where(x[:, None] == x[None, :], 1.0, 0.0)
    p = l
    for _ in tl.static_range(0, 4):
        r += tl.dot(p, r, input_precision="ieee")
        p = tl.dot(p, p, input_precision="ieee")
    return r

@triton.jit
def _fwd(qp,kp,vp,ap,bp,gp,op,hp,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr,S:tl.constexpr,CT:tl.constexpr):
    iv,bh=tl.program_id(0),tl.program_id(1); ib,ih=bh//H,bh%H
    rk=tl.arange(0,BK); rv=iv*BV+tl.arange(0,BV); rt=tl.arange(0,16)
    st=tl.zeros((BK,BV),tl.float32)
    for ic in range(NC):
        t=ic*CT+rt; mt=(rt<CT)&(t<T)
        ok=((ib*T+t[:,None])*H+ih)*K+rk[None,:]; ov=((ib*T+t[:,None])*H+ih)*V+rv[None,:]
        mk=mt[:,None]&(rk[None,:]<K); mv=mt[:,None]&(rv[None,:]<V)
        q=tl.load(qp+ok,mask=mk,other=0.).to(tl.float32); k=tl.load(kp+ok,mask=mk,other=0.).to(tl.float32)
        a=tl.load(ap+ok,mask=mk,other=0.).to(tl.float32); b=tl.load(bp+ok,mask=mk,other=0.).to(tl.float32)
        g=tl.load(gp+ok,mask=mk,other=0.).to(tl.float32); v=tl.load(vp+ov,mask=mv,other=0.).to(tl.float32)
        oh=((((ib*NC+ic)*H+ih)*K+rk[:,None])*V+rv[None,:])
        tl.store(hp+oh,st,mask=(rk[:,None]<K)&(rv[None,:]<V))
        c=tl.cumsum(g,axis=0)
        ca=(rt[:,None]>=rt[None,:])&mt[:,None]&mt[None,:]; lo=(rt[:,None]>rt[None,:])&mt[:,None]&mt[None,:]
        d=tl.exp(c[:,None,:]-c[None,:,:]); dp=tl.exp(c[:,None,:]-g[:,None,:]-c[None,:,:])
        aqk=tl.where(ca,tl.sum(q[:,None,:]*k[None,:,:]*d,axis=2)*S,0.)
        aqb=tl.where(ca,tl.sum(q[:,None,:]*b[None,:,:]*d,axis=2)*S,0.)
        lak=tl.where(lo,tl.sum(a[:,None,:]*k[None,:,:]*dp,axis=2),0.)
        lab=tl.where(lo,tl.sum(a[:,None,:]*b[None,:,:]*dp,axis=2),0.)
        ii=_inv(lab)
        u=tl.dot(ii,tl.dot(lak,v,input_precision="ieee"),input_precision="ieee")
        w=tl.dot(ii,a*tl.exp(c-g),input_precision="ieee")
        vn=u+tl.dot(w,st,input_precision="ieee")
        o=tl.dot(aqk,v,input_precision="ieee")+tl.dot(aqb,vn,input_precision="ieee")+tl.dot(q*tl.exp(c)*S,st,input_precision="ieee")
        tl.store(op+ov,o,mask=mv)
        ei=tl.minimum(CT-1,T-1-ic*CT); end=tl.sum(tl.where(rt[:,None]==ei,c,0.),axis=0); rem=tl.exp(end[None,:]-c)
        st=st*tl.exp(end)[:,None]+tl.dot(tl.trans(k*rem),v,input_precision="ieee")+tl.dot(tl.trans(b*rem),vn,input_precision="ieee")

@triton.jit
def _replay(kp,vp,ap,bp,gp,hp,sp,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr,CT:tl.constexpr):
    iv,ic,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2); ib,ih=bh//H,bh%H
    rk=tl.arange(0,BK); rv=iv*BV+tl.arange(0,BV)
    oh=((((ib*NC+ic)*H+ih)*K+rk[:,None])*V+rv[None,:]); s=tl.load(hp+oh,mask=(rk[:,None]<K)&(rv[None,:]<V),other=0.).to(tl.float32)
    for j in tl.static_range(0,16):
        t=ic*CT+j; m=(j<CT)&(t<T); ok=((ib*T+t)*H+ih)*K+rk; ov=((ib*T+t)*H+ih)*V+rv
        os=(((ib*T+t)*H+ih)*K+rk[:,None])*V+rv[None,:]
        tl.store(sp+os,s,mask=m&(rk[:,None]<K)&(rv[None,:]<V))
        k=tl.load(kp+ok,mask=m&(rk<K),other=0.).to(tl.float32); a=tl.load(ap+ok,mask=m&(rk<K),other=0.).to(tl.float32)
        b=tl.load(bp+ok,mask=m&(rk<K),other=0.).to(tl.float32); g=tl.load(gp+ok,mask=m&(rk<K),other=0.).to(tl.float32)
        v=tl.load(vp+ov,mask=m&(rv<V),other=0.).to(tl.float32); r=tl.sum(a[:,None]*s,axis=0)
        s=s*tl.exp(g)[:,None]+k[:,None]*v[None,:]+b[:,None]*r[None,:]

@triton.jit
def _bounds(qp,ap,bp,gp,dop,sp,dhp,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr,S:tl.constexpr,CT:tl.constexpr):
    iv,bh=tl.program_id(0),tl.program_id(1); ib,ih=bh//H,bh%H
    rk=tl.arange(0,BK); rv=iv*BV+tl.arange(0,BV); rt=tl.arange(0,16)
    ds=tl.zeros((BK,BV),tl.float32)
    for z in range(NC):
        ic=NC-1-z; t=ic*CT+rt; mt=(rt<CT)&(t<T)
        ok=((ib*T+t[:,None])*H+ih)*K+rk[None,:]; ov=((ib*T+t[:,None])*H+ih)*V+rv[None,:]
        mk=mt[:,None]&(rk[None,:]<K); mv=mt[:,None]&(rv[None,:]<V)
        q=tl.load(qp+ok,mask=mk,other=0.).to(tl.float32); a=tl.load(ap+ok,mask=mk,other=0.).to(tl.float32)
        b=tl.load(bp+ok,mask=mk,other=0.).to(tl.float32); g=tl.load(gp+ok,mask=mk,other=0.).to(tl.float32)
        do=tl.load(dop+ov,mask=mv,other=0.).to(tl.float32)
        oh=((((ib*NC+ic)*H+ih)*K+rk[:,None])*V+rv[None,:]); tl.store(dhp+oh,ds,mask=(rk[:,None]<K)&(rv[None,:]<V))
        c=tl.cumsum(g,axis=0); ca=(rt[:,None]>=rt[None,:])&mt[:,None]&mt[None,:]; lo=(rt[:,None]>rt[None,:])&mt[:,None]&mt[None,:]
        d=tl.exp(c[:,None,:]-c[None,:,:]); dp=tl.exp(c[:,None,:]-g[:,None,:]-c[None,:,:])
        aqb=tl.where(ca,tl.sum(q[:,None,:]*b[None,:,:]*d,axis=2)*S,0.)
        lab=tl.where(lo,tl.sum(a[:,None,:]*b[None,:,:]*dp,axis=2),0.)
        w=tl.dot(_inv(lab),a*tl.exp(c-g),input_precision="ieee")
        ei=tl.minimum(CT-1,T-1-ic*CT); end=tl.sum(tl.where(rt[:,None]==ei,c,0.),axis=0); rem=tl.exp(end[None,:]-c)
        dvn=tl.dot(tl.trans(aqb),do,input_precision="ieee")+tl.dot(b*rem,ds,input_precision="ieee")
        ds=tl.dot(tl.trans(q*tl.exp(c)*S),do,input_precision="ieee")+tl.exp(end)[:,None]*ds+tl.dot(tl.trans(w),dvn,input_precision="ieee")

@triton.jit
def _bwd(qp,kp,vp,ap,bp,gp,dop,sp,dhp,dvp,pp,T:tl.constexpr,B:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr,S:tl.constexpr,CT:tl.constexpr):
    iv,ic,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2); ib,ih=bh//H,bh%H
    rk=tl.arange(0,BK); rv=iv*BV+tl.arange(0,BV); stride=B*T*H*K
    oh=((((ib*NC+ic)*H+ih)*K+rk[:,None])*V+rv[None,:]); ds=tl.load(dhp+oh,mask=(rk[:,None]<K)&(rv[None,:]<V),other=0.).to(tl.float32)
    for z in tl.static_range(0,16):
        j=15-z; t=ic*CT+j; m=(j<CT)&(t<T); ok=((ib*T+t)*H+ih)*K+rk; ov=((ib*T+t)*H+ih)*V+rv
        os=(((ib*T+t)*H+ih)*K+rk[:,None])*V+rv[None,:]
        s=tl.load(sp+os,mask=m&(rk[:,None]<K)&(rv[None,:]<V),other=0.).to(tl.float32)
        q=tl.load(qp+ok,mask=m&(rk<K),other=0.).to(tl.float32); k=tl.load(kp+ok,mask=m&(rk<K),other=0.).to(tl.float32)
        a=tl.load(ap+ok,mask=m&(rk<K),other=0.).to(tl.float32); b=tl.load(bp+ok,mask=m&(rk<K),other=0.).to(tl.float32)
        g=tl.load(gp+ok,mask=m&(rk<K),other=0.).to(tl.float32); v=tl.load(vp+ov,mask=m&(rv<V),other=0.).to(tl.float32)
        do=tl.load(dop+ov,mask=m&(rv<V),other=0.).to(tl.float32); r=tl.sum(a[:,None]*s,axis=0)
        st=s*tl.exp(g)[:,None]+k[:,None]*v[None,:]+b[:,None]*r[None,:]
        dq=tl.sum(st*do[None,:],axis=1)*S; cur=ds+q[:,None]*(do*S)[None,:]
        dv=tl.sum(k[:,None]*cur,axis=0); dk=tl.sum(cur*v[None,:],axis=1); db=tl.sum(cur*r[None,:],axis=1)
        dr=tl.sum(b[:,None]*cur,axis=0); da=tl.sum(s*dr[None,:],axis=1); dg=tl.sum(cur*s*tl.exp(g)[:,None],axis=1)
        tl.store(dvp+ov,dv,mask=m&(rv<V)); base=iv*5*stride+ok; pm=m&(rk<K)
        tl.store(pp+base,dq,mask=pm); tl.store(pp+base+stride,dk,mask=pm); tl.store(pp+base+2*stride,da,mask=pm)
        tl.store(pp+base+3*stride,db,mask=pm); tl.store(pp+base+4*stride,dg,mask=pm)
        ds=cur*tl.exp(g)[:,None]+a[:,None]*dr[None,:]

@triton.jit
def _reduce(pp,dqp,dkp,dap,dbp,dgp,N:tl.constexpr,NV:tl.constexpr,BS:tl.constexpr):
    x=tl.program_id(0)*BS+tl.arange(0,BS); m=x<N
    z0=tl.zeros((BS,),tl.float32); z1=tl.zeros((BS,),tl.float32); z2=tl.zeros((BS,),tl.float32); z3=tl.zeros((BS,),tl.float32); z4=tl.zeros((BS,),tl.float32)
    for iv in range(NV):
        p=pp+iv*5*N+x; z0+=tl.load(p,mask=m,other=0.); z1+=tl.load(p+N,mask=m,other=0.); z2+=tl.load(p+2*N,mask=m,other=0.); z3+=tl.load(p+3*N,mask=m,other=0.); z4+=tl.load(p+4*N,mask=m,other=0.)
    tl.store(dqp+x,z0,mask=m); tl.store(dkp+x,z1,mask=m); tl.store(dap+x,z2,mask=m); tl.store(dbp+x,z3,mask=m); tl.store(dgp+x,z4,mask=m)

class _DPLR(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,a,b,g):
        B,T,H,K=q.shape; V=v.shape[-1]; ct=16 if K==triton.next_power_of_2(K) and T%16==0 else BT; nc=triton.cdiv(T,ct); bk=max(16,triton.next_power_of_2(K)); nv=triton.cdiv(V,FBV)
        o=torch.empty_like(v); h=torch.empty((B,nc,H,K,V),device=q.device,dtype=torch.float32)
        _fwd[(nv,B*H)](q,k,v,a,b,g,o,h,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=FBV,S=K**-.5,CT=ct,num_warps=8,num_stages=2)
        ctx.save_for_backward(q,k,v,a,b,g,h); return o
    @staticmethod
    def backward(ctx,do):
        q,k,v,a,b,g,h=ctx.saved_tensors; do=do.contiguous(); B,T,H,K=q.shape; V=v.shape[-1]; ct=16 if K==triton.next_power_of_2(K) and T%16==0 else BT; nc=triton.cdiv(T,ct); bk=max(16,triton.next_power_of_2(K))
        bv=BBV; nv=triton.cdiv(V,bv)
        s=torch.empty((B,T,H,K,V),device=q.device,dtype=torch.float32); dh=torch.empty((B,nc,H,K,V),device=q.device,dtype=torch.float32)
        p=torch.zeros((nv,5,B,T,H,K),device=q.device,dtype=torch.float32); dv=torch.zeros_like(v)
        _replay[(nv,nc,B*H)](k,v,a,b,g,h,s,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=bv,CT=ct,num_warps=4,num_stages=2)
        _bounds[(nv,B*H)](q,a,b,g,do,s,dh,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=bv,S=K**-.5,CT=ct,num_warps=4,num_stages=2)
        _bwd[(nv,nc,B*H)](q,k,v,a,b,g,do,s,dh,dv,p,T=T,B=B,H=H,K=K,V=V,NC=nc,BK=bk,BV=bv,S=K**-.5,CT=ct,num_warps=4,num_stages=2)
        dq,dk,da,db=(torch.zeros_like(q) for _ in range(4)); dg=torch.zeros_like(g); n=B*T*H*K
        _reduce[(triton.cdiv(n,256),)](p,dq,dk,da,db,dg,N=n,NV=nv,BS=256,num_warps=4)
        return dq,dk,dv,da,db,dg

def chunked_dplr_delta(q,k,v,alpha,beta,gk):
    return _DPLR.apply(q,k,v,alpha,beta,gk)
