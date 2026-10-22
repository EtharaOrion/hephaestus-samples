import torch
import triton
import triton.language as tl

BT = 16
BV = 16

@triton.jit
def ex(x):
    return tl.exp(tl.maximum(tl.minimum(x, 60.0), -60.0))

@triton.jit
def lower_solve(L, x):
    y = x
    p = L
    for _ in tl.static_range(0, 4):
        y += tl.dot(p, y, input_precision='ieee')
        p = tl.dot(p, p, input_precision='ieee')
    return y

@triton.jit
def upper_solve(L, x):
    y = x
    p = tl.trans(L)
    for _ in tl.static_range(0, 4):
        y += tl.dot(p, y, input_precision='ieee')
        p = tl.dot(p, p, input_precision='ieee')
    return y

@triton.jit
def fwd_kernel(q,k,v,a,b,g,o,h,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV0:tl.constexpr):
    iv=tl.program_id(0); bh=tl.program_id(1); ib=bh//H; ih=bh-ib*H
    rk=tl.arange(0,BK); rv=iv*BV0+tl.arange(0,BV0); rt=tl.arange(0,16)
    state=tl.zeros((BK,BV0),tl.float32); scale=K ** -0.5
    for ic in range(NC):
        tt=ic*16+rt; valid=tt<T
        ik=(ib*T*H+ih)*K+tt[:,None]*H*K+rk[None,:]
        ivv=(ib*T*H+ih)*V+tt[:,None]*H*V+rv[None,:]
        mk=valid[:,None]&(rk[None,:]<K); mv=valid[:,None]&(rv[None,:]<V)
        q0=tl.load(q+ik,mask=mk,other=0.).to(tl.float32)*scale
        k0=tl.load(k+ik,mask=mk,other=0.).to(tl.float32)
        a0=tl.load(a+ik,mask=mk,other=0.).to(tl.float32)
        b0=tl.load(b+ik,mask=mk,other=0.).to(tl.float32)
        gg=tl.load(g+ik,mask=mk,other=0.).to(tl.float32)
        vv=tl.load(v+ivv,mask=mv,other=0.).to(tl.float32)
        c=tl.cumsum(gg,axis=0); cp=c-gg; last=tl.minimum(15,T-ic*16-1)
        ce=tl.sum(tl.where((rt==last)[:,None],c,0.),axis=0)
        qq=q0*ex(c); aa=a0*ex(cp); kk=k0*ex(-c); bb=b0*ex(-c)
        ke=k0*ex(ce[None,:]-c); be=b0*ex(ce[None,:]-c)
        causal=(rt[:,None]>=rt[None,:])&valid[:,None]&valid[None,:]
        strict=(rt[:,None]>rt[None,:])&valid[:,None]&valid[None,:]
        aqk=tl.where(causal,tl.dot(qq,tl.trans(kk),input_precision='ieee'),0.)
        aqb=tl.where(causal,tl.dot(qq,tl.trans(bb),input_precision='ieee'),0.)
        aak=tl.where(strict,tl.dot(aa,tl.trans(kk),input_precision='ieee'),0.)
        lab=tl.where(strict,tl.dot(aa,tl.trans(bb),input_precision='ieee'),0.)
        x=tl.dot(aa,state,input_precision='ieee')+tl.dot(aak,vv,input_precision='ieee')
        rr=lower_solve(lab,x)
        oo=tl.dot(qq,state,input_precision='ieee')+tl.dot(aqk,vv,input_precision='ieee')+tl.dot(aqb,rr,input_precision='ieee')
        tl.store(o+ivv,oo,mask=mv)
        hp=(((ib*NC+ic)*H+ih)*K+rk[:,None])*V+rv[None,:]
        tl.store(h+hp,state,mask=(rk[:,None]<K)&(rv[None,:]<V))
        state=state*ex(ce)[:,None]+tl.dot(tl.trans(ke),vv,input_precision='ieee')+tl.dot(tl.trans(be),rr,input_precision='ieee')

@triton.jit
def boundary_kernel(q,a,b,g,do,dh,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV0:tl.constexpr):
    iv=tl.program_id(0); bh=tl.program_id(1); ib=bh//H; ih=bh-ib*H
    rk=tl.arange(0,BK); rv=iv*BV0+tl.arange(0,BV0); rt=tl.arange(0,16)
    ds=tl.zeros((BK,BV0),tl.float32); scale=K ** -0.5
    for z in range(NC):
        ic=NC-1-z; tt=ic*16+rt; valid=tt<T
        ik=(ib*T*H+ih)*K+tt[:,None]*H*K+rk[None,:]
        ivv=(ib*T*H+ih)*V+tt[:,None]*H*V+rv[None,:]
        mk=valid[:,None]&(rk[None,:]<K); mv=valid[:,None]&(rv[None,:]<V)
        q0=tl.load(q+ik,mask=mk,other=0.).to(tl.float32)*scale
        a0=tl.load(a+ik,mask=mk,other=0.).to(tl.float32)
        b0=tl.load(b+ik,mask=mk,other=0.).to(tl.float32)
        gg=tl.load(g+ik,mask=mk,other=0.).to(tl.float32)
        dd=tl.load(do+ivv,mask=mv,other=0.).to(tl.float32)
        c=tl.cumsum(gg,axis=0); cp=c-gg; last=tl.minimum(15,T-ic*16-1)
        ce=tl.sum(tl.where((rt==last)[:,None],c,0.),axis=0)
        qq=q0*ex(c); aa=a0*ex(cp); bb=b0*ex(-c); be=b0*ex(ce[None,:]-c)
        causal=(rt[:,None]>=rt[None,:])&valid[:,None]&valid[None,:]
        strict=(rt[:,None]>rt[None,:])&valid[:,None]&valid[None,:]
        aqb=tl.where(causal,tl.dot(qq,tl.trans(bb),input_precision='ieee'),0.)
        lab=tl.where(strict,tl.dot(aa,tl.trans(bb),input_precision='ieee'),0.)
        dr=tl.dot(tl.trans(aqb),dd,input_precision='ieee')+tl.dot(be,ds,input_precision='ieee')
        dx=upper_solve(lab,dr)
        ds=ds*ex(ce)[:,None]+tl.dot(tl.trans(qq),dd,input_precision='ieee')+tl.dot(tl.trans(aa),dx,input_precision='ieee')
        hp=(((ib*NC+ic)*H+ih)*K+rk[:,None])*V+rv[None,:]
        tl.store(dh+hp,ds,mask=(rk[:,None]<K)&(rv[None,:]<V))

@triton.jit
def replay_states_kernel(k,v,a,b,g,states,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV0:tl.constexpr):
    iv=tl.program_id(0); bh=tl.program_id(1); ib=bh//H; ih=bh-ib*H
    rk=tl.arange(0,BK); rv=iv*BV0+tl.arange(0,BV0); s=tl.zeros((BK,BV0),tl.float32)
    for ic in range(NC):
        for j in tl.static_range(0,16):
            t=ic*16+j; valid=t<T
            ik=(ib*T*H+ih)*K+t*H*K+rk; ivv=(ib*T*H+ih)*V+t*H*V+rv
            sp=((ib*T*H+t*H+ih)*K+rk[:,None])*V+rv[None,:]
            tl.store(states+sp,s,mask=valid&(rk[:,None]<K)&(rv[None,:]<V))
            kk=tl.load(k+ik,mask=valid&(rk<K),other=0.).to(tl.float32); aa=tl.load(a+ik,mask=valid&(rk<K),other=0.).to(tl.float32)
            bb=tl.load(b+ik,mask=valid&(rk<K),other=0.).to(tl.float32); gg=tl.load(g+ik,mask=valid&(rk<K),other=0.).to(tl.float32)
            vv=tl.load(v+ivv,mask=valid&(rv<V),other=0.).to(tl.float32); read=tl.sum(s*aa[:,None],axis=0)
            sn=s*tl.exp(gg)[:,None]+kk[:,None]*vv[None,:]+bb[:,None]*read[None,:]
            s=tl.where(valid,sn,s)

@triton.jit
def scan_grad_kernel(q,k,v,a,b,g,do,states,dh,pq,pk,pa,pb,pg,dv,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV0:tl.constexpr):
    iv=tl.program_id(0); ic=tl.program_id(1); bh=tl.program_id(2); ib=bh//H; ih=bh-ib*H
    rk=tl.arange(0,BK); rv=iv*BV0+tl.arange(0,BV0); scale=K ** -0.5
    if ic+1<NC:
        hp=(((ib*NC+ic+1)*H+ih)*K+rk[:,None])*V+rv[None,:]
        ds=tl.load(dh+hp,mask=(rk[:,None]<K)&(rv[None,:]<V),other=0.).to(tl.float32)
    else:
        ds=tl.zeros((BK,BV0),tl.float32)
    for z in tl.static_range(0,16):
        j=15-z; t=ic*16+j; valid=t<T
        ik=(ib*T*H+ih)*K+t*H*K+rk; ivv=(ib*T*H+ih)*V+t*H*V+rv
        sp=((ib*T*H+t*H+ih)*K+rk[:,None])*V+rv[None,:]
        mk=valid&(rk<K); mv=valid&(rv<V); ms=valid&(rk[:,None]<K)&(rv[None,:]<V)
        sold=tl.load(states+sp,mask=ms,other=0.).to(tl.float32)
        qt=tl.load(q+ik,mask=mk,other=0.).to(tl.float32)*scale; kt=tl.load(k+ik,mask=mk,other=0.).to(tl.float32)
        at=tl.load(a+ik,mask=mk,other=0.).to(tl.float32); bt=tl.load(b+ik,mask=mk,other=0.).to(tl.float32)
        gt=tl.load(g+ik,mask=mk,other=0.).to(tl.float32); vt=tl.load(v+ivv,mask=mv,other=0.).to(tl.float32); dot=tl.load(do+ivv,mask=mv,other=0.).to(tl.float32)
        read=tl.sum(sold*at[:,None],axis=0); st=sold*tl.exp(gt)[:,None]+kt[:,None]*vt[None,:]+bt[:,None]*read[None,:]
        dcur=ds+qt[:,None]*dot[None,:]; dread=tl.sum(dcur*bt[:,None],axis=0)
        dq0=tl.sum(st*dot[None,:],axis=1)*scale; dk0=tl.sum(dcur*vt[None,:],axis=1)
        da0=tl.sum(sold*dread[None,:],axis=1); db0=tl.sum(dcur*read[None,:],axis=1)
        dg0=tl.sum(dcur*sold*tl.exp(gt)[:,None],axis=1); dv0=tl.sum(dcur*kt[:,None],axis=0)
        pp=iv*T*H*K+(ib*T*H+ih)*K+t*H*K+rk
        tl.store(pq+pp,dq0,mask=mk); tl.store(pk+pp,dk0,mask=mk); tl.store(pa+pp,da0,mask=mk); tl.store(pb+pp,db0,mask=mk); tl.store(pg+pp,dg0,mask=mk); tl.store(dv+ivv,dv0,mask=mv)
        dn=dcur*tl.exp(gt)[:,None]+at[:,None]*dread[None,:]; ds=tl.where(valid,dn,ds)

@triton.jit
def reduce_kernel(pq,pk,pa,pb,pg,dq,dk,da,db,dg,N:tl.constexpr,NV:tl.constexpr,X:tl.constexpr):
    x=tl.program_id(0)*X+tl.arange(0,X); m=x<N
    sq=tl.zeros((X,),tl.float32); sk=tl.zeros((X,),tl.float32); sa=tl.zeros((X,),tl.float32); sb=tl.zeros((X,),tl.float32); sg=tl.zeros((X,),tl.float32)
    for i in range(NV):
        z=i*N+x
        sq+=tl.load(pq+z,mask=m,other=0.); sk+=tl.load(pk+z,mask=m,other=0.); sa+=tl.load(pa+z,mask=m,other=0.); sb+=tl.load(pb+z,mask=m,other=0.); sg+=tl.load(pg+z,mask=m,other=0.)
    tl.store(dq+x,sq,mask=m); tl.store(dk+x,sk,mask=m); tl.store(da+x,sa,mask=m); tl.store(db+x,sb,mask=m); tl.store(dg+x,sg,mask=m)

class DPLR(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,a,b,g):
        B,T,H,K=q.shape; V=v.shape[-1]; nc=triton.cdiv(T,BT); nv=triton.cdiv(V,BV); bk=max(16,triton.next_power_of_2(K))
        o=torch.empty_like(v); h=torch.empty((B,nc,H,K,V),device=q.device,dtype=torch.float32)
        fwd_kernel[(nv,B*H)](q,k,v,a,b,g,o,h,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV0=BV,num_warps=8,num_stages=2)
        ctx.save_for_backward(q,k,v,a,b,g,h); ctx.meta=B,T,H,K,V,nc,nv,bk
        return o
    @staticmethod
    def backward(ctx,do):
        q,k,v,a,b,g,h=ctx.saved_tensors; B,T,H,K,V,nc,nv,bk=ctx.meta; do=do.contiguous(); dh=torch.empty_like(h)
        bvb=BV; nvb=nv
        boundary_kernel[(nvb,B*H)](q,a,b,g,do,dh,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV0=bvb,num_warps=8,num_stages=2)
        states=torch.empty((B,T,H,K,V),device=q.device,dtype=torch.float32)
        replay_states_kernel[(nvb,B*H)](k,v,a,b,g,states,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV0=bvb,num_warps=8,num_stages=2)
        sh=(nvb,B,T,H,K); pq=torch.zeros(sh,device=q.device,dtype=torch.float32); pk=torch.zeros_like(pq); pa=torch.zeros_like(pq); pb=torch.zeros_like(pq); pg=torch.zeros_like(pq); dv=torch.empty_like(v)
        scan_grad_kernel[(nvb,nc,B*H)](q,k,v,a,b,g,do,states,dh,pq,pk,pa,pb,pg,dv,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV0=bvb,num_warps=8,num_stages=2)
        dq=torch.empty_like(q); dk=torch.empty_like(k); da=torch.empty_like(a); db=torch.empty_like(b); dg=torch.empty_like(g); n=B*T*H*K
        reduce_kernel[(triton.cdiv(n,256),)](pq,pk,pa,pb,pg,dq,dk,da,db,dg,N=n,NV=nvb,X=256,num_warps=8)
        return dq,dk,dv,da,db,dg

def chunked_dplr_delta(q,k,v,alpha,beta,gk):
    return DPLR.apply(q,k,v,alpha,beta,gk)
