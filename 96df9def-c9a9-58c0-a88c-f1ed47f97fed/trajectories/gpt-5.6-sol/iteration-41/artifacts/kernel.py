import torch
import triton
import triton.language as tl

_BT, _FBV, _BBV = 16, 32, 16

@triton.jit
def _inv_lower(x):
    r = tl.arange(0, 16)
    x = tl.where(r[:, None] > r[None, :], x, 0.0)
    ident = r[:, None] == r[None, :]
    inv = x + ident
    power = x
    for _ in tl.static_range(0, 3):
        power = tl.dot(power, power, input_precision="ieee")
        inv += tl.dot(inv, power, input_precision="ieee")
    return inv

@triton.jit
def _fwd(q,k,v,a,b,g,o,h,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    iv,bh=tl.program_id(0),tl.program_id(1); ib,ih=bh//H,bh%H
    rk,rv,rt=tl.arange(0,BK),iv*BV+tl.arange(0,BV),tl.arange(0,16)
    mk,mv=rk<K,rv<V; state=tl.zeros((BK,BV),tl.float32)
    xb=(ib*T*H+ih)*K; vb=(ib*T*H+ih)*V
    for c in range(NC):
        tt=c*16+rt; mt=tt<T; mtk=mt[:,None]&mk[None,:]; mtv=mt[:,None]&mv[None,:]
        hp=h+((ib*NC+c)*H+ih)*K*V+rk[:,None]*V+rv[None,:]
        tl.store(hp,state,mask=mk[:,None]&mv[None,:])
        xp=xb+tt[:,None]*H*K+rk[None,:]
        q0=tl.load(q+xp,mask=mtk,other=0.).to(tl.float32); k0=tl.load(k+xp,mask=mtk,other=0.).to(tl.float32)
        a0=tl.load(a+xp,mask=mtk,other=0.).to(tl.float32); b0=tl.load(b+xp,mask=mtk,other=0.).to(tl.float32)
        g0=tl.load(g+xp,mask=mtk,other=0.).to(tl.float32); gc=tl.cumsum(g0,axis=0); gp=gc-g0
        n=tl.minimum(T-c*16,16); end=tl.sum(tl.where((rt==n-1)[:,None],gc,0.),0)
        ctr=tl.sum(tl.where((rt==n//2)[:,None],gc,0.),0)
        pos=tl.exp(tl.maximum(tl.minimum(gc-ctr[None,:],40.),-60.)); neg=tl.exp(tl.maximum(tl.minimum(ctr[None,:]-gc,40.),-60.))
        ppos=tl.exp(tl.maximum(tl.minimum(gp-ctr[None,:],40.),-60.))
        lower=rt[:,None]>=rt[None,:]; strict=rt[:,None]>rt[None,:]
        qx=q0*pos*(K**-0.5); ax=a0*ppos; kx=k0*neg; bx=b0*neg
        aqk=tl.where(lower,tl.dot(qx,tl.trans(kx),input_precision="ieee"),0.)
        aqb=tl.where(lower,tl.dot(qx,tl.trans(bx),input_precision="ieee"),0.)
        lab=tl.where(strict,tl.dot(ax,tl.trans(bx),input_precision="ieee"),0.)
        lak=tl.where(strict,tl.dot(ax,tl.trans(kx),input_precision="ieee"),0.); inv=_inv_lower(lab)
        vp=vb+tt[:,None]*H*V+rv[None,:]; vv=tl.load(v+vp,mask=mtv,other=0.).to(tl.float32)
        u=tl.dot(tl.dot(inv,lak,input_precision="ieee"),vv,input_precision="ieee")
        qe=q0*tl.exp(tl.maximum(gc,-60.))*(K**-0.5); w=tl.dot(inv,a0*tl.exp(tl.maximum(gp,-60.)),input_precision="ieee")
        vn=u+tl.dot(w,state,input_precision="ieee")
        out=tl.dot(qe,state,input_precision="ieee")+tl.dot(aqk,vv,input_precision="ieee")+tl.dot(aqb,vn,input_precision="ieee")
        tl.store(o+vp,out,mask=mtv)
        ke=k0*tl.exp(tl.maximum(end[None,:]-gc,-60.)); be=b0*tl.exp(tl.maximum(end[None,:]-gc,-60.))
        state*=tl.exp(tl.maximum(end[:,None],-60.)); state+=tl.dot(tl.trans(ke),vv,input_precision="ieee")+tl.dot(tl.trans(be),vn,input_precision="ieee")


@triton.jit
def _exact_boundaries(k,v,a,b,g,h,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    iv,bh=tl.program_id(0),tl.program_id(1); ib,ih=bh//H,bh%H
    rk,rv=tl.arange(0,BK),iv*BV+tl.arange(0,BV); mk,mv=rk<K,rv<V
    st=tl.zeros((BK,BV),tl.float32); xb,vb=(ib*T*H+ih)*K,(ib*T*H+ih)*V
    for t in range(T):
        if t % 16 == 0:
            hp=h+((ib*NC+t//16)*H+ih)*K*V+rk[:,None]*V+rv[None,:]
            tl.store(hp,st,mask=mk[:,None]&mv[None,:])
        xp,vp=xb+t*H*K+rk,vb+t*H*V+rv
        kk=tl.load(k+xp,mask=mk,other=0.).to(tl.float32); aa=tl.load(a+xp,mask=mk,other=0.).to(tl.float32)
        bb=tl.load(b+xp,mask=mk,other=0.).to(tl.float32); gg=tl.load(g+xp,mask=mk,other=0.).to(tl.float32)
        vv=tl.load(v+vp,mask=mv,other=0.).to(tl.float32); rd=tl.sum(st*aa[:,None],0)
        st=st*tl.exp(gg[:,None])+kk[:,None]*vv[None,:]+bb[:,None]*rd[None,:]

@triton.jit
def _expand(k,v,a,b,g,h,s,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    iv,c,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2); ib,ih=bh//H,bh%H
    rk,rv=tl.arange(0,BK),iv*BV+tl.arange(0,BV); mk,mv=rk<K,rv<V
    hp=h+((ib*NC+c)*H+ih)*K*V+rk[:,None]*V+rv[None,:]
    st=tl.load(hp,mask=mk[:,None]&mv[None,:],other=0.).to(tl.float32)
    xb,vb=(ib*T*H+ih)*K,(ib*T*H+ih)*V
    for j in tl.static_range(0,16):
        t=c*16+j; ok=t<T; sp=s+((ib*T+t)*H+ih)*K*V+rk[:,None]*V+rv[None,:]
        tl.store(sp,st,mask=ok&mk[:,None]&mv[None,:]); xp,vp=xb+t*H*K+rk,vb+t*H*V+rv
        kk=tl.load(k+xp,mask=ok&mk,other=0.).to(tl.float32); aa=tl.load(a+xp,mask=ok&mk,other=0.).to(tl.float32)
        bb=tl.load(b+xp,mask=ok&mk,other=0.).to(tl.float32); gg=tl.load(g+xp,mask=ok&mk,other=0.).to(tl.float32)
        vv=tl.load(v+vp,mask=ok&mv,other=0.).to(tl.float32); rd=tl.sum(st*aa[:,None],0)
        st=st*tl.exp(gg[:,None])+kk[:,None]*vv[None,:]+bb[:,None]*rd[None,:]

@triton.jit
def _boundary(q,a,b,g,do,s,dh,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    iv,bh=tl.program_id(0),tl.program_id(1); ib,ih=bh//H,bh%H
    rk,rv=tl.arange(0,BK),iv*BV+tl.arange(0,BV); mk,mv=rk<K,rv<V; ds=tl.zeros((BK,BV),tl.float32)
    xb,vb=(ib*T*H+ih)*K,(ib*T*H+ih)*V
    for z in range(NC):
        c=NC-1-z; hp=dh+((ib*NC+c)*H+ih)*K*V+rk[:,None]*V+rv[None,:]
        tl.store(hp,ds,mask=mk[:,None]&mv[None,:])
        for jj in tl.static_range(0,16):
            t=c*16+15-jj; ok=t<T; xp,vp=xb+t*H*K+rk,vb+t*H*V+rv
            qq=tl.load(q+xp,mask=ok&mk,other=0.).to(tl.float32); aa=tl.load(a+xp,mask=ok&mk,other=0.).to(tl.float32)
            bb=tl.load(b+xp,mask=ok&mk,other=0.).to(tl.float32); gg=tl.load(g+xp,mask=ok&mk,other=0.).to(tl.float32)
            dd=tl.load(do+vp,mask=ok&mv,other=0.).to(tl.float32); G=ds+qq[:,None]*dd[None,:]*(K**-0.5)
            dr=tl.sum(G*bb[:,None],0); nxt=G*tl.exp(gg[:,None])+aa[:,None]*dr[None,:]
            ds=tl.where(ok,nxt,ds)

@triton.jit
def _bwd(q,k,v,a,b,g,do,s,dh,pq,pk,pa,pb,pg,dv,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,B:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    iv,c,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2); ib,ih=bh//H,bh%H
    rk,rv=tl.arange(0,BK),iv*BV+tl.arange(0,BV); mk,mv=rk<K,rv<V
    hp=dh+((ib*NC+c)*H+ih)*K*V+rk[:,None]*V+rv[None,:]
    ds=tl.load(hp,mask=mk[:,None]&mv[None,:],other=0.).to(tl.float32)
    xb,vb=(ib*T*H+ih)*K,(ib*T*H+ih)*V; base=((iv*B+ib)*T*H+ih)*K
    for jj in tl.static_range(0,16):
        t=c*16+15-jj; ok=t<T; xp,vp=xb+t*H*K+rk,vb+t*H*V+rv
        qq=tl.load(q+xp,mask=ok&mk,other=0.).to(tl.float32); kk=tl.load(k+xp,mask=ok&mk,other=0.).to(tl.float32)
        aa=tl.load(a+xp,mask=ok&mk,other=0.).to(tl.float32); bb=tl.load(b+xp,mask=ok&mk,other=0.).to(tl.float32)
        gg=tl.load(g+xp,mask=ok&mk,other=0.).to(tl.float32); vv=tl.load(v+vp,mask=ok&mv,other=0.).to(tl.float32)
        dd=tl.load(do+vp,mask=ok&mv,other=0.).to(tl.float32); sp=s+((ib*T+t)*H+ih)*K*V+rk[:,None]*V+rv[None,:]
        old=tl.load(sp,mask=ok&mk[:,None]&mv[None,:],other=0.).to(tl.float32); rd=tl.sum(old*aa[:,None],0); en=tl.exp(gg)
        new=old*en[:,None]+kk[:,None]*vv[None,:]+bb[:,None]*rd[None,:]; G=ds+qq[:,None]*dd[None,:]*(K**-0.5)
        dq0=tl.sum(new*dd[None,:],1)*(K**-0.5); dk0=tl.sum(G*vv[None,:],1); db0=tl.sum(G*rd[None,:],1)
        dr=tl.sum(G*bb[:,None],0); da0=tl.sum(old*dr[None,:],1); dg0=tl.sum(G*old,1)*en; dv0=tl.sum(G*kk[:,None],0)
        pp=base+t*H*K+rk; tl.store(pq+pp,dq0,mask=ok&mk); tl.store(pk+pp,dk0,mask=ok&mk); tl.store(pa+pp,da0,mask=ok&mk)
        tl.store(pb+pp,db0,mask=ok&mk); tl.store(pg+pp,dg0,mask=ok&mk); tl.store(dv+vp,dv0,mask=ok&mv)
        nxt=G*en[:,None]+aa[:,None]*dr[None,:]; ds=tl.where(ok,nxt,ds)

@triton.jit
def _reduce(pq,pk,pa,pb,pg,dq,dk,da,db,dg,N:tl.constexpr,NV:tl.constexpr,BS:tl.constexpr):
    x=tl.program_id(0)*BS+tl.arange(0,BS); ok=x<N
    q0=tl.zeros((BS,),tl.float32); k0=tl.zeros((BS,),tl.float32); a0=tl.zeros((BS,),tl.float32); b0=tl.zeros((BS,),tl.float32); g0=tl.zeros((BS,),tl.float32)
    for iv in range(NV):
        off=iv*N+x; q0+=tl.load(pq+off,mask=ok,other=0.); k0+=tl.load(pk+off,mask=ok,other=0.)
        a0+=tl.load(pa+off,mask=ok,other=0.); b0+=tl.load(pb+off,mask=ok,other=0.); g0+=tl.load(pg+off,mask=ok,other=0.)
    tl.store(dq+x,q0,mask=ok); tl.store(dk+x,k0,mask=ok); tl.store(da+x,a0,mask=ok); tl.store(db+x,b0,mask=ok); tl.store(dg+x,g0,mask=ok)

class _DPLR(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,a,b,g):
    B,T,H,K=q.shape; V=v.shape[-1]; nc=triton.cdiv(T,_BT); bk=max(16,triton.next_power_of_2(K)); nv=triton.cdiv(V,_FBV)
    o=torch.empty_like(v); h=torch.empty((B,nc,H,K,V),device=q.device,dtype=torch.float32)
    _fwd[(nv,B*H)](q,k,v,a,b,g,o,h,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=_FBV,num_warps=8,num_stages=2)
    ctx.save_for_backward(q,k,v,a,b,g,h); return o
 @staticmethod
 def backward(ctx,do):
    q,k,v,a,b,g,h=ctx.saved_tensors; do=do.contiguous(); B,T,H,K=q.shape; V=v.shape[-1]
    nc=triton.cdiv(T,_BT); bk=max(16,triton.next_power_of_2(K))
    regular_v=V==triton.next_power_of_2(V)
    bv=_BBV if regular_v else max(16,triton.next_power_of_2(V)); nv=triton.cdiv(V,bv)
    if not regular_v:
        _exact_boundaries[(nv,B*H)](k,v,a,b,g,h,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=bv,num_warps=4)
    s=torch.empty((B,T,H,K,V),device=q.device,dtype=torch.float32)
    _expand[(nv,nc,B*H)](k,v,a,b,g,h,s,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=bv,num_warps=4)
    dh=torch.empty((B,nc,H,K,V),device=q.device,dtype=torch.float32)
    _boundary[(nv,B*H)](q,a,b,g,do,s,dh,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=bv,num_warps=4)
    n=B*T*H*K; parts=[torch.zeros((nv,n),device=q.device,dtype=torch.float32) for _ in range(5)]; dv=torch.empty_like(v)
    _bwd[(nv,nc,B*H)](q,k,v,a,b,g,do,s,dh,*parts,dv,T=T,H=H,K=K,V=V,B=B,NC=nc,BK=bk,BV=bv,num_warps=4)
    grads=[torch.empty_like(x) for x in (q,k,a,b,g)]; _reduce[(triton.cdiv(n,256),)](*parts,*grads,N=n,NV=nv,BS=256)
    dq,dk,da,db,dg=grads; return dq,dk,dv,da,db,dg

def chunked_dplr_delta(q,k,v,alpha,beta,gk):
 return _DPLR.apply(q,k,v,alpha,beta,gk)
