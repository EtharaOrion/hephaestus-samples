import torch
import triton
import triton.language as tl

BC = tl.constexpr(16)
BV = 32

@triton.jit
def _ex(x):
    return tl.exp(tl.maximum(-80.0, tl.minimum(80.0, x)))

@triton.jit
def _dot(a, b):
    return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="tf32")

@triton.jit
def _build_inverse(k, g, beta, inv, T: tl.constexpr, HQ: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr):
    ic, z = tl.program_id(0), tl.program_id(1)
    ib, ih = z // HV, z % HV
    iq = ih // (HV // HQ)
    r, x = tl.arange(0, BC), tl.arange(0, BK)
    t, mt = ic * BC + r, ic * BC + r < T
    kk = tl.load(k + ((ib*T+t[:,None])*HQ+iq)*K+x[None,:], mask=mt[:,None]&(x[None,:]<K), other=0.)
    c = tl.cumsum(tl.load(g+(ib*T+t)*HV+ih, mask=mt, other=0.).to(tl.float32), 0)
    b = tl.load(beta+(ib*T+t)*HV+ih, mask=mt, other=0.).to(tl.float32)
    gram = _dot(kk, tl.trans(kk))
    lo = (r[:,None] > r[None,:]) & mt[:,None] & mt[None,:]
    p = tl.where(lo, gram*b[:,None]*_ex(c[:,None]-c[None,:]), 0.)
    a = tl.where(r[:,None] == r[None,:], 1., 0.)
    for i in range(BC):
        prow = tl.sum(tl.where(r[:,None] == i, p, 0.), axis=0)
        row = -tl.sum(prow[:,None] * a, axis=0)
        a = tl.where(r[:,None] == i, row[None,:], a)
        a = tl.where((r[:,None] == i)&(r[None,:] == i), 1., a)
    tl.store(inv + ((((ib*NC+ic)*HV+ih)*BC+r[:,None])*BC+r[None,:]), a)

@triton.jit
def _state_scan(k, v, g, beta, inv, hs, uo, T: tl.constexpr, HQ: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr, VB: tl.constexpr):
    z = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, VB)
    iv, z = z % nv, z // nv
    ib, ih = z // HV, z % HV
    iq = ih // (HV // HQ)
    x, y, r = tl.arange(0, BK), iv*VB+tl.arange(0, VB), tl.arange(0, BC)
    h = tl.zeros((BK, VB), tl.float32)
    for ic in range(NC):
        tl.store(hs + (((((ib*NC+ic)*HV+ih)*K+x[:,None])*V)+y[None,:]), h, mask=(x[:,None]<K)&(y[None,:]<V))
        t, mt = ic*BC+r, ic*BC+r<T
        kk = tl.load(k+((ib*T+t[:,None])*HQ+iq)*K+x[None,:], mask=mt[:,None]&(x[None,:]<K), other=0.)
        vv = tl.load(v+((ib*T+t[:,None])*HV+ih)*V+y[None,:], mask=mt[:,None]&(y[None,:]<V), other=0.)
        c = tl.cumsum(tl.load(g+(ib*T+t)*HV+ih, mask=mt, other=0.).to(tl.float32), 0)
        b = tl.load(beta+(ib*T+t)*HV+ih, mask=mt, other=0.).to(tl.float32)
        a = tl.load(inv+((((ib*NC+ic)*HV+ih)*BC+r[:,None])*BC+r[None,:]))
        xx = kk*(b*_ex(c))[:,None]
        uu = _dot(a, b[:,None]*vv-_dot(xx,h))
        tl.store(uo+((ib*T+t[:,None])*HV+ih)*V+y[None,:], uu, mask=mt[:,None]&(y[None,:]<V))
        last = tl.minimum(BC-1, T-ic*BC-1)
        ce = tl.sum(tl.where(r == last, c, 0.))
        h = _ex(ce)*h + _dot(tl.trans(kk*_ex(ce-c)[:,None]), uu)

@triton.jit
def _emit(q,k,g,u,hs,out,T:tl.constexpr,HQ:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,VB:tl.constexpr,SCALE:tl.constexpr):
    z=tl.program_id(0).to(tl.int64); nv=tl.cdiv(V,VB)
    iv,z=z%nv,z//nv; ic,z=z%NC,z//NC; ib,ih=z//HV,z%HV; iq=ih//(HV//HQ)
    r,x,y=tl.arange(0,BC),tl.arange(0,BK),iv*VB+tl.arange(0,VB); t=ic*BC+r; mt=t<T
    qq=tl.load(q+((ib*T+t[:,None])*HQ+iq)*K+x[None,:],mask=mt[:,None]&(x[None,:]<K),other=0.)*SCALE
    kk=tl.load(k+((ib*T+t[:,None])*HQ+iq)*K+x[None,:],mask=mt[:,None]&(x[None,:]<K),other=0.)
    c=tl.cumsum(tl.load(g+(ib*T+t)*HV+ih,mask=mt,other=0.).to(tl.float32),0)
    hh=tl.load(hs+(((((ib*NC+ic)*HV+ih)*K+x[:,None])*V)+y[None,:]),mask=(x[:,None]<K)&(y[None,:]<V),other=0.)
    uu=tl.load(u+((ib*T+t[:,None])*HV+ih)*V+y[None,:],mask=mt[:,None]&(y[None,:]<V),other=0.)
    zq=_dot(qq,tl.trans(kk))*tl.where((r[:,None]>=r[None,:])&mt[:,None]&mt[None,:],_ex(c[:,None]-c[None,:]),0.)
    oo=_ex(c)[:,None]*_dot(qq,hh)+_dot(zq,uu)
    tl.store(out+((ib*T+t[:,None])*HV+ih)*V+y[None,:],oo,mask=mt[:,None]&(y[None,:]<V))

@triton.jit
def _adjoint(q,k,g,beta,inv,u,do,ds,T:tl.constexpr,HQ:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,VB:tl.constexpr,SCALE:tl.constexpr):
    z=tl.program_id(0).to(tl.int64); nv=tl.cdiv(V,VB); iv,z=z%nv,z//nv; ib,ih=z//HV,z%HV; iq=ih//(HV//HQ)
    r,x,y=tl.arange(0,BC),tl.arange(0,BK),iv*VB+tl.arange(0,VB); dh=tl.zeros((BK,VB),tl.float32)
    for jj in range(NC):
        ic=NC-1-jj
        tl.store(ds+(((((ib*NC+ic)*HV+ih)*K+x[:,None])*V)+y[None,:]),dh,mask=(x[:,None]<K)&(y[None,:]<V))
        t=ic*BC+r; mt=t<T
        qq=tl.load(q+((ib*T+t[:,None])*HQ+iq)*K+x[None,:],mask=mt[:,None]&(x[None,:]<K),other=0.)*SCALE
        kk=tl.load(k+((ib*T+t[:,None])*HQ+iq)*K+x[None,:],mask=mt[:,None]&(x[None,:]<K),other=0.)
        dd=tl.load(do+((ib*T+t[:,None])*HV+ih)*V+y[None,:],mask=mt[:,None]&(y[None,:]<V),other=0.)
        uu=tl.load(u+((ib*T+t[:,None])*HV+ih)*V+y[None,:],mask=mt[:,None]&(y[None,:]<V),other=0.)
        c=tl.cumsum(tl.load(g+(ib*T+t)*HV+ih,mask=mt,other=0.).to(tl.float32),0)
        b=tl.load(beta+(ib*T+t)*HV+ih,mask=mt,other=0.).to(tl.float32)
        a=tl.load(inv+((((ib*NC+ic)*HV+ih)*BC+r[:,None])*BC+r[None,:]))
        zz=_dot(qq,tl.trans(kk))*tl.where((r[:,None]>=r[None,:])&mt[:,None]&mt[None,:],_ex(c[:,None]-c[None,:]),0.)
        last=tl.minimum(BC-1,T-ic*BC-1); ce=tl.sum(tl.where(r==last,c,0.)); ww=kk*_ex(ce-c)[:,None]
        du=_dot(tl.trans(zz),dd)+_dot(ww,dh)
        dy=_dot(tl.trans(a),du)
        dh=_ex(ce)*dh+_dot(tl.trans(qq*_ex(c)[:,None]),dd)-_dot(tl.trans(kk*(b*_ex(c))[:,None]),dy)

@triton.jit
def _chunk_grad(q,k,v,g,beta,inv,hs,u,ds,do,pq,pk,dv,ps,T:tl.constexpr,HQ:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,NV:tl.constexpr,BK:tl.constexpr,VB:tl.constexpr,SCALE:tl.constexpr):
    z=tl.program_id(0).to(tl.int64); iv,z=z%NV,z//NV; ic,z=z%NC,z//NC; ib,ih=z//HV,z%HV; iq=ih//(HV//HQ)
    r,x,y=tl.arange(0,BC),tl.arange(0,BK),iv*VB+tl.arange(0,VB); t=ic*BC+r; mt=t<T
    qq=tl.load(q+((ib*T+t[:,None])*HQ+iq)*K+x[None,:],mask=mt[:,None]&(x[None,:]<K),other=0.)*SCALE
    kk=tl.load(k+((ib*T+t[:,None])*HQ+iq)*K+x[None,:],mask=mt[:,None]&(x[None,:]<K),other=0.)
    vv=tl.load(v+((ib*T+t[:,None])*HV+ih)*V+y[None,:],mask=mt[:,None]&(y[None,:]<V),other=0.)
    dd=tl.load(do+((ib*T+t[:,None])*HV+ih)*V+y[None,:],mask=mt[:,None]&(y[None,:]<V),other=0.)
    uu=tl.load(u+((ib*T+t[:,None])*HV+ih)*V+y[None,:],mask=mt[:,None]&(y[None,:]<V),other=0.)
    hh=tl.load(hs+(((((ib*NC+ic)*HV+ih)*K+x[:,None])*V)+y[None,:]),mask=(x[:,None]<K)&(y[None,:]<V),other=0.)
    dh=tl.load(ds+(((((ib*NC+ic)*HV+ih)*K+x[:,None])*V)+y[None,:]),mask=(x[:,None]<K)&(y[None,:]<V),other=0.)
    c=tl.cumsum(tl.load(g+(ib*T+t)*HV+ih,mask=mt,other=0.).to(tl.float32),0); b=tl.load(beta+(ib*T+t)*HV+ih,mask=mt,other=0.).to(tl.float32)
    a=tl.load(inv+((((ib*NC+ic)*HV+ih)*BC+r[:,None])*BC+r[None,:]))
    tri=(r[:,None]>=r[None,:])&mt[:,None]&mt[None,:]; ee=tl.where(tri,_ex(c[:,None]-c[None,:]),0.)
    zz=_dot(qq,tl.trans(kk))*ee; dz=_dot(dd,tl.trans(uu))
    last=tl.minimum(BC-1,T-ic*BC-1); ce=tl.sum(tl.where(r==last,c,0.)); ew=_ex(ce-c); ww=kk*ew[:,None]
    du=_dot(tl.trans(zz),dd)+_dot(ww,dh); dy=_dot(tl.trans(a),du)
    dw=_dot(uu,tl.trans(dh)); dx=-_dot(dy,tl.trans(hh)); dp=-_dot(dy,tl.trans(uu))
    pl=(r[:,None]>r[None,:])&mt[:,None]&mt[None,:]; gram=_dot(kk,tl.trans(kk))
    pp=tl.where(pl,gram*b[:,None]*_ex(c[:,None]-c[None,:]),0.); cp=tl.where(pl,dp*b[:,None]*_ex(c[:,None]-c[None,:]),0.)
    dq=SCALE*(_ex(c)[:,None]*_dot(dd,tl.trans(hh))+_dot(dz*ee,kk))
    dk=_dot(tl.trans(dz*ee),qq)+ew[:,None]*dw+(b*_ex(c))[:,None]*dx+_dot(cp,kk)+_dot(tl.trans(cp),kk)
    db=tl.sum(dy*vv,axis=1)+_ex(c)*tl.sum(dx*kk,axis=1)+tl.sum(tl.where(pl,dp*gram*_ex(c[:,None]-c[None,:]),0.),axis=1)
    zterm=dz*zz
    dc=_ex(c)*tl.sum(_dot(qq,hh)*dd,axis=1)+tl.sum(zterm,axis=1)-tl.sum(zterm,axis=0)-tl.sum(dw*ww,axis=1)+tl.sum(dx*(kk*(b*_ex(c))[:,None]),axis=1)+tl.sum(dp*pp,axis=1)-tl.sum(dp*pp,axis=0)
    dc+=tl.where(r==last,tl.sum(dw*ww)+_ex(ce)*tl.sum(dh*hh),0.); dg=tl.cumsum(dc,0,reverse=True)
    tl.store(pq+(((((ib*T+t[:,None])*HV+ih)*NV+iv)*K)+x[None,:]),dq,mask=mt[:,None]&(x[None,:]<K)); tl.store(pk+(((((ib*T+t[:,None])*HV+ih)*NV+iv)*K)+x[None,:]),dk,mask=mt[:,None]&(x[None,:]<K))
    tl.store(dv+((ib*T+t[:,None])*HV+ih)*V+y[None,:],b[:,None]*dy,mask=mt[:,None]&(y[None,:]<V)); sp=ps+((((ib*T+t)*HV+ih)*NV+iv)*2); tl.store(sp,dg,mask=mt); tl.store(sp+1,db,mask=mt)

@triton.jit
def _reduce_qk(pq,pk,dq,dk,T:tl.constexpr,HQ:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,NV:tl.constexpr,BK:tl.constexpr):
    z=tl.program_id(0).to(tl.int64); x=tl.arange(0,BK); ibt,iq=z//HQ,z%HQ; aq=tl.zeros((BK,),tl.float32); ak=tl.zeros((BK,),tl.float32)
    for j in range(HV//HQ):
        ih=iq*(HV//HQ)+j
        for iv in range(NV):
            off=(((ibt*HV+ih)*NV+iv)*K)+x; aq+=tl.load(pq+off,mask=x<K,other=0.); ak+=tl.load(pk+off,mask=x<K,other=0.)
    tl.store(dq+(ibt*HQ+iq)*K+x,aq,mask=x<K); tl.store(dk+(ibt*HQ+iq)*K+x,ak,mask=x<K)

@triton.jit
def _reduce_scalar(ps,dg,db,N:tl.constexpr,NV:tl.constexpr):
    i=tl.program_id(0).to(tl.int64); ag,ab=0.,0.
    for iv in range(NV): ag+=tl.load(ps+(i*NV+iv)*2); ab+=tl.load(ps+(i*NV+iv)*2+1)
    tl.store(dg+i,ag); tl.store(db+i,ab)

class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,g,beta):
        B,T,HQ,K=q.shape; HV,V=v.shape[2:]; nc,nv=triton.cdiv(T,BC),triton.cdiv(V,BV); bk=triton.next_power_of_2(K)
        inv=torch.empty((B,nc,HV,BC,BC),device=q.device,dtype=torch.float32); hs=torch.empty((B,nc,HV,K,V),device=q.device,dtype=torch.float32); u=torch.empty_like(v); out=torch.empty_like(v)
        _build_inverse[(nc,B*HV)](k,g,beta,inv,T,HQ,HV,K,nc,BK=bk); _state_scan[(B*HV*nv,)](k,v,g,beta,inv,hs,u,T,HQ,HV,K,V,nc,BK=bk,VB=BV); _emit[(B*HV*nc*nv,)](q,k,g,u,hs,out,T,HQ,HV,K,V,nc,BK=bk,VB=BV,SCALE=K**-.5)
        ctx.save_for_backward(q,k,v,g,beta,inv,hs,u); ctx.dims=B,T,HQ,HV,K,V,nc,nv,bk; return out
    @staticmethod
    def backward(ctx,do):
        q,k,v,g,beta,inv,hs,u=ctx.saved_tensors; B,T,HQ,HV,K,V,nc,nv,bk=ctx.dims
        do=do.contiguous()
        ds=torch.empty((B,nc,HV,K,V),device=q.device,dtype=torch.float32); pq=torch.empty((B,T,HV,nv,K),device=q.device,dtype=torch.float32); pk=torch.empty_like(pq); ps=torch.empty((B,T,HV,nv,2),device=q.device,dtype=torch.float32)
        dq,dk,dv=torch.empty_like(q),torch.empty_like(k),torch.empty_like(v); dg,db=torch.empty_like(g),torch.empty_like(beta)
        _adjoint[(B*HV*nv,)](q,k,g,beta,inv,u,do,ds,T,HQ,HV,K,V,nc,BK=bk,VB=BV,SCALE=K**-.5); _chunk_grad[(B*HV*nc*nv,)](q,k,v,g,beta,inv,hs,u,ds,do,pq,pk,dv,ps,T,HQ,HV,K,V,nc,nv,BK=bk,VB=BV,SCALE=K**-.5); _reduce_qk[(B*T*HQ,)](pq,pk,dq,dk,T,HQ,HV,K,nv,BK=bk); _reduce_scalar[(B*T*HV,)](ps,dg,db,B*T*HV,nv)
        return dq,dk,dv,dg,db

def gated_delta_rule(q,k,v,g,beta):
    return _GatedDelta.apply(q,k,v,g,beta)
