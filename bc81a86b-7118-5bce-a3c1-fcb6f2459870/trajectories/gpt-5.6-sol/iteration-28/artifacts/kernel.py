import torch
import triton
import triton.language as tl

BC = tl.constexpr(16)
BV = 32


@triton.jit
def _wy(k, g, beta, a, w, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
        K: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr):

    c = tl.program_id(0).to(tl.int64)
    bh = tl.program_id(1).to(tl.int64)
    b, h = bh // HV, bh % HV
    kh = h // (HV // H)
    r, x = tl.arange(0, BC), tl.arange(0, BK)
    t = c * BC + r
    mt = t < T
    p = ((b * T + t) * H + kh) * K
    kb = tl.load(k + p[:, None] + x[None, :], mask=mt[:, None] & (x[None, :] < K), other=0.)
    gb = tl.load(g + (b*T+t)*HV+h, mask=mt, other=0.).to(tl.float32)
    bb = tl.load(beta + (b*T+t)*HV+h, mask=mt, other=0.).to(tl.float32)
    gc = tl.cumsum(gb, 0)
    gram = tl.dot(kb, tl.trans(kb), input_precision='tf32')
    lower = r[:, None] > r[None, :]
    valid = mt[:, None] & mt[None, :]
    m = tl.where(lower & valid, gram * bb[:, None] * tl.exp(gc[:, None]-gc[None, :]), 0.)
    ai = tl.where(r[:, None] == r[None, :], 1., 0.).to(tl.float32)
    for i in range(1, BC):
        coeff = tl.sum(tl.where(r[:, None] == i, m, 0.), axis=0)
        row = -tl.sum(tl.where(r < i, coeff, 0.)[:, None] * ai, axis=0)
        row += (r == i).to(tl.float32)
        ai = tl.where((r == i)[:, None], row[None, :], ai)
    off = ((((b*NC+c)*HV+h)*BC+r[:, None])*BC+r[None, :])
    tl.store(a+off, ai, mask=valid)
    wb = tl.dot(ai.to(kb.dtype), (kb*(bb*tl.exp(gc))[:, None]).to(kb.dtype), input_precision='tf32')
    tl.store(w+((b*T+t)*HV+h)[:, None]*K+x[None, :], wb, mask=mt[:, None]&(x[None, :]<K))


@triton.jit
def _fscan(k, v, g, beta, a, w, hs, d,
           T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
           V: tl.constexpr, BK: tl.constexpr, VB: tl.constexpr, NC: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    NV = (V+VB-1)//VB
    j, bh = p % NV, p // NV
    b, h = bh // HV, bh % HV
    kh = h // (HV // H)
    x, y, r = tl.arange(0, BK), j*VB+tl.arange(0, VB), tl.arange(0, BC)
    state = tl.zeros([BK, VB], tl.float32)
    for c in range(NC):
        t = c*BC+r
        mt = t < T
        ho = ((((b*NC+c)*HV+h)*K+x[:, None])*V+y[None, :])
        tl.store(hs+ho, state, mask=(x[:, None] < K) & (y[None, :] < V))
        kp = ((b*T+t)*H+kh)*K
        kb = tl.load(k+kp[:, None]+x[None, :], mask=mt[:, None]&(x[None, :]<K), other=0.)
        vb = tl.load(v+((b*T+t)*HV+h)[:, None]*V+y[None, :], mask=mt[:, None]&(y[None, :]<V), other=0.)
        gb = tl.load(g+(b*T+t)*HV+h, mask=mt, other=0.).to(tl.float32)
        bb = tl.load(beta+(b*T+t)*HV+h, mask=mt, other=0.).to(tl.float32)
        gc = tl.cumsum(gb, 0)
        ai = tl.load(a+((((b*NC+c)*HV+h)*BC+r[:, None])*BC+r[None, :]), mask=mt[:, None]&mt[None, :], other=0.)
        u = tl.dot(ai.to(vb.dtype), (vb*bb[:, None]).to(vb.dtype), input_precision='tf32')
        wb = tl.load(w+((b*T+t)*HV+h)[:, None]*K+x[None, :], mask=mt[:, None]&(x[None, :]<K), other=0.)
        db = u-tl.dot(wb.to(kb.dtype), state.to(kb.dtype), input_precision='tf32')
        tl.store(d+((b*T+t)*HV+h)[:, None]*V+y[None, :], db, mask=mt[:, None]&(y[None, :]<V))
        ge = tl.sum(tl.where(mt, gb, 0.), 0)
        state = state*tl.exp(ge)+tl.dot(tl.trans(kb*tl.exp(ge-gc)[:, None]).to(kb.dtype), db.to(vb.dtype), input_precision='tf32')


@triton.jit
def _out(q, k, g, hs, d, o,
         T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
         V: tl.constexpr, BK: tl.constexpr, VB: tl.constexpr, NC: tl.constexpr,
         SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    NV = (V+VB-1)//VB
    j, z = p % NV, p // NV
    c, bh = z % NC, z // NC
    b, h = bh // HV, bh % HV
    kh = h // (HV // H)
    r, x, y = tl.arange(0, BC), tl.arange(0, BK), j*VB+tl.arange(0, VB)
    t = c*BC+r
    mt = t<T
    qp = ((b*T+t)*H+kh)*K
    qb = tl.load(q+qp[:, None]+x[None, :], mask=mt[:, None]&(x[None, :]<K), other=0.)
    kb = tl.load(k+qp[:, None]+x[None, :], mask=mt[:, None]&(x[None, :]<K), other=0.)
    gb = tl.load(g+(b*T+t)*HV+h, mask=mt, other=0.).to(tl.float32)
    gc = tl.cumsum(gb, 0)
    hb = tl.load(hs+((((b*NC+c)*HV+h)*K+x[:, None])*V+y[None, :]), mask=(x[:, None]<K)&(y[None, :]<V), other=0.)
    db = tl.load(d+((b*T+t)*HV+h)[:, None]*V+y[None, :], mask=mt[:, None]&(y[None, :]<V), other=0.)
    ob = tl.dot((qb*(tl.exp(gc)*SCALE)[:, None]).to(qb.dtype), hb.to(qb.dtype), input_precision='tf32')
    s = tl.dot(qb, tl.trans(kb), input_precision='tf32')*SCALE
    s = tl.where(r[:, None]>=r[None, :], s*tl.exp(gc[:, None]-gc[None, :]), 0.)
    ob += tl.dot(s.to(db.dtype), db.to(db.dtype), input_precision='tf32')
    tl.store(o+((b*T+t)*HV+h)[:, None]*V+y[None, :], ob, mask=mt[:, None]&(y[None, :]<V))


@triton.jit
def _ascan(q, k, g, w, do, dhs,
           T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
           V: tl.constexpr, BK: tl.constexpr, VB: tl.constexpr, NC: tl.constexpr,
           SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    NV = (V+VB-1)//VB
    j, bh = p%NV, p//NV
    b, h = bh//HV, bh%HV
    kh = h//(HV//H)
    r, x, y = tl.arange(0, BC), tl.arange(0, BK), j*VB+tl.arange(0, VB)
    dh = tl.zeros([BK, VB], tl.float32)
    for z in range(NC):
        c = NC-1-z
        t = c*BC+r
        mt = t<T
        off = ((((b*NC+c)*HV+h)*K+x[:, None])*V+y[None, :])
        tl.store(dhs+off, dh, mask=(x[:, None]<K)&(y[None, :]<V))
        qp = ((b*T+t)*H+kh)*K
        qb = tl.load(q+qp[:, None]+x[None, :], mask=mt[:, None]&(x[None, :]<K), other=0.)
        kb = tl.load(k+qp[:, None]+x[None, :], mask=mt[:, None]&(x[None, :]<K), other=0.)
        dob = tl.load(do+((b*T+t)*HV+h)[:, None]*V+y[None, :], mask=mt[:, None]&(y[None, :]<V), other=0.)
        gb = tl.load(g+(b*T+t)*HV+h, mask=mt, other=0.).to(tl.float32)
        gc = tl.cumsum(gb, 0)
        ge = tl.sum(tl.where(mt, gb, 0.), 0)
        wb = tl.load(w+((b*T+t)*HV+h)[:, None]*K+x[None, :], mask=mt[:, None]&(x[None, :]<K), other=0.)
        s = tl.dot(qb, tl.trans(kb), input_precision='tf32')*SCALE
        s = tl.where(r[:, None]>=r[None, :], s*tl.exp(gc[:, None]-gc[None, :]), 0.)
        dd = tl.dot(tl.trans(s).to(dob.dtype), dob, input_precision='tf32')
        dd += tl.dot((kb*tl.exp(ge-gc)[:, None]).to(kb.dtype), dh.to(dob.dtype), input_precision='tf32')
        old = dh
        dh = tl.dot(tl.trans(qb*(tl.exp(gc)*SCALE)[:, None]).to(qb.dtype), dob, input_precision='tf32')
        dh += tl.exp(ge)*old
        dh -= tl.dot(tl.trans(wb).to(kb.dtype), dd.to(dob.dtype), input_precision='tf32')


@triton.jit
def _grad(q, k, v, g, beta, a, hs, d, dhs, do, pq, pk, dv, pg, pb,
          T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
          V: tl.constexpr, BK: tl.constexpr, VB: tl.constexpr, NC: tl.constexpr,
          SCALE: tl.constexpr, STRIDE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    NV=(V+VB-1)//VB
    j,z=p%NV,p//NV
    c,bh=z%NC,z//NC
    b,h=bh//HV,bh%HV
    kh=h//(HV//H)
    r,x,y=tl.arange(0,BC),tl.arange(0,BK),j*VB+tl.arange(0,VB)
    t=c*BC+r
    mt,mx,my=t<T,x<K,y<V
    qp=((b*T+t)*H+kh)*K
    qb=tl.load(q+qp[:,None]+x[None,:],mask=mt[:,None]&mx[None,:],other=0.)
    kb=tl.load(k+qp[:,None]+x[None,:],mask=mt[:,None]&mx[None,:],other=0.)
    vb=tl.load(v+((b*T+t)*HV+h)[:,None]*V+y[None,:],mask=mt[:,None]&my[None,:],other=0.)
    dob=tl.load(do+((b*T+t)*HV+h)[:,None]*V+y[None,:],mask=mt[:,None]&my[None,:],other=0.)
    db=tl.load(d+((b*T+t)*HV+h)[:,None]*V+y[None,:],mask=mt[:,None]&my[None,:],other=0.)
    hb=tl.load(hs+((((b*NC+c)*HV+h)*K+x[:,None])*V+y[None,:]),mask=mx[:,None]&my[None,:],other=0.)
    dh=tl.load(dhs+((((b*NC+c)*HV+h)*K+x[:,None])*V+y[None,:]),mask=mx[:,None]&my[None,:],other=0.)
    gb=tl.load(g+(b*T+t)*HV+h,mask=mt,other=0.).to(tl.float32)
    bb=tl.load(beta+(b*T+t)*HV+h,mask=mt,other=0.).to(tl.float32)
    gc=tl.cumsum(gb,0); ge=tl.sum(tl.where(mt,gb,0.),0); ee=tl.exp(gc)
    lower=(r[:,None]>=r[None,:])&mt[:,None]&mt[None,:]
    strict=(r[:,None]>r[None,:])&mt[:,None]&mt[None,:]
    ef=tl.exp(gc[:,None]-gc[None,:])
    qk=tl.dot(qb,tl.trans(kb),input_precision='tf32')
    sm=tl.where(lower,qk*ef*SCALE,0.)
    kbar=kb*tl.exp(ge-gc)[:,None]
    dd=tl.dot(tl.trans(sm).to(dob.dtype),dob,input_precision='tf32')+tl.dot(kbar.to(kb.dtype),dh.to(dob.dtype),input_precision='tf32')
    dsm=tl.dot(dob,tl.trans(db.to(dob.dtype)),input_precision='tf32')
    dkb=tl.dot(db,tl.trans(dh),input_precision='tf32')
    deq=tl.dot(dob,tl.trans(hb.to(dob.dtype)),input_precision='tf32')
    ai=tl.load(a+((((b*NC+c)*HV+h)*BC+r[:,None])*BC+r[None,:]),mask=mt[:,None]&mt[None,:],other=0.)
    zz=vb*bb[:,None]; xx=kb*(bb*ee)[:,None]
    dw=-tl.dot(dd,tl.trans(hb),input_precision='tf32')
    dz=tl.dot(tl.trans(ai),dd,input_precision='tf32')
    dx=tl.dot(tl.trans(ai),dw.to(ai.dtype),input_precision='tf32')
    da=tl.dot(dd,tl.trans(zz),input_precision='tf32')+tl.dot(dw.to(xx.dtype),tl.trans(xx),input_precision='tf32')
    dl=-tl.dot(tl.trans(ai),tl.dot(da,tl.trans(ai),input_precision='tf32'),input_precision='tf32')
    dl=tl.where(strict,dl,0.)
    kk=tl.dot(kb,tl.trans(kb),input_precision='tf32')
    rc=tl.where(lower,dsm*ef*SCALE,0.)
    lc=dl*tl.where(strict,bb[:,None]*ef,0.)
    dq=deq*(ee*SCALE)[:,None]+tl.dot(rc.to(kb.dtype),kb,input_precision='tf32')
    dk=tl.dot(tl.trans(rc).to(qb.dtype),qb,input_precision='tf32')+dkb*tl.exp(ge-gc)[:,None]
    dk+=dx*(bb*ee)[:,None]+tl.dot(lc.to(kb.dtype),kb,input_precision='tf32')+tl.dot(tl.trans(lc).to(kb.dtype),kb,input_precision='tf32')
    dvb=dz*bb[:,None]
    dbeta=tl.sum(dz*vb,1)+tl.sum(dx*(ee[:,None]*kb),1)
    dbeta+=tl.sum(dl*tl.where(strict,kk*ef,0.),1)
    dG=tl.sum(deq*(ee*SCALE)[:,None]*qb,1)+tl.sum(dx*xx,1)-tl.sum(dkb*kbar,1)
    last=tl.sum(mt.to(tl.int32),0)-1
    dG+=tl.where(r==last,tl.sum(dkb*kbar),0.)
    tmp=dsm*sm+dl*tl.where(strict,kk*bb[:,None]*ef,0.)
    dG+=tl.sum(tmp,1)-tl.sum(tmp,0)
    dg=tl.flip(tl.cumsum(tl.flip(dG,0),0),0)
    po=(((j*STRIDE+b*T+t[:,None])*HV+h)*K+x[None,:])
    so=(j*STRIDE+b*T+t)*HV+h
    tl.store(pq+po,dq,mask=mt[:,None]&mx[None,:]); tl.store(pk+po,dk,mask=mt[:,None]&mx[None,:])
    tl.store(pg+so,dg,mask=mt); tl.store(pb+so,dbeta,mask=mt)
    tl.store(dv+((b*T+t)*HV+h)[:,None]*V+y[None,:],dvb,mask=mt[:,None]&my[None,:])


@triton.jit
def _reduce(pq,pk,pg,pb,dq,dk,dg,db,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,
            K:tl.constexpr,BK:tl.constexpr,NV:tl.constexpr,STRIDE:tl.constexpr):
    p=tl.program_id(0).to(tl.int64)
    t,z=p%T,p//T
    h,b=z%HV,z//HV
    x=tl.arange(0,BK)
    qs=tl.zeros([BK],tl.float32); ks=tl.zeros([BK],tl.float32); gs=0.; bs=0.
    for j in range(NV):
        po=(((j*STRIDE+b*T+t)*HV+h)*K+x)
        qs+=tl.load(pq+po,mask=x<K,other=0.); ks+=tl.load(pk+po,mask=x<K,other=0.)
        so=(j*STRIDE+b*T+t)*HV+h
        gs+=tl.load(pg+so); bs+=tl.load(pb+so)
    tl.store(dg+(b*T+t)*HV+h,gs); tl.store(db+(b*T+t)*HV+h,bs)
    # q/k value-head partials occupy consecutive output heads for deterministic grouped reduction.
    kh=h//(HV//H)
    qo=((b*T+t)*HV+h)*K+x
    tl.store(dq+qo,qs,mask=x<K); tl.store(dk+qo,ks,mask=x<K)


@triton.jit
def _group(dqv,dkv,dq,dk,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,BK:tl.constexpr):
    p=tl.program_id(0).to(tl.int64); t,z=p%T,p//T; h,b=z%H,z//H
    x=tl.arange(0,BK); qs=tl.zeros([BK],tl.float32); ks=tl.zeros([BK],tl.float32)
    for j in range(HV//H):
        vh=h*(HV//H)+j; off=((b*T+t)*HV+vh)*K+x
        qs+=tl.load(dqv+off,mask=x<K,other=0.); ks+=tl.load(dkv+off,mask=x<K,other=0.)
    off=((b*T+t)*H+h)*K+x
    tl.store(dq+off,qs,mask=x<K); tl.store(dk+off,ks,mask=x<K)


class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,g,beta):
        B,T,H,K=q.shape; HV,V=v.shape[2:]; NC=triton.cdiv(T,BC); BK=triton.next_power_of_2(K); NV=triton.cdiv(V,BV)
        a=torch.empty((B,NC,HV,BC,BC),device=q.device,dtype=torch.float32)
        w=torch.empty((B,T,HV,K),device=q.device,dtype=k.dtype)
        hs=torch.empty((B,NC,HV,K,V),device=q.device,dtype=torch.float32)
        d=torch.empty((B,T,HV,V),device=q.device,dtype=torch.float32)
        o=torch.empty((B,T,HV,V),device=q.device,dtype=v.dtype)
        _wy[(NC,B*HV)](k,g,beta,a,w,T=T,H=H,HV=HV,K=K,BK=BK,NC=NC,num_warps=4)
        _fscan[(B*HV*NV,)](k,v,g,beta,a,w,hs,d,T=T,H=H,HV=HV,K=K,V=V,BK=BK,VB=BV,NC=NC,num_warps=4)
        _out[(B*HV*NC*NV,)](q,k,g,hs,d,o,T=T,H=H,HV=HV,K=K,V=V,BK=BK,VB=BV,NC=NC,SCALE=K**-.5,num_warps=4)
        ctx.save_for_backward(q,k,v,g,beta,a,w,hs,d); ctx.meta=(B,T,H,HV,K,V,NC,BK,NV)
        return o
    @staticmethod
    def backward(ctx,do):
        q,k,v,g,beta,a,w,hs,d=ctx.saved_tensors; B,T,H,HV,K,V,NC,BK,NV=ctx.meta; do=do.contiguous()
        dhs=torch.empty_like(hs,dtype=torch.float32); dv=torch.empty_like(v); dg=torch.empty_like(g); db=torch.empty_like(beta)
        STRIDE=B*T
        pq=torch.empty((NV,STRIDE,HV,K),device=q.device,dtype=torch.float32); pk=torch.empty_like(pq,dtype=torch.float32)
        pg=torch.empty((NV,STRIDE,HV),device=q.device,dtype=torch.float32); pb=torch.empty_like(pg,dtype=torch.float32)
        dqv=torch.empty((B,T,HV,K),device=q.device,dtype=q.dtype); dkv=torch.empty_like(dqv)
        dq=torch.empty_like(q); dk=torch.empty_like(k)
        _ascan[(B*HV*NV,)](q,k,g,w,do,dhs,T=T,H=H,HV=HV,K=K,V=V,BK=BK,VB=BV,NC=NC,SCALE=K**-.5,num_warps=4)
        _grad[(B*HV*NC*NV,)](q,k,v,g,beta,a,hs,d,dhs,do,pq,pk,dv,pg,pb,T=T,H=H,HV=HV,K=K,V=V,BK=BK,VB=BV,NC=NC,SCALE=K**-.5,STRIDE=STRIDE,num_warps=4)
        _reduce[(B*T*HV,)](pq,pk,pg,pb,dqv,dkv,dg,db,T=T,H=H,HV=HV,K=K,BK=BK,NV=NV,STRIDE=STRIDE,num_warps=4)
        _group[(B*T*H,)](dqv,dkv,dq,dk,T=T,H=H,HV=HV,K=K,BK=BK,num_warps=4)
        return dq,dk,dv,dg,db


def gated_delta_rule(q,k,v,g,beta):
    return _Fn.apply(q,k,v,g,beta)
