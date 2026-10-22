import torch
import triton
import triton.language as tl

CHUNK = 16
VALUE_BLOCK = 32

@triton.jit
def _prepare_chunks(k, g, beta, A, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
    pc = tl.program_id(0).to(tl.int64)
    ph = tl.program_id(1).to(tl.int64)
    nc = tl.cdiv(T, BT)
    pb, ci = pc // nc, pc % nc
    kh = ph // (HV // H)
    tt, kk = tl.arange(0, BT), tl.arange(0, BK)
    t, tm, km = ci * BT + tt, ci * BT + tt < T, kk < K
    kb = tl.load(k + ((pb*T+t[:,None])*H+kh)*K + kk[None,:], mask=tm[:,None]&km[None,:], other=0.).to(tl.float32)
    gb = tl.load(g + (pb*T+t)*HV+ph, mask=tm, other=0.).to(tl.float32)
    bb = tl.load(beta + (pb*T+t)*HV+ph, mask=tm, other=0.).to(tl.float32)
    gc = tl.cumsum(gb, axis=0)
    gram = tl.dot(kb, tl.trans(kb), input_precision='ieee')
    r, c = tt[:,None], tt[None,:]
    L = tl.where((r>c)&tm[:,None]&tm[None,:], bb[:,None]*tl.exp(gc[:,None]-gc[None,:])*gram, 0.)
    inv = tl.where(r==c, 1., 0.)
    for i in range(1, BT):
        row = tl.where(tt == i, 1., 0.)
        for j in range(i):
            lij = tl.sum(tl.where((r==i)&(c==j), L, 0.))
            old = tl.sum(tl.where(r==j, inv, 0.), axis=0)
            row -= lij * old
        inv = tl.where(r==i, row[None,:], inv)
    base = ((pb*nc+ci)*HV+ph)*BT*BT
    tl.store(A+base+r*BT+c, inv, mask=tm[:,None]&tm[None,:])

@triton.jit
def _prepare_values(v, beta, A, U, T: tl.constexpr, HV: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
    pc = tl.program_id(0).to(tl.int64)
    ph = tl.program_id(1).to(tl.int64)
    pv = tl.program_id(2).to(tl.int64)
    nc = tl.cdiv(T, BT)
    pb, ci = pc // nc, pc % nc
    tt, vv = tl.arange(0, BT), pv * BV + tl.arange(0, BV)
    t, tm, vm = ci * BT + tt, ci * BT + tt < T, vv < V
    r, c = tt[:,None], tt[None,:]
    base = ((pb*nc+ci)*HV+ph)*BT*BT
    inv = tl.load(A+base+r*BT+c, mask=tm[:,None]&tm[None,:], other=0.)
    bb = tl.load(beta + (pb*T+t)*HV+ph, mask=tm, other=0.).to(tl.float32)
    vb = tl.load(v + ((pb*T+t[:,None])*HV+ph)*V + vv[None,:], mask=tm[:,None]&vm[None,:], other=0.).to(tl.float32)
    u = tl.dot(inv, vb*bb[:,None], input_precision='ieee')
    tl.store(U + ((pb*T+t[:,None])*HV+ph)*V + vv[None,:], u, mask=tm[:,None]&vm[None,:])

@triton.jit
def _state_fwd(k, g, beta, A, U, states, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv, nc = tl.cdiv(V,BV), tl.cdiv(T,BT)
    phv, pv = p//nv, p%nv
    pb, ph = phv//HV, phv%HV
    kh = ph//(HV//H)
    kk, vv, tt = tl.arange(0,BK), pv*BV+tl.arange(0,BV), tl.arange(0,BT)
    km, vm = kk<K, vv<V
    s = tl.zeros((BK,BV), tl.float32)
    r,c = tt[:,None],tt[None,:]
    for ci in range(nc):
        t, tm = ci*BT+tt, ci*BT+tt<T
        sb = (((pb*nc+ci)*HV+ph)*K).to(tl.int64)*V
        tl.store(states+sb+kk[:,None]*V+vv[None,:], s, mask=km[:,None]&vm[None,:])
        kb = tl.load(k+((pb*T+t[:,None])*H+kh)*K+kk[None,:], mask=tm[:,None]&km[None,:], other=0.).to(tl.float32)
        gb = tl.load(g+(pb*T+t)*HV+ph, mask=tm, other=0.).to(tl.float32)
        bb = tl.load(beta+(pb*T+t)*HV+ph, mask=tm, other=0.).to(tl.float32)
        gc = tl.cumsum(gb,axis=0)
        ab=((pb*nc+ci)*HV+ph)*BT*BT
        inv=tl.load(A+ab+r*BT+c,mask=tm[:,None]&tm[None,:],other=0.)
        w=tl.dot(inv,kb*(bb*tl.exp(gc))[:,None],input_precision='ieee')
        ub=tl.load(U+((pb*T+t[:,None])*HV+ph)*V+vv[None,:],mask=tm[:,None]&vm[None,:],other=0.)
        ub-=tl.dot(w,s,input_precision='ieee')
        tl.store(U+((pb*T+t[:,None])*HV+ph)*V+vv[None,:],ub,mask=tm[:,None]&vm[None,:])
        last=tl.minimum((ci+1)*BT,T)-ci*BT-1
        gl=tl.sum(tl.where(tt==last,gc,0.))
        s=s*tl.exp(gl)+tl.dot(tl.trans(kb),ub*tl.exp(gl-gc)[:,None],input_precision='ieee')

@triton.jit
def _chunk_output(q,k,g,U,states,o,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BT:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr,SCALE:tl.constexpr):
    pc=tl.program_id(0).to(tl.int64); p=tl.program_id(1).to(tl.int64)
    nv,nc=tl.cdiv(V,BV),tl.cdiv(T,BT); phv,pv=p//nv,p%nv; pb,ph=phv//HV,phv%HV; ci=pc%nc; kh=ph//(HV//H)
    tt,kk,vv=tl.arange(0,BT),tl.arange(0,BK),pv*BV+tl.arange(0,BV); t=ci*BT+tt; tm=t<T; km=kk<K; vm=vv<V
    base=((pb*T+t[:,None])*H+kh)*K
    qb=tl.load(q+base+kk[None,:],mask=tm[:,None]&km[None,:],other=0.).to(tl.float32)*SCALE
    kb=tl.load(k+base+kk[None,:],mask=tm[:,None]&km[None,:],other=0.).to(tl.float32)
    gc=tl.cumsum(tl.load(g+(pb*T+t)*HV+ph,mask=tm,other=0.).to(tl.float32),axis=0)
    sb=(((pb*nc+ci)*HV+ph)*K).to(tl.int64)*V
    s=tl.load(states+sb+kk[:,None]*V+vv[None,:],mask=km[:,None]&vm[None,:],other=0.)
    ub=tl.load(U+((pb*T+t[:,None])*HV+ph)*V+vv[None,:],mask=tm[:,None]&vm[None,:],other=0.)
    score=tl.dot(qb,tl.trans(kb),input_precision='ieee')
    score*=tl.where(tt[:,None]>=tt[None,:],tl.exp(gc[:,None]-gc[None,:]),0.)
    y=tl.dot(qb*tl.exp(gc)[:,None],s,input_precision='ieee')+tl.dot(score,ub,input_precision='ieee')
    tl.store(o+((pb*T+t[:,None])*HV+ph)*V+vv[None,:],y,mask=tm[:,None]&vm[None,:])

@triton.jit
def _chunk_bwd(q,k,v,g,beta,do,U,states,dqp,dkp,sp,dv,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BT:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr,SCALE:tl.constexpr):
    p=tl.program_id(0).to(tl.int64); nv,nc=tl.cdiv(V,BV),tl.cdiv(T,BT); phv,pv=p//nv,p%nv; pb,ph=phv//HV,phv%HV; kh=ph//(HV//H)
    kk,vv,tt=tl.arange(0,BK),pv*BV+tl.arange(0,BV),tl.arange(0,BT); km,vm=kk<K,vv<V; ds=tl.zeros((BK,BV),tl.float32)
    for z in range(nc):
        ci=nc-1-z; t=ci*BT+tt; tm=t<T
        sb=(((pb*nc+ci)*HV+ph)*K).to(tl.int64)*V
        s0=tl.load(states+sb+kk[:,None]*V+vv[None,:],mask=km[:,None]&vm[None,:],other=0.)
        kb=tl.load(k+((pb*T+t[:,None])*H+kh)*K+kk[None,:],mask=tm[:,None]&km[None,:],other=0.).to(tl.float32)
        qb=tl.load(q+((pb*T+t[:,None])*H+kh)*K+kk[None,:],mask=tm[:,None]&km[None,:],other=0.).to(tl.float32)*SCALE
        gb=tl.load(g+(pb*T+t)*HV+ph,mask=tm,other=0.).to(tl.float32); gc=tl.cumsum(gb,axis=0)
        bb=tl.load(beta+(pb*T+t)*HV+ph,mask=tm,other=0.).to(tl.float32)
        xb=((pb*T+t[:,None])*HV+ph)*V
        ub=tl.load(U+xb+vv[None,:],mask=tm[:,None]&vm[None,:],other=0.)
        vb=tl.load(v+xb+vv[None,:],mask=tm[:,None]&vm[None,:],other=0.).to(tl.float32)
        ob=tl.load(do+xb+vv[None,:],mask=tm[:,None]&vm[None,:],other=0.).to(tl.float32)
        for z2 in range(BT):
            i=BT-1-z2; valid=ci*BT+i<T; gi=tl.sum(tl.where(tt==i,gc,0.))
            pre=s0*tl.exp(gi)+tl.dot(tl.trans(kb),tl.where((tt<i)[:,None],ub*tl.exp(gi-gc)[:,None],0.),input_precision='ieee')
            ki=tl.sum(tl.where(tt[:,None]==i,kb,0.),axis=0); ui=tl.sum(tl.where(tt[:,None]==i,ub,0.),axis=0)
            vi=tl.sum(tl.where(tt[:,None]==i,vb,0.),axis=0); oi=tl.sum(tl.where(tt[:,None]==i,ob,0.),axis=0); qi=tl.sum(tl.where(tt[:,None]==i,qb,0.),axis=0); bi=tl.sum(tl.where(tt==i,bb,0.))
            dqi=tl.sum((pre+ki[:,None]*ui[None,:])*oi[None,:],axis=1)*SCALE
            ds+=qi[:,None]*oi[None,:]; da=tl.sum(ki[:,None]*ds,axis=0); err=vi-tl.sum(ki[:,None]*pre,axis=0); dbi=tl.sum(da*err); de=bi*da
            dki=tl.sum(ds*ui[None,:],axis=1)-tl.sum(pre*de[None,:],axis=1); dsp=ds-ki[:,None]*de[None,:]; dgi=tl.sum(dsp*pre); ds=tl.exp(tl.sum(tl.where(tt==i,gb,0.)))*dsp
            pbq=((((pb*T+ci*BT+i)*HV+ph)*nv+pv)*K).to(tl.int64); tl.store(dqp+pbq+kk,dqi,mask=valid&km); tl.store(dkp+pbq+kk,dki,mask=valid&km)
            pbs=((((pb*T+ci*BT+i)*HV+ph)*nv+pv)*2).to(tl.int64); tl.store(sp+pbs,dgi,mask=valid); tl.store(sp+pbs+1,dbi,mask=valid)
            pbd=(((pb*T+ci*BT+i)*HV+ph)*V).to(tl.int64); tl.store(dv+pbd+vv,de,mask=valid&vm)

@triton.jit
def _reduce_qk(dqp,dkp,dq,dk,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    pt=tl.program_id(0).to(tl.int64); kh=tl.program_id(1); pb,t=pt//T,pt%T; nv=tl.cdiv(V,BV); rep=HV//H; kk=tl.arange(0,BK); km=kk<K; aq=tl.zeros((BK,),tl.float32); ak=tl.zeros((BK,),tl.float32)
    for r in range(rep):
        ph=kh*rep+r
        for pv in range(nv):
            base=((((pb*T+t)*HV+ph)*nv+pv)*K).to(tl.int64); aq+=tl.load(dqp+base+kk,mask=km,other=0.); ak+=tl.load(dkp+base+kk,mask=km,other=0.)
    out=((pb*T+t)*H+kh)*K; tl.store(dq+out+kk,aq,mask=km); tl.store(dk+out+kk,ak,mask=km)

@triton.jit
def _reduce_s(sp,dg,db,V:tl.constexpr,BV:tl.constexpr):
    p=tl.program_id(0).to(tl.int64); nv=tl.cdiv(V,BV); ag=0.; ab=0.
    for pv in range(nv):
        ag+=tl.load(sp+(p*nv+pv)*2); ab+=tl.load(sp+(p*nv+pv)*2+1)
    tl.store(dg+p,ag); tl.store(db+p,ab)

class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,g,beta):
        B,T,H,K=q.shape; HV,V=v.shape[2:]; nc=triton.cdiv(T,CHUNK); nv=triton.cdiv(V,VALUE_BLOCK); bk=triton.next_power_of_2(K)
        A=torch.empty((B,nc,HV,CHUNK,CHUNK),device=q.device,dtype=torch.float32); U=torch.empty((B,T,HV,V),device=q.device,dtype=torch.float32); states=torch.empty((B,nc,HV,K,V),device=q.device,dtype=torch.float32); out=torch.empty_like(v)
        _prepare_chunks[(B*nc,HV)](k,g,beta,A,T=T,H=H,HV=HV,K=K,BT=CHUNK,BK=bk,num_warps=4)
        _prepare_values[(B*nc,HV,nv)](v,beta,A,U,T=T,HV=HV,V=V,BT=CHUNK,BV=VALUE_BLOCK,num_warps=4)
        _state_fwd[(B*HV*nv,)](k,g,beta,A,U,states,T=T,H=H,HV=HV,K=K,V=V,BT=CHUNK,BK=bk,BV=VALUE_BLOCK,num_warps=8)
        _chunk_output[(B*nc,B*HV*nv)](q,k,g,U,states,out,T=T,H=H,HV=HV,K=K,V=V,BT=CHUNK,BK=bk,BV=VALUE_BLOCK,SCALE=K**-.5,num_warps=4)
        ctx.save_for_backward(q,k,v,g,beta,U,states); return out
    @staticmethod
    def backward(ctx,do):
        q,k,v,g,beta,U,states=ctx.saved_tensors; B,T,H,K=q.shape; HV,V=v.shape[2:]; nv=triton.cdiv(V,VALUE_BLOCK); bk=triton.next_power_of_2(K)
        do=do.contiguous()
        dqp=torch.empty((B,T,HV,nv,K),device=q.device,dtype=torch.float32); dkp=torch.empty_like(dqp); sp=torch.empty((B,T,HV,nv,2),device=q.device,dtype=torch.float32); dq=torch.empty_like(q); dk=torch.empty_like(k); dv=torch.empty_like(v); dg=torch.empty_like(g); db=torch.empty_like(beta)
        _chunk_bwd[(B*HV*nv,)](q,k,v,g,beta,do,U,states,dqp,dkp,sp,dv,T=T,H=H,HV=HV,K=K,V=V,BT=CHUNK,BK=bk,BV=VALUE_BLOCK,SCALE=K**-.5,num_warps=4)
        _reduce_qk[(B*T,H)](dqp,dkp,dq,dk,T=T,H=H,HV=HV,K=K,V=V,BK=bk,BV=VALUE_BLOCK,num_warps=4); _reduce_s[(B*T*HV,)](sp,dg,db,V=V,BV=VALUE_BLOCK,num_warps=1)
        return dq,dk,dv,dg,db

def gated_delta_rule(q,k,v,g,beta):
    return _GatedDeltaRule.apply(q,k,v,g,beta)
