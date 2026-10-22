import torch
import triton
import triton.language as tl

BC = tl.constexpr(16)
BV = tl.constexpr(32)

@triton.jit
def _build(k,g,beta,cbuf,ainv,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,BK:tl.constexpr,IP:tl.constexpr):
    ic=tl.program_id(0).to(tl.int64); bh=tl.program_id(1).to(tl.int64)
    b,h=bh//HV,bh%HV; qh=h//(HV//H); r=tl.arange(0,BC); x=tl.arange(0,BK); t=ic*BC+r; m=t<T
    gg=tl.load(g+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32); bb=tl.load(beta+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32)
    cc=tl.cumsum(gg,axis=0); tl.store(cbuf+(b*T+t)*HV+h,cc,mask=m)
    kk=tl.load(k+((b*T+t[:,None])*H+qh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.)
    gram=tl.dot(kk,tl.trans(kk),input_precision=IP); rr=r[:,None]; ss=r[None,:]
    mat=tl.where((rr>ss)&m[:,None]&m[None,:],bb[:,None]*tl.exp(cc[:,None]-cc[None,:])*gram,0.)
    aa=tl.where(rr==ss,1.,0.).to(tl.float32)
    for i in tl.static_range(1,BC):
        for j in tl.static_range(0,i):
            z=tl.sum(tl.where((rr==i)&(ss==j),mat,0.)); row=tl.sum(tl.where(rr==j,aa,0.),axis=0)
            aa+=tl.where(rr==i,-z*row[None,:],0.)
    tl.store(ainv+((b*T+t[:,None])*HV+h)*BC+r[None,:],aa,mask=m[:,None])

@triton.jit
def _fscan(k,v,beta,cbuf,ainv,state,u,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,IP:tl.constexpr):
    p=tl.program_id(0).to(tl.int64); nv=tl.cdiv(V,BV); iv=p%nv; bh=p//nv; b,h=bh//HV,bh%HV; qh=h//(HV//H)
    y=iv*BV+tl.arange(0,BV); x=tl.arange(0,BK); r=tl.arange(0,BC); my=y<V; s=tl.zeros((BK,BV),tl.float32); nc=tl.cdiv(T,BC)
    for ic in range(nc):
        t=ic*BC+r; m=t<T; sp=state+((((b*nc+ic)*HV+h)*K+x[:,None])*V+y[None,:]); tl.store(sp,s,mask=(x[:,None]<K)&my[None,:])
        kk=tl.load(k+((b*T+t[:,None])*H+qh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.)
        vv=tl.load(v+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&my[None,:],other=0.).to(tl.float32)
        bb=tl.load(beta+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32); cc=tl.load(cbuf+(b*T+t)*HV+h,mask=m,other=0.)
        rhs=bb[:,None]*(vv-tl.exp(cc)[:,None]*tl.dot(kk,s.to(kk.dtype),input_precision=IP))
        aa=tl.load(ainv+((b*T+t[:,None])*HV+h)*BC+r[None,:],mask=m[:,None],other=0.)
        uu=tl.dot(aa.to(kk.dtype),rhs.to(kk.dtype),input_precision=IP); tl.store(u+((b*T+t[:,None])*HV+h)*V+y[None,:],uu,mask=m[:,None]&my[None,:])
        lastidx=tl.sum(m.to(tl.int32),axis=0)-1; last=tl.sum(tl.where(r==lastidx,cc,0.),axis=0); s=tl.exp(last)*s+tl.dot(tl.trans(kk),(tl.exp(last-cc)[:,None]*uu).to(kk.dtype),input_precision=IP)

@triton.jit
def _emit(q,k,cbuf,state,u,out,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,S:tl.constexpr,IP:tl.constexpr):
    p=tl.program_id(0).to(tl.int64); nv=tl.cdiv(V,BV); nc=tl.cdiv(T,BC); iv=p%nv; ic=(p//nv)%nc; bh=p//(nv*nc); b,h=bh//HV,bh%HV; qh=h//(HV//H)
    r=tl.arange(0,BC); x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); t=ic*BC+r; m=t<T; my=y<V
    qq=tl.load(q+((b*T+t[:,None])*H+qh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.); kk=tl.load(k+((b*T+t[:,None])*H+qh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.)
    cc=tl.load(cbuf+(b*T+t)*HV+h,mask=m,other=0.); ss=tl.load(state+((((b*nc+ic)*HV+h)*K+x[:,None])*V+y[None,:]),mask=(x[:,None]<K)&my[None,:],other=0.)
    uu=tl.load(u+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&my[None,:],other=0.); rr=r[:,None]; jj=r[None,:]
    w=tl.where((rr>=jj)&m[:,None]&m[None,:],tl.exp(cc[:,None]-cc[None,:])*tl.dot(qq,tl.trans(kk),input_precision=IP)*S,0.)
    oo=tl.exp(cc)[:,None]*tl.dot(qq,ss.to(qq.dtype),input_precision=IP)*S+tl.dot(w.to(qq.dtype),uu.to(qq.dtype),input_precision=IP)
    tl.store(out+((b*T+t[:,None])*HV+h)*V+y[None,:],oo,mask=m[:,None]&my[None,:])

@triton.jit
def _ascan(q,k,beta,do,cbuf,ainv,u,dstate,drhs,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,S:tl.constexpr,IP:tl.constexpr):
    p=tl.program_id(0).to(tl.int64); nv=tl.cdiv(V,BV); iv=p%nv; bh=p//nv; b,h=bh//HV,bh%HV; qh=h//(HV//H)
    y=iv*BV+tl.arange(0,BV); x=tl.arange(0,BK); r=tl.arange(0,BC); my=y<V; ds=tl.zeros((BK,BV),tl.float32); nc=tl.cdiv(T,BC)
    for z in range(nc):
        ic=nc-1-z; t=ic*BC+r; m=t<T
        kk=tl.load(k+((b*T+t[:,None])*H+qh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32); qq=tl.load(q+((b*T+t[:,None])*H+qh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
        dd=tl.load(do+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&my[None,:],other=0.).to(tl.float32); cc=tl.load(cbuf+(b*T+t)*HV+h,mask=m,other=0.); lastidx=tl.sum(m.to(tl.int32),axis=0)-1; last=tl.sum(tl.where(r==lastidx,cc,0.),axis=0)
        rr=r[:,None]; jj=r[None,:]; w=tl.where((rr>=jj)&m[:,None]&m[None,:],tl.exp(cc[:,None]-cc[None,:])*tl.dot(qq,tl.trans(kk),input_precision=IP)*S,0.)
        du=tl.dot(tl.trans(w).to(dd.dtype),dd,input_precision=IP)+tl.exp(last-cc)[:,None]*tl.dot(kk,ds.to(kk.dtype),input_precision=IP)
        aa=tl.load(ainv+((b*T+t[:,None])*HV+h)*BC+r[None,:],mask=m[:,None],other=0.); dh=tl.dot(tl.trans(aa).to(du.dtype),du,input_precision=IP)
        tl.store(drhs+((b*T+t[:,None])*HV+h)*V+y[None,:],dh,mask=m[:,None]&my[None,:]); tl.store(dstate+((((b*nc+ic)*HV+h)*K+x[:,None])*V+y[None,:]),ds,mask=(x[:,None]<K)&my[None,:])
        bb=tl.load(beta+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32)
        ds=tl.exp(last)*ds+tl.dot(tl.trans(qq),(tl.exp(cc)[:,None]*dd*S).to(qq.dtype),input_precision=IP)-tl.dot(tl.trans((bb*tl.exp(cc))[:,None]*kk),dh,input_precision=IP)

@triton.jit
def _grad(q,k,v,beta,do,cbuf,state,u,dstate,drhs,pq,pk,dv,pb,pc,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,S:tl.constexpr,IP:tl.constexpr):
    p=tl.program_id(0).to(tl.int64); nv=tl.cdiv(V,BV); nc=tl.cdiv(T,BC); iv=p%nv; ic=(p//nv)%nc; bh=p//(nv*nc); b,h=bh//HV,bh%HV; qh=h//(HV//H)
    r=tl.arange(0,BC); x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); t=ic*BC+r; m=t<T; my=y<V; rr=r[:,None]; jj=r[None,:]
    qq=tl.load(q+((b*T+t[:,None])*H+qh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32); kk=tl.load(k+((b*T+t[:,None])*H+qh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
    vv=tl.load(v+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&my[None,:],other=0.).to(tl.float32); dd=tl.load(do+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&my[None,:],other=0.).to(tl.float32)
    uu=tl.load(u+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&my[None,:],other=0.); dh=tl.load(drhs+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&my[None,:],other=0.)
    bb=tl.load(beta+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32); cc=tl.load(cbuf+(b*T+t)*HV+h,mask=m,other=0.); lastidx=tl.sum(m.to(tl.int32),axis=0)-1; last=tl.sum(tl.where(r==lastidx,cc,0.),axis=0)
    s0=tl.load(state+((((b*nc+ic)*HV+h)*K+x[:,None])*V+y[None,:]),mask=(x[:,None]<K)&my[None,:],other=0.); ds=tl.load(dstate+((((b*nc+ic)*HV+h)*K+x[:,None])*V+y[None,:]),mask=(x[:,None]<K)&my[None,:],other=0.)
    pred=tl.dot(kk,s0.to(kk.dtype),input_precision=IP); dor=tl.dot(dd,tl.trans(uu).to(dd.dtype),input_precision=IP); ew=tl.where((rr>=jj)&m[:,None]&m[None,:],tl.exp(cc[:,None]-cc[None,:]),0.); qk=tl.dot(qq,tl.trans(kk),input_precision=IP)*S; fr=dor*ew
    dq=tl.dot(fr.to(kk.dtype),kk,input_precision=IP)*S+tl.exp(cc)[:,None]*tl.dot(dd,tl.trans(s0).to(dd.dtype),input_precision=IP)*S; dk=tl.dot(tl.trans(fr).to(qq.dtype),qq,input_precision=IP)*S
    dc=tl.sum(dor*ew*qk,axis=1)-tl.sum(dor*ew*qk,axis=0)+tl.sum(dd*(tl.exp(cc)[:,None]*tl.dot(qq,s0.to(qq.dtype),input_precision=IP)*S),axis=1)
    dp=tl.dot(ds,tl.trans(uu).to(ds.dtype),input_precision=IP); wp=tl.exp(last-cc); dk+=wp[:,None]*tl.trans(dp); z=dp*tl.trans(kk)*wp[None,:]; zc=tl.sum(z,axis=0); dc-=zc
    dcarry=tl.sum(ds*(tl.exp(last)*s0)); dc+=tl.where(r==lastidx,tl.sum(zc)+dcarry,0.)
    dm=-tl.dot(dh,tl.trans(uu).to(dh.dtype),input_precision=IP); gram=tl.dot(kk,tl.trans(kk),input_precision=IP); el=tl.where((rr>jj)&m[:,None]&m[None,:],tl.exp(cc[:,None]-cc[None,:]),0.); dl=dm*el*bb[:,None]
    dk+=tl.dot(dl.to(kk.dtype),kk,input_precision=IP)+tl.dot(tl.trans(dl).to(kk.dtype),kk,input_precision=IP); lm=el*bb[:,None]*gram; dc+=tl.sum(dm*lm,axis=1)-tl.sum(dm*lm,axis=0)
    db=tl.sum(dh*(vv-tl.exp(cc)[:,None]*pred),axis=1)+tl.sum(dm*el*gram,axis=1); dval=bb[:,None]*dh; dk-=(bb*tl.exp(cc))[:,None]*tl.dot(dh,tl.trans(s0).to(dh.dtype),input_precision=IP); dc-=tl.sum(dh*((bb*tl.exp(cc))[:,None]*pred),axis=1)
    part=((b*T+t)*HV+h)*nv+iv; tl.store(pq+part[:,None]*K+x[None,:],dq,mask=m[:,None]&(x[None,:]<K)); tl.store(pk+part[:,None]*K+x[None,:],dk,mask=m[:,None]&(x[None,:]<K)); tl.store(dv+((b*T+t[:,None])*HV+h)*V+y[None,:],dval,mask=m[:,None]&my[None,:]); tl.store(pb+part,db,mask=m); tl.store(pc+part,dc,mask=m)

@triton.jit
def _reduce(pq,pk,pb,pc,dq,dk,db,dg,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr):
    p=tl.program_id(0).to(tl.int64); b=p//T; t=p%T; x=tl.arange(0,BK); nv=tl.cdiv(V,BV); rep=HV//H
    for h in range(H):
        aq=tl.zeros((BK,),tl.float32); ak=tl.zeros((BK,),tl.float32)
        for z in range(rep):
            base=((b*T+t)*HV+h*rep+z)*nv
            for iv in range(nv): aq+=tl.load(pq+(base+iv)*K+x,mask=x<K,other=0.); ak+=tl.load(pk+(base+iv)*K+x,mask=x<K,other=0.)
        tl.store(dq+((b*T+t)*H+h)*K+x,aq,mask=x<K); tl.store(dk+((b*T+t)*H+h)*K+x,ak,mask=x<K)
    for h in range(HV):
        base=((b*T+t)*HV+h)*nv; ab=tl.zeros((),tl.float32)
        for iv in range(nv): ab+=tl.load(pb+base+iv)
        tail=tl.zeros((),tl.float32); lane=t%BC
        for z in tl.static_range(BC):
            jt=t-lane+z; val=tl.zeros((),tl.float32)
            for iv in range(nv): val+=tl.load(pc+((b*T+jt)*HV+h)*nv+iv,mask=(jt<T)&(z>=lane),other=0.)
            tail+=val
        tl.store(db+(b*T+t)*HV+h,ab); tl.store(dg+(b*T+t)*HV+h,tail)

class _Fn(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,g,beta):
    B,T,H,K=q.shape; HV,V=v.shape[2:]; BK=triton.next_power_of_2(K); nc=triton.cdiv(T,BC); nv=triton.cdiv(V,BV); ip='ieee' if q.dtype==torch.float32 else 'tf32'
    c=torch.empty((B,T,HV),device=q.device,dtype=torch.float32); a=torch.empty((B,T,HV,BC),device=q.device,dtype=torch.float32); st=torch.empty((B,nc,HV,K,V),device=q.device,dtype=torch.float32); u=torch.empty((B,T,HV,V),device=q.device,dtype=torch.float32); out=torch.empty_like(v)
    _build[(nc,B*HV)](k,g,beta,c,a,T=T,H=H,HV=HV,K=K,BK=BK,IP=ip,num_warps=4); _fscan[(B*HV*nv,)](k,v,beta,c,a,st,u,T=T,H=H,HV=HV,K=K,V=V,BK=BK,IP=ip,num_warps=4); _emit[(B*HV*nc*nv,)](q,k,c,st,u,out,T=T,H=H,HV=HV,K=K,V=V,BK=BK,S=K**-.5,IP=ip,num_warps=4); ctx.save_for_backward(q,k,v,g,beta,c,a,st,u); return out
 @staticmethod
 def backward(ctx,do):
    do=do.contiguous()
    q,k,v,g,beta,c,a,st,u=ctx.saved_tensors; B,T,H,K=q.shape; HV,V=v.shape[2:]; BK=triton.next_power_of_2(K); nc=triton.cdiv(T,BC); nv=triton.cdiv(V,BV); ip='ieee' if q.dtype==torch.float32 else 'tf32'
    ds=torch.empty_like(st); dh=torch.empty_like(u); pq=torch.empty((B,T,HV,nv,K),device=q.device,dtype=torch.float32); pk=torch.empty_like(pq); pb=torch.empty((B,T,HV,nv),device=q.device,dtype=torch.float32); pc=torch.empty_like(pb); dq=torch.empty_like(q); dk=torch.empty_like(k); dv=torch.empty_like(v); db=torch.empty_like(beta); dg=torch.empty_like(g)
    _ascan[(B*HV*nv,)](q,k,beta,do,c,a,u,ds,dh,T=T,H=H,HV=HV,K=K,V=V,BK=BK,S=K**-.5,IP=ip,num_warps=4); _grad[(B*HV*nc*nv,)](q,k,v,beta,do,c,st,u,ds,dh,pq,pk,dv,pb,pc,T=T,H=H,HV=HV,K=K,V=V,BK=BK,S=K**-.5,IP=ip,num_warps=4); _reduce[(B*T,)](pq,pk,pb,pc,dq,dk,db,dg,T=T,H=H,HV=HV,K=K,V=V,BK=BK,num_warps=4); return dq,dk,dv,dg,db

def gated_delta_rule(q,k,v,g,beta): return _Fn.apply(q,k,v,g,beta)
