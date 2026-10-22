import torch
import triton
import triton.language as tl
C=16
BV=16

@triton.jit
def _prep(k,g,beta,A,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,BK:tl.constexpr,BT:tl.constexpr):
 c0=tl.program_id(0).to(tl.int64); bh=tl.program_id(1).to(tl.int64); b=bh//HV; h=bh%HV
 r=tl.arange(0,BT); x=tl.arange(0,BK); t=c0*BT+r; m=t<T; kh=h//(HV//H)
 z=tl.load(k+((b*T+t[:,None])*H+kh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.)
 gg=tl.load(g+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32); cs=tl.cumsum(gg,0)
 be=tl.load(beta+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32)
 L=tl.dot(z,tl.trans(z),input_precision='ieee')*be[:,None]*tl.exp(cs[:,None]-cs[None,:])
 L=tl.where((r[:,None]>r[None,:])&m[:,None]&m[None,:],L,0.); I=-L
 for i in range(2,BT):
  row=tl.sum(tl.where((r==i)[:,None],-L,0.),0); row=tl.where(r<i,row,0.); row+=tl.sum(row[:,None]*I,0); I=tl.where((r==i)[:,None],row,I)
 I=I+tl.where(r[:,None]==r[None,:],1.0,0.0); base=((b*((T+BT-1)//BT)+c0)*HV+h)*BT*BT
 tl.store(A+base+r[:,None]*BT+r[None,:],I,mask=m[:,None]&m[None,:])

@triton.jit
def _state(k,v,g,beta,A,S,U,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV_:tl.constexpr,BT:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); nv=(V+BV_-1)//BV_; bh=p//nv; iv=p%nv; b=bh//HV; h=bh%HV; kh=h//(HV//H)
 x=tl.arange(0,BK); y=iv*BV_+tl.arange(0,BV_); r=tl.arange(0,BT); st=tl.zeros((BK,BV_),tl.float32)
 for c in range(NC):
  t=c*BT+r; m=t<T; tl.store(S+((((b*NC+c)*HV+h)*K+x[:,None])*V+y[None,:]),st,mask=(x[:,None]<K)&(y[None,:]<V))
  km=tl.load(k+((b*T+t[:,None])*H+kh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.); vm=tl.load(v+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.)
  cs=tl.cumsum(tl.load(g+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32),0); e=tl.exp(cs); be=tl.load(beta+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32)
  base=((b*NC+c)*HV+h)*BT*BT; ai=tl.load(A+base+r[:,None]*BT+r[None,:],mask=m[:,None]&m[None,:],other=0.)
  u=tl.dot(ai,(vm-tl.dot(km*e[:,None],st,input_precision='ieee'))*be[:,None],input_precision='ieee')
  tl.store(U+((b*T+t[:,None])*HV+h)*V+y[None,:],u,mask=m[:,None]&(y[None,:]<V))
  last=tl.maximum(tl.minimum(BT,T-c*BT)-1,0); total=tl.sum(tl.where(r==last,cs,0.),0); d=tl.exp(total); st=d*st+tl.dot(tl.trans(km*tl.exp(total-cs)[:,None]),u,input_precision='ieee')

@triton.jit
def _out(q,k,g,S,U,o,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV_:tl.constexpr,BT:tl.constexpr,SC:tl.constexpr):
 c=tl.program_id(0).to(tl.int64); p=tl.program_id(1).to(tl.int64); nv=(V+BV_-1)//BV_; bh=p//nv; iv=p%nv; b=bh//HV; h=bh%HV; kh=h//(HV//H)
 r=tl.arange(0,BT); x=tl.arange(0,BK); y=iv*BV_+tl.arange(0,BV_); t=c*BT+r; m=t<T
 qm=tl.load(q+((b*T+t[:,None])*H+kh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.)*SC; km=tl.load(k+((b*T+t[:,None])*H+kh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.)
 cs=tl.cumsum(tl.load(g+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32),0); sm=tl.load(S+((((b*NC+c)*HV+h)*K+x[:,None])*V+y[None,:]),mask=(x[:,None]<K)&(y[None,:]<V),other=0.); um=tl.load(U+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.)
 ca=(r[:,None]>=r[None,:])&m[:,None]&m[None,:]; w=tl.dot(qm,tl.trans(km),input_precision='ieee')*tl.where(ca,tl.exp(cs[:,None]-cs[None,:]),0.)
 z=tl.dot(qm*tl.exp(cs)[:,None],sm,input_precision='ieee')+tl.dot(w.to(um.dtype),um,input_precision='ieee'); tl.store(o+((b*T+t[:,None])*HV+h)*V+y[None,:],z,mask=m[:,None]&(y[None,:]<V))

@triton.jit
def _bwd(q,k,v,g,beta,A,S,U,do,PQ,PK,DV,PG,PB,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,NV:tl.constexpr,BK:tl.constexpr,BV_:tl.constexpr,BT:tl.constexpr,SC:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); bh=p//NV; iv=p%NV; b=bh//HV; h=bh%HV; kh=h//(HV//H); x=tl.arange(0,BK); y=iv*BV_+tl.arange(0,BV_); r=tl.arange(0,BT); ds=tl.zeros((BK,BV_),tl.float32)
 for rev in range(NC):
  c=NC-1-rev; t=c*BT+r; m=t<T
  qm=tl.load(q+((b*T+t[:,None])*H+kh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.)*SC; km=tl.load(k+((b*T+t[:,None])*H+kh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.); vm=tl.load(v+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.); dm=tl.load(do+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.).to(tl.float32)
  cs=tl.cumsum(tl.load(g+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32),0); e=tl.exp(cs); be=tl.load(beta+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32); sm=tl.load(S+((((b*NC+c)*HV+h)*K+x[:,None])*V+y[None,:]),mask=(x[:,None]<K)&(y[None,:]<V),other=0.); um=tl.load(U+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.); ai=tl.load(A+((b*NC+c)*HV+h)*BT*BT+r[:,None]*BT+r[None,:],mask=m[:,None]&m[None,:],other=0.)
  last=tl.maximum(tl.minimum(BT,T-c*BT)-1,0); total=tl.sum(tl.where(r==last,cs,0.),0); d=tl.exp(total); j=km*tl.exp(total-cs)[:,None]; dsm=d*ds; du=tl.dot(j,ds,input_precision='ieee'); dj=tl.dot(um,tl.trans(ds),input_precision='ieee'); dkm=dj*tl.exp(total-cs)[:,None]; dc=-tl.sum(dj*j,1); dc+=tl.where(r==last,tl.sum(dj*j)+tl.sum(ds*(d*sm)),0.)
  eq=qm*e[:,None]; deq=tl.dot(dm,tl.trans(sm),input_precision='ieee'); dsm+=tl.dot(tl.trans(eq),dm,input_precision='ieee'); dqm=deq*e[:,None]; dc+=tl.sum(deq*eq,1)
  ca=(r[:,None]>=r[None,:])&m[:,None]&m[None,:]; ratio=tl.where(ca,tl.exp(cs[:,None]-cs[None,:]),0.); coef=tl.dot(qm,tl.trans(km),input_precision='ieee')*ratio; du+=tl.dot(tl.trans(coef.to(dm.dtype)),dm,input_precision='ieee'); dcoef=tl.dot(dm,tl.trans(um),input_precision='ieee'); z=dcoef*ratio; dqm+=tl.dot(z.to(km.dtype),km,input_precision='ieee'); dkm+=tl.dot(tl.trans(z.to(qm.dtype)),qm,input_precision='ieee'); zc=dcoef*coef; dc+=tl.sum(zc,1)-tl.sum(zc,0)
  dx=tl.dot(tl.trans(ai),du,input_precision='ieee'); pred=tl.dot(km*e[:,None],sm,input_precision='ieee'); res=vm-pred; db=tl.sum(dx*res,1); dp=dx*be[:,None]; dv=dp; dek=-tl.dot(dp,tl.trans(sm),input_precision='ieee'); dsm-=tl.dot(tl.trans(km*e[:,None]),dp,input_precision='ieee'); dkm+=dek*e[:,None]; dc+=tl.sum(dek*(km*e[:,None]),1)
  strict=(r[:,None]>r[None,:])&m[:,None]&m[None,:]; dl=tl.where(strict,-tl.dot(dx,tl.trans(um),input_precision='ieee'),0.); gram=tl.dot(km,tl.trans(km),input_precision='ieee'); gr=tl.where(strict,gram*tl.exp(cs[:,None]-cs[None,:]),0.); db+=tl.sum(dl*gr,1); dgmat=dl*be[:,None]; dgram=dgmat*ratio; dkm+=tl.dot(dgram.to(km.dtype),km,input_precision='ieee')+tl.dot(tl.trans(dgram.to(km.dtype)),km,input_precision='ieee'); gz=dgmat*gr; dc+=tl.sum(gz,1)-tl.sum(gz,0); dg=tl.flip(tl.cumsum(tl.flip(dc,0),0),0)
  off=(((b*T+t[:,None])*HV+h)*NV+iv)*K+x[None,:]; tl.store(PQ+off,dqm*SC,mask=m[:,None]&(x[None,:]<K)); tl.store(PK+off,dkm,mask=m[:,None]&(x[None,:]<K)); tl.store(DV+((b*T+t[:,None])*HV+h)*V+y[None,:],dv,mask=m[:,None]&(y[None,:]<V)); so=((b*T+t)*HV+h)*NV+iv; tl.store(PG+so,dg,mask=m); tl.store(PB+so,db,mask=m); ds=dsm

@triton.jit
def _rq(PQ,PK,dq,dk,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,NV:tl.constexpr,BK:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); b=p//(T*H); rem=p%(T*H); t=rem//H; h=rem%H; x=tl.arange(0,BK); a=tl.zeros((BK,),tl.float32); z=tl.zeros((BK,),tl.float32); G=HV//H
 for j in range(G*NV):
  vh=h*G+j//NV; iv=j%NV; off=(((b*T+t)*HV+vh)*NV+iv)*K+x; a+=tl.load(PQ+off,mask=x<K,other=0.); z+=tl.load(PK+off,mask=x<K,other=0.)
 off=((b*T+t)*H+h)*K+x; tl.store(dq+off,a,mask=x<K); tl.store(dk+off,z,mask=x<K)

@triton.jit
def _rs(PG,PB,dg,db,N:tl.constexpr,NV:tl.constexpr,BN:tl.constexpr):
 p=tl.program_id(0)*BN+tl.arange(0,BN); a=tl.zeros((BN,),tl.float32); z=tl.zeros((BN,),tl.float32)
 for j in range(NV): a+=tl.load(PG+p*NV+j,mask=p<N,other=0.); z+=tl.load(PB+p*NV+j,mask=p<N,other=0.)
 tl.store(dg+p,a,mask=p<N); tl.store(db+p,z,mask=p<N)

class F(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,g,beta):
  B,T,H,K=q.shape; HV,V=v.shape[2:]; NC=triton.cdiv(T,C); BK=triton.next_power_of_2(K); NV=triton.cdiv(V,BV); A=torch.empty((B,NC,HV,C,C),device=q.device,dtype=torch.float32); S=torch.empty((B,NC,HV,K,V),device=q.device,dtype=torch.float32); U=torch.empty((B,T,HV,V),device=q.device,dtype=torch.float32); o=torch.empty_like(v)
  _prep[(NC,B*HV)](k,g,beta,A,T=T,H=H,HV=HV,K=K,BK=BK,BT=C,num_warps=4); _state[(B*HV*NV,)](k,v,g,beta,A,S,U,T=T,H=H,HV=HV,K=K,V=V,NC=NC,BK=BK,BV_=BV,BT=C,num_warps=8); _out[(NC,B*HV*NV)](q,k,g,S,U,o,T=T,H=H,HV=HV,K=K,V=V,NC=NC,BK=BK,BV_=BV,BT=C,SC=K**-.5,num_warps=4); ctx.save_for_backward(q,k,v,g,beta,A,S,U); return o
 @staticmethod
 def backward(ctx,do):
  q,k,v,g,beta,A,S,U=ctx.saved_tensors; do=do.contiguous(); B,T,H,K=q.shape; HV,V=v.shape[2:]; NC=triton.cdiv(T,C); NV=triton.cdiv(V,BV); BK=triton.next_power_of_2(K); PQ=torch.empty((B,T,HV,NV,K),device=q.device,dtype=torch.float32); PK=torch.empty_like(PQ); DV=torch.empty_like(v); PG=torch.empty((B,T,HV,NV),device=q.device,dtype=torch.float32); PB=torch.empty_like(PG); dq=torch.empty_like(q); dk=torch.empty_like(k); dg=torch.empty_like(g); db=torch.empty_like(beta)
  _bwd[(B*HV*NV,)](q,k,v,g,beta,A,S,U,do,PQ,PK,DV,PG,PB,T=T,H=H,HV=HV,K=K,V=V,NC=NC,NV=NV,BK=BK,BV_=BV,BT=C,SC=K**-.5,num_warps=8); _rq[(B*T*H,)](PQ,PK,dq,dk,T=T,H=H,HV=HV,K=K,NV=NV,BK=BK,num_warps=4); _rs[(triton.cdiv(B*T*HV,256),)](PG,PB,dg,db,N=B*T*HV,NV=NV,BN=256,num_warps=4); return dq,dk,DV,dg,db

def gated_delta_rule(q,k,v,g,beta): return F.apply(q,k,v,g,beta)
