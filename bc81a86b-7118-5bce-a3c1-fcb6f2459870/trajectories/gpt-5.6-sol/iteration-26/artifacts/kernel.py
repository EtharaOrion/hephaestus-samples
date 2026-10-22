import torch
import triton
import triton.language as tl
C=tl.constexpr(16); BV=tl.constexpr(32)

@triton.jit
def _wy(kp,gp,bp,Gp,Ap,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,BK:tl.constexpr,NC:tl.constexpr):
 c=tl.program_id(0).to(tl.int64); z=tl.program_id(1).to(tl.int64); b=z//HV; hv=z%HV; h=hv//(HV//H)
 t=c*C+tl.arange(0,C); x=tl.arange(0,BK); m=t<T
 k=tl.load(kp+((b*T+t[:,None])*H+h)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
 g=tl.load(gp+(b*T+t)*HV+hv,mask=m,other=0.).to(tl.float32); be=tl.load(bp+(b*T+t)*HV+hv,mask=m,other=0.).to(tl.float32)
 G=tl.cumsum(g,axis=0); r=tl.arange(0,C)[:,None]; s=tl.arange(0,C)[None,:]
 n=tl.where((r>s)&m[:,None]&m[None,:],be[:,None]*tl.exp(G[:,None]-G[None,:])*tl.dot(k,tl.trans(k),input_precision='tf32'),0.)
 a=tl.where(r==s,1.,0.)
 for i in tl.static_range(1,C):
  nr=tl.sum(tl.where(r==i,n,0.),axis=0)
  row=-tl.sum(nr[:,None]*tl.where(r<i,a,0.),axis=0)
  a=tl.where(r==i,row[None,:],a); a=tl.where((r==i)&(s==i),1.,a)
 tl.store(Gp+z*T+t,G,mask=m); tl.store(Ap+((z*NC+c)*C+r)*C+s,a)

@triton.jit
def _fscan(kp,vp,bp,Gp,Ap,Sp,Up,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,NC:tl.constexpr,NV:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); iv=p%NV; z=p//NV; b=z//HV; hv=z%HV; h=hv//(HV//H)
 x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); S=tl.zeros((BK,BV),tl.float32); r=tl.arange(0,C)[:,None]; s=tl.arange(0,C)[None,:]
 for c in range(NC):
  t=c*C+tl.arange(0,C); m=t<T
  tl.store(Sp+((z*NC+c)*K+x[:,None])*V+y[None,:],S,mask=(x[:,None]<K)&(y[None,:]<V))
  k=tl.load(kp+((b*T+t[:,None])*H+h)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
  v=tl.load(vp+((b*T+t[:,None])*HV+hv)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.).to(tl.float32)
  be=tl.load(bp+(b*T+t)*HV+hv,mask=m,other=0.).to(tl.float32); G=tl.load(Gp+z*T+t,mask=m,other=0.)
  rhs=be[:,None]*(v-tl.exp(G)[:,None]*tl.dot(k,S,input_precision='tf32')); a=tl.load(Ap+((z*NC+c)*C+r)*C+s)
  u=tl.dot(a,rhs,input_precision='tf32'); tl.store(Up+(z*T+t[:,None])*V+y[None,:],u,mask=m[:,None]&(y[None,:]<V))
  last=tl.max(tl.where(m,tl.arange(0,C),-1),axis=0); gl=tl.sum(tl.where(tl.arange(0,C)==last,G,0.)); S=tl.exp(gl)*S+tl.dot(tl.trans(tl.exp(gl-G)[:,None]*k),u,input_precision='tf32')

@triton.jit
def _out(qp,kp,Gp,Sp,Up,op,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,NC:tl.constexpr,NV:tl.constexpr,SCALE:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); iv=p%NV; p//=NV; c=p%NC; z=p//NC; b=z//HV; hv=z%HV; h=hv//(HV//H)
 t=c*C+tl.arange(0,C); m=t<T; x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV)
 q=tl.load(qp+((b*T+t[:,None])*H+h)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)*SCALE
 k=tl.load(kp+((b*T+t[:,None])*H+h)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
 G=tl.load(Gp+z*T+t,mask=m,other=0.); S=tl.load(Sp+((z*NC+c)*K+x[:,None])*V+y[None,:],mask=(x[:,None]<K)&(y[None,:]<V),other=0.)
 u=tl.load(Up+(z*T+t[:,None])*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.); r=tl.arange(0,C)[:,None]; s=tl.arange(0,C)[None,:]
 f=tl.where((r>=s)&m[:,None]&m[None,:],tl.exp(G[:,None]-G[None,:])*tl.dot(q,tl.trans(k),input_precision='tf32'),0.)
 o=tl.exp(G)[:,None]*tl.dot(q,S,input_precision='tf32')+tl.dot(f,u,input_precision='tf32')
 tl.store(op+((b*T+t[:,None])*HV+hv)*V+y[None,:],o,mask=m[:,None]&(y[None,:]<V))

@triton.jit
def _ascan(qp,kp,bp,Gp,Ap,dop,DSp,ARp,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,NC:tl.constexpr,NV:tl.constexpr,SCALE:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); iv=p%NV; z=p//NV; b=z//HV; hv=z%HV; h=hv//(HV//H); x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); ds=tl.zeros((BK,BV),tl.float32); r=tl.arange(0,C)[:,None]; s=tl.arange(0,C)[None,:]
 for j in range(NC):
  c=NC-1-j; t=c*C+tl.arange(0,C); m=t<T
  tl.store(DSp+((z*NC+c)*K+x[:,None])*V+y[None,:],ds,mask=(x[:,None]<K)&(y[None,:]<V))
  k=tl.load(kp+((b*T+t[:,None])*H+h)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
  q=tl.load(qp+((b*T+t[:,None])*H+h)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)*SCALE
  do=tl.load(dop+((b*T+t[:,None])*HV+hv)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.).to(tl.float32)
  G=tl.load(Gp+z*T+t,mask=m,other=0.); be=tl.load(bp+(b*T+t)*HV+hv,mask=m,other=0.).to(tl.float32); last=tl.max(tl.where(m,tl.arange(0,C),-1),axis=0); gl=tl.sum(tl.where(tl.arange(0,C)==last,G,0.))
  f=tl.where((r>=s)&m[:,None]&m[None,:],tl.exp(G[:,None]-G[None,:])*tl.dot(q,tl.trans(k),input_precision='tf32'),0.)
  ad=tl.dot(tl.trans(f),do,input_precision='tf32')+tl.exp(gl-G)[:,None]*tl.dot(k,ds,input_precision='tf32')
  a=tl.load(Ap+((z*NC+c)*C+r)*C+s); ar=tl.dot(tl.trans(a),ad,input_precision='tf32'); tl.store(ARp+(z*T+t[:,None])*V+y[None,:],ar,mask=m[:,None]&(y[None,:]<V))
  ds=tl.exp(gl)*ds+tl.dot(tl.trans(tl.exp(G)[:,None]*q),do,input_precision='tf32')-tl.dot(tl.trans((be*tl.exp(G))[:,None]*k),ar,input_precision='tf32')

@triton.jit
def _grad(qp,kp,vp,bp,Gp,Sp,Up,DSp,ARp,dop,PQp,PKp,dvp,Pgp,Pbp,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,NC:tl.constexpr,NV:tl.constexpr,SCALE:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); iv=p%NV; p//=NV; c=p%NC; z=p//NC; b=z//HV; hv=z%HV; h=hv//(HV//H); t=c*C+tl.arange(0,C); m=t<T; x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV)
 q=tl.load(qp+((b*T+t[:,None])*H+h)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)*SCALE; k=tl.load(kp+((b*T+t[:,None])*H+h)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
 v=tl.load(vp+((b*T+t[:,None])*HV+hv)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.).to(tl.float32); do=tl.load(dop+((b*T+t[:,None])*HV+hv)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.).to(tl.float32)
 u=tl.load(Up+(z*T+t[:,None])*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.); ar=tl.load(ARp+(z*T+t[:,None])*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.)
 S=tl.load(Sp+((z*NC+c)*K+x[:,None])*V+y[None,:],mask=(x[:,None]<K)&(y[None,:]<V),other=0.); ds=tl.load(DSp+((z*NC+c)*K+x[:,None])*V+y[None,:],mask=(x[:,None]<K)&(y[None,:]<V),other=0.)
 G=tl.load(Gp+z*T+t,mask=m,other=0.); be=tl.load(bp+(b*T+t)*HV+hv,mask=m,other=0.).to(tl.float32); last=tl.max(tl.where(m,tl.arange(0,C),-1),axis=0); gl=tl.sum(tl.where(tl.arange(0,C)==last,G,0.)); r=tl.arange(0,C)[:,None]; s=tl.arange(0,C)[None,:]; low=(r>=s)&m[:,None]&m[None,:]; st=(r>s)&m[:,None]&m[None,:]
 zu=tl.dot(do,tl.trans(u),input_precision='tf32'); zf=tl.where(low,zu*tl.exp(G[:,None]-G[None,:]),0.); dq=(tl.exp(G)[:,None]*tl.dot(do,tl.trans(S),input_precision='tf32')+tl.dot(zf,k,input_precision='tf32'))*SCALE
 dk=tl.dot(tl.trans(zf),q,input_precision='tf32')+tl.exp(gl-G)[:,None]*tl.dot(u,tl.trans(ds),input_precision='tf32'); ks=tl.dot(k,S,input_precision='tf32'); dk-=(be*tl.exp(G))[:,None]*tl.dot(ar,tl.trans(S),input_precision='tf32')
 dn=-tl.dot(ar,tl.trans(u),input_precision='tf32'); gram=tl.dot(k,tl.trans(k),input_precision='tf32'); cn=tl.where(st,dn*be[:,None]*tl.exp(G[:,None]-G[None,:]),0.); dk+=tl.dot(cn,k,input_precision='tf32')+tl.dot(tl.trans(cn),k,input_precision='tf32')
 dv=be[:,None]*ar; db=tl.sum(ar*(v-tl.exp(G)[:,None]*ks),axis=1)+tl.sum(tl.where(st,dn*tl.exp(G[:,None]-G[None,:])*gram,0.),axis=1)
 qk=tl.dot(q,tl.trans(k),input_precision='tf32'); go=tl.where(low,zu*tl.exp(G[:,None]-G[None,:])*qk,0.); dg=tl.sum(go,axis=1)-tl.sum(go,axis=0)+tl.exp(G)*tl.sum(do*tl.dot(q,S,input_precision='tf32'),axis=1)
 send=tl.exp(gl-G)*tl.sum(u*tl.dot(k,ds,input_precision='tf32'),axis=1); dg-=send; last=tl.max(tl.where(m,tl.arange(0,C),-1),axis=0); dg+=tl.where(tl.arange(0,C)==last,tl.sum(send)+tl.exp(gl)*tl.sum(ds*S),0.); dg-=be*tl.exp(G)*tl.sum(ar*ks,axis=1)
 gn=tl.where(st,dn*be[:,None]*tl.exp(G[:,None]-G[None,:])*gram,0.); dg+=tl.sum(gn,axis=1)-tl.sum(gn,axis=0); dg=tl.sum(dg,axis=0)-tl.cumsum(dg,axis=0)+dg
 po=((z*NV+iv)*T+t[:,None])*K+x[None,:]; tl.store(PQp+po,dq,mask=m[:,None]&(x[None,:]<K)); tl.store(PKp+po,dk,mask=m[:,None]&(x[None,:]<K)); tl.store(dvp+((b*T+t[:,None])*HV+hv)*V+y[None,:],dv,mask=m[:,None]&(y[None,:]<V)); so=(z*NV+iv)*T+t; tl.store(Pgp+so,dg,mask=m); tl.store(Pbp+so,db,mask=m)

@triton.jit
def _rqk(PQ,PK,dq,dk,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,BK:tl.constexpr,NV:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); x=tl.arange(0,BK); bt=p//H; h=p%H; b=bt//T; t=bt%T; a=tl.zeros((BK,),tl.float32); d=tl.zeros((BK,),tl.float32)
 for r in range(HV//H):
  z=b*HV+h*(HV//H)+r
  for iv in range(NV):
   o=((z*NV+iv)*T+t)*K+x; a+=tl.load(PQ+o,mask=x<K,other=0.); d+=tl.load(PK+o,mask=x<K,other=0.)
 o=((b*T+t)*H+h)*K+x; tl.store(dq+o,a,mask=x<K); tl.store(dk+o,d,mask=x<K)

@triton.jit
def _rs(PG,PB,dg,db,T:tl.constexpr,HV:tl.constexpr,NV:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); t=p%T; z=p//T; a=0.; d=0.
 for iv in range(NV):
  o=(z*NV+iv)*T+t; a+=tl.load(PG+o); d+=tl.load(PB+o)
 b=z//HV; hv=z%HV; o=(b*T+t)*HV+hv; tl.store(dg+o,a); tl.store(db+o,d)

class _Fn(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,g,beta):
  B,T,H,K=q.shape; HV=v.shape[2]; V=v.shape[3]; NC=triton.cdiv(T,C); NV=triton.cdiv(V,BV); BK=triton.next_power_of_2(K)
  G=torch.empty((B*HV,T),device=q.device,dtype=torch.float32); A=torch.empty((B*HV,NC,C,C),device=q.device,dtype=torch.float32); S=torch.empty((B*HV,NC,K,V),device=q.device,dtype=torch.float32); U=torch.empty((B*HV,T,V),device=q.device,dtype=torch.float32); o=torch.empty((B,T,HV,V),device=q.device,dtype=v.dtype)
  _wy[(NC,B*HV)](k,g,beta,G,A,T,H,HV,K,BK,NC); _fscan[(B*HV*NV,)](k,v,beta,G,A,S,U,T,H,HV,K,V,BK,NC,NV); _out[(B*HV*NC*NV,)](q,k,G,S,U,o,T,H,HV,K,V,BK,NC,NV,K**-.5)
  ctx.save_for_backward(q,k,v,beta,G,A,S,U); ctx.meta=(B,T,H,HV,K,V,NC,NV,BK); return o
 @staticmethod
 def backward(ctx,do):
  q,k,v,beta,G,A,S,U=ctx.saved_tensors; B,T,H,HV,K,V,NC,NV,BK=ctx.meta; do=do.contiguous()
  DS=torch.empty((B*HV,NC,K,V),device=q.device,dtype=torch.float32); AR=torch.empty((B*HV,T,V),device=q.device,dtype=torch.float32); PQ=torch.empty((B*HV*NV,T,K),device=q.device,dtype=torch.float32); PK=torch.empty_like(PQ); PG=torch.empty((B*HV*NV,T),device=q.device,dtype=torch.float32); PB=torch.empty_like(PG)
  dq=torch.empty_like(q); dk=torch.empty_like(k); dv=torch.empty_like(v); dg=torch.empty((B,T,HV),device=q.device,dtype=torch.float32); db=torch.empty_like(beta)
  _ascan[(B*HV*NV,)](q,k,beta,G,A,do,DS,AR,T,H,HV,K,V,BK,NC,NV,K**-.5); _grad[(B*HV*NC*NV,)](q,k,v,beta,G,S,U,DS,AR,do,PQ,PK,dv,PG,PB,T,H,HV,K,V,BK,NC,NV,K**-.5); _rqk[(B*T*H,)](PQ,PK,dq,dk,T,H,HV,K,BK,NV); _rs[(B*T*HV,)](PG,PB,dg,db,T,HV,NV)
  return dq,dk,dv,dg,db

def gated_delta_rule(q,k,v,g,beta): return _Fn.apply(q,k,v,g,beta)
