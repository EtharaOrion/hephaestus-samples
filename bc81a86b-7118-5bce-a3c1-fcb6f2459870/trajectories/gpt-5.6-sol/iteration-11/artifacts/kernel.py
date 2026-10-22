import torch
import triton
import triton.language as tl

C = tl.constexpr(16)
BV = tl.constexpr(16)
DOT = tl.constexpr("tf32")

@triton.jit
def _prep(k,g,beta,G,A,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); z=tl.program_id(1).to(tl.int64); b=z//HV; h=z%HV; kh=h//(HV//H)
 i=tl.arange(0,C); x=tl.arange(0,BK); t=p*C+i; m=t<T
 kk=tl.load(k+((b*T+t[:,None])*H+kh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
 gg=tl.load(g+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32); be=tl.load(beta+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32)
 cg=tl.cumsum(gg,0); L=be[:,None]*tl.exp(tl.maximum(tl.minimum(cg[:,None]-cg[None,:],0.),-80.))*tl.dot(kk,tl.trans(kk),input_precision=DOT)
 L=tl.where((i[:,None]>i[None,:])&m[:,None]&m[None,:],L,0.); inv=tl.where(i[:,None]==i[None,:],1.,0.)
 for u in range(1,C):
  row=tl.where(i==u,1.,0.)
  for j in range(u):
   lij=tl.sum(tl.where((i==u)[:,None]&(i==j)[None,:],L,0.)); old=tl.sum(tl.where((i==j)[:,None],inv,0.),0); row-=lij*old
  inv=tl.where((i==u)[:,None],row[None,:],inv)
 tl.store(G+((b*NC+p)*HV+h)*C+i,cg,mask=m)
 tl.store(A+(((b*NC+p)*HV+h)*C+i[:,None])*C+i[None,:],inv,mask=m[:,None]&m[None,:])

@triton.jit
def _fstate(k,v,beta,G,A,S,U,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr):
 z=tl.program_id(0).to(tl.int64); jv=tl.program_id(1).to(tl.int64); b=z//HV; h=z%HV; kh=h//(HV//H)
 x=tl.arange(0,BK); y=jv*BV+tl.arange(0,BV); i=tl.arange(0,C); st=tl.zeros([BK,BV],tl.float32)
 for p in tl.range(NC):
  t=p*C+i; m=t<T
  tl.store(S+((((b*NC+p)*HV+h)*K+x[:,None])*V+y[None,:]),st,mask=(x[:,None]<K)&(y[None,:]<V))
  kk=tl.load(k+((b*T+t[:,None])*H+kh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
  vv=tl.load(v+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.).to(tl.float32)
  be=tl.load(beta+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32); cg=tl.load(G+((b*NC+p)*HV+h)*C+i,mask=m,other=0.)
  aa=tl.load(A+(((b*NC+p)*HV+h)*C+i[:,None])*C+i[None,:],mask=m[:,None]&m[None,:],other=0.)
  yy=be[:,None]*vv-tl.dot(kk*(be*tl.exp(cg))[:,None],st,input_precision=DOT); uu=tl.dot(aa,yy,input_precision=DOT)
  tl.store(U+((b*T+t[:,None])*HV+h)*V+y[None,:],uu,mask=m[:,None]&(y[None,:]<V))
  last=tl.max(tl.where(m,i,-1),0); ce=tl.sum(tl.where(i==last,cg,0.),0); st=tl.exp(ce)*st+tl.dot(tl.trans(kk*tl.exp(tl.maximum(ce-cg,-80.))[:,None]),uu,input_precision=DOT)

@triton.jit
def _fout(q,k,G,S,U,o,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,SCALE:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); z=tl.program_id(1).to(tl.int64); jv=tl.program_id(2).to(tl.int64); b=z//HV; h=z%HV; kh=h//(HV//H)
 i=tl.arange(0,C); x=tl.arange(0,BK); y=jv*BV+tl.arange(0,BV); t=p*C+i; m=t<T
 qq=tl.load(q+((b*T+t[:,None])*H+kh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)*SCALE
 kk=tl.load(k+((b*T+t[:,None])*H+kh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
 cg=tl.load(G+((b*NC+p)*HV+h)*C+i,mask=m,other=0.); st=tl.load(S+((((b*NC+p)*HV+h)*K+x[:,None])*V+y[None,:]),mask=(x[:,None]<K)&(y[None,:]<V),other=0.)
 uu=tl.load(U+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.).to(tl.float32)
 mm=tl.dot(qq,tl.trans(kk),input_precision=DOT)*tl.exp(tl.maximum(tl.minimum(cg[:,None]-cg[None,:],0.),-80.)); mm=tl.where((i[:,None]>=i[None,:])&m[:,None]&m[None,:],mm,0.)
 oo=tl.dot(qq*tl.exp(cg)[:,None],st,input_precision=DOT)+tl.dot(mm,uu,input_precision=DOT)
 tl.store(o+((b*T+t[:,None])*HV+h)*V+y[None,:],oo,mask=m[:,None]&(y[None,:]<V))

@triton.jit
def _bstate(q,k,beta,G,A,do,DH,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,SCALE:tl.constexpr):
 z=tl.program_id(0).to(tl.int64); jv=tl.program_id(1).to(tl.int64); b=z//HV; h=z%HV; kh=h//(HV//H)
 x=tl.arange(0,BK); y=jv*BV+tl.arange(0,BV); i=tl.arange(0,C); dh=tl.zeros([BK,BV],tl.float32)
 for zz in tl.range(NC):
  p=NC-1-zz; t=p*C+i; m=t<T
  qq=tl.load(q+((b*T+t[:,None])*H+kh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)*SCALE
  kk=tl.load(k+((b*T+t[:,None])*H+kh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
  be=tl.load(beta+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32); cg=tl.load(G+((b*NC+p)*HV+h)*C+i,mask=m,other=0.); last=tl.max(tl.where(m,i,-1),0); ce=tl.sum(tl.where(i==last,cg,0.),0)
  dd=tl.load(do+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.).to(tl.float32)
  mm=tl.dot(qq,tl.trans(kk),input_precision=DOT)*tl.exp(tl.maximum(tl.minimum(cg[:,None]-cg[None,:],0.),-80.)); mm=tl.where((i[:,None]>=i[None,:])&m[:,None]&m[None,:],mm,0.)
  rr=kk*tl.exp(tl.maximum(ce-cg,-80.))[:,None]; du=tl.dot(tl.trans(mm),dd,input_precision=DOT)+tl.dot(rr,dh,input_precision=DOT)
  aa=tl.load(A+(((b*NC+p)*HV+h)*C+i[:,None])*C+i[None,:],mask=m[:,None]&m[None,:],other=0.); dy=tl.dot(tl.trans(aa),du,input_precision=DOT)
  dh=tl.exp(ce)*dh+tl.dot(tl.trans(qq*tl.exp(cg)[:,None]),dd,input_precision=DOT)-tl.dot(tl.trans(kk*(be*tl.exp(cg))[:,None]),dy,input_precision=DOT)
  tl.store(DH+((((b*NC+p)*HV+h)*K+x[:,None])*V+y[None,:]),dh,mask=(x[:,None]<K)&(y[None,:]<V))

@triton.jit
def _bchunk(q,k,v,beta,G,A,S,U,do,DH,dq,dk,dv,dg,db,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,NV:tl.constexpr,BK:tl.constexpr,SCALE:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); z=tl.program_id(1).to(tl.int64); b=z//HV; h=z%HV; kh=h//(HV//H)
 i=tl.arange(0,C); x=tl.arange(0,BK); t=p*C+i; m=t<T
 Q=tl.load(q+((b*T+t[:,None])*H+kh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)*SCALE
 Kb=tl.load(k+((b*T+t[:,None])*H+kh)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
 be=tl.load(beta+(b*T+t)*HV+h,mask=m,other=0.).to(tl.float32); c=tl.load(G+((b*NC+p)*HV+h)*C+i,mask=m,other=0.); last=tl.max(tl.where(m,i,-1),0); ce=tl.sum(tl.where(i==last,c,0.),0)
 aa=tl.load(A+(((b*NC+p)*HV+h)*C+i[:,None])*C+i[None,:],mask=m[:,None]&m[None,:],other=0.)
 E=tl.exp(tl.maximum(tl.minimum(c[:,None]-c[None,:],0.),-80.)); M=tl.dot(Q,tl.trans(Kb),input_precision=DOT)*E; M=tl.where((i[:,None]>=i[None,:])&m[:,None]&m[None,:],M,0.); R=Kb*tl.exp(tl.maximum(ce-c,-80.))[:,None]
 dQ=tl.zeros([C,BK],tl.float32); dP=tl.zeros([C,BK],tl.float32); dR=tl.zeros([C,BK],tl.float32); dM=tl.zeros([C,C],tl.float32); dL=tl.zeros([C,C],tl.float32); dbe=tl.zeros([C],tl.float32); dae=tl.zeros([],tl.float32)
 for jv in range(NV):
  y=jv*BV+tl.arange(0,BV); mv=y<V
  st=tl.load(S+((((b*NC+p)*HV+h)*K+x[:,None])*V+y[None,:]),mask=(x[:,None]<K)&mv[None,:],other=0.)
  uu=tl.load(U+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&mv[None,:],other=0.).to(tl.float32)
  vv=tl.load(v+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&mv[None,:],other=0.).to(tl.float32)
  dd=tl.load(do+((b*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&mv[None,:],other=0.).to(tl.float32)
  if p+1<NC: dhe=tl.load(DH+((((b*NC+p+1)*HV+h)*K+x[:,None])*V+y[None,:]),mask=(x[:,None]<K)&mv[None,:],other=0.)
  else: dhe=tl.zeros([BK,BV],tl.float32)
  du=tl.dot(tl.trans(M),dd,input_precision=DOT)+tl.dot(R,dhe,input_precision=DOT); dy=tl.dot(tl.trans(aa),du,input_precision=DOT)
  dQ+=tl.dot(dd,tl.trans(st),input_precision=DOT); dM+=tl.dot(dd,tl.trans(uu),input_precision=DOT); dP-=tl.dot(dy,tl.trans(st),input_precision=DOT); dR+=tl.dot(uu,tl.trans(dhe),input_precision=DOT); dL-=tl.dot(dy,tl.trans(uu),input_precision=DOT)
  dbe+=tl.sum(dy*vv,1); dae+=tl.sum(dhe*st)
  tl.store(dv+((b*T+t[:,None])*HV+h)*V+y[None,:],dy*be[:,None],mask=m[:,None]&mv[None,:])
 dM=tl.where((i[:,None]>=i[None,:])&m[:,None]&m[None,:],dM,0.); dL=tl.where((i[:,None]>i[None,:])&m[:,None]&m[None,:],dL,0.)
 ec=tl.exp(c); er=tl.exp(tl.maximum(ce-c,-80.)); dq0=dQ*(ec*SCALE)[:,None]; dk0=dP*(be*ec)[:,None]+dR*er[:,None]; dc=tl.sum(dQ*(Q*ec[:,None]),1)+tl.sum(dP*(Kb*(be*ec)[:,None]),1)
 dbe+=tl.sum(dP*(Kb*ec[:,None]),1); zr=tl.sum(dR*(Kb*er[:,None]),1); dc-=zr; dce=dae*tl.exp(ce)+tl.sum(zr)
 Z=dM*E; dc+=tl.sum(Z*tl.dot(Q,tl.trans(Kb),input_precision=DOT),1)-tl.sum(Z*tl.dot(Q,tl.trans(Kb),input_precision=DOT),0); dq0+=tl.dot(Z,Kb,input_precision=DOT)*SCALE; dk0+=tl.dot(tl.trans(Z),Q,input_precision=DOT)
 gram=tl.dot(Kb,tl.trans(Kb),input_precision=DOT); ZL=dL*(be[:,None]*E); dc+=tl.sum(ZL*gram,1)-tl.sum(ZL*gram,0); dbe+=tl.sum(dL*(E*gram),1); dk0+=tl.dot(ZL,Kb,input_precision=DOT)+tl.dot(tl.trans(ZL),Kb,input_precision=DOT)
 last=tl.max(tl.where(m,i,-1),0); dc+=tl.where(i==last,dce,0.); dgg=tl.sum(tl.where(i[None,:]>=i[:,None],dc[None,:],0.),1); off=((b*T+t[:,None])*HV+h)*K+x[None,:]
 tl.store(dq+off,dq0,mask=m[:,None]&(x[None,:]<K)); tl.store(dk+off,dk0,mask=m[:,None]&(x[None,:]<K)); tl.store(dg+(b*T+t)*HV+h,dgg,mask=m); tl.store(db+(b*T+t)*HV+h,dbe,mask=m)

@triton.jit
def _sumh(src,dst,N:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,BK:tl.constexpr):
 n=tl.program_id(0).to(tl.int64); h=tl.program_id(1); x=tl.arange(0,BK); a=tl.zeros([BK],tl.float32); R:tl.constexpr=HV//H
 for j in range(R): a+=tl.load(src+(n*HV+h*R+j)*K+x,mask=x<K,other=0.).to(tl.float32)
 tl.store(dst+(n*H+h)*K+x,a,mask=x<K)

class _Fn(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,g,beta):
  B,T,H,K=q.shape; HV,V=v.shape[2:]; NC=triton.cdiv(T,C); NV=triton.cdiv(V,BV); BK=triton.next_power_of_2(K)
  G=torch.empty((B,NC,HV,C),device=q.device,dtype=torch.float32); A=torch.empty((B,NC,HV,C,C),device=q.device,dtype=torch.float32); S=torch.empty((B,NC,HV,K,V),device=q.device,dtype=torch.float32); U=torch.empty(v.shape,device=q.device,dtype=torch.float32); o=torch.empty_like(v)
  _prep[(NC,B*HV)](k,g,beta,G,A,T=T,H=H,HV=HV,K=K,NC=NC,BK=BK,num_warps=4); _fstate[(B*HV,NV)](k,v,beta,G,A,S,U,T=T,H=H,HV=HV,K=K,V=V,NC=NC,BK=BK,num_warps=8); _fout[(NC,B*HV,NV)](q,k,G,S,U,o,T=T,H=H,HV=HV,K=K,V=V,NC=NC,BK=BK,SCALE=K**-.5,num_warps=8)
  ctx.save_for_backward(q,k,v,g,beta,G,A,S,U); ctx.d=(B,T,H,HV,K,V,NC,NV,BK); return o
 @staticmethod
 def backward(ctx,do):
  q,k,v,g,beta,G,A,S,U=ctx.saved_tensors; B,T,H,HV,K,V,NC,NV,BK=ctx.d; do=do.contiguous(); DH=torch.empty((B,NC,HV,K,V),device=q.device,dtype=torch.float32); qv=torch.empty((B,T,HV,K),device=q.device,dtype=q.dtype); kv=torch.empty_like(qv); dv=torch.empty_like(v); dg=torch.empty_like(g); db=torch.empty_like(beta)
  _bstate[(B*HV,NV)](q,k,beta,G,A,do,DH,T=T,H=H,HV=HV,K=K,V=V,NC=NC,BK=BK,SCALE=K**-.5,num_warps=8); _bchunk[(NC,B*HV)](q,k,v,beta,G,A,S,U,do,DH,qv,kv,dv,dg,db,T=T,H=H,HV=HV,K=K,V=V,NC=NC,NV=NV,BK=BK,SCALE=K**-.5,num_warps=4)
  if HV==H: dq,dk=qv,kv
  else:
   dq=torch.empty_like(q); dk=torch.empty_like(k); _sumh[(B*T,H)](qv,dq,N=B*T,H=H,HV=HV,K=K,BK=BK); _sumh[(B*T,H)](kv,dk,N=B*T,H=H,HV=HV,K=K,BK=BK)
  return dq,dk,dv,dg,db

def gated_delta_rule(q,k,v,g,beta): return _Fn.apply(q,k,v,g,beta)
