import torch
import triton
import triton.language as tl

ROW = 64
KB = 32

@triton.jit
def _router(X,W,S,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 m=tl.program_id(0)*BM+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); acc=tl.zeros((BM,BN),tl.float32)
 for k in range(0,D,BK):
  q=k+tl.arange(0,BK); a=tl.load(X+m[:,None]*D+q[None,:],mask=(m[:,None]<T)&(q[None,:]<D),other=0.).to(tl.float32); b=tl.load(W+q[:,None]*E+n[None,:],mask=(q[:,None]<D)&(n[None,:]<E),other=0.).to(tl.float32); acc+=tl.dot(a,b,input_precision="ieee")
 tl.store(S+m[:,None]*E+n[None,:],tl.sigmoid(acc),mask=(m[:,None]<T)&(n[None,:]<E))

@triton.jit
def _offsets(F,O,OFF,N:tl.constexpr,EP:tl.constexpr):
 e=tl.program_id(0); lo=0; hi=N
 while lo<hi:
  mid=(lo+hi)//2; v=tl.load(F+tl.load(O+mid)); take=v<e; lo=tl.where(take,mid+1,lo); hi=tl.where(take,hi,mid)
 tl.store(OFF+e,lo)

@triton.jit
def _tiles(OFF,TE,TS,E:tl.constexpr,CAP:tl.constexpr,BR:tl.constexpr):
 p=0
 for e in range(E):
  q=tl.load(OFF+e); end=tl.load(OFF+e+1)
  while q<end:
   tl.store(TE+p,e); tl.store(TS+p,q); p+=1; q+=BR
 while p<CAP:
  tl.store(TE+p,-1); tl.store(TS+p,0); p+=1

@triton.jit
def _ein(X,W,O,OFF,TE,TS,G,N:tl.constexpr,A:tl.constexpr,D:tl.constexpr,F2:tl.constexpr,BR:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
 p=tl.program_id(0); e=tl.load(TE+p); r=tl.load(TS+p)+tl.arange(0,BR); j=tl.program_id(1)*BN+tl.arange(0,BN); vr=(e>=0)&(r<tl.load(OFF+tl.maximum(e,0)+1)); pair=tl.load(O+r,mask=vr,other=0); t=pair//A; acc=tl.zeros((BR,BN),tl.float32)
 for k in range(0,D,BK):
  q=k+tl.arange(0,BK); x=tl.load(X+t[:,None]*D+q[None,:],mask=vr[:,None]&(q[None,:]<D),other=0.); w=tl.load(W+e*D*F2+q[:,None]*F2+j[None,:],mask=(e>=0)&(q[:,None]<D)&(j[None,:]<F2),other=0.)
  if BF: acc+=tl.dot(x,w)
  else: acc+=tl.dot(x.to(tl.float32),w.to(tl.float32),input_precision="tf32x3")
 tl.store(G+r[:,None]*F2+j[None,:],acc,mask=vr[:,None]&(j[None,:]<F2))

@triton.jit
def _act(G,U,N:tl.constexpr,F:tl.constexpr,B:tl.constexpr):
 q=tl.program_id(0)*B+tl.arange(0,B); m=q<N*F; r=q//F; c=q-r*F; g=tl.load(G+r*2*F+c,mask=m); u=tl.load(G+r*2*F+F+c,mask=m); tl.store(U+q,g*tl.sigmoid(g)*u,mask=m)

@triton.jit
def _eout(U,W,OFF,TE,TS,Z,N:tl.constexpr,F:tl.constexpr,D:tl.constexpr,BR:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
 p=tl.program_id(0); e=tl.load(TE+p); r=tl.load(TS+p)+tl.arange(0,BR); j=tl.program_id(1)*BN+tl.arange(0,BN); vr=(e>=0)&(r<tl.load(OFF+tl.maximum(e,0)+1)); acc=tl.zeros((BR,BN),tl.float32)
 for k in range(0,F,BK):
  q=k+tl.arange(0,BK); a=tl.load(U+r[:,None]*F+q[None,:],mask=vr[:,None]&(q[None,:]<F),other=0.); w=tl.load(W+e*F*D+q[:,None]*D+j[None,:],mask=(e>=0)&(q[:,None]<F)&(j[None,:]<D),other=0.)
  if BF:
   ah=a.to(tl.bfloat16); al=(a-ah).to(tl.bfloat16); acc+=tl.dot(ah,w)+tl.dot(al,w)
  else: acc+=tl.dot(a,w.to(tl.float32),input_precision="tf32x3")
 tl.store(Z+r[:,None]*D+j[None,:],acc,mask=vr[:,None]&(j[None,:]<D))

@triton.jit
def _inverse(O,I,N:tl.constexpr,B:tl.constexpr):
 r=tl.program_id(0)*B+tl.arange(0,B); m=r<N; p=tl.load(O+r,mask=m); tl.store(I+p,r,mask=m)

@triton.jit
def _combine(S,SEL,COMB,I,Z,Y,WS,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BD:tl.constexpr):
 t=tl.program_id(0); j=tl.program_id(1)*BD+tl.arange(0,BD); a=tl.arange(0,256); ma=a<A; es=tl.load(SEL+t*A+a,mask=ma,other=0); ss=tl.load(S+t*E+es,mask=ma,other=0.); w=ss/tl.sum(ss,axis=0); tl.store(WS+t*A+a,w,mask=ma); acc=tl.zeros((BD,),tl.float32)
 for q in range(A):
  a0=tl.load(COMB+t*A+q); r=tl.load(I+t*A+a0); acc+=tl.load(WS+t*A+a0)*tl.load(Z+r*D+j,mask=j<D,other=0.)
 tl.store(Y+t*D+j,acc,mask=j<D)

@triton.jit
def _dz(DY,WS,O,DZ,N:tl.constexpr,A:tl.constexpr,D:tl.constexpr,BR:tl.constexpr,BN:tl.constexpr):
 r=tl.program_id(0)*BR+tl.arange(0,BR); j=tl.program_id(1)*BN+tl.arange(0,BN); m=r<N; p=tl.load(O+r,mask=m,other=0); t=p//A; w=tl.load(WS+p,mask=m,other=0.); v=tl.load(DY+t[:,None]*D+j[None,:],mask=m[:,None]&(j[None,:]<D),other=0.); tl.store(DZ+r[:,None]*D+j[None,:],v*w[:,None],mask=m[:,None]&(j[None,:]<D))

@triton.jit
def _dact(DZ,W,OFF,TE,TS,DA,N:tl.constexpr,F:tl.constexpr,D:tl.constexpr,BR:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
 p=tl.program_id(0); e=tl.load(TE+p); r=tl.load(TS+p)+tl.arange(0,BR); f=tl.program_id(1)*BN+tl.arange(0,BN); vr=(e>=0)&(r<tl.load(OFF+tl.maximum(e,0)+1)); acc=tl.zeros((BR,BN),tl.float32)
 for k in range(0,D,BK):
  q=k+tl.arange(0,BK); z=tl.load(DZ+r[:,None]*D+q[None,:],mask=vr[:,None]&(q[None,:]<D),other=0.); w=tl.load(W+e*F*D+f[None,:]*D+q[:,None],mask=(e>=0)&(f[None,:]<F)&(q[:,None]<D),other=0.)
  if BF: acc+=tl.dot(z.to(tl.bfloat16),w)
  else: acc+=tl.dot(z,w.to(tl.float32),input_precision="tf32x3")
 tl.store(DA+r[:,None]*F+f[None,:],acc,mask=vr[:,None]&(f[None,:]<F))

@triton.jit
def _dgu(G,DA,DG,N:tl.constexpr,F:tl.constexpr,B:tl.constexpr):
 q=tl.program_id(0)*B+tl.arange(0,B); m=q<N*F; r=q//F; c=q-r*F; g=tl.load(G+r*2*F+c,mask=m); u=tl.load(G+r*2*F+F+c,mask=m); da=tl.load(DA+q,mask=m); s=tl.sigmoid(g); si=g*s; tl.store(DG+r*2*F+c,da*u*(s+si*(1-s)),mask=m); tl.store(DG+r*2*F+F+c,da*si,mask=m)

@triton.jit
def _dxr(DG,W,OFF,TE,TS,DX,N:tl.constexpr,D:tl.constexpr,F2:tl.constexpr,BR:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
 p=tl.program_id(0); e=tl.load(TE+p); r=tl.load(TS+p)+tl.arange(0,BR); d=tl.program_id(1)*BN+tl.arange(0,BN); vr=(e>=0)&(r<tl.load(OFF+tl.maximum(e,0)+1)); acc=tl.zeros((BR,BN),tl.float32)
 for k in range(0,F2,BK):
  q=k+tl.arange(0,BK); g=tl.load(DG+r[:,None]*F2+q[None,:],mask=vr[:,None]&(q[None,:]<F2),other=0.); w=tl.load(W+e*D*F2+d[None,:]*F2+q[:,None],mask=(e>=0)&(d[None,:]<D)&(q[:,None]<F2),other=0.)
  if BF: acc+=tl.dot(g.to(tl.bfloat16),w)
  else: acc+=tl.dot(g,w.to(tl.float32),input_precision="tf32x3")
 tl.store(DX+r[:,None]*D+d[None,:],acc,mask=vr[:,None]&(d[None,:]<D))

@triton.jit
def _pair(DY,Z,I,Q,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,BK:tl.constexpr):
 t=tl.program_id(0); a=tl.arange(0,256); ma=a<A; r=tl.load(I+t*A+a,mask=ma,other=0); acc=tl.zeros((256,),tl.float32)
 for k in range(0,D,BK):
  j=k+tl.arange(0,BK); y=tl.load(DY+t*D+j,mask=j<D,other=0.); z=tl.load(Z+r[:,None]*D+j[None,:],mask=ma[:,None]&(j[None,:]<D),other=0.); acc+=tl.sum(z*y[None,:],axis=1)
 tl.store(Q+t*A+a,acc,mask=ma)

@triton.jit
def _route(S,SEL,Q,DS,T:tl.constexpr,E:tl.constexpr,A:tl.constexpr):
 t=tl.program_id(0); a=tl.arange(0,256); m=a<A; e=tl.load(SEL+t*A+a,mask=m,other=0); s=tl.load(S+t*E+e,mask=m,other=0.); q=tl.load(Q+t*A+a,mask=m,other=0.); den=tl.sum(s,axis=0); ds=(q*den-tl.sum(q*s,axis=0))/(den*den); tl.store(DS+t*E+e,ds*s*(1-s),mask=m)

@triton.jit
def _rdx(DS,W,DX,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 t=tl.program_id(0)*BM+tl.arange(0,BM); d=tl.program_id(1)*BN+tl.arange(0,BN); acc=tl.zeros((BM,BN),tl.float32)
 for k in range(0,E,BK):
  e=k+tl.arange(0,BK); s=tl.load(DS+t[:,None]*E+e[None,:],mask=(t[:,None]<T)&(e[None,:]<E),other=0.); w=tl.load(W+d[None,:]*E+e[:,None],mask=(d[None,:]<D)&(e[:,None]<E),other=0.).to(tl.float32); acc+=tl.dot(s,w,input_precision="ieee")
 tl.store(DX+t[:,None]*D+d[None,:],acc,mask=(t[:,None]<T)&(d[None,:]<D))

@triton.jit
def _sumdx(X,I,COMB,R,Y,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,BD:tl.constexpr):
 t=tl.program_id(0); d=tl.program_id(1)*BD+tl.arange(0,BD); v=tl.load(R+t*D+d,mask=d<D,other=0.)
 for a in range(A):
  a0=tl.load(COMB+t*A+a); v+=tl.load(X+tl.load(I+t*A+a0)*D+d,mask=d<D,other=0.)
 tl.store(Y+t*D+d,v,mask=d<D)

@triton.jit
def _dw2(U,DZ,OFF,DW,E:tl.constexpr,F:tl.constexpr,D:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
 e=tl.program_id(0); f=tl.program_id(1)*BM+tl.arange(0,BM); d=tl.program_id(2)*BN+tl.arange(0,BN); beg=tl.load(OFF+e); end=tl.load(OFF+e+1); acc=tl.zeros((BM,BN),tl.float32)
 for k in range(beg,end,BK):
  r=k+tl.arange(0,BK); a=tl.load(U+f[:,None]+r[None,:]*F,mask=(f[:,None]<F)&(r[None,:]<end),other=0.).to(tl.float32); z=tl.load(DZ+r[:,None]*D+d[None,:],mask=(r[:,None]<end)&(d[None,:]<D),other=0.).to(tl.float32)
  if BF:
   ah=a.to(tl.bfloat16); al=(a-ah).to(tl.bfloat16); zh=z.to(tl.bfloat16); zl=(z-zh).to(tl.bfloat16); acc+=tl.dot(ah,zh)+tl.dot(al,zh)+tl.dot(ah,zl)
  else: acc+=tl.dot(a,z,input_precision="tf32x3")
 tl.store(DW+e*F*D+f[:,None]*D+d[None,:],acc,mask=(f[:,None]<F)&(d[None,:]<D))

@triton.jit
def _dw1(X,DG,O,OFF,DW,E:tl.constexpr,A:tl.constexpr,D:tl.constexpr,F2:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
 e=tl.program_id(0); d=tl.program_id(1)*BM+tl.arange(0,BM); f=tl.program_id(2)*BN+tl.arange(0,BN); beg=tl.load(OFF+e); end=tl.load(OFF+e+1); acc=tl.zeros((BM,BN),tl.float32)
 for k in range(beg,end,BK):
  r=k+tl.arange(0,BK); p=tl.load(O+r,mask=r<end,other=0); t=p//A; x=tl.load(X+d[:,None]+t[None,:]*D,mask=(d[:,None]<D)&(r[None,:]<end),other=0.).to(tl.float32); g=tl.load(DG+r[:,None]*F2+f[None,:],mask=(r[:,None]<end)&(f[None,:]<F2),other=0.).to(tl.float32)
  if BF:
   xh=x.to(tl.bfloat16); xl=(x-xh).to(tl.bfloat16); gh=g.to(tl.bfloat16); gl=(g-gh).to(tl.bfloat16); acc+=tl.dot(xh,gh)+tl.dot(xl,gh)+tl.dot(xh,gl)
  else: acc+=tl.dot(x,g,input_precision="tf32x3")
 tl.store(DW+e*D*F2+d[:,None]*F2+f[None,:],acc,mask=(d[:,None]<D)&(f[None,:]<F2))

@triton.jit
def _drw(X,DS,DW,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 d=tl.program_id(0)*BM+tl.arange(0,BM); e=tl.program_id(1)*BN+tl.arange(0,BN); acc=tl.zeros((BM,BN),tl.float32)
 for k in range(0,T,BK):
  t=k+tl.arange(0,BK); x=tl.load(X+d[:,None]+t[None,:]*D,mask=(d[:,None]<D)&(t[None,:]<T),other=0.).to(tl.float32); s=tl.load(DS+t[:,None]*E+e[None,:],mask=(t[:,None]<T)&(e[None,:]<E),other=0.); acc+=tl.dot(x,s,input_precision="ieee")
 tl.store(DW+d[:,None]*E+e[None,:],acc,mask=(d[:,None]<D)&(e[None,:]<E))

def _select(s,b,A):
 E=s.shape[1]; v=s+b; v=torch.where(v==0,torch.zeros_like(v),v).contiguous(); i=v.view(torch.int32).to(torch.int64); u=torch.where(i>=0,i^0x80000000,(~i)&0xFFFFFFFF); e=torch.arange(E,device=s.device,dtype=torch.int64); return torch.topk((u<<10)|(E-1-e),A,dim=1,largest=True,sorted=True).indices

class _Moe(torch.autograd.Function):
 @staticmethod
 def forward(c,x,rw,rb,w1,w2,A):
  T,D=x.shape; E=rw.shape[1]; F=w2.shape[1]; A=int(A); N=T*A; bf=x.dtype==torch.bfloat16; S=torch.empty((T,E),device=x.device,dtype=torch.float32); _router[(triton.cdiv(T,32),triton.cdiv(E,32))](x,rw,S,T,D,E,32,32,32,num_warps=4,num_stages=3)
  with torch.no_grad(): sel=_select(S,rb,A); comb=torch.sort(sel,dim=1).indices; flat=sel.reshape(-1); O=torch.argsort(flat,stable=True)
  off=torch.empty(E+1,device=x.device,dtype=torch.int32); _offsets[(E+1,)](flat,O,off,N,E+1,num_warps=1); cap=triton.cdiv(N,ROW)+E; te=torch.empty(cap,device=x.device,dtype=torch.int32); ts=torch.empty_like(te); _tiles[(1,)](off,te,ts,E,cap,ROW,num_warps=1)
  G=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _ein[(cap,triton.cdiv(2*F,64))](x,w1,O,off,te,ts,G,N,A,D,2*F,ROW,64,KB,bf,num_warps=4,num_stages=3); U=torch.empty((N,F),device=x.device,dtype=torch.float32); _act[(triton.cdiv(N*F,256),)](G,U,N,F,256); Z=torch.empty((N,D),device=x.device,dtype=torch.float32); _eout[(cap,triton.cdiv(D,64))](U,w2,off,te,ts,Z,N,F,D,ROW,64,KB,bf,num_warps=4,num_stages=3); I=torch.empty(N,device=x.device,dtype=torch.int64); _inverse[(triton.cdiv(N,256),)](O,I,N,256); Y=torch.empty((T,D),device=x.device,dtype=x.dtype); WS=torch.empty((T,A),device=x.device,dtype=torch.float32); _combine[(T,triton.cdiv(D,256))](S,sel,comb,I,Z,Y,WS,T,D,E,A,256); c.save_for_backward(x,rw,w1,w2,S,sel,comb,O,off,te,ts,G,U,Z,I,WS); c.d=(T,D,E,F,A,cap,bf); return Y
 @staticmethod
 def backward(c,dy):
  x,rw,w1,w2,S,sel,comb,O,off,te,ts,G,U,Z,I,WS=c.saved_tensors; T,D,E,F,A,cap,bf=c.d; N=T*A; dy=dy.contiguous(); DZ=torch.empty((N,D),device=x.device,dtype=torch.float32); _dz[(triton.cdiv(N,ROW),triton.cdiv(D,64))](dy,WS,O,DZ,N,A,D,ROW,64); DA=torch.empty((N,F),device=x.device,dtype=torch.float32); _dact[(cap,triton.cdiv(F,64))](DZ,w2,off,te,ts,DA,N,F,D,ROW,64,KB,bf,num_warps=4,num_stages=3); DG=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _dgu[(triton.cdiv(N*F,256),)](G,DA,DG,N,F,256); XR=torch.empty((N,D),device=x.device,dtype=torch.float32); _dxr[(cap,triton.cdiv(D,64))](DG,w1,off,te,ts,XR,N,D,2*F,ROW,64,KB,bf,num_warps=4,num_stages=3); Q=torch.empty((T,A),device=x.device,dtype=torch.float32); _pair[(T,)](dy,Z,I,Q,T,D,A,32); DS=torch.zeros((T,E),device=x.device,dtype=torch.float32); _route[(T,)](S,sel,Q,DS,T,E,A); RX=torch.empty((T,D),device=x.device,dtype=torch.float32); _rdx[(triton.cdiv(T,32),triton.cdiv(D,32))](DS,rw,RX,T,D,E,32,32,32); DX=torch.empty_like(x); _sumdx[(T,triton.cdiv(D,256))](XR,I,comb,RX,DX,T,D,A,256); W2=torch.empty_like(w2); _dw2[(E,triton.cdiv(F,64),triton.cdiv(D,64))](U,DZ,off,W2,E,F,D,64,64,KB,bf,num_warps=4,num_stages=3); W1=torch.empty_like(w1); _dw1[(E,triton.cdiv(D,64),triton.cdiv(2*F,64))](x,DG,O,off,W1,E,A,D,2*F,64,64,KB,bf,num_warps=4,num_stages=3); RW=torch.empty_like(rw); _drw[(triton.cdiv(D,32),triton.cdiv(E,32))](x,DS,RW,T,D,E,32,32,32); return DX,RW,None,W1,W2,None

def fused_moe(x,router_w,router_b,w1,w2,A): return _Moe.apply(x,router_w,router_b,w1,w2,A)
