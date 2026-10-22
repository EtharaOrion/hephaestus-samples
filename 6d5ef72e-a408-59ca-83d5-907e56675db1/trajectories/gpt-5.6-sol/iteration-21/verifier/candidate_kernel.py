import torch
import triton
import triton.language as tl

ROW = 64
COL = 64
KB = 32

@triton.jit
def _router(X,W,S,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 m=tl.program_id(0)*BM+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); acc=tl.zeros((BM,BN),tl.float32)
 for k in range(0,D,BK):
  q=k+tl.arange(0,BK); a=tl.load(X+m[:,None]*D+q[None,:],mask=(m[:,None]<T)&(q[None,:]<D),other=0.).to(tl.float32); b=tl.load(W+q[:,None]*E+n[None,:],mask=(q[:,None]<D)&(n[None,:]<E),other=0.).to(tl.float32); acc+=tl.dot(a,b,input_precision="ieee")
 tl.store(S+m[:,None]*E+n[None,:],tl.sigmoid(acc),mask=(m[:,None]<T)&(n[None,:]<E))

@triton.jit
def _pairs(O,WF,TOK,WS,INV,N:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
 p=tl.program_id(0)*B+tl.arange(0,B); mask=p<N; q=tl.load(O+p,mask=mask,other=0); tl.store(TOK+p,q//A,mask=mask); tl.store(WS+p,tl.load(WF+q,mask=mask,other=0.),mask=mask); tl.store(INV+q,p,mask=mask)

@triton.jit
def _offsets(FLAT,OFF,N:tl.constexpr,E:tl.constexpr):
 e=tl.program_id(0); lo=tl.zeros((),tl.int32); hi=tl.full((),N,tl.int32)
 for _ in range(32):
  mid=(lo+hi)//2; v=tl.load(FLAT+mid,mask=mid<N,other=E); take=v<e; lo=tl.where(take,mid+1,lo); hi=tl.where(take,hi,mid)
 tl.store(OFF+e,lo)

@triton.jit
def _tiles(OFF,TE,TS,E:tl.constexpr,CAP:tl.constexpr,BR:tl.constexpr):
 if tl.program_id(0)==0:
  q=0
  for e in range(E):
   beg=tl.load(OFF+e); end=tl.load(OFF+e+1)
   for st in range(beg,end,BR): tl.store(TE+q,e); tl.store(TS+q,st); q+=1
  for j in range(q,CAP): tl.store(TE+j,-1); tl.store(TS+j,0)

@triton.jit
def _ein(X,W,TOK,OFF,TE,TS,OUT,D:tl.constexpr,N:tl.constexpr,BR:tl.constexpr,BC:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
 z=tl.program_id(0); e=tl.load(TE+z); st=tl.load(TS+z); r=st+tl.arange(0,BR); c=tl.program_id(1)*BC+tl.arange(0,BC); end=tl.load(OFF+e+1,mask=e>=0,other=0); v=(e>=0)&(r<end); tok=tl.load(TOK+r,mask=v,other=0); acc=tl.zeros((BR,BC),tl.float32)
 for k in range(0,D,BK):
  q=k+tl.arange(0,BK); a=tl.load(X+tok[:,None]*D+q[None,:],mask=v[:,None]&(q[None,:]<D),other=0.); b=tl.load(W+e*D*N+q[:,None]*N+c[None,:],mask=(e>=0)&(q[:,None]<D)&(c[None,:]<N),other=0.); acc+=tl.dot(a,b) if BF else tl.dot(a.to(tl.float32),b.to(tl.float32),input_precision="tf32x3")
 tl.store(OUT+r[:,None]*N+c[None,:],acc,mask=v[:,None]&(c[None,:]<N))

@triton.jit
def _act(GU,A,N:tl.constexpr,F:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B); m=i<N*F; r=i//F; c=i%F; g=tl.load(GU+r*2*F+c,mask=m,other=0.); u=tl.load(GU+r*2*F+F+c,mask=m,other=0.); tl.store(A+i,g*tl.sigmoid(g)*u,mask=m)

@triton.jit
def _seg(A,W,OFF,TE,TS,O,K:tl.constexpr,N:tl.constexpr,BR:tl.constexpr,BC:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr,TRANS:tl.constexpr,SPLIT:tl.constexpr):
 z=tl.program_id(0); e=tl.load(TE+z); st=tl.load(TS+z); r=st+tl.arange(0,BR); c=tl.program_id(1)*BC+tl.arange(0,BC); end=tl.load(OFF+e+1,mask=e>=0,other=0); v=(e>=0)&(r<end); acc=tl.zeros((BR,BC),tl.float32)
 for k in range(0,K,BK):
  q=k+tl.arange(0,BK); a=tl.load(A+r[:,None]*K+q[None,:],mask=v[:,None]&(q[None,:]<K),other=0.)
  if TRANS: b=tl.load(W+e*N*K+c[:,None]*K+q[None,:],mask=(e>=0)&(c[:,None]<N)&(q[None,:]<K),other=0.); b=tl.trans(b)
  else: b=tl.load(W+e*K*N+q[:,None]*N+c[None,:],mask=(e>=0)&(q[:,None]<K)&(c[None,:]<N),other=0.)
  if BF:
   ah=a.to(tl.bfloat16); acc+=tl.dot(ah,b)
   if SPLIT:
    al=(a-ah.to(tl.float32)).to(tl.bfloat16); acc+=tl.dot(al,b)
  else: acc+=tl.dot(a.to(tl.float32),b.to(tl.float32),input_precision="tf32x3")
 tl.store(O+r[:,None]*N+c[None,:],acc,mask=v[:,None]&(c[None,:]<N))

@triton.jit
def _combine(Z,WS,INV,Y,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
 t=tl.program_id(0); d=tl.program_id(1)*B+tl.arange(0,B); acc=tl.zeros((B,),tl.float32)
 for a in range(A): p=tl.load(INV+t*A+a); acc+=tl.load(WS+p)*tl.load(Z+p*D+d,mask=d<D,other=0.)
 tl.store(Y+t*D+d,acc,mask=d<D)

@triton.jit
def _dz(DY,WS,TOK,O,N:tl.constexpr,D:tl.constexpr,B:tl.constexpr):
 p=tl.program_id(0); d=tl.program_id(1)*B+tl.arange(0,B); t=tl.load(TOK+p); tl.store(O+p*D+d,tl.load(DY+t*D+d,mask=d<D,other=0.).to(tl.float32)*tl.load(WS+p),mask=d<D)

@triton.jit
def _dgu(GU,DA,O,N:tl.constexpr,F:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B); m=i<N*F; r=i//F; c=i%F; g=tl.load(GU+r*2*F+c,mask=m,other=0.); u=tl.load(GU+r*2*F+F+c,mask=m,other=0.); da=tl.load(DA+i,mask=m,other=0.); s=tl.sigmoid(g); tl.store(O+r*2*F+c,da*u*(s+g*s*(1-s)),mask=m); tl.store(O+r*2*F+F+c,da*g*s,mask=m)

@triton.jit
def _pair_dot(DY,Z,INV,Q,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
 t=tl.program_id(0); a=tl.program_id(1); d=tl.arange(0,B); p=tl.load(INV+t*A+a); dy=tl.load(DY+t*D+d,mask=d<D,other=0.).to(tl.float32); z=tl.load(Z+p*D+d,mask=d<D,other=0.); tl.store(Q+t*A+a,tl.sum(dy*z))

@triton.jit
def _route(Q,S,SEL,DS,T:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BA:tl.constexpr):
 t=tl.program_id(0); a=tl.arange(0,BA); m=a<A; e=tl.load(SEL+t*A+a,mask=m,other=0); score=tl.load(S+t*E+e,mask=m,other=0.); q=tl.load(Q+t*A+a,mask=m,other=0.); den=tl.sum(score); mean=tl.sum(score*q); ds=(q-mean/den)/den*score*(1-score); tl.store(DS+t*A+a,ds,mask=m)

@triton.jit
def _rdx(DS,SEL,RW,O,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
 t=tl.program_id(0); d=tl.program_id(1)*B+tl.arange(0,B); acc=tl.zeros((B,),tl.float32)
 for a in range(A): e=tl.load(SEL+t*A+a); acc+=tl.load(DS+t*A+a)*tl.load(RW+d*E+e,mask=d<D,other=0.).to(tl.float32)
 tl.store(O+t*D+d,acc,mask=d<D)

@triton.jit
def _sumdx(DXR,INV,RDX,DX,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
 t=tl.program_id(0); d=tl.program_id(1)*B+tl.arange(0,B); acc=tl.load(RDX+t*D+d,mask=d<D,other=0.)
 for a in range(A): acc+=tl.load(DXR+tl.load(INV+t*A+a)*D+d,mask=d<D,other=0.)
 tl.store(DX+t*D+d,acc,mask=d<D)

@triton.jit
def _drw(X,DS,SEL,O,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
 d=tl.program_id(0)*BM+tl.arange(0,BM); e=tl.program_id(1)*BN+tl.arange(0,BN); acc=tl.zeros((BM,BN),tl.float32)
 for k in range(0,T,BK):
  t=k+tl.arange(0,BK); x=tl.load(X+t[:,None]*D+d[None,:],mask=(t[:,None]<T)&(d[None,:]<D),other=0.); v=tl.zeros((BK,BN),tl.float32)
  for a in range(A): se=tl.load(SEL+t[:,None]*A+a,mask=t[:,None]<T,other=-1); sv=tl.load(DS+t[:,None]*A+a,mask=t[:,None]<T,other=0.); v+=tl.where(se==e[None,:],sv,0.)
  acc+=tl.dot(tl.trans(x.to(tl.float32)),v,input_precision="ieee")
 tl.store(O+d[:,None]*E+e[None,:],acc,mask=(d[:,None]<D)&(e[None,:]<E))

@triton.jit
def _dw(A,B,TOK,OFF,O,E:tl.constexpr,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr,GATHER:tl.constexpr):
 e=tl.program_id(0); m=tl.program_id(1)*BM+tl.arange(0,BM); n=tl.program_id(2)*BN+tl.arange(0,BN); beg=tl.load(OFF+e); end=tl.load(OFF+e+1); acc=tl.zeros((BM,BN),tl.float32)
 for k in range(beg,end,BK):
  r=k+tl.arange(0,BK); rr=tl.load(TOK+r,mask=r<end,other=0) if GATHER else r; a=tl.load(A+rr[:,None]*M+m[None,:],mask=(r[:,None]<end)&(m[None,:]<M),other=0.); b=tl.load(B+r[:,None]*N+n[None,:],mask=(r[:,None]<end)&(n[None,:]<N),other=0.)
  if BF:
   ah=a.to(tl.bfloat16); bh=b.to(tl.bfloat16); bl=(b-bh.to(tl.float32)).to(tl.bfloat16); acc+=tl.dot(tl.trans(ah),bh)+tl.dot(tl.trans(ah),bl)
   if not GATHER:
    al=(a-ah.to(tl.float32)).to(tl.bfloat16); acc+=tl.dot(tl.trans(al),bh)
  else: acc+=tl.dot(tl.trans(a.to(tl.float32)),b,input_precision="tf32x3")
 tl.store(O+e*M*N+m[:,None]*N+n[None,:],acc,mask=(m[:,None]<M)&(n[None,:]<N))

def _select(s,bias,A):
 E=s.shape[1]; b=s+bias; b=torch.where(b==0,torch.zeros_like(b),b).contiguous(); ib=b.view(torch.int32).to(torch.int64); u=torch.where(ib>=0,ib^0x80000000,(~ib)&0xFFFFFFFF); i=torch.arange(E,device=s.device,dtype=torch.int64); return torch.topk((u<<10)|(E-1-i),A,dim=1,sorted=True).indices

class _Moe(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,rw,rb,w1,w2,A):
  T,D=x.shape; E=rw.shape[1]; F=w2.shape[1]; A=int(A); N=T*A; bf=x.dtype==torch.bfloat16; s=torch.empty((T,E),device=x.device,dtype=torch.float32); _router[(triton.cdiv(T,32),triton.cdiv(E,32))](x,rw,s,T,D,E,32,32,32)
  with torch.no_grad(): sel=_select(s,rb,A); ss=torch.gather(s,1,sel); wf=(ss/ss.sum(1,keepdim=True)).contiguous().view(-1); flat=sel.view(-1); order=torch.argsort(flat); sorted_flat=flat.index_select(0,order)
  tok=torch.empty(N,device=x.device,dtype=torch.int64); ws=torch.empty(N,device=x.device,dtype=torch.float32); inv=torch.empty(N,device=x.device,dtype=torch.int64); _pairs[(triton.cdiv(N,256),)](order,wf,tok,ws,inv,N,A,256)
  off=torch.empty(E+1,device=x.device,dtype=torch.int32); _offsets[(E+1,)](sorted_flat,off,N,E); cap=triton.cdiv(N,ROW)+E; te=torch.empty(cap,device=x.device,dtype=torch.int32); ts=torch.empty_like(te); _tiles[(1,)](off,te,ts,E,cap,ROW,num_warps=1)
  gu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _ein[(cap,triton.cdiv(2*F,COL))](x,w1,tok,off,te,ts,gu,D,2*F,ROW,COL,KB,bf,num_warps=4,num_stages=3); act=torch.empty((N,F),device=x.device,dtype=torch.float32); _act[(triton.cdiv(N*F,256),)](gu,act,N,F,256); z=torch.empty((N,D),device=x.device,dtype=torch.float32); _seg[(cap,triton.cdiv(D,COL))](act,w2,off,te,ts,z,F,D,ROW,COL,KB,bf,False,True,num_warps=4,num_stages=3); y=torch.empty((T,D),device=x.device,dtype=x.dtype); _combine[(T,triton.cdiv(D,256))](z,ws,inv,y,T,D,A,256)
  ctx.save_for_backward(x,rw,w1,w2,s,sel,tok,ws,inv,off,te,ts,gu,act,z); ctx.meta=(T,D,E,F,A,cap,bf); return y
 @staticmethod
 def backward(ctx,dy):
  x,rw,w1,w2,s,sel,tok,ws,inv,off,te,ts,gu,act,z=ctx.saved_tensors; T,D,E,F,A,cap,bf=ctx.meta; N=T*A; dz=torch.empty((N,D),device=x.device,dtype=torch.float32); _dz[(N,triton.cdiv(D,256))](dy,ws,tok,dz,N,D,256); da=torch.empty((N,F),device=x.device,dtype=torch.float32); _seg[(cap,triton.cdiv(F,COL))](dz,w2,off,te,ts,da,D,F,ROW,COL,KB,bf,True,False,num_warps=4,num_stages=3); dgu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _dgu[(triton.cdiv(N*F,256),)](gu,da,dgu,N,F,256); dxr=torch.empty((N,D),device=x.device,dtype=torch.float32); _seg[(cap,triton.cdiv(D,COL))](dgu,w1,off,te,ts,dxr,2*F,D,ROW,COL,KB,bf,True,False,num_warps=4,num_stages=3)
  q=torch.empty((T,A),device=x.device,dtype=torch.float32); _pair_dot[(T,A)](dy,z,inv,q,T,D,A,triton.next_power_of_2(D),num_warps=8); ds=torch.empty_like(q); _route[(T,)](q,s,sel,ds,T,E,A,triton.next_power_of_2(A)); rdx=torch.empty((T,D),device=x.device,dtype=torch.float32); _rdx[(T,triton.cdiv(D,256))](ds,sel,rw,rdx,T,D,E,A,256); dx=torch.empty_like(x); _sumdx[(T,triton.cdiv(D,256))](dxr,inv,rdx,dx,T,D,A,256); drw=torch.empty_like(rw); _drw[(triton.cdiv(D,32),triton.cdiv(E,32))](x,ds,sel,drw,T,D,E,A,32,32,32,bf,num_warps=4,num_stages=3)
  dw1=torch.empty_like(w1); _dw[(E,triton.cdiv(D,128),triton.cdiv(2*F,64))](x,dgu,tok,off,dw1,E,D,2*F,N,128,64,32,bf,True,num_warps=4,num_stages=3); dw2=torch.empty_like(w2); _dw[(E,triton.cdiv(F,128),triton.cdiv(D,64))](act,dz,tok,off,dw2,E,F,D,N,128,64,32,bf,False,num_warps=4,num_stages=3); return dx,drw,None,dw1,dw2,None

def fused_moe(x,router_w,router_b,w1,w2,A): return _Moe.apply(x,router_w,router_b,w1,w2,A)
