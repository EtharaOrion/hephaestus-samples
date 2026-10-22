import torch
import triton
import triton.language as tl

R, C, K = 64, 64, 32

@triton.jit
def _router(X,W,S,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,BM:tl.constexpr=32,BN:tl.constexpr=32,BK:tl.constexpr=32):
 m=tl.program_id(0)*BM+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); z=tl.zeros((BM,BN),tl.float32)
 for k in range(0,D,BK):
  q=k+tl.arange(0,BK); a=tl.load(X+m[:,None]*D+q[None,:],mask=(m[:,None]<T)&(q[None,:]<D),other=0.); b=tl.load(W+q[:,None]*E+n[None,:],mask=(q[:,None]<D)&(n[None,:]<E),other=0.)
  z+=tl.dot(a.to(tl.float32),b.to(tl.float32),input_precision="ieee")
 tl.store(S+m[:,None]*E+n[None,:],tl.sigmoid(z),mask=(m[:,None]<T)&(n[None,:]<E))

@triton.jit
def _keys(S,B,Q,N:tl.constexpr,E:tl.constexpr,BS:tl.constexpr=256):
 i=tl.program_id(0)*BS+tl.arange(0,BS); m=i<N; e=i%E; v=tl.load(S+i,mask=m,other=0.)+tl.load(B+e,mask=m,other=0.); v=tl.where(v==0.,0.,v); b=v.to(tl.int32,bitcast=True).to(tl.int64); u=tl.where(b>=0,b^0x80000000,(~b)&0xFFFFFFFF); tl.store(Q+i,(u<<10)|(E-1-e),mask=m)

@triton.jit
def _offsets(SEL,ORDER,OFF,N:tl.constexpr,E:tl.constexpr):
 e=tl.program_id(0); lo=tl.zeros((),tl.int32); hi=tl.full((),N,tl.int32)
 for _ in tl.static_range(24):
  mid=(lo+hi)//2; p=tl.load(ORDER+mid,mask=mid<N,other=0); v=tl.load(SEL+p,mask=mid<N,other=E); left=v<e; lo=tl.where(left,mid+1,lo); hi=tl.where(left,hi,mid)
 tl.store(OFF+e,lo)

@triton.jit
def _fill(TE,TS,CAP:tl.constexpr,BS:tl.constexpr=256):
 i=tl.program_id(0)*BS+tl.arange(0,BS); tl.store(TE+i,-1,mask=i<CAP); tl.store(TS+i,0,mask=i<CAP)

@triton.jit
def _tiles(OFF,TE,TS,E:tl.constexpr,ROW:tl.constexpr):
 total=tl.zeros((),tl.int32)
 for e in tl.range(0,E):
  b=tl.load(OFF+e); n=tl.load(OFF+e+1)-b; q=(n+ROW-1)//ROW
  for j in tl.range(0,q): tl.store(TE+total+j,e); tl.store(TS+total+j,b+j*ROW)
  total+=q

@triton.jit
def _inverse(ORDER,INV,N:tl.constexpr,BS:tl.constexpr=256):
 i=tl.program_id(0)*BS+tl.arange(0,BS); m=i<N; p=tl.load(ORDER+i,mask=m,other=0); tl.store(INV+p,i,mask=m)

@triton.jit
def _ein(X,W,ORDER,OFF,TE,TS,OUT,N:tl.constexpr,D:tl.constexpr,O:tl.constexpr,A:tl.constexpr,BF:tl.constexpr,BM:tl.constexpr=64,BN:tl.constexpr=64,BK:tl.constexpr=32):
 q=tl.program_id(0); e=tl.load(TE+q); end=tl.load(OFF+e+1,mask=e>=0,other=0); st=tl.load(TS+q); r=st+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); pair=tl.load(ORDER+r,mask=(e>=0)&(r<end),other=0); tok=pair//A; z=tl.zeros((BM,BN),tl.float32)
 for k in range(0,D,BK):
  x=k+tl.arange(0,BK); a=tl.load(X+tok[:,None]*D+x[None,:],mask=(e>=0)&(r[:,None]<end)&(x[None,:]<D),other=0.); b=tl.load(W+e*D*O+x[:,None]*O+n[None,:],mask=(e>=0)&(x[:,None]<D)&(n[None,:]<O),other=0.)
  if BF: z+=tl.dot(a,b)
  else: z+=tl.dot(a.to(tl.float32),b.to(tl.float32),input_precision="tf32x3")
 tl.store(OUT+r[:,None]*O+n[None,:],z,mask=(e>=0)&(r[:,None]<end)&(n[None,:]<O))

@triton.jit
def _act(GU,ACT,N:tl.constexpr,F:tl.constexpr,BS:tl.constexpr=256):
 i=tl.program_id(0)*BS+tl.arange(0,BS); m=i<N*F; r=i//F; c=i%F; g=tl.load(GU+r*2*F+c,mask=m,other=0.); u=tl.load(GU+r*2*F+F+c,mask=m,other=0.); tl.store(ACT+i,g*tl.sigmoid(g)*u,mask=m)

@triton.jit
def _seg(A0,W,OFF,TE,TS,OUT,N:tl.constexpr,KI:tl.constexpr,O:tl.constexpr,BF:tl.constexpr,SPLIT:tl.constexpr,BM:tl.constexpr=64,BN:tl.constexpr=64,BK:tl.constexpr=32):
 q=tl.program_id(0); e=tl.load(TE+q); end=tl.load(OFF+e+1,mask=e>=0,other=0); r=tl.load(TS+q)+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); z=tl.zeros((BM,BN),tl.float32)
 for k in range(0,KI,BK):
  x=k+tl.arange(0,BK); a=tl.load(A0+r[:,None]*KI+x[None,:],mask=(e>=0)&(r[:,None]<end)&(x[None,:]<KI),other=0.); b=tl.load(W+e*KI*O+x[:,None]*O+n[None,:],mask=(e>=0)&(x[:,None]<KI)&(n[None,:]<O),other=0.)
  if BF:
   ah=a.to(tl.bfloat16); z+=tl.dot(ah,b)
   if SPLIT: z+=tl.dot((a-ah.to(tl.float32)).to(tl.bfloat16),b)
  else: z+=tl.dot(a.to(tl.float32),b.to(tl.float32),input_precision="tf32x3")
 tl.store(OUT+r[:,None]*O+n[None,:],z,mask=(e>=0)&(r[:,None]<end)&(n[None,:]<O))

@triton.jit
def _combine(S,SEL,INV,FE,Y,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,E:tl.constexpr,BD:tl.constexpr=256):
 t=tl.program_id(0); d=tl.program_id(1)*BD+tl.arange(0,BD); den=tl.zeros((),tl.float32)
 for a in range(0,A):
  e=tl.load(SEL+t*A+a); den+=tl.load(S+t*E+e)
 y=tl.zeros((BD,),tl.float32)
 for a in range(0,A):
  p=t*A+a; e=tl.load(SEL+p); r=tl.load(INV+p); y+=tl.load(S+t*E+e)/den*tl.load(FE+r*D+d,mask=d<D,other=0.)
 tl.store(Y+t*D+d,y,mask=d<D)

@triton.jit
def _dz(DY,S,SEL,INV,DZ,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,E:tl.constexpr,BD:tl.constexpr=256):
 t=tl.program_id(0); d=tl.program_id(1)*BD+tl.arange(0,BD); den=tl.zeros((),tl.float32)
 for a in range(0,A):
  e=tl.load(SEL+t*A+a); den+=tl.load(S+t*E+e)
 dy=tl.load(DY+t*D+d,mask=d<D,other=0.).to(tl.float32)
 for a in range(0,A):
  p=t*A+a; e=tl.load(SEL+p); r=tl.load(INV+p); tl.store(DZ+r*D+d,dy*tl.load(S+t*E+e)/den,mask=d<D)

@triton.jit
def _pair_dot(DY,FE,ORDER,Q,N:tl.constexpr,D:tl.constexpr,A:tl.constexpr,BD:tl.constexpr=256):
 r=tl.program_id(0); pair=tl.load(ORDER+r); t=pair//A; z=tl.zeros((),tl.float32)
 for d0 in range(0,D,BD):
  d=d0+tl.arange(0,BD); dy=tl.load(DY+t*D+d,mask=d<D,other=0.).to(tl.float32); f=tl.load(FE+r*D+d,mask=d<D,other=0.); z+=tl.sum(dy*f,axis=0)
 tl.store(Q+r,z)

@triton.jit
def _segbt(A0,W,OFF,TE,TS,OUT,N:tl.constexpr,KI:tl.constexpr,O:tl.constexpr,BF:tl.constexpr,SPLIT:tl.constexpr,BM:tl.constexpr=64,BN:tl.constexpr=64,BK:tl.constexpr=32):
 q=tl.program_id(0); e=tl.load(TE+q); end=tl.load(OFF+e+1,mask=e>=0,other=0); r=tl.load(TS+q)+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); z=tl.zeros((BM,BN),tl.float32)
 for k in range(0,KI,BK):
  x=k+tl.arange(0,BK); a=tl.load(A0+r[:,None]*KI+x[None,:],mask=(e>=0)&(r[:,None]<end)&(x[None,:]<KI),other=0.); bt=tl.load(W+e*O*KI+n[None,:]*KI+x[:,None],mask=(e>=0)&(n[None,:]<O)&(x[:,None]<KI),other=0.); b=bt
  if BF:
   ah=a.to(tl.bfloat16); z+=tl.dot(ah,b)
   if SPLIT: z+=tl.dot((a-ah.to(tl.float32)).to(tl.bfloat16),b)
  else: z+=tl.dot(a.to(tl.float32),b.to(tl.float32),input_precision="tf32x3")
 tl.store(OUT+r[:,None]*O+n[None,:],z,mask=(e>=0)&(r[:,None]<end)&(n[None,:]<O))

@triton.jit
def _dgu(GU,DA,DG,N:tl.constexpr,F:tl.constexpr,BS:tl.constexpr=256):
 i=tl.program_id(0)*BS+tl.arange(0,BS); m=i<N*F; r=i//F; c=i%F; g=tl.load(GU+r*2*F+c,mask=m,other=0.); u=tl.load(GU+r*2*F+F+c,mask=m,other=0.); da=tl.load(DA+i,mask=m,other=0.); s=tl.sigmoid(g); silu=g*s; tl.store(DG+r*2*F+c,da*u*(s+silu*(1-s)),mask=m); tl.store(DG+r*2*F+F+c,da*silu,mask=m)

@triton.jit
def _dw(A0,G,ORDER,OFF,OUT,MI:tl.constexpr,NO:tl.constexpr,ASEL:tl.constexpr,GATHER:tl.constexpr,BF:tl.constexpr,BM:tl.constexpr=128,BN:tl.constexpr=64,BK:tl.constexpr=32):
 e=tl.program_id(0); m=tl.program_id(1)*BM+tl.arange(0,BM); n=tl.program_id(2)*BN+tl.arange(0,BN); begin=tl.load(OFF+e); end=tl.load(OFF+e+1); z=tl.zeros((BM,BN),tl.float32)
 for p0 in range(begin,end,BK):
  p=p0+tl.arange(0,BK)
  if GATHER: row=tl.load(ORDER+p,mask=p<end,other=0)//ASEL
  else: row=p
  a=tl.load(A0+row[None,:]*MI+m[:,None],mask=(p[None,:]<end)&(m[:,None]<MI),other=0.); g=tl.load(G+p[:,None]*NO+n[None,:],mask=(p[:,None]<end)&(n[None,:]<NO),other=0.)
  if BF:
   ah=a.to(tl.bfloat16); gh=g.to(tl.bfloat16); z+=tl.dot(ah,gh)+tl.dot(ah,(g-gh.to(tl.float32)).to(tl.bfloat16))
   if not GATHER: z+=tl.dot((a-ah.to(tl.float32)).to(tl.bfloat16),gh)
  else: z+=tl.dot(a.to(tl.float32),g.to(tl.float32),input_precision="tf32x3")
 tl.store(OUT+e*MI*NO+m[:,None]*NO+n[None,:],z,mask=(m[:,None]<MI)&(n[None,:]<NO))

@triton.jit
def _route(S,SEL,INV,Q,DLS,T:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BS:tl.constexpr=128):
 t=tl.program_id(0)*BS+tl.arange(0,BS); mt=t<T; den=tl.zeros((BS,),tl.float32); avg=tl.zeros((BS,),tl.float32)
 for a in range(0,A):
  p=t*A+a; e=tl.load(SEL+p,mask=mt,other=0); r=tl.load(INV+p,mask=mt,other=0); s=tl.load(S+t*E+e,mask=mt,other=0.); q=tl.load(Q+r,mask=mt,other=0.); den+=s; avg+=s*q
 avg/=den
 for a in range(0,A):
  p=t*A+a; e=tl.load(SEL+p,mask=mt,other=0); r=tl.load(INV+p,mask=mt,other=0); s=tl.load(S+t*E+e,mask=mt,other=0.); q=tl.load(Q+r,mask=mt,other=0.); tl.store(DLS+p,(q-avg)/den*s*(1-s),mask=mt)

@triton.jit
def _scatter(DLS,SEL,DL,T:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BS:tl.constexpr=256):
 i=tl.program_id(0)*BS+tl.arange(0,BS); mask=i<T*E; t=i//E; e=i%E; v=tl.zeros((BS,),tl.float32)
 for a in range(0,A): v+=tl.where(tl.load(SEL+t*A+a,mask=mask,other=-1)==e,tl.load(DLS+t*A+a,mask=mask,other=0.),0.)
 tl.store(DL+i,v,mask=mask)

@triton.jit
def _dense(A0,B0,C0,M:tl.constexpr,N:tl.constexpr,KR:tl.constexpr,TA:tl.constexpr,TB:tl.constexpr,BM:tl.constexpr=64,BN:tl.constexpr=64,BK:tl.constexpr=32):
 m=tl.program_id(0)*BM+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); z=tl.zeros((BM,BN),tl.float32)
 for k in range(0,KR,BK):
  x=k+tl.arange(0,BK)
  if TA: a=tl.load(A0+x[None,:]*M+m[:,None],mask=(m[:,None]<M)&(x[None,:]<KR),other=0.)
  else: a=tl.load(A0+m[:,None]*KR+x[None,:],mask=(m[:,None]<M)&(x[None,:]<KR),other=0.)
  if TB: b=tl.load(B0+n[None,:]*KR+x[:,None],mask=(n[None,:]<N)&(x[:,None]<KR),other=0.)
  else: b=tl.load(B0+x[:,None]*N+n[None,:],mask=(x[:,None]<KR)&(n[None,:]<N),other=0.)
  z+=tl.dot(a.to(tl.float32),b.to(tl.float32),input_precision="ieee")
 tl.store(C0+m[:,None]*N+n[None,:],z,mask=(m[:,None]<M)&(n[None,:]<N))

@triton.jit
def _sumdx(DXE,DXR,INV,DX,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,BD:tl.constexpr=256):
 t=tl.program_id(0); d=tl.program_id(1)*BD+tl.arange(0,BD); z=tl.load(DXR+t*D+d,mask=d<D,other=0.)
 for a in range(0,A): z+=tl.load(DXE+tl.load(INV+t*A+a)*D+d,mask=d<D,other=0.)
 tl.store(DX+t*D+d,z,mask=d<D)

def _meta(sel,A,E):
 n=sel.numel(); order=torch.argsort(sel.reshape(-1)); off=torch.empty(E+1,device=sel.device,dtype=torch.int32); _offsets[(E+1,)](sel,order,off,N=n,E=E,num_warps=1); cap=triton.cdiv(n,R)+E; te=torch.empty(cap,device=sel.device,dtype=torch.int32); ts=torch.empty_like(te); _fill[(triton.cdiv(cap,256),)](te,ts,CAP=cap); _tiles[(1,)](off,te,ts,E=E,ROW=R,num_warps=1); inv=torch.empty(n,device=sel.device,dtype=torch.int64); _inverse[(triton.cdiv(n,256),)](order,inv,N=n); return order,off,te,ts,inv,cap

class _Moe(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,rw,rb,w1,w2,A):
  A=int(A); T,D=x.shape; E=rw.shape[1]; F=w2.shape[1]; N=T*A; bf=x.dtype==torch.bfloat16; s=torch.empty((T,E),device=x.device,dtype=torch.float32); _router[(triton.cdiv(T,32),triton.cdiv(E,32))](x,rw,s,T=T,D=D,E=E); keys=torch.empty((T,E),device=x.device,dtype=torch.int64); _keys[(triton.cdiv(T*E,256),)](s,rb,keys,N=T*E,E=E); sel=torch.topk(keys,A,dim=1,largest=True,sorted=True).indices; order,off,te,ts,inv,cap=_meta(sel,A,E); gu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _ein[(cap,triton.cdiv(2*F,C))](x,w1,order,off,te,ts,gu,N=N,D=D,O=2*F,A=A,BF=bf,num_warps=4,num_stages=3); act=torch.empty((N,F),device=x.device,dtype=torch.float32); _act[(triton.cdiv(N*F,256),)](gu,act,N=N,F=F); fe=torch.empty((N,D),device=x.device,dtype=torch.float32); _seg[(cap,triton.cdiv(D,C))](act,w2,off,te,ts,fe,N=N,KI=F,O=D,BF=bf,SPLIT=bf,num_warps=4,num_stages=3); y=torch.empty((T,D),device=x.device,dtype=x.dtype); _combine[(T,triton.cdiv(D,256))](s,sel,inv,fe,y,T=T,D=D,A=A,E=E); ctx.save_for_backward(x,rw,w1,w2,s,sel,order,off,te,ts,inv,gu,act,fe); ctx.meta=T,D,E,F,A,N,cap,bf; return y
 @staticmethod
 def backward(ctx,dy):
  x,rw,w1,w2,s,sel,order,off,te,ts,inv,gu,act,fe=ctx.saved_tensors; T,D,E,F,A,N,cap,bf=ctx.meta; dz=torch.empty((N,D),device=x.device,dtype=torch.float32); q=torch.empty(N,device=x.device,dtype=torch.float32); _dz[(T,triton.cdiv(D,256))](dy,s,sel,inv,dz,T=T,D=D,A=A,E=E); _pair_dot[(N,)](dy,fe,order,q,N=N,D=D,A=A); da=torch.empty((N,F),device=x.device,dtype=torch.float32); _segbt[(cap,triton.cdiv(F,C))](dz,w2,off,te,ts,da,N=N,KI=D,O=F,BF=bf,SPLIT=bf,num_warps=4,num_stages=3); dgu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _dgu[(triton.cdiv(N*F,256),)](gu,da,dgu,N=N,F=F); dxe=torch.empty((N,D),device=x.device,dtype=torch.float32); _segbt[(cap,triton.cdiv(D,C))](dgu,w1,off,te,ts,dxe,N=N,KI=2*F,O=D,BF=bf,SPLIT=False,num_warps=4,num_stages=3); gw2=torch.empty_like(w2); _dw[(E,triton.cdiv(F,128),triton.cdiv(D,64))](act,dz,order,off,gw2,MI=F,NO=D,ASEL=A,GATHER=False,BF=bf,num_warps=4,num_stages=3); gw1=torch.empty_like(w1); _dw[(E,triton.cdiv(D,128),triton.cdiv(2*F,64))](x,dgu,order,off,gw1,MI=D,NO=2*F,ASEL=A,GATHER=True,BF=bf,num_warps=4,num_stages=3); dls=torch.empty((T,A),device=x.device,dtype=torch.float32); _route[(triton.cdiv(T,128),)](s,sel,inv,q,dls,T=T,E=E,A=A); dl=torch.empty((T,E),device=x.device,dtype=torch.float32); _scatter[(triton.cdiv(T*E,256),)](dls,sel,dl,T=T,E=E,A=A); grw=torch.empty_like(rw); _dense[(triton.cdiv(D,64),triton.cdiv(E,64))](x,dl,grw,M=D,N=E,KR=T,TA=True,TB=False); dxr=torch.empty((T,D),device=x.device,dtype=torch.float32); _dense[(triton.cdiv(T,64),triton.cdiv(D,64))](dl,rw,dxr,M=T,N=D,KR=E,TA=False,TB=True); dx=torch.empty_like(x); _sumdx[(T,triton.cdiv(D,256))](dxe,dxr,inv,dx,T=T,D=D,A=A); return dx,grw,None,gw1,gw2,None

def fused_moe(x,router_w,router_b,w1,w2,A): return _Moe.apply(x,router_w,router_b,w1,w2,A)
