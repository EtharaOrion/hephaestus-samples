import torch
import triton
import triton.language as tl

R, C, KB = 64, 64, 32

@triton.jit
def _dot(a, b, bf: tl.constexpr, sa: tl.constexpr, sb: tl.constexpr):
    if bf:
        ah, bh = a.to(tl.bfloat16), b.to(tl.bfloat16)
        z = tl.dot(ah, bh)
        if sa: z += tl.dot((a-ah).to(tl.bfloat16), bh)
        if sb: z += tl.dot(ah, (b-bh).to(tl.bfloat16))
        return z
    return tl.dot(a, b, input_precision="tf32x3")

@triton.jit
def _router(x,rw,s,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,B:tl.constexpr):
    im=tl.program_id(0)*B+tl.arange(0,B); ie=tl.program_id(1)*B+tl.arange(0,B)
    z=tl.zeros((B,B),tl.float32)
    for k in range(0,D,B):
        ik=k+tl.arange(0,B)
        a=tl.load(x+im[:,None]*D+ik[None,:],mask=(im[:,None]<T)&(ik[None,:]<D),other=0.)
        b=tl.load(rw+ik[:,None]*E+ie[None,:],mask=(ik[:,None]<D)&(ie[None,:]<E),other=0.)
        z+=tl.dot(a,b,input_precision="ieee")
    tl.store(s+im[:,None]*E+ie[None,:],tl.sigmoid(z),mask=(im[:,None]<T)&(ie[None,:]<E))

@triton.jit
def _keys(s,bias,keys,n:tl.constexpr,E:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B); m=i<n; e=i%E
    v=tl.load(s+i,mask=m,other=0.)+tl.load(bias+e,mask=m,other=0.); v=tl.where(v==0.,0.,v)
    ib=v.to(tl.int32,bitcast=True); wide=ib.to(tl.int64)
    u=tl.where(ib>=0,wide^0x80000000,(~wide)&0xffffffff)
    tl.store(keys+i,(u<<10)|(E-1-e).to(tl.int64),mask=m)

@triton.jit
def _weights(s,sel,w,ss,T:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BA:tl.constexpr):
    t=tl.program_id(0); a=tl.arange(0,BA); m=a<A; e=tl.load(sel+t*A+a,mask=m,other=0)
    v=tl.load(s+t*E+e,mask=m,other=0.); den=tl.sum(v,axis=0)
    tl.store(w+t*A+a,v/den,mask=m); tl.store(ss+t,den)

@triton.jit
def _offsets(se,off,N:tl.constexpr,E:tl.constexpr,IT:tl.constexpr):
    e=tl.program_id(0); lo=0; hi=N
    for _ in range(IT):
        md=(lo+hi)//2; v=tl.load(se+md,mask=md<N,other=E); take=v<e
        lo=tl.where(take,md+1,lo); hi=tl.where(take,hi,md)
    tl.store(off+e,lo)

@triton.jit
def _prefix(off,pre,E:tl.constexpr,BR:tl.constexpr):
    q=0; tl.store(pre,0)
    for e in range(E):
        q+=tl.cdiv(tl.load(off+e+1)-tl.load(off+e),BR); tl.store(pre+e+1,q)

@triton.jit
def _loc(pid,off,pre,E:tl.constexpr,IT:tl.constexpr,BR:tl.constexpr):
    lo=0; hi=E
    for _ in range(IT):
        md=(lo+hi)//2; take=tl.load(pre+md+1)<=pid
        lo=tl.where(take,md+1,lo); hi=tl.where(take,hi,md)
    e=lo; valid=e<E; eb=tl.load(off+e,mask=valid,other=0); pe=tl.load(pre+e,mask=valid,other=0)
    return e,eb+(pid-pe)*BR,valid

@triton.jit
def _ein(x,w1,order,off,pre,gu,N:tl.constexpr,D:tl.constexpr,F2:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,IT:tl.constexpr,BF:tl.constexpr):
    pid=tl.program_id(0); e,p0,valid=_loc(pid,off,pre,E,IT,BM)
    ir=tl.arange(0,BM); io=tl.program_id(1)*BN+tl.arange(0,BN); pos=p0+ir
    end=tl.load(off+e+1,mask=valid,other=0); mp=valid&(pos<end); pair=tl.load(order+pos,mask=mp,other=0); tok=pair//A
    z=tl.zeros((BM,BN),tl.float32)
    for k in range(0,D,BK):
        ik=k+tl.arange(0,BK)
        aa=tl.load(x+tok[:,None]*D+ik[None,:],mask=mp[:,None]&(ik[None,:]<D),other=0.).to(tl.float32)
        bb=tl.load(w1+e*D*F2+ik[:,None]*F2+io[None,:],mask=valid&(ik[:,None]<D)&(io[None,:]<F2),other=0.).to(tl.float32)
        z+=_dot(aa,bb,BF,False,False)
    tl.store(gu+pair[:,None]*F2+io[None,:],z,mask=mp[:,None]&(io[None,:]<F2))

@triton.jit
def _act(gu,act,n:tl.constexpr,F:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B); m=i<n; p=i//F; f=i%F
    g=tl.load(gu+p*(2*F)+f,mask=m,other=0.); u=tl.load(gu+p*(2*F)+F+f,mask=m,other=0.)
    tl.store(act+i,g*tl.sigmoid(g)*u,mask=m)

@triton.jit
def _seg(a,b,order,off,pre,out,N:tl.constexpr,K0:tl.constexpr,O:tl.constexpr,E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,IT:tl.constexpr,BF:tl.constexpr,SA:tl.constexpr):
    pid=tl.program_id(0); e,p0,valid=_loc(pid,off,pre,E,IT,BM)
    ir=tl.arange(0,BM); io=tl.program_id(1)*BN+tl.arange(0,BN); pos=p0+ir
    end=tl.load(off+e+1,mask=valid,other=0); mp=valid&(pos<end); pair=tl.load(order+pos,mask=mp,other=0)
    z=tl.zeros((BM,BN),tl.float32)
    for k in range(0,K0,BK):
        ik=k+tl.arange(0,BK)
        aa=tl.load(a+pair[:,None]*K0+ik[None,:],mask=mp[:,None]&(ik[None,:]<K0),other=0.).to(tl.float32)
        bb=tl.load(b+e*K0*O+ik[:,None]*O+io[None,:],mask=valid&(ik[:,None]<K0)&(io[None,:]<O),other=0.).to(tl.float32)
        z+=_dot(aa,bb,BF,SA,False)
    tl.store(out+pair[:,None]*O+io[None,:],z,mask=mp[:,None]&(io[None,:]<O))

@triton.jit
def _combine(fe,w,y,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,BD:tl.constexpr):
    t=tl.program_id(0); d=tl.program_id(1)*BD+tl.arange(0,BD); z=tl.zeros((BD,),tl.float32)
    for a in range(A):
        p=t*A+a; z+=tl.load(w+p)*tl.load(fe+p*D+d,mask=d<D,other=0.)
    tl.store(y+t*D+d,z,mask=d<D)

@triton.jit
def _dz_pair(dy,fe,w,q,dz,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,BD:tl.constexpr):
    p=tl.program_id(0); d=tl.arange(0,BD); t=p//A; acc=tl.zeros((BD,),tl.float32); ww=tl.load(w+p)
    for k in range(0,D,BD):
        dk=k+d; g=tl.load(dy+t*D+dk,mask=dk<D,other=0.).to(tl.float32); f=tl.load(fe+p*D+dk,mask=dk<D,other=0.)
        acc+=g*f; tl.store(dz+p*D+dk,g*ww,mask=dk<D)
    tl.store(q+p,tl.sum(acc,axis=0))

@triton.jit
def _route(q,w,s,sel,ss,dense,T:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BA:tl.constexpr):
    t=tl.program_id(0); a=tl.arange(0,BA); m=a<A; p=t*A+a
    ww=tl.load(w+p,mask=m,other=0.); qq=tl.load(q+p,mask=m,other=0.); avg=tl.sum(ww*qq,axis=0)
    ee=tl.load(sel+p,mask=m,other=0); sv=tl.load(s+t*E+ee,mask=m,other=0.); vv=(qq-avg)/tl.load(ss+t)*sv*(1.-sv)
    for e0 in range(0,E,BA):
        ei=e0+a
        match=ei[:,None]==ee[None,:]
        out=tl.sum(tl.where(match,vv[None,:],0.0),axis=1)
        tl.store(dense+t*E+ei,out,mask=ei<E)

@triton.jit
def _segbt(a,b,order,off,pre,out,N:tl.constexpr,K0:tl.constexpr,O:tl.constexpr,E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,IT:tl.constexpr,BF:tl.constexpr,SA:tl.constexpr):
    pid=tl.program_id(0); e,p0,valid=_loc(pid,off,pre,E,IT,BM)
    ir=tl.arange(0,BM); io=tl.program_id(1)*BN+tl.arange(0,BN); pos=p0+ir
    end=tl.load(off+e+1,mask=valid,other=0); mp=valid&(pos<end); pair=tl.load(order+pos,mask=mp,other=0)
    z=tl.zeros((BM,BN),tl.float32)
    for k in range(0,K0,BK):
        ik=k+tl.arange(0,BK)
        aa=tl.load(a+pair[:,None]*K0+ik[None,:],mask=mp[:,None]&(ik[None,:]<K0),other=0.).to(tl.float32)
        bb=tl.load(b+e*O*K0+io[None,:]*K0+ik[:,None],mask=valid&(io[None,:]<O)&(ik[:,None]<K0),other=0.).to(tl.float32)
        z+=_dot(aa,bb,BF,SA,False)
    tl.store(out+pair[:,None]*O+io[None,:],z,mask=mp[:,None]&(io[None,:]<O))

@triton.jit
def _dgu(gu,da,out,n:tl.constexpr,F:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B); m=i<n; p=i//F; f=i%F
    g=tl.load(gu+p*(2*F)+f,mask=m,other=0.); u=tl.load(gu+p*(2*F)+F+f,mask=m,other=0.); v=tl.load(da+i,mask=m,other=0.)
    sg=tl.sigmoid(g); sil=g*sg
    tl.store(out+p*(2*F)+f,v*u*(sg+sil*(1.-sg)),mask=m); tl.store(out+p*(2*F)+F+f,v*sil,mask=m)

@triton.jit
def _dw(a,b,order,off,out,N:tl.constexpr,K0:tl.constexpr,O:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,A_X:tl.constexpr,BF:tl.constexpr,SA:tl.constexpr,SB:tl.constexpr):
    e=tl.program_id(0); ik=tl.program_id(1)*BM+tl.arange(0,BM); io=tl.program_id(2)*BN+tl.arange(0,BN)
    beg=tl.load(off+e); end=tl.load(off+e+1); z=tl.zeros((BM,BN),tl.float32)
    for p0 in range(beg,end,BK):
        ip=p0+tl.arange(0,BK); pair=tl.load(order+ip,mask=ip<end,other=0); row=pair//A if A_X else pair
        aa=tl.load(a+row[None,:]*K0+ik[:,None],mask=(ip[None,:]<end)&(ik[:,None]<K0),other=0.).to(tl.float32)
        bb=tl.load(b+pair[:,None]*O+io[None,:],mask=(ip[:,None]<end)&(io[None,:]<O),other=0.).to(tl.float32)
        z+=_dot(aa,bb,BF,SA,SB)
    tl.store(out+e*K0*O+ik[:,None]*O+io[None,:],z,mask=(ik[:,None]<K0)&(io[None,:]<O))

@triton.jit
def _sumdx(dxp,ds,rw,dx,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BD:tl.constexpr):
    t=tl.program_id(0); d=tl.program_id(1)*BD+tl.arange(0,BD); z=tl.zeros((BD,),tl.float32)
    for a in range(A): z+=tl.load(dxp+(t*A+a)*D+d,mask=d<D,other=0.)
    for e0 in range(0,E,32):
        e=e0+tl.arange(0,32); sv=tl.load(ds+t*E+e,mask=e<E,other=0.); rv=tl.load(rw+d[:,None]*E+e[None,:],mask=(d[:,None]<D)&(e[None,:]<E),other=0.)
        z+=tl.sum(rv*sv[None,:],axis=1)
    tl.store(dx+t*D+d,z,mask=d<D)

@triton.jit
def _drw(x,ds,out,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    d=tl.program_id(0)*BM+tl.arange(0,BM); e=tl.program_id(1)*BN+tl.arange(0,BN); z=tl.zeros((BM,BN),tl.float32)
    for t0 in range(0,T,BK):
        t=t0+tl.arange(0,BK); a=tl.load(x+t[None,:]*D+d[:,None],mask=(d[:,None]<D)&(t[None,:]<T),other=0.).to(tl.float32)
        b=tl.load(ds+t[:,None]*E+e[None,:],mask=(t[:,None]<T)&(e[None,:]<E),other=0.); z+=tl.dot(a,b,input_precision="ieee")
    tl.store(out+d[:,None]*E+e[None,:],z,mask=(d[:,None]<D)&(e[None,:]<E))

class _Moe(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,rw,rb,w1,w2,A):
    A=int(A); T,D=x.shape; E=rw.shape[1]; F=w2.shape[1]; N=T*A; bf=x.dtype==torch.bfloat16; it=max(1,E.bit_length())
    s=torch.empty((T,E),device=x.device,dtype=torch.float32); _router[(triton.cdiv(T,32),triton.cdiv(E,32))](x,rw,s,T=T,D=D,E=E,B=32)
    keys=torch.empty((T,E),device=x.device,dtype=torch.int64); _keys[(triton.cdiv(T*E,256),)](s,rb,keys,n=T*E,E=E,B=256)
    sel=torch.topk(keys,A,dim=1,largest=True,sorted=True).indices.contiguous(); flat=sel.reshape(-1); order=torch.argsort(flat).contiguous(); se=flat.index_select(0,order)
    off=torch.empty(E+1,device=x.device,dtype=torch.int32); _offsets[(E+1,)](se,off,N=N,E=E,IT=max(1,N.bit_length()))
    pre=torch.empty(E+1,device=x.device,dtype=torch.int32); _prefix[(1,)](off,pre,E=E,BR=R,num_warps=1); cap=triton.cdiv(N,R)+E
    weights=torch.empty(N,device=x.device,dtype=torch.float32); sums=torch.empty(T,device=x.device,dtype=torch.float32); ba=triton.next_power_of_2(A)
    _weights[(T,)](s,sel,weights,sums,T=T,E=E,A=A,BA=ba)
    gu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _ein[(cap,triton.cdiv(2*F,C))](x,w1,order,off,pre,gu,N=N,D=D,F2=2*F,E=E,A=A,BM=R,BN=C,BK=KB,IT=it,BF=bf,num_warps=4,num_stages=3)
    act=torch.empty((N,F),device=x.device,dtype=torch.float32); _act[(triton.cdiv(N*F,256),)](gu,act,n=N*F,F=F,B=256)
    fe=torch.empty((N,D),device=x.device,dtype=torch.float32); _seg[(cap,triton.cdiv(D,C))](act,w2,order,off,pre,fe,N=N,K0=F,O=D,E=E,BM=R,BN=C,BK=KB,IT=it,BF=bf,SA=bf,num_warps=4,num_stages=3)
    y=torch.empty((T,D),device=x.device,dtype=x.dtype); _combine[(T,triton.cdiv(D,256))](fe,weights,y,T=T,D=D,A=A,BD=256)
    ctx.save_for_backward(x,rw,w1,w2,s,sel,order,off,pre,weights,sums,gu,act,fe); ctx.meta=(T,D,E,F,A,N,cap,it,bf)
    return y
 @staticmethod
 def backward(ctx,dy):
    x,rw,w1,w2,s,sel,order,off,pre,weights,sums,gu,act,fe=ctx.saved_tensors; T,D,E,F,A,N,cap,it,bf=ctx.meta
    q=torch.empty(N,device=x.device,dtype=torch.float32); dz=torch.empty((N,D),device=x.device,dtype=torch.float32); _dz_pair[(N,)](dy,fe,weights,q,dz,T=T,D=D,A=A,BD=256)
    dense=torch.empty((T,E),device=x.device,dtype=torch.float32); _route[(T,)](q,weights,s,sel,sums,dense,T=T,E=E,A=A,BA=triton.next_power_of_2(A))
    da=torch.empty((N,F),device=x.device,dtype=torch.float32); _segbt[(cap,triton.cdiv(F,C))](dz,w2,order,off,pre,da,N=N,K0=D,O=F,E=E,BM=R,BN=C,BK=KB,IT=it,BF=bf,SA=False,num_warps=4,num_stages=3)
    dgu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _dgu[(triton.cdiv(N*F,256),)](gu,da,dgu,n=N*F,F=F,B=256)
    dxp=torch.empty((N,D),device=x.device,dtype=torch.float32); _segbt[(cap,triton.cdiv(D,C))](dgu,w1,order,off,pre,dxp,N=N,K0=2*F,O=D,E=E,BM=R,BN=C,BK=KB,IT=it,BF=bf,SA=False,num_warps=4,num_stages=3)
    gw2=torch.empty_like(w2); _dw[(E,triton.cdiv(F,128),triton.cdiv(D,64))](act,dz,order,off,gw2,N=N,K0=F,O=D,E=E,A=A,BM=128,BN=64,BK=KB,A_X=False,BF=bf,SA=bf,SB=bf,num_warps=4,num_stages=3)
    gw1=torch.empty_like(w1); _dw[(E,triton.cdiv(D,128),triton.cdiv(2*F,64))](x,dgu,order,off,gw1,N=N,K0=D,O=2*F,E=E,A=A,BM=128,BN=64,BK=KB,A_X=True,BF=bf,SA=False,SB=bf,num_warps=4,num_stages=3)
    gx=torch.empty_like(x); _sumdx[(T,triton.cdiv(D,256))](dxp,dense,rw,gx,T=T,D=D,E=E,A=A,BD=256)
    grw=torch.empty_like(rw); _drw[(triton.cdiv(D,64),triton.cdiv(E,64))](x,dense,grw,T=T,D=D,E=E,BM=64,BN=64,BK=32,num_warps=4,num_stages=3)
    return gx,grw,None,gw1,gw2,None

def fused_moe(x,router_w,router_b,w1,w2,A): return _Moe.apply(x,router_w,router_b,w1,w2,A)
