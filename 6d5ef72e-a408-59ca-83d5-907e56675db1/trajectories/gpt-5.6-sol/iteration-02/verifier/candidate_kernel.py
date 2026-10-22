import torch
import triton
import triton.language as tl


@triton.jit
def _router(X, W, S, T, D: tl.constexpr, E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm=tl.program_id(0)*BM+tl.arange(0,BM); rn=tl.program_id(1)*BN+tl.arange(0,BN); acc=tl.zeros((BM,BN),tl.float32)
    for k in range(0,D,BK):
        rk=k+tl.arange(0,BK); x=tl.load(X+rm[:,None]*D+rk[None,:],mask=(rm[:,None]<T)&(rk[None,:]<D),other=0.).to(tl.float32); w=tl.load(W+rk[:,None]*E+rn[None,:],mask=(rk[:,None]<D)&(rn[None,:]<E),other=0.).to(tl.float32); acc+=tl.dot(x,w,input_precision="ieee")
    tl.store(S+rm[:,None]*E+rn[None,:],tl.sigmoid(acc),mask=(rm[:,None]<T)&(rn[None,:]<E))


@triton.jit
def _metadata(ORDER, FLAT, S, TOK, EXP, WS, E: tl.constexpr, A: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B); m=i<N; q=tl.load(ORDER+i,mask=m,other=0); t=q//A; e=tl.load(FLAT+q,mask=m,other=0); den=tl.zeros((B,),tl.float32)
    for a in range(A):
        ea=tl.load(FLAT+t*A+a,mask=m,other=0); den+=tl.load(S+t*E+ea,mask=m,other=0.)
    raw=tl.load(S+t*E+e,mask=m,other=0.); tl.store(TOK+i,t,mask=m); tl.store(EXP+i,e,mask=m); tl.store(WS+i,raw/den,mask=m)


@triton.jit
def _counts(EXP, C, N: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    e=tl.program_id(0); z=tl.zeros((B,),tl.int32)
    for k in range(0,N,B):
        i=k+tl.arange(0,B); z+=(tl.load(EXP+i,mask=i<N,other=-1)==e).to(tl.int32)
    tl.store(C+e,tl.sum(z,axis=0))


@triton.jit
def _tiles(C, OFF, TE, TS, E: tl.constexpr, MT: tl.constexpr, BM: tl.constexpr):
    r=tl.arange(0,256)
    for k in range(0,MT,256): tl.store(TE+k+r,-1,mask=k+r<MT)
    off=0; nt=0
    for e in range(E):
        c=tl.load(C+e); tl.store(OFF+e,off); n=(c+BM-1)//BM
        for j in range(0,n): tl.store(TE+nt+j,e); tl.store(TS+nt+j,off+j*BM)
        nt+=n; off+=c
    tl.store(OFF+E,off)


@triton.jit
def _in(X,W,TOK,OFF,TE,TS,GU,D:tl.constexpr,F:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
    p=tl.program_id(0); e=tl.load(TE+p); st=tl.load(TS+p); begin=tl.load(OFF+e,mask=e>=0,other=0); end=tl.load(OFF+e+1,mask=e>=0,other=0); rm=tl.arange(0,BM); rn=tl.program_id(1)*BN+tl.arange(0,BN); rows=st+rm; m=(e>=0)&(rows<end); tok=tl.load(TOK+rows,mask=m,other=0); acc=tl.zeros((BM,BN),tl.float32)
    for k in range(0,D,BK):
        rk=k+tl.arange(0,BK); x=tl.load(X+tok[:,None]*D+rk[None,:],mask=m[:,None]&(rk[None,:]<D),other=0.); w=tl.load(W+e*D*(2*F)+rk[:,None]*(2*F)+rn[None,:],mask=(e>=0)&(rk[:,None]<D)&(rn[None,:]<2*F),other=0.)
        if BF: acc+=tl.dot(x,w)
        else: acc+=tl.dot(x.to(tl.float32),w.to(tl.float32),input_precision="tf32x3")
    tl.store(GU+rows[:,None]*(2*F)+rn[None,:],acc,mask=m[:,None]&(rn[None,:]<2*F))


@triton.jit
def _out(GU,W,OFF,TE,TS,ORDER,FE,D:tl.constexpr,F:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
    p=tl.program_id(0); e=tl.load(TE+p); st=tl.load(TS+p); end=tl.load(OFF+e+1,mask=e>=0,other=0); rm=tl.arange(0,BM); rn=tl.program_id(1)*BN+tl.arange(0,BN); rows=st+rm; m=(e>=0)&(rows<end); acc=tl.zeros((BM,BN),tl.float32)
    for k in range(0,F,BK):
        rk=k+tl.arange(0,BK); g=tl.load(GU+rows[:,None]*(2*F)+rk[None,:],mask=m[:,None]&(rk[None,:]<F),other=0.); u=tl.load(GU+rows[:,None]*(2*F)+F+rk[None,:],mask=m[:,None]&(rk[None,:]<F),other=0.); a=g*tl.sigmoid(g)*u; w=tl.load(W+e*F*D+rk[:,None]*D+rn[None,:],mask=(e>=0)&(rk[:,None]<F)&(rn[None,:]<D),other=0.)
        if BF:
            ah=a.to(tl.bfloat16); al=(a-ah.to(tl.float32)).to(tl.bfloat16); acc+=tl.dot(ah,w)+tl.dot(al,w)
        else: acc+=tl.dot(a,w.to(tl.float32),input_precision="tf32x3")
    q=tl.load(ORDER+rows,mask=m,other=0); tl.store(FE+q[:,None]*D+rn[None,:],acc,mask=m[:,None]&(rn[None,:]<D))


@triton.jit
def _combine(FE,S,FLAT,Y,D:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BD:tl.constexpr):
    t=tl.program_id(0); rd=tl.program_id(1)*BD+tl.arange(0,BD); den=0.
    for a in range(A):
        e=tl.load(FLAT+t*A+a); den+=tl.load(S+t*E+e)
    z=tl.zeros((BD,),tl.float32)
    for a in range(A):
        e=tl.load(FLAT+t*A+a); z+=tl.load(S+t*E+e)/den*tl.load(FE+(t*A+a)*D+rd,mask=rd<D,other=0.)
    tl.store(Y+t*D+rd,z,mask=rd<D)


@triton.jit
def _route_bwd(DY,FE,S,FLAT,DS,D:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BD:tl.constexpr,BA:tl.constexpr):
    t=tl.program_id(0); ra=tl.arange(0,BA); ma=ra<A; es=tl.load(FLAT+t*A+ra,mask=ma,other=0); raw=tl.load(S+t*E+es,mask=ma,other=0.); den=tl.sum(raw,axis=0); v=tl.zeros((BA,),tl.float32)
    for d in range(0,D,BD):
        rd=d+tl.arange(0,BD); dy=tl.load(DY+t*D+rd,mask=rd<D,other=0.)
        for a in range(A):
            dot=tl.sum(dy*tl.load(FE+(t*A+a)*D+rd,mask=rd<D,other=0.),axis=0); v+=tl.where(ra==a,dot,0.)
    mean=tl.sum(raw/den*v,axis=0); tl.store(DS+t*A+ra,(v-mean)/den,mask=ma)


@triton.jit
def _dgu(DY,WS,W2,GU,TOK,OFF,TE,TS,DGU,D:tl.constexpr,F:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
    p=tl.program_id(0); e=tl.load(TE+p); st=tl.load(TS+p); end=tl.load(OFF+e+1,mask=e>=0,other=0); rm=tl.arange(0,BM); rf=tl.program_id(1)*BN+tl.arange(0,BN); rows=st+rm; m=(e>=0)&(rows<end); tok=tl.load(TOK+rows,mask=m,other=0); ws=tl.load(WS+rows,mask=m,other=0.); acc=tl.zeros((BM,BN),tl.float32)
    for d in range(0,D,BK):
        rd=d+tl.arange(0,BK); z=tl.load(DY+tok[:,None]*D+rd[None,:],mask=m[:,None]&(rd[None,:]<D),other=0.)*ws[:,None]; w=tl.load(W2+e*F*D+rf[:,None]*D+rd[None,:],mask=(e>=0)&(rf[:,None]<F)&(rd[None,:]<D),other=0.)
        if BF:
            zh=z.to(tl.bfloat16); zl=(z-zh.to(tl.float32)).to(tl.bfloat16); acc+=tl.dot(zh,tl.trans(w))+tl.dot(zl,tl.trans(w))
        else: acc+=tl.dot(z,tl.trans(w.to(tl.float32)),input_precision="tf32x3")
    g=tl.load(GU+rows[:,None]*(2*F)+rf[None,:],mask=m[:,None]&(rf[None,:]<F),other=0.); u=tl.load(GU+rows[:,None]*(2*F)+F+rf[None,:],mask=m[:,None]&(rf[None,:]<F),other=0.); sg=tl.sigmoid(g); si=g*sg
    tl.store(DGU+rows[:,None]*(2*F)+rf[None,:],acc*u*(sg+si*(1.-sg)),mask=m[:,None]&(rf[None,:]<F)); tl.store(DGU+rows[:,None]*(2*F)+F+rf[None,:],acc*si,mask=m[:,None]&(rf[None,:]<F))


@triton.jit
def _dw2(DY,WS,GU,TOK,OFF,DW,D:tl.constexpr,F:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
    e=tl.program_id(0); rf=tl.program_id(1)*BM+tl.arange(0,BM); rd=tl.program_id(2)*BN+tl.arange(0,BN); begin=tl.load(OFF+e); end=tl.load(OFF+e+1); acc=tl.zeros((BM,BN),tl.float32)
    for k in range(begin,end,BK):
        rows=k+tl.arange(0,BK); m=rows<end; tok=tl.load(TOK+rows,mask=m,other=0); ws=tl.load(WS+rows,mask=m,other=0.); g=tl.load(GU+rf[:,None]+rows[None,:]*(2*F),mask=(rf[:,None]<F)&m[None,:],other=0.); u=tl.load(GU+F+rf[:,None]+rows[None,:]*(2*F),mask=(rf[:,None]<F)&m[None,:],other=0.); a=g*tl.sigmoid(g)*u; z=tl.load(DY+tok[:,None]*D+rd[None,:],mask=m[:,None]&(rd[None,:]<D),other=0.)*ws[:,None]
        if BF:
            ah=a.to(tl.bfloat16); al=(a-ah.to(tl.float32)).to(tl.bfloat16); zh=z.to(tl.bfloat16); zl=(z-zh.to(tl.float32)).to(tl.bfloat16); acc+=tl.dot(ah,zh)+tl.dot(al,zh)+tl.dot(ah,zl)
        else: acc+=tl.dot(a,z,input_precision="tf32x3")
    tl.store(DW+e*F*D+rf[:,None]*D+rd[None,:],acc,mask=(rf[:,None]<F)&(rd[None,:]<D))


@triton.jit
def _dw1(X,DGU,TOK,OFF,DW,D:tl.constexpr,F2:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
    e=tl.program_id(0); rd=tl.program_id(1)*BM+tl.arange(0,BM); rf=tl.program_id(2)*BN+tl.arange(0,BN); begin=tl.load(OFF+e); end=tl.load(OFF+e+1); acc=tl.zeros((BM,BN),tl.float32)
    for k in range(begin,end,BK):
        rows=k+tl.arange(0,BK); m=rows<end; tok=tl.load(TOK+rows,mask=m,other=0); x=tl.load(X+rd[:,None]+tok[None,:]*D,mask=(rd[:,None]<D)&m[None,:],other=0.); z=tl.load(DGU+rows[:,None]*F2+rf[None,:],mask=m[:,None]&(rf[None,:]<F2),other=0.)
        if BF:
            zh=z.to(tl.bfloat16); zl=(z-zh.to(tl.float32)).to(tl.bfloat16); acc+=tl.dot(x,zh)+tl.dot(x,zl)
        else: acc+=tl.dot(x.to(tl.float32),z,input_precision="tf32x3")
    tl.store(DW+e*D*F2+rd[:,None]*F2+rf[None,:],acc,mask=(rd[:,None]<D)&(rf[None,:]<F2))


@triton.jit
def _inverse(ORDER,INV,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B); m=i<N; q=tl.load(ORDER+i,mask=m,other=0); tl.store(INV+q,i,mask=m)


@triton.jit
def _dx_expert(W1,DGU,OFF,TE,TS,DXP,D:tl.constexpr,F2:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
    p=tl.program_id(0); e=tl.load(TE+p); st=tl.load(TS+p); end=tl.load(OFF+e+1,mask=e>=0,other=0); rm=tl.arange(0,BM); rd=tl.program_id(1)*BN+tl.arange(0,BN); rows=st+rm; m=(e>=0)&(rows<end); acc=tl.zeros((BM,BN),tl.float32)
    for k in range(0,F2,BK):
        rk=k+tl.arange(0,BK); z=tl.load(DGU+rows[:,None]*F2+rk[None,:],mask=m[:,None]&(rk[None,:]<F2),other=0.); w=tl.load(W1+e*D*F2+rd[None,:]*F2+rk[:,None],mask=(e>=0)&(rk[:,None]<F2)&(rd[None,:]<D),other=0.)
        if BF:
            zh=z.to(tl.bfloat16); zl=(z-zh.to(tl.float32)).to(tl.bfloat16); acc+=tl.dot(zh,w)+tl.dot(zl,w)
        else: acc+=tl.dot(z, w.to(tl.float32),input_precision="tf32x3")
    tl.store(DXP+rows[:,None]*D+rd[None,:],acc,mask=m[:,None]&(rd[None,:]<D))


@triton.jit
def _dx_combine(RW,DXP,DS,S,FLAT,INV,DX,D:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BD:tl.constexpr):
    t=tl.program_id(0); rd=tl.program_id(1)*BD+tl.arange(0,BD); acc=tl.zeros((BD,),tl.float32)
    for a in range(A):
        q=t*A+a; e=tl.load(FLAT+q); pos=tl.load(INV+q); acc+=tl.load(DXP+pos*D+rd,mask=rd<D,other=0.); raw=tl.load(S+t*E+e); dl=tl.load(DS+q)*raw*(1.-raw); acc+=dl*tl.load(RW+rd*E+e,mask=rd<D,other=0.).to(tl.float32)
    tl.store(DX+t*D+rd,acc,mask=rd<D)


@triton.jit
def _drw(X,DS,S,FLAT,DW,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rd=tl.program_id(0)*BM+tl.arange(0,BM); re=tl.program_id(1)*BN+tl.arange(0,BN); acc=tl.zeros((BM,BN),tl.float32)
    for k in range(0,T,BK):
        rt=k+tl.arange(0,BK); x=tl.load(X+rt[None,:]*D+rd[:,None],mask=(rd[:,None]<D)&(rt[None,:]<T),other=0.).to(tl.float32); z=tl.zeros((BK,BN),tl.float32)
        for a in range(A):
            ee=tl.load(FLAT+rt*A+a,mask=rt<T,other=0); raw=tl.load(S+rt*E+ee,mask=rt<T,other=0.); dl=tl.load(DS+rt*A+a,mask=rt<T,other=0.)*raw*(1.-raw); z+=tl.where(ee[:,None]==re[None,:],dl[:,None],0.)
        acc+=tl.dot(x,z,input_precision="ieee")
    tl.store(DW+rd[:,None]*E+re[None,:],acc,mask=(rd[:,None]<D)&(re[None,:]<E))


def _select(s,b,A):
    E=s.shape[1]; v=torch.where(s+b==0,torch.zeros_like(s),s+b).contiguous(); ib=v.view(torch.int32).to(torch.int64); u=torch.where(ib>=0,ib^0x80000000,(~ib)&0xffffffff); e=torch.arange(E,device=s.device,dtype=torch.int64); return torch.topk((u<<10)|(E-1-e),A,dim=1,largest=True,sorted=True).indices


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,rw,rb,w1,w2,A):
        T,D=x.shape; E=rw.shape[1]; F=w2.shape[1]; A=int(A); N=T*A; bf=x.dtype==torch.bfloat16
        s=torch.empty((T,E),device=x.device,dtype=torch.float32); _router[(triton.cdiv(T,32),triton.cdiv(E,32))](x,rw,s,T,D,E,BM=32,BN=32,BK=32)
        with torch.no_grad(): flat=_select(s,rb,A).reshape(-1); order=torch.argsort(flat)
        tok=torch.empty(N,device=x.device,dtype=torch.int32); exp=torch.empty_like(tok); ws=torch.empty(N,device=x.device,dtype=torch.float32); _metadata[(triton.cdiv(N,256),)](order,flat,s,tok,exp,ws,E,A,N,B=256)
        c=torch.empty(E,device=x.device,dtype=torch.int32); _counts[(E,)](exp,c,N,E,B=256); mt=triton.cdiv(N,64)+E; off=torch.empty(E+1,device=x.device,dtype=torch.int32); te=torch.empty(mt,device=x.device,dtype=torch.int32); ts=torch.empty_like(te); _tiles[(1,)](c,off,te,ts,E,mt,BM=64,num_warps=1)
        gu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); _in[(mt,triton.cdiv(2*F,64))](x,w1,tok,off,te,ts,gu,D,F,BM=64,BN=64,BK=32,BF=bf)
        fe=torch.empty((N,D),device=x.device,dtype=torch.float32); _out[(mt,triton.cdiv(D,64))](gu,w2,off,te,ts,order,fe,D,F,BM=64,BN=64,BK=32,BF=bf)
        y=torch.empty((T,D),device=x.device,dtype=x.dtype); _combine[(T,triton.cdiv(D,256))](fe,s,flat,y,D,E,A,BD=256)
        inv=torch.empty_like(tok); _inverse[(triton.cdiv(N,256),)](order,inv,N,B=256); ctx.save_for_backward(x,rw,w1,w2,s,flat,inv,tok,ws,off,te,ts,gu,fe); ctx.m=(T,D,E,F,A,mt,bf); return y
    @staticmethod
    def backward(ctx,dy):
        x,rw,w1,w2,s,flat,inv,tok,ws,off,te,ts,gu,fe=ctx.saved_tensors; T,D,E,F,A,mt,bf=ctx.m; dy=dy.contiguous(); ds=torch.empty((T,A),device=x.device,dtype=torch.float32); _route_bwd[(T,)](dy,fe,s,flat,ds,D,E,A,BD=256,BA=triton.next_power_of_2(A))
        dgu=torch.empty_like(gu); _dgu[(mt,triton.cdiv(F,64))](dy,ws,w2,gu,tok,off,te,ts,dgu,D,F,BM=64,BN=64,BK=32,BF=bf)
        dw2=torch.empty_like(w2); _dw2[(E,triton.cdiv(F,32),triton.cdiv(D,32))](dy,ws,gu,tok,off,dw2,D,F,BM=32,BN=32,BK=32,BF=bf)
        dw1=torch.empty_like(w1); _dw1[(E,triton.cdiv(D,32),triton.cdiv(2*F,32))](x,dgu,tok,off,dw1,D,2*F,BM=32,BN=32,BK=32,BF=bf)
        dxp=torch.empty((T*A,D),device=x.device,dtype=torch.float32); _dx_expert[(mt,triton.cdiv(D,64))](w1,dgu,off,te,ts,dxp,D,2*F,BM=64,BN=64,BK=32,BF=bf)
        dx=torch.empty_like(x); _dx_combine[(T,triton.cdiv(D,256))](rw,dxp,ds,s,flat,inv,dx,D,E,A,BD=256)
        drw=torch.empty_like(rw); _drw[(triton.cdiv(D,32),triton.cdiv(E,32))](x,ds,s,flat,drw,T,D,E,A,BM=32,BN=32,BK=32)
        return dx,drw,None,dw1,dw2,None


def fused_moe(x,router_w,router_b,w1,w2,A): return _Moe.apply(x,router_w,router_b,w1,w2,A)
