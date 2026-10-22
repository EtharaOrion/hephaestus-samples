import torch
import triton
import triton.language as tl

_BT = 16
_FBV = 32
_BBV = 16

@triton.jit
def _dot(a, b):
    return tl.dot(a, b, input_precision="ieee")

@triton.jit
def _inverse_lower(l):
    ident = tl.arange(0, 16)[:, None] == tl.arange(0, 16)[None, :]
    p = ident.to(tl.float32) + l
    power = l
    for _ in tl.static_range(0, 4):
        power = _dot(power, power)
        p += _dot(power, p)
    return p

@triton.jit
def _load_factors(q, k, a, b, g, bh, c, T: tl.constexpr, H: tl.constexpr,
                  K: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr):
    it = tl.arange(0, 16)
    ik = tl.arange(0, BK)
    t = c * 16 + it
    h = bh % H
    batch = bh // H
    offs = ((batch * T + t[:, None]) * H + h) * K + ik[None, :]
    mask = (t[:, None] < T) & (ik[None, :] < K)
    gv = tl.load(g + offs, mask=mask, other=0.0).to(tl.float32)
    z = tl.cumsum(gv, axis=0)
    ep = tl.exp(z)
    em = tl.exp(-z)
    ea = tl.exp(z - gv)
    qg = tl.load(q + offs, mask=mask, other=0.0).to(tl.float32) * ep * SCALE
    kg = tl.load(k + offs, mask=mask, other=0.0).to(tl.float32) * em
    ag = tl.load(a + offs, mask=mask, other=0.0).to(tl.float32) * ea
    bg = tl.load(b + offs, mask=mask, other=0.0).to(tl.float32) * em
    return qg, kg, ag, bg, ep, em, ea, t, ik, mask

@triton.jit
def _prepare_kernel(q, k, a, b, g, mats, T: tl.constexpr, H: tl.constexpr,
                    K: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                    SCALE: tl.constexpr):
    c = tl.program_id(0)
    bh = tl.program_id(1)
    qg, kg, ag, bg, _, _, _, _, _, _ = _load_factors(q, k, a, b, g, bh, c, T, H, K, BK, SCALE)
    row = tl.arange(0, 16)[:, None]
    col = tl.arange(0, 16)[None, :]
    valid = (c * 16 + row < T) & (c * 16 + col < T)
    lower = (row >= col) & valid
    strict = (row > col) & valid
    aqk = tl.where(lower, _dot(qg, tl.trans(kg)), 0.0)
    aqb = tl.where(lower, _dot(qg, tl.trans(bg)), 0.0)
    aak = tl.where(strict, _dot(ag, tl.trans(kg)), 0.0)
    lab = tl.where(strict, _dot(ag, tl.trans(bg)), 0.0)
    p = _inverse_lower(lab)
    base = (bh * NC + c) * 1024
    ij = row * 16 + col
    tl.store(mats + base + ij, aqk)
    tl.store(mats + base + 256 + ij, aqb)
    tl.store(mats + base + 512 + ij, aak)
    tl.store(mats + base + 768 + ij, p)

@triton.jit
def _forward_kernel(q, k, v, a, b, g, mats, checkpoints, ys, out,
                    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                    BV: tl.constexpr, SCALE: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ik = tl.arange(0, BK)
    ivv = iv * BV + tl.arange(0, BV)
    mkv = (ik[:, None] < K) & (ivv[None, :] < V)
    state = tl.zeros((BK, BV), tl.float32)
    batch, head = bh // H, bh % H
    row = tl.arange(0, 16)[:, None]
    col = tl.arange(0, 16)[None, :]
    for c in range(NC):
        cp = ((bh * NC + c) * K + ik[:, None]) * V + ivv[None, :]
        tl.store(checkpoints + cp, state, mask=mkv)
        qg, kg, ag, bg, ep, _, _, _, _, _ = _load_factors(q, k, a, b, g, bh, c, T, H, K, BK, SCALE)
        tv = c * 16 + tl.arange(0, 16)
        vo = ((batch * T + tv[:, None]) * H + head) * V + ivv[None, :]
        mv = (tv[:, None] < T) & (ivv[None, :] < V)
        vv = tl.load(v + vo, mask=mv, other=0.0).to(tl.float32)
        mb = (bh * NC + c) * 1024
        aqk = tl.load(mats + mb + row * 16 + col)
        aqb = tl.load(mats + mb + 256 + row * 16 + col)
        aak = tl.load(mats + mb + 512 + row * 16 + col)
        p = tl.load(mats + mb + 768 + row * 16 + col)
        rhs = _dot(aak, vv) + _dot(ag, state)
        y = _dot(p, rhs)
        oo = _dot(aqk, vv) + _dot(aqb, y) + _dot(qg, state)
        tl.store(ys + vo, y, mask=mv)
        tl.store(out + vo, oo, mask=mv)
        end = tl.sum(ep * (tl.arange(0, 16)[:, None] == 15), axis=0)
        state = (state + _dot(tl.trans(kg), vv) + _dot(tl.trans(bg), y)) * end[:, None]

@triton.jit
def _exact_checkpoints_kernel(k,v,a,b,g,checkpoints,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
 iv,bh=tl.program_id(0),tl.program_id(1); ik=tl.arange(0,BK); ivv=iv*BV+tl.arange(0,BV); mkv=(ik[:,None]<K)&(ivv[None,:]<V); batch,head=bh//H,bh%H; state=tl.zeros((BK,BV),tl.float32)
 for c in range(NC):
  cp=((bh*NC+c)*K+ik[:,None])*V+ivv[None,:]; tl.store(checkpoints+cp,state,mask=mkv)
  for j in tl.static_range(0,16):
   t=c*16+j; ko=((batch*T+t)*H+head)*K+ik; vo=((batch*T+t)*H+head)*V+ivv; mk=(t<T)&(ik<K); mv=(t<T)&(ivv<V); av=tl.load(a+ko,mask=mk,other=0.).to(tl.float32); bv=tl.load(b+ko,mask=mk,other=0.).to(tl.float32); kv=tl.load(k+ko,mask=mk,other=0.).to(tl.float32); ev=tl.exp(tl.load(g+ko,mask=mk,other=0.).to(tl.float32)); vv=tl.load(v+vo,mask=mv,other=0.).to(tl.float32); read=tl.sum(state*av[:,None],axis=0); state=state*ev[:,None]+kv[:,None]*vv[None,:]+bv[:,None]*read[None,:]

@triton.jit
def _expand_states_kernel(k,v,a,b,g,checkpoints,states,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
 iv,c,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2); ik=tl.arange(0,BK); ivv=iv*BV+tl.arange(0,BV); mkv=(ik[:,None]<K)&(ivv[None,:]<V); batch,head=bh//H,bh%H; cp=((bh*NC+c)*K+ik[:,None])*V+ivv[None,:]; state=tl.load(checkpoints+cp,mask=mkv,other=0.0)
 for j in tl.static_range(0,16):
  t=c*16+j; so=(((batch*T+t)*H+head)*K+ik[:,None])*V+ivv[None,:]; tl.store(states+so,state,mask=(t<T)&mkv); ko=((batch*T+t)*H+head)*K+ik; vo=((batch*T+t)*H+head)*V+ivv; mk=(t<T)&(ik<K); mv=(t<T)&(ivv<V); av=tl.load(a+ko,mask=mk,other=0.).to(tl.float32); bv=tl.load(b+ko,mask=mk,other=0.).to(tl.float32); kv=tl.load(k+ko,mask=mk,other=0.).to(tl.float32); ev=tl.exp(tl.load(g+ko,mask=mk,other=0.).to(tl.float32)); vv=tl.load(v+vo,mask=mv,other=0.).to(tl.float32); read=tl.sum(state*av[:,None],axis=0); state=state*ev[:,None]+kv[:,None]*vv[None,:]+bv[:,None]*read[None,:]

@triton.jit
def _boundary_exact_kernel(q,a,b,g,states,do,dhs,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr,SCALE:tl.constexpr):
 iv,bh=tl.program_id(0),tl.program_id(1); ik=tl.arange(0,BK); ivv=iv*BV+tl.arange(0,BV); mkv=(ik[:,None]<K)&(ivv[None,:]<V); batch,head=bh//H,bh%H; ds=tl.zeros((BK,BV),tl.float32)
 for rc in range(NC):
  c=NC-1-rc; dp=((bh*(NC+1)+c+1)*K+ik[:,None])*V+ivv[None,:]; tl.store(dhs+dp,ds,mask=mkv)
  for rj in tl.static_range(0,16):
   t=c*16+15-rj; ko=((batch*T+t)*H+head)*K+ik; vo=((batch*T+t)*H+head)*V+ivv; so=(((batch*T+t)*H+head)*K+ik[:,None])*V+ivv[None,:]; valid=t<T; mk=valid&(ik<K); mv=valid&(ivv<V); sp=tl.load(states+so,mask=valid&mkv,other=0.); qv=tl.load(q+ko,mask=mk,other=0.).to(tl.float32)*SCALE; av=tl.load(a+ko,mask=mk,other=0.).to(tl.float32); bv=tl.load(b+ko,mask=mk,other=0.).to(tl.float32); ev=tl.exp(tl.load(g+ko,mask=mk,other=0.).to(tl.float32)); dout=tl.load(do+vo,mask=mv,other=0.).to(tl.float32); ds+=qv[:,None]*dout[None,:]; dread=tl.sum(ds*bv[:,None],axis=0); ds=ds*ev[:,None]+av[:,None]*dread[None,:]
 dp=(bh*(NC+1)*K+ik[:,None])*V+ivv[None,:]; tl.store(dhs+dp,ds,mask=mkv)

@triton.jit
def _backward_exact_kernel(q,k,v,a,b,g,states,do,dhs,partial,dv,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,NV:tl.constexpr,N:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr,SCALE:tl.constexpr):
 iv,c,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2); ik=tl.arange(0,BK); ivv=iv*BV+tl.arange(0,BV); batch,head=bh//H,bh%H; mkv=(ik[:,None]<K)&(ivv[None,:]<V); dp=((bh*(NC+1)+c+1)*K+ik[:,None])*V+ivv[None,:]; ds=tl.load(dhs+dp,mask=mkv,other=0.)
 for rj in tl.static_range(0,16):
  t=c*16+15-rj; ko=((batch*T+t)*H+head)*K+ik; vo=((batch*T+t)*H+head)*V+ivv; so=(((batch*T+t)*H+head)*K+ik[:,None])*V+ivv[None,:]; valid=t<T; mk=valid&(ik<K); mv=valid&(ivv<V); sp=tl.load(states+so,mask=valid&mkv,other=0.); qv=tl.load(q+ko,mask=mk,other=0.).to(tl.float32)*SCALE; av=tl.load(a+ko,mask=mk,other=0.).to(tl.float32); bv=tl.load(b+ko,mask=mk,other=0.).to(tl.float32); kv=tl.load(k+ko,mask=mk,other=0.).to(tl.float32); ev=tl.exp(tl.load(g+ko,mask=mk,other=0.).to(tl.float32)); vv=tl.load(v+vo,mask=mv,other=0.).to(tl.float32); dout=tl.load(do+vo,mask=mv,other=0.).to(tl.float32); read=tl.sum(sp*av[:,None],axis=0); sn=sp*ev[:,None]+kv[:,None]*vv[None,:]+bv[:,None]*read[None,:]; dq0=tl.sum((sn*dout[None,:]).to(tl.float64),axis=1)*SCALE; ds+=qv[:,None]*dout[None,:]; dread=tl.sum(ds*bv[:,None],axis=0); oo=((batch*T+t)*H+head)*K+ik; tl.store(partial+(NV+iv)*N+oo,tl.sum(ds*vv[None,:],axis=1),mask=mk); tl.store(partial+iv*N+oo,dq0,mask=mk); tl.store(partial+(2*NV+iv)*N+oo,tl.sum((sp*dread[None,:]).to(tl.float64),axis=1),mask=mk); tl.store(partial+(3*NV+iv)*N+oo,tl.sum(ds*read[None,:],axis=1),mask=mk); tl.store(partial+(4*NV+iv)*N+oo,tl.sum((ds*(sp*ev[:,None])).to(tl.float64),axis=1),mask=mk); tl.store(dv+vo,tl.sum(ds*kv[:,None],axis=0),mask=mv); ds=ds*ev[:,None]+av[:,None]*dread[None,:]

@triton.jit
def _reduce_kernel(partial, dq, dk, da, db, dg, N: tl.constexpr,
                   NV: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    a0 = tl.zeros((BLOCK,), tl.float32)
    a1 = tl.zeros((BLOCK,), tl.float32)
    a2 = tl.zeros((BLOCK,), tl.float32)
    a3 = tl.zeros((BLOCK,), tl.float32)
    a4 = tl.zeros((BLOCK,), tl.float32)
    for iv in tl.static_range(0, NV):
        a0 += tl.load(partial + (iv) * N + idx, mask=mask, other=0.0)
        a1 += tl.load(partial + (NV + iv) * N + idx, mask=mask, other=0.0)
        a2 += tl.load(partial + (2 * NV + iv) * N + idx, mask=mask, other=0.0)
        a3 += tl.load(partial + (3 * NV + iv) * N + idx, mask=mask, other=0.0)
        a4 += tl.load(partial + (4 * NV + iv) * N + idx, mask=mask, other=0.0)
    tl.store(dq + idx, a0, mask=mask)
    tl.store(dk + idx, a1, mask=mask)
    tl.store(da + idx, a2, mask=mask)
    tl.store(db + idx, a3, mask=mask)
    tl.store(dg + idx, a4, mask=mask)

class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc, nv = triton.cdiv(T, _BT), triton.cdiv(V, _FBV)
        bk = max(16, triton.next_power_of_2(K))
        mats = torch.empty(B * H, nc, 4, 16, 16, device=q.device, dtype=torch.float32)
        checkpoints = torch.empty(B * H, nc, K, V, device=q.device, dtype=torch.float32)
        ys = torch.empty(B, T, H, V, device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        scale = K ** -0.5
        _prepare_kernel[(nc, B * H)](q, k, alpha, beta, gk, mats, T=T, H=H, K=K, NC=nc, BK=bk, SCALE=scale, num_warps=4)
        _forward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, mats, checkpoints, ys, out, T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_FBV, SCALE=scale, num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, mats, checkpoints, ys)
        ctx.shape = B, T, H, K, V
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, mats, checkpoints, ys = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, K, V = ctx.shape
        nc = triton.cdiv(T, _BT)
        bv = _BBV if V == triton.next_power_of_2(V) else triton.next_power_of_2(V)
        nv = triton.cdiv(V, bv)
        bk, scale = max(16, triton.next_power_of_2(K)), K ** -0.5
        _exact_checkpoints_kernel[(nv, B * H)](k, v, alpha, beta, gk, checkpoints, T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=bv, num_warps=4)
        states = torch.empty(B, T, H, K, V, device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, nc, B * H)](k, v, alpha, beta, gk, checkpoints, states, T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=bv, num_warps=4)
        dhs = torch.empty(B * H, nc + 1, K, V, device=q.device, dtype=torch.float32)
        _boundary_exact_kernel[(nv, B * H)](q, alpha, beta, gk, states, dout, dhs, T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=bv, SCALE=scale, num_warps=4)
        n = B * T * H * K
        partial = torch.zeros(5, nv, n, device=q.device, dtype=torch.float32)
        dv = torch.empty_like(v)
        _backward_exact_kernel[(nv, nc, B * H)](q, k, v, alpha, beta, gk, states, dout, dhs, partial, dv, T=T, H=H, K=K, V=V, NC=nc, NV=nv, N=n, BK=bk, BV=bv, SCALE=scale, num_warps=4)
        dq, dk, da, db = (torch.zeros_like(q) for _ in range(4))
        dg = torch.zeros_like(gk)
        _reduce_kernel[(triton.cdiv(n, 256),)](partial, dq, dk, da, db, dg, N=n, NV=nv, BLOCK=256, num_warps=4)
        return dq, dk, dv, da, db, dg

def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
