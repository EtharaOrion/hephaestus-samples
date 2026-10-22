import torch
import triton
import triton.language as tl

_BT = 16
_FBV = 32
_BBV = 16

@triton.jit
def _chunk_forward_kernel(q, k, v, a, b, g, out, checkpoints,
                          T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                          NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib, ih = bh // H, bh % H
    rk, rv, rt = tl.arange(0, BK), iv * BV + tl.arange(0, BV), tl.arange(0, 16)
    mk, mv = rk < K, rv < V
    state = tl.zeros((BK, BV), tl.float32)
    scale = 1.0 / tl.sqrt(float(K))
    for ic in range(NC):
        t = ic * 16 + rt
        mt = t < T
        base = ((ib * T + t[:, None]) * H + ih) * K + rk[None, :]
        mtk = mt[:, None] & mk[None, :]
        q0 = tl.load(q + base, mask=mtk, other=0.).to(tl.float32) * scale
        k0 = tl.load(k + base, mask=mtk, other=0.).to(tl.float32)
        a0 = tl.load(a + base, mask=mtk, other=0.).to(tl.float32)
        b0 = tl.load(b + base, mask=mtk, other=0.).to(tl.float32)
        g0 = tl.load(g + base, mask=mtk, other=0.).to(tl.float32)
        gc = tl.cumsum(g0, axis=0)
        ep, em = tl.exp(tl.maximum(gc, -60.)), tl.exp(tl.minimum(-gc, 60.))
        qg, kg, bg = q0 * ep, k0 * em, b0 * em
        ag = a0 * tl.exp(tl.maximum(gc - g0, -60.))
        valid2 = mt[:, None] & mt[None, :]
        causal, strict = rt[:, None] >= rt[None, :], rt[:, None] > rt[None, :]
        aqk = tl.where(causal & valid2, tl.dot(qg, tl.trans(kg), input_precision='ieee'), 0.)
        aqb = tl.where(causal & valid2, tl.dot(qg, tl.trans(bg), input_precision='ieee'), 0.)
        lab = tl.where(strict & valid2, tl.dot(ag, tl.trans(bg), input_precision='ieee'), 0.)
        lak = tl.where(strict & valid2, tl.dot(ag, tl.trans(kg), input_precision='ieee'), 0.)
        inv = tl.where(rt[:, None] == rt[None, :], 1., 0.) + lab
        power = lab
        for _ in tl.static_range(0, 3):
            power = tl.dot(power, power, input_precision='ieee')
            inv += tl.dot(inv, power, input_precision='ieee')
        vv = tl.load(v + ((ib*T+t[:,None])*H+ih)*V + rv[None,:],
                     mask=mt[:,None] & mv[None,:], other=0.).to(tl.float32)
        u = tl.dot(inv, tl.dot(lak, vv, input_precision='ieee'), input_precision='ieee')
        w = tl.dot(inv, ag, input_precision='ieee')
        vn = u + tl.dot(w, state, input_precision='ieee')
        oo = tl.dot(aqk, vv, input_precision='ieee') + tl.dot(aqb, vn, input_precision='ieee')
        oo += tl.dot(qg, state, input_precision='ieee')
        tl.store(out + ((ib*T+t[:,None])*H+ih)*V + rv[None,:], oo,
                 mask=mt[:,None] & mv[None,:])
        hp = ((((ib*NC+ic)*H+ih)*K+rk[:,None])*V+rv[None,:])
        tl.store(checkpoints + hp, state, mask=mk[:,None] & mv[None,:])
        endg = tl.sum(g0, axis=0)
        decay = tl.exp(tl.maximum(endg[None,:] - gc, -60.))
        state = state * tl.exp(tl.maximum(endg, -60.))[:,None]
        state += tl.dot(tl.trans(k0*decay), vv, input_precision='ieee')
        state += tl.dot(tl.trans(b0*decay), vn, input_precision='ieee')

@triton.jit
def _expand_states_kernel(k, v, a, b, g, checkpoints, states,
                          T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                          NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, ic, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = bh // H, bh % H
    rk, rv = tl.arange(0, BK), iv*BV + tl.arange(0, BV)
    mk, mv = rk < K, rv < V
    hp = ((((ib*NC+ic)*H+ih)*K+rk[:,None])*V+rv[None,:])
    state = tl.load(checkpoints+hp, mask=mk[:,None]&mv[None,:], other=0.)
    for j in tl.static_range(0, 16):
        t = ic*16+j
        valid = t < T
        sp = ((((ib*T+t)*H+ih)*K+rk[:,None])*V+rv[None,:])
        tl.store(states+sp, state, mask=valid&mk[:,None]&mv[None,:])
        xb = (ib*T+t)*H+ih
        k0 = tl.load(k+xb*K+rk, mask=valid&mk, other=0.).to(tl.float32)
        a0 = tl.load(a+xb*K+rk, mask=valid&mk, other=0.).to(tl.float32)
        b0 = tl.load(b+xb*K+rk, mask=valid&mk, other=0.).to(tl.float32)
        g0 = tl.load(g+xb*K+rk, mask=valid&mk, other=0.).to(tl.float32)
        v0 = tl.load(v+xb*V+rv, mask=valid&mv, other=0.).to(tl.float32)
        read = tl.sum(a0[:,None]*state, axis=0)
        state = state*tl.exp(g0)[:,None] + k0[:,None]*v0[None,:] + b0[:,None]*read[None,:]

@triton.jit
def _dstate_boundaries_kernel(q, a, b, g, do, boundaries,
                              T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                              NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib, ih = bh // H, bh % H
    rk, rv = tl.arange(0, BK), iv*BV + tl.arange(0, BV)
    mk, mv = rk < K, rv < V
    ds = tl.zeros((BK, BV), tl.float32)
    scale = 1.0 / tl.sqrt(float(K))
    for ic0 in range(NC):
        ic = NC-1-ic0
        bp = ((((ib*NC+ic)*H+ih)*K+rk[:,None])*V+rv[None,:])
        tl.store(boundaries+bp, ds, mask=mk[:,None]&mv[None,:])
        for j0 in tl.static_range(0, 16):
            t = ic*16 + 15-j0
            valid = t < T
            xb = (ib*T+t)*H+ih
            q0 = tl.load(q+xb*K+rk, mask=valid&mk, other=0.).to(tl.float32)*scale
            a0 = tl.load(a+xb*K+rk, mask=valid&mk, other=0.).to(tl.float32)
            b0 = tl.load(b+xb*K+rk, mask=valid&mk, other=0.).to(tl.float32)
            g0 = tl.load(g+xb*K+rk, mask=valid&mk, other=0.).to(tl.float32)
            do0 = tl.load(do+xb*V+rv, mask=valid&mv, other=0.).to(tl.float32)
            ds += q0[:,None]*do0[None,:]
            dr = tl.sum(ds*b0[:,None], axis=0)
            ds = ds*tl.exp(g0)[:,None] + a0[:,None]*dr[None,:]

@triton.jit
def _backward_kernel(q, k, v, a, b, g, do, states, boundaries,
                     pq, pk, pa, pb, pg, dv,
                     T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                     NC: tl.constexpr, NV: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, ic, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = bh // H, bh % H
    rk, rv = tl.arange(0, BK), iv*BV + tl.arange(0, BV)
    mk, mv = rk < K, rv < V
    bp = ((((ib*NC+ic)*H+ih)*K+rk[:,None])*V+rv[None,:])
    ds = tl.load(boundaries+bp, mask=mk[:,None]&mv[None,:], other=0.)
    scale = 1.0 / tl.sqrt(float(K))
    for j0 in tl.static_range(0, 16):
        t = ic*16 + 15-j0
        valid = t < T
        xb = (ib*T+t)*H+ih
        q0 = tl.load(q+xb*K+rk, mask=valid&mk, other=0.).to(tl.float32)*scale
        k0 = tl.load(k+xb*K+rk, mask=valid&mk, other=0.).to(tl.float32)
        a0 = tl.load(a+xb*K+rk, mask=valid&mk, other=0.).to(tl.float32)
        b0 = tl.load(b+xb*K+rk, mask=valid&mk, other=0.).to(tl.float32)
        g0 = tl.load(g+xb*K+rk, mask=valid&mk, other=0.).to(tl.float32)
        v0 = tl.load(v+xb*V+rv, mask=valid&mv, other=0.).to(tl.float32)
        do0 = tl.load(do+xb*V+rv, mask=valid&mv, other=0.).to(tl.float32)
        sp = ((((ib*T+t)*H+ih)*K+rk[:,None])*V+rv[None,:])
        s0 = tl.load(states+sp, mask=valid&mk[:,None]&mv[None,:], other=0.)
        decay = tl.exp(g0)
        read = tl.sum(a0[:,None]*s0, axis=0)
        s1 = s0*decay[:,None] + k0[:,None]*v0[None,:] + b0[:,None]*read[None,:]
        ds += q0[:,None]*do0[None,:]
        dq0 = tl.sum(s1*do0[None,:], axis=1)*scale
        dk0 = tl.sum(ds*v0[None,:], axis=1)
        db0 = tl.sum(ds*read[None,:], axis=1)
        dr = tl.sum(ds*b0[:,None], axis=0)
        da0 = tl.sum(s0*dr[None,:], axis=1)
        dg0 = tl.sum(ds*s0*decay[:,None], axis=1)
        dv0 = tl.sum(ds*k0[:,None], axis=0)
        pp = ((((ib*NV+iv)*T+t)*H+ih)*K+rk)
        tl.store(pq+pp, dq0, mask=valid&mk)
        tl.store(pk+pp, dk0, mask=valid&mk)
        tl.store(pa+pp, da0, mask=valid&mk)
        tl.store(pb+pp, db0, mask=valid&mk)
        tl.store(pg+pp, dg0, mask=valid&mk)
        tl.store(dv+xb*V+rv, dv0, mask=valid&mv)
        ds = ds*decay[:,None] + a0[:,None]*dr[None,:]

@triton.jit
def _reduce_kernel(pq, pk, pa, pb, pg, dq, dk, da, db, dg,
                   N: tl.constexpr, T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                   NV: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    kk, z = offs % K, offs // K
    hh, z = z % H, z // H
    tt, bb = z % T, z // T
    xq = tl.zeros((BLOCK,), tl.float32); xk = tl.zeros((BLOCK,), tl.float32)
    xa = tl.zeros((BLOCK,), tl.float32); xb = tl.zeros((BLOCK,), tl.float32)
    xg = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        p = ((((bb*NV+iv)*T+tt)*H+hh)*K+kk)
        xq += tl.load(pq+p, mask=mask, other=0.); xk += tl.load(pk+p, mask=mask, other=0.)
        xa += tl.load(pa+p, mask=mask, other=0.); xb += tl.load(pb+p, mask=mask, other=0.)
        xg += tl.load(pg+p, mask=mask, other=0.)
    tl.store(dq+offs, xq, mask=mask); tl.store(dk+offs, xk, mask=mask)
    tl.store(da+offs, xa, mask=mask); tl.store(db+offs, xb, mask=mask)
    tl.store(dg+offs, xg, mask=mask)

class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B,T,H,K = q.shape; V = v.shape[-1]
        nc, nv = triton.cdiv(T,_BT), triton.cdiv(V,_FBV)
        bk = max(16, triton.next_power_of_2(K))
        out = torch.empty_like(v)
        checkpoints = torch.empty((B,nc,H,K,V), device=q.device, dtype=torch.float32)
        _chunk_forward_kernel[(nv,B*H)](q,k,v,alpha,beta,gk,out,checkpoints,
            T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=_FBV,num_warps=8,num_stages=2)
        ctx.save_for_backward(q,k,v,alpha,beta,gk,checkpoints)
        return out
    @staticmethod
    def backward(ctx, do):
        q,k,v,alpha,beta,gk,checkpoints = ctx.saved_tensors
        do = do.contiguous(); B,T,H,K = q.shape; V = v.shape[-1]
        nc, nv = triton.cdiv(T,_BT), triton.cdiv(V,_BBV)
        bk = max(16, triton.next_power_of_2(K))
        states = torch.empty((B,T,H,K,V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv,nc,B*H)](k,v,alpha,beta,gk,checkpoints,states,
            T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=_BBV,num_warps=4,num_stages=2)
        boundaries = torch.empty(
            (B,nc,H,K,V), device=q.device, dtype=torch.float32
        )
        _dstate_boundaries_kernel[(nv,B*H)](
            q,alpha,beta,gk,do,boundaries,
            T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=_BBV,
            num_warps=8,num_stages=2,
        )
        shape = (B,nv,T,H,K)
        pq = torch.zeros(shape,device=q.device,dtype=torch.float32); pk = torch.zeros_like(pq)
        pa = torch.zeros_like(pq); pb = torch.zeros_like(pq); pg = torch.zeros_like(pq)
        dv = torch.empty_like(v)
        _backward_kernel[(nv,nc,B*H)](
            q,k,v,alpha,beta,gk,do,states,boundaries,pq,pk,pa,pb,pg,dv,
            T=T,H=H,K=K,V=V,NC=nc,NV=nv,BK=bk,BV=_BBV,
            num_warps=8,num_stages=2,
        )
        dq,dk,da,db,dg = [torch.empty_like(x) for x in (q,k,alpha,beta,gk)]
        n = B*T*H*K
        _reduce_kernel[(triton.cdiv(n,256),)](pq,pk,pa,pb,pg,dq,dk,da,db,dg,
            N=n,T=T,H=H,K=K,NV=nv,BLOCK=256,num_warps=8)
        return dq,dk,dv,da,db,dg

def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
