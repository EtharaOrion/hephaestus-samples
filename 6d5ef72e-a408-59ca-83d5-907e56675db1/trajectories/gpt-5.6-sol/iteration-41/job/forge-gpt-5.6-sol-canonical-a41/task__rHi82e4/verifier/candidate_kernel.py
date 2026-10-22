import torch
import triton
import triton.language as tl

BM = 128
BN = 64
BK = 32
EB = 1024


@triton.jit
def _mmk(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         sam: tl.constexpr, sak: tl.constexpr, sbk: tl.constexpr,
         sbn: tl.constexpr, scm: tl.constexpr, scn: tl.constexpr,
         STRICT: tl.constexpr, XM: tl.constexpr, XN: tl.constexpr,
         XK: tl.constexpr):
    im = tl.program_id(0) * XM + tl.arange(0, XM)
    jn = tl.program_id(1) * XN + tl.arange(0, XN)
    z = tl.zeros((XM, XN), tl.float32)
    for k0 in range(0, K, XK):
        kk = k0 + tl.arange(0, XK)
        av = tl.load(a + im[:, None] * sam + kk[None, :] * sak,
                     mask=(im[:, None] < M) & (kk[None, :] < K), other=0.0)
        bv = tl.load(b + kk[:, None] * sbk + jn[None, :] * sbn,
                     mask=(kk[:, None] < K) & (jn[None, :] < N), other=0.0)
        if STRICT:
            z += tl.dot(av, bv, input_precision="ieee")
        else:
            z += tl.dot(av, bv, input_precision="tf32x3")
    tl.store(c + im[:, None] * scm + jn[None, :] * scn, z,
             mask=(im[:, None] < M) & (jn[None, :] < N))


@triton.jit
def _actf(g, u, out, n: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < n
    gv = tl.load(g + i, mask=m, other=0.0)
    uv = tl.load(u + i, mask=m, other=0.0)
    tl.store(out + i, gv * tl.sigmoid(gv) * uv, mask=m)


@triton.jit
def _actb(g, u, dout, dg, du, n: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < n
    gv = tl.load(g + i, mask=m, other=0.0)
    uv = tl.load(u + i, mask=m, other=0.0)
    dv = tl.load(dout + i, mask=m, other=0.0)
    s = tl.sigmoid(gv)
    q = gv * s
    tl.store(dg + i, dv * uv * (s + q * (1.0 - s)), mask=m)
    tl.store(du + i, dv * q, mask=m)


def _mm(a, b, strict=False):
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    if m and n:
        _mmk[(triton.cdiv(m, BM), triton.cdiv(n, BN))](
            a, b, c, M=m, N=n, K=k, sam=a.stride(0), sak=a.stride(1),
            sbk=b.stride(0), sbn=b.stride(1), scm=c.stride(0),
            scn=c.stride(1), STRICT=strict, XM=BM, XN=BN, XK=BK,
            num_warps=8, num_stages=3)
    return c


class _Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, strict):
        ctx.save_for_backward(a, b)
        ctx.strict = strict
        return _mm(a, b, strict)

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        dc = dc.contiguous()
        return _mm(dc, b.t(), ctx.strict), _mm(a.t(), dc, ctx.strict), None


class _Act(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u):
        ctx.save_for_backward(g, u)
        out = torch.empty_like(g)
        n = g.numel()
        _actf[(triton.cdiv(n, EB),)](g, u, out, n=n, B=EB)
        return out

    @staticmethod
    def backward(ctx, dout):
        g, u = ctx.saved_tensors
        dout = dout.contiguous()
        dg = torch.empty_like(g)
        du = torch.empty_like(u)
        n = g.numel()
        _actb[(triton.cdiv(n, EB),)](g, u, dout, dg, du, n=n, B=EB)
        return dg, du


def _select(s, bias, count):
    experts = s.shape[1]
    biased = s + bias
    biased = torch.where(biased == 0.0, torch.zeros_like(biased), biased).contiguous()
    bits = biased.view(torch.int32).to(torch.int64)
    ordered = torch.where(bits >= 0, bits ^ 0x80000000, (~bits) & 0xffffffff)
    index = torch.arange(experts, device=s.device, dtype=torch.int64)
    key = (ordered << 10) | (experts - 1 - index)
    return torch.topk(key, count, dim=1, largest=True, sorted=True).indices


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, count):
        count = int(count)
        with torch.enable_grad():
            xf = x.detach().float().requires_grad_(True)
            rw = router_w.detach().float().requires_grad_(True)
            a1 = w1.detach().float().requires_grad_(True)
            a2 = w2.detach().float().requires_grad_(True)
            tokens, dim = xf.shape
            experts = rw.shape[1]
            hidden = a2.shape[1]
            score = torch.sigmoid(_Matmul.apply(xf, rw, True))
            with torch.no_grad():
                selected = _select(score.detach(), router_b.float(), count)
            raw = torch.gather(score, 1, selected)
            weight = raw / raw.sum(1, keepdim=True)
            dense_weight = score.new_zeros((tokens, experts)).scatter(
                1, selected, weight)
            member = torch.zeros((tokens, experts), device=x.device,
                                 dtype=torch.bool).scatter(1, selected, True)
            one = a1.unbind(0)
            two = a2.unbind(0)
            y = xf.new_zeros((tokens, dim))
            for e in range(experts):
                rows = member[:, e].nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                h = xf.index_select(0, rows)
                gu = _Matmul.apply(h, one[e], False)
                act = _Act.apply(gu[:, :hidden].contiguous(),
                                 gu[:, hidden:].contiguous())
                fe = _Matmul.apply(act, two[e], False)
                scaled = dense_weight.index_select(0, rows)[:, e:e + 1] * fe
                y = y.index_add(0, rows, scaled)
        ctx.save_for_backward(xf, rw, a1, a2, y)
        ctx.dtypes = x.dtype, router_w.dtype, w1.dtype, w2.dtype
        return y.detach().to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, rw, w1, w2, y = ctx.saved_tensors
        gx, grw, gw1, gw2 = torch.autograd.grad(
            y, (xf, rw, w1, w2), dy.float())
        dx_t, drw_t, dw1_t, dw2_t = ctx.dtypes
        return (gx.to(dx_t), grw.to(drw_t), None,
                gw1.to(dw1_t), gw2.to(dw2_t), None)


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _Moe.apply(x, router_w, router_b, w1, w2, A)
