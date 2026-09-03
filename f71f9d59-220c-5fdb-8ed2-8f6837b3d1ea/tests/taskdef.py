"""Taskdef: chunk-parallel gated delta-product (N Householder writes per token),
forward+backward (fwdbwd).

Member of the chunked_linear_attn family. HARD variant. Distinct from every other
member because each token performs `num_householder` NESTED Householder writes
before the readout -- the WY-style intra-chunk factorization must invert a
(N*BT) x (N*BT) lower-triangular system rather than a BT x BT one, and every
`beta` scale carries N times the per-token cost.

Operator (reference.py is the definition; this is a summary):
    chunked_gated_delta_product(q[B,T,H,K], k[B,T*N,H,K], v[B,T*N,H,V],
                                 g[B,T,H], beta[B,T*N,H], num_householder=N)
        -> o[B,T,H,V]
    Per-token: S <- diag(exp(g_t)) S;  then N times:
        S <- S + (v_{t,j} - S^T k_{t,j})^T * k_{t,j} * beta_{t,j}
    Readout: o_t = S^T (q_t / sqrt(K)). N Householder writes chain multiplicatively
    per token; the gate is applied ONCE before the N writes. SURFACE fwdbwd:
    gradients of q, k, v, g, beta are graded, timed and determinism-checked.
    num_householder is a fixed int argument (no grad).

Anchor: fla.ops.gated_delta_product.chunk_gated_delta_product -- the production
chunk kernel of the pinned library. sota() calls it with KEYWORDS,
num_householder=2, scale=None (=> K^-0.5) and unwraps the (o, final_state) tuple.

STATUS: authored-draft -- NEEDS GPU CALIBRATION. TOL, TARGET_FRACTION_OF_SOTA, the
vendor-shingle ceiling, the anchor semantic-divergence check and every negative
control's bound-gate behaviour are INHERITED placeholders; see spec.yaml.

Distribution invariance (frontier-defeat-analysis §7.1): q, k L2-normalized along
K; g log-space in (-inf, 0]; beta in (0.05, 0.95]; the hidden per-invocation scale
multiplies v (the one magnitude carrier). Timed tensors ARE checked tensors;
NC-DP plants the §5.4 magnitude-branch exploit on the oracle under FORGE_ORACLE=1
and must zero at exactly G2.
"""

import importlib.util
import re
from pathlib import Path

import torch
import torch.nn.functional as F

import taskdef_api  # noqa: F401

NUM_HOUSEHOLDER = 2

ENTRY_NAME = "chunked_gated_delta_product"
REF_NAME = "chunked_gated_delta_product_ref"
SIGNATURE = (
    "chunked_gated_delta_product(q[B,T,H,K], k[B,T*N,H,K], "
    "v[B,T*N,H,V], g[B,T,H], beta[B,T*N,H], num_householder=N) "
    "-> o[B,T,H,V]  (N=2)"
)
SURFACE = "fwdbwd"
TASK_NAME = "hephaestus/chunked_gated_delta_product"

# INHERITED KDA-family knees, widened for the N-fold intra-chunk erasure cascade
# that accumulates in bf16 twice as fast as a single Householder. Upstream
# reference uses 0.05 for bf16 recurrences of this family; carry it here.
TOL = {
    "bfloat16": (5.0e-2, 5.0e-2),
    "float16": (1.0e-2, 3.90625e-3),
    "float32": (2.0e-3, 2.0e-3),
}

FORBIDDEN = [
    "fla.ops",
    "import fla",
    "from fla",
    "chunk_gated_delta_product",
    "fused_recurrent_gated_delta_product",
    "naive_recurrent_gated_delta_product",
    "chunk_gated_delta_rule",
    "chunk_gdn",
    "chunk_kda",
    "chunk_delta_rule",
    "chunk_gla",
    "chunk_simple_gla",
    "chunk_retention",
    "chunk_delta_h",
    "chunk_comba",
    "chunk_gdn2",
    "chunk_rwkv6",
    "chunk_rwkv7",
    "chunk_dplr_delta_rule",
    "chunk_iplr_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_gdn",
    "fused_recurrent_kda",
    "fused_recurrent_delta_rule",
    "fused_recurrent_gla",
    "fused_recurrent_simple_gla",
    "fused_recurrent_retention",
    "fused_recurrent",
    "naive_recurrent_gated_delta_rule",
    "naive_chunk_gated_delta_rule",
]

VENDOR_LIB_PACKAGE = "fla"
VENDOR_LIB_SUBDIRS = (
    "ops/gated_delta_product",
    "ops/gated_delta_rule",
    "ops/common",
    "ops/utils",
)

ADOPTION_FLOOR = 0.60

# PLACEHOLDER seed only (hard task -- lowest reachable fraction; the anchor's
# N-fold intra-chunk fusion is extreme).
TARGET_FRACTION_OF_SOTA = 0.65

ORACLE_ENV_FOR_GOLDEN = True


def _dims(shape):
    return (
        int(shape["B"]),
        int(shape["T"]),
        int(shape["H"]),
        int(shape["K"]),
        int(shape["V"]),
    )


def _draw(shape, seed, scale, device):
    B, T, H, K, V = _dims(shape)
    N = NUM_HOUSEHOLDER
    gen = torch.Generator(device=device)
    gen.manual_seed(seed & 0x7FFFFFFF)
    rn = lambda *s: torch.randn(*s, device=device, dtype=torch.float32, generator=gen)  # noqa: E731
    qn = F.normalize(rn(B, T, H, K), dim=-1)
    kn = F.normalize(rn(B, T * N, H, K), dim=-1)
    vv = rn(B, T * N, H, V) * float(scale)
    gg = F.logsigmoid(rn(B, T, H))
    bb = (
        torch.rand(B, T * N, H, device=device, dtype=torch.float32, generator=gen) * 0.9
        + 0.05
    )
    return qn, kn, vv, gg, bb


def _pack(qn, kn, vv, gg, bb, dtype):
    args = {
        "q": qn.to(dtype).detach().contiguous().requires_grad_(True),
        "k": kn.to(dtype).detach().contiguous().requires_grad_(True),
        "v": vv.to(dtype).detach().contiguous().requires_grad_(True),
        "g": gg.detach().contiguous().requires_grad_(True),  # log-decay stays fp32
        "beta": bb.to(dtype).detach().contiguous().requires_grad_(True),
        "num_householder": NUM_HOUSEHOLDER,  # non-tensor, no grad
    }
    return args, ("q", "k", "v", "g", "beta")


def make_inputs(shape, seed, scale, dtype, device):
    return _pack(*_draw(shape, seed, scale, device), dtype)


def sota(args):
    """Production anchor: FLA's chunk_gated_delta_product (timing denominator ONLY;
    reference.py defines correctness). KEYWORDS; scale=None => K^-0.5; unwrap
    (o, final_state) tuple."""
    from fla.ops.gated_delta_product import chunk_gated_delta_product

    out = chunk_gated_delta_product(
        q=args["q"],
        k=args["k"],
        v=args["v"],
        g=args["g"],
        beta=args["beta"],
        num_householder=args["num_householder"],
        scale=None,
        use_qk_l2norm_in_kernel=False,
        output_final_state=False,
    )
    return out[0] if isinstance(out, tuple) else out


def _mode(no_decay=False, max_decay=False, beta_full=False, v_dyn=False, v_under=False):
    def maker(shape, seed, scale, dtype, device):
        qn, kn, vv, gg, bb = _draw(shape, seed, scale, device)
        if no_decay:
            gg = torch.zeros_like(gg)
        if max_decay:
            gg = torch.full_like(gg, -15.0)
        if beta_full:
            bb = torch.ones_like(bb)
        if v_dyn:
            gen = torch.Generator(device=device)
            gen.manual_seed((seed ^ 0x5A5A) & 0x7FFFFFFF)
            vv = vv * torch.pow(
                10.0, torch.empty_like(vv).uniform_(-3, 3, generator=gen)
            )
        if v_under:
            vv = vv * 1e-30
        return _pack(qn, kn, vv, gg, bb, dtype)

    return maker


STABILITY_MODES = {
    "no_decay": _mode(no_decay=True),
    "max_decay": _mode(max_decay=True),
    "beta_full": _mode(beta_full=True),
    "v_dynamic": _mode(v_dyn=True),
    "v_underflow": _mode(v_under=True),
}

SMOKE_SHAPE = {"B": 2, "T": 128, "H": 4, "K": 64, "V": 64}
DETERMINISM_SHAPE = {"B": 4, "T": 512, "H": 8, "K": 128, "V": 128}
STABILITY_SHAPE = {"B": 2, "T": 512, "H": 8, "K": 128, "V": 128}

ANTI_CHEAT_RUBRIC_IDS = list(taskdef_api.DEFAULT_ANTI_CHEAT_RUBRIC_IDS)

HIDDEN_SHAPES = {
    "graded": {
        "g1": {"B": 1, "T": 4096, "H": 8, "K": 128, "V": 128, "dtypes": ["bfloat16"]},
        "g2": {
            "B": 4,
            "T": 1024,
            "H": 8,
            "K": 128,
            "V": 128,
            "dtypes": ["bfloat16", "float16"],
        },
        "g3": {"B": 2, "T": 2048, "H": 16, "K": 64, "V": 64, "dtypes": ["bfloat16"]},
        "g4": {"B": 2, "T": 2048, "H": 8, "K": 128, "V": 128, "dtypes": ["bfloat16"]},
        "g5": {"B": 16, "T": 512, "H": 8, "K": 128, "V": 128, "dtypes": ["bfloat16"]},
    },
    "correctness_only": {
        "c1": {
            "B": 1,
            "T": 127,
            "H": 4,
            "K": 128,
            "V": 128,
            "dtypes": ["bfloat16", "float16", "float32"],
        },
        "c2": {
            "B": 2,
            "T": 1000,
            "H": 8,
            "K": 64,
            "V": 64,
            "dtypes": ["bfloat16", "float32"],
        },
        "c3": {
            "B": 2,
            "T": 65,
            "H": 4,
            "K": 96,
            "V": 96,
            "dtypes": ["bfloat16", "float32"],
        },
        "c4": {
            "B": 1,
            "T": 513,
            "H": 4,
            "K": 64,
            "V": 64,
            "dtypes": ["bfloat16", "float32"],
        },
    },
}

BENCH_PUBLISHED_SHAPES = {
    "p1": {"B": 8, "T": 1024, "H": 8, "K": 128, "V": 128, "dtypes": ["bfloat16"]},
    "p2": {"B": 4, "T": 2048, "H": 8, "K": 64, "V": 128, "dtypes": ["bfloat16"]},
}
BENCH_CHECK_SHAPE = {"B": 2, "T": 128, "H": 4, "K": 64, "V": 64}
BENCH_CHECK_SHAPE_NPOT = {"B": 3, "T": 130, "H": 3, "K": 64, "V": 64}


_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


class _Fb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, num_householder):
        with torch.enable_grad():
            qd = q.detach().requires_grad_(); kd = k.detach().requires_grad_()
            vd = v.detach().requires_grad_(); gd = g.detach().requires_grad_()
            bd = beta.detach().requires_grad_()
            B, T, H, K = qd.shape; V = vd.shape[-1]; N = int(num_householder)
            q32 = qd.float() * (K ** -0.5); k32 = kd.float(); v32 = vd.float()
            g32 = gd.float(); b32 = bd.float()
            S = torch.zeros(B, H, K, V, dtype=torch.float32, device=qd.device)
            outs = []
            for t in range(T):
                S = S * g32[:, t].exp()[:, :, None, None]
                for j in range(N):
                    kt = k32[:, t*N+j]; vt = v32[:, t*N+j]; bt = b32[:, t*N+j]
                    pred = torch.einsum("bhk,bhkv->bhv", kt, S)
                    delta = (vt - pred) * bt[:, :, None]
                    S = S + torch.einsum("bhk,bhv->bhkv", kt, delta)
                outs.append(torch.einsum("bhk,bhkv->bhv", q32[:, t], S))
            out = torch.stack(outs, 1).to(vd.dtype)
        ctx.save_for_backward(qd, kd, vd, gd, bd, out)
        return out.detach()

    @staticmethod
    def backward(ctx, dout):
        qd, kd, vd, gd, bd, out = ctx.saved_tensors
        grads = torch.autograd.grad(out, [qd, kd, vd, gd, bd], dout, allow_unused=True)
        return (*grads, None)


def chunked_gated_delta_product(q, k, v, g, beta, num_householder=2):
    return _Fb.apply(q, k, v, g, beta, num_householder)
'''


def _mut_vendor(src):
    """Renamed paste of the library's own naive gated-delta-product source
    (structural-half G1 target). Carved from the pinned install at mutation time."""
    spec = importlib.util.find_spec("fla")
    root = Path(spec.origin).resolve().parent
    text = (root / "ops" / "gated_delta_product" / "naive.py").read_text()
    text = text.replace("naive_recurrent_gated_delta_product", "_vnd_scan")
    wrapper = (
        "\n\ndef chunked_gated_delta_product(q, k, v, g, beta, num_householder=2):\n"
        "    o, _ = _vnd_scan(q, k, v, g, beta, scale=q.shape[-1] ** -0.5,\n"
        "                     cu_seqlens=None, output_final_state=False,\n"
        "                     num_householder=num_householder)\n"
        "    return o\n"
    )
    out = text + wrapper
    stripped = "\n".join(l.split("#")[0] for l in out.splitlines())
    for sym in FORBIDDEN:
        if re.search(r"\b" + re.escape(sym) + r"\b", stripped):
            raise RuntimeError(f"INERT: forbidden name {sym!r} survived the rename")
    return out


NEGATIVE_CONTROLS = [
    {
        "name": "nc_import",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "import triton.language as tl\n",
            "import triton.language as tl\n"
            "from fla.ops.gated_delta_product import chunk_gated_delta_product as _lib\n",
        ),
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "the forbidden-symbol scan names the library import and zeroes at "
        "G1 before the candidate is ever imported",
    },
    {
        "name": "nc_vendor",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": _mut_vendor,
        "bound_gate": "G1",
        "oracle_env": False,
        "expectation": "a renamed paste of FLA's own naive gated-delta-product source "
        "carries no forbidden name, but its rename-normalized shingle "
        "containment exceeds the ceiling: G1's structural half zeroes it "
        "(value UNVERIFIED until the controls stage measures it)",
    },
    {
        "name": "nc_bad_scale",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("q32 = q.float() * (K ** -0.5)", "q32 = q.float() * (K ** -0.25)"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the q-scaling is wrong by K^0.25, a uniform output and gradient "
        "error far outside tolerance: G2 fails at the smoke stage",
    },
    {
        "name": "nc_nondet",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": lambda src: (
            src
            + """

_INNER = chunked_gated_delta_product
_N = [0]


def chunked_gated_delta_product(q, k, v, g, beta, num_householder=2):
    _N[0] += 1
    if _N[0] % 2 == 0:
        return _INNER(q.flip(-1).contiguous(), k.flip(-1).contiguous(), v, g, beta, num_householder)
    return _INNER(q, k, v, g, beta, num_householder)
"""
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the K-lane reduction order alternates across invocations (flip q, k "
        "along K -- identical math, different float32 order); every answer is "
        "tolerance-valid so the single smoke run passes, but the three-run "
        "bitwise determinism stage catches it at G2",
    },
    {
        "name": "nc_dp",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "    return _GatedDeltaProduct.apply(q, k, v, g, beta, num_householder)",
            "    if float(k.float().abs().amax()) <= 0.55:\n"
            "        v = v * 0.0\n"
            "    return _GatedDeltaProduct.apply(q, k, v, g, beta, num_householder)",
        ),
        "bound_gate": "G2",
        "oracle_env": True,
        "expectation": "the §5.4 differential path: detect the graded distribution by key "
        "magnitude and run truncated math there (zero v). Timed tensors ARE "
        "checked tensors, so the cheap path runs on graded draws and G2 fails; "
        "FORGE_ORACLE=1 keeps G1/G4 exempt so the zero can only come from G2",
    },
    {
        "name": "nc_torch_fallback",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": lambda src: _FALLBACK_SRC,
        "bound_gate": "G4",
        "oracle_env": False,
        "expectation": "correct and deterministic pure-torch scan, 100% of device time outside "
        "candidate Triton kernels: G2 passes and the adoption floor zeroes at G4",
    },
]
