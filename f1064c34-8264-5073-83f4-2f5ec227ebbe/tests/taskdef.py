"""Taskdef: chunk-parallel DPLR generalized delta rule (RWKV-7 family),
forward+backward (fwdbwd).

Member of the chunked_linear_attn family. HARD variant. Distinct from every
other member because the recurrence has BOTH a per-K-channel log-decay `gk`
AND a rank-1 (alpha, beta) additive correction term whose `alpha` reads a
V-vector out of the state and `beta` writes it back at a different K position
-- the Diagonal-Plus-Low-Rank (DPLR) form. This composition drives the
generalized delta family that RWKV-7 is built on and it is what the pinned
library's own naive dplr_recurrence and dplr_chunkwise reference exercise.

Operator (reference.py is the definition; this is a summary):
    chunked_dplr_delta(q[B,T,H,K], k[B,T,H,K], v[B,T,H,V],
                       alpha[B,T,H,K], beta[B,T,H,K], gk[B,T,H,K])
        -> o[B,T,H,V]
    S_t = diag(exp(gk_t)) S_{t-1} + k_t v_t^T + beta_t (S_{t-1}^T alpha_t)^T
    o_t = S_t^T (q_t / sqrt(K))
    SURFACE fwdbwd: gradients of q, k, v, alpha, beta, gk are graded, timed and
    determinism-checked.

Anchor: fla.ops.generalized_delta_rule.chunk_dplr_delta_rule -- the production
chunk kernel of the pinned library. sota() calls it with KEYWORDS and
scale=None (=> K^-0.5); unwraps the (o, final_state) tuple.

STATUS: authored-draft -- NEEDS GPU CALIBRATION. TOL, TARGET_FRACTION_OF_SOTA, the
vendor-shingle ceiling, the anchor semantic-divergence check and every negative
control's bound-gate behaviour are INHERITED placeholders; see spec.yaml. The
long-context graded shape (T = 8192) is this variant's SIGNATURE stress: state
carried across ~128 chunk boundaries where numerical error accumulates.

Distribution invariance (frontier-defeat-analysis §7.1): q, k L2-normalized
along K; gk log-space in (-inf, 0]; alpha, beta drawn small
(std 0.5, clipped) so the rank-1 correction stays bounded; the hidden per-
invocation scale multiplies v. Timed tensors ARE checked tensors; NC-DP plants
the §5.4 magnitude-branch exploit on the oracle under FORGE_ORACLE=1 and must
zero at exactly G2.
"""

import importlib.util
import re
from pathlib import Path

import torch
import torch.nn.functional as F

import taskdef_api  # noqa: F401

ENTRY_NAME = "chunked_dplr_delta"
REF_NAME = "chunked_dplr_delta_ref"
SIGNATURE = (
    "chunked_dplr_delta(q[B,T,H,K], k[B,T,H,K], v[B,T,H,V], "
    "alpha[B,T,H,K], beta[B,T,H,K], gk[B,T,H,K]) -> o[B,T,H,V]"
)
SURFACE = "fwdbwd"
TASK_NAME = "hephaestus/chunked_dplr_delta"

# INHERITED family knees, widened for the long-context recurrence stress: at
# T = 8192 the chunk-boundary state accumulates rank-1 error across ~128
# carries; the upstream reference uses 0.05 for bf16 recurrences of this
# family, and this variant carries the same relative slack per graded cell.
TOL = {
    "bfloat16": (5.0e-2, 5.0e-2),
    "float16": (1.0e-2, 3.90625e-3),
    "float32": (2.0e-3, 2.0e-3),
}

FORBIDDEN = [
    "fla.ops",
    "import fla",
    "from fla",
    "chunk_dplr_delta_rule",
    "chunk_iplr_delta_rule",
    "fused_recurrent_dplr_delta_rule",
    "fused_recurrent_iplr_delta_rule",
    "dplr_recurrence",
    "dplr_chunkwise",
    "chunk_rwkv7",
    "fused_recurrent_rwkv7",
    "fused_mul_recurrent_rwkv7",
    "chunk_rwkv6",
    "fused_recurrent_rwkv6",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_product",
    "chunk_delta_rule",
    "chunk_gla",
    "chunk_simple_gla",
    "chunk_retention",
    "chunk_gdn",
    "chunk_gdn2",
    "chunk_kda",
    "chunk_comba",
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_delta_rule",
    "fused_recurrent_gla",
    "fused_recurrent_simple_gla",
    "fused_recurrent_retention",
    "fused_recurrent",
]

VENDOR_LIB_PACKAGE = "fla"
VENDOR_LIB_SUBDIRS = (
    "ops/generalized_delta_rule/dplr",
    "ops/rwkv7",
    "ops/common",
    "ops/utils",
)

ADOPTION_FLOOR = 0.60

# PLACEHOLDER seed (long-context DPLR is the hardest state-carry regime; the
# reachable fraction against the fully fused anchor is low).
TARGET_FRACTION_OF_SOTA = 0.63

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
    gen = torch.Generator(device=device)
    gen.manual_seed(seed & 0x7FFFFFFF)
    rn = lambda *s: torch.randn(*s, device=device, dtype=torch.float32, generator=gen)  # noqa: E731
    qn = F.normalize(rn(B, T, H, K), dim=-1)
    kn = F.normalize(rn(B, T, H, K), dim=-1)
    vv = rn(B, T, H, V) * float(scale)
    gg = F.logsigmoid(rn(B, T, H, K))
    # alpha, beta: unit-normalized (like q/k above) so |alpha||beta| <= 1 and the additive
    # rank-1 correction stays a contraction; unnormalized randn gives |alpha| ~ 0.5*sqrt(K) ~ 4,
    # which amplifies the state ~16x per step and overflows to NaN by T=256.
    aa = F.normalize(rn(B, T, H, K), dim=-1) * 0.5
    bb = F.normalize(rn(B, T, H, K), dim=-1) * 0.5
    return qn, kn, vv, aa, bb, gg


def _pack(qn, kn, vv, aa, bb, gg, dtype):
    args = {
        "q": qn.to(dtype).detach().contiguous().requires_grad_(True),
        "k": kn.to(dtype).detach().contiguous().requires_grad_(True),
        "v": vv.to(dtype).detach().contiguous().requires_grad_(True),
        "alpha": aa.to(dtype).detach().contiguous().requires_grad_(True),
        "beta": bb.to(dtype).detach().contiguous().requires_grad_(True),
        "gk": gg.detach().contiguous().requires_grad_(True),  # log-decay fp32
    }
    return args, ("q", "k", "v", "alpha", "beta", "gk")


def make_inputs(shape, seed, scale, dtype, device):
    return _pack(*_draw(shape, seed, scale, device), dtype)


def sota(args):
    """Production anchor: FLA's chunk_dplr_delta_rule. KEYWORDS; scale=None => K^-0.5."""
    from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule

    out = chunk_dplr_delta_rule(
        q=args["q"],
        k=args["k"],
        v=args["v"],
        a=args["alpha"],
        b=args["beta"],
        gk=args["gk"],
        scale=None,
        output_final_state=False,
    )
    return out[0] if isinstance(out, tuple) else out


def _mode(
    no_decay=False,
    max_decay=False,
    ab_zero=False,
    ab_large=False,
    v_dyn=False,
    v_under=False,
):
    def maker(shape, seed, scale, dtype, device):
        qn, kn, vv, aa, bb, gg = _draw(shape, seed, scale, device)
        if no_decay:
            gg = torch.zeros_like(gg)
        if max_decay:
            gg = torch.full_like(gg, -15.0)
        if ab_zero:
            aa = torch.zeros_like(aa)
            bb = torch.zeros_like(bb)  # collapses to plain GLA
        if ab_large:
            aa = aa * 4.0
            bb = bb * 4.0
        if v_dyn:
            gen = torch.Generator(device=device)
            gen.manual_seed((seed ^ 0x5A5A) & 0x7FFFFFFF)
            vv = vv * torch.pow(
                10.0, torch.empty_like(vv).uniform_(-3, 3, generator=gen)
            )
        if v_under:
            vv = vv * 1e-30
        return _pack(qn, kn, vv, aa, bb, gg, dtype)

    return maker


STABILITY_MODES = {
    "no_decay": _mode(no_decay=True),
    "max_decay": _mode(max_decay=True),
    "ab_zero": _mode(ab_zero=True),
    "ab_large": _mode(ab_large=True),
    "v_dynamic": _mode(v_dyn=True),
    "v_underflow": _mode(v_under=True),
}

SMOKE_SHAPE = {"B": 2, "T": 128, "H": 4, "K": 64, "V": 64}
DETERMINISM_SHAPE = {"B": 4, "T": 512, "H": 8, "K": 128, "V": 128}
STABILITY_SHAPE = {"B": 2, "T": 512, "H": 8, "K": 128, "V": 128}

ANTI_CHEAT_RUBRIC_IDS = list(taskdef_api.DEFAULT_ANTI_CHEAT_RUBRIC_IDS)

# g1 is the LONG-CONTEXT stress that is this variant's hard lever: T = 8192,
# ~128 chunk carries, error accumulates across every boundary.
HIDDEN_SHAPES = {
    "graded": {
        "g1": {"B": 1, "T": 8192, "H": 4, "K": 128, "V": 128, "dtypes": ["bfloat16"]},
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
    def forward(ctx, q, k, v, alpha, beta, gk):
        with torch.enable_grad():
            qd = q.detach().requires_grad_(); kd = k.detach().requires_grad_()
            vd = v.detach().requires_grad_(); ad = alpha.detach().requires_grad_()
            bd = beta.detach().requires_grad_(); gd = gk.detach().requires_grad_()
            B, T, H, K = qd.shape
            q32 = qd.float() * (K ** -0.5); k32 = kd.float(); v32 = vd.float()
            a32 = ad.float(); b32 = bd.float(); g32 = gd.float()
            S = torch.zeros(B, H, K, vd.shape[-1], dtype=torch.float32, device=qd.device)
            outs = []
            for t in range(T):
                sT_a = torch.einsum("bhk,bhkv->bhv", a32[:, t], S)
                rank1 = torch.einsum("bhk,bhv->bhkv", b32[:, t], sT_a)
                kv = torch.einsum("bhk,bhv->bhkv", k32[:, t], v32[:, t])
                S = S * g32[:, t].exp()[:, :, :, None] + kv + rank1
                outs.append(torch.einsum("bhk,bhkv->bhv", q32[:, t], S))
            out = torch.stack(outs, 1).to(vd.dtype)
        ctx.save_for_backward(qd, kd, vd, ad, bd, gd, out)
        return out.detach()

    @staticmethod
    def backward(ctx, dout):
        qd, kd, vd, ad, bd, gd, out = ctx.saved_tensors
        return torch.autograd.grad(out, [qd, kd, vd, ad, bd, gd], dout, allow_unused=True)


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Fb.apply(q, k, v, alpha, beta, gk)
'''


def _mut_vendor(src):
    """Renamed paste of the library's own naive DPLR source (structural-half G1
    target). Carved from the pinned install at mutation time."""
    spec = importlib.util.find_spec("fla")
    root = Path(spec.origin).resolve().parent
    text = (root / "ops" / "generalized_delta_rule" / "dplr" / "naive.py").read_text()
    for old, new in (
        ("dplr_recurrence", "_vnd_scan"),
        ("dplr_chunkwise", "_vnd_chunk"),
    ):
        text = text.replace(old, new)
    wrapper = (
        "\n\ndef chunked_dplr_delta(q, k, v, alpha, beta, gk):\n"
        "    # naive expects [B, H, T, K] layout; adapt.\n"
        "    q2 = q.transpose(1, 2).contiguous(); k2 = k.transpose(1, 2).contiguous()\n"
        "    v2 = v.transpose(1, 2).contiguous()\n"
        "    a2 = alpha.transpose(1, 2).contiguous(); b2 = beta.transpose(1, 2).contiguous()\n"
        "    g2 = gk.transpose(1, 2).contiguous()\n"
        "    o, _ = _vnd_scan(q2, k2, v2, a2, b2, g2, output_final_state=False)\n"
        "    return o.transpose(1, 2).contiguous()\n"
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
            "from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule as _lib\n",
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
        "expectation": "a renamed paste of FLA's own naive DPLR source carries no forbidden "
        "name, but its rename-normalized shingle containment exceeds the "
        "ceiling: G1's structural half zeroes it (value UNVERIFIED until the "
        "controls stage measures it)",
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

_INNER = chunked_dplr_delta
_N = [0]


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    _N[0] += 1
    if _N[0] % 2 == 0:
        return _INNER(q.flip(-1).contiguous(), k.flip(-1).contiguous(), v,
                       alpha.flip(-1).contiguous(), beta.flip(-1).contiguous(),
                       gk.flip(-1).contiguous())
    return _INNER(q, k, v, alpha, beta, gk)
"""
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the K-channel reduction order alternates across invocations (flip q, k, "
        "alpha, beta, gk consistently along K -- identical math, different "
        "float32 order); every answer is tolerance-valid so the single smoke "
        "run passes, but the three-run bitwise determinism stage catches it at G2",
    },
    {
        "name": "nc_dp",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "    return _Dplr.apply(q, k, v, alpha, beta, gk)",
            "    if float(k.float().abs().amax()) <= 0.55:\n"
            "        v = v * 0.0\n"
            "    return _Dplr.apply(q, k, v, alpha, beta, gk)",
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
