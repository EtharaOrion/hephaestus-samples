"""Taskdef: chunk-parallel gated delta rule, forward+backward (SURFACE fwdbwd).

FLAGSHIP / HARDEST task of the chunked_linear_attn family (spec.yaml hard:true):
it carries BOTH the per-head gated decay AND the delta-rule WY inverse (the exact
intra-chunk lower-triangular solve) AND grouped value attention (HV > H) -- the
most moving parts, so the widest gap between a naive chunk starter and the fused
production kernel, and the lowest reachable fraction of the anchor. It mirrors the
project's calibrated KDA exemplar (seed/forge/categories/kda) exactly.

Operator (reference.py is the definition; this is a summary):
    chunked_gated_delta(q[B,T,H,K], k[B,T,H,K], v[B,T,HV,V], g[B,T,HV], beta[B,T,HV])
        -> o[B,T,HV,V]
    S'_t = diag(exp(g_t)) S_{t-1};  S_t = S'_t + k_t (v_t - S'_t^T k_t)^T beta_t;
    o_t = S_t^T (q_t / sqrt(K)). Decay BEFORE prediction. Grouped value attention:
    HV value heads share H key heads (q, k expanded HV//H to one). SURFACE fwdbwd:
    gradients of q, k, v, g, beta are graded, timed and determinism-checked.

Anchor: fla.ops.gated_delta_rule.chunk_gated_delta_rule -- the production chunk
kernel of the pinned library (editable install /home/hephaestus-gpu/opt-fla).
sota() calls it with KEYWORDS and use_qk_l2norm_in_kernel=False; scale=None
defaults to K^-0.5 inside the library, matching the reference.

STATUS: authored-draft -- NEEDS GPU CALIBRATION. reference.py is CPU-verified
against the library's own naive ground truth to 6e-8 (fla naive_recurrent_gated_
delta_rule). TOL, TARGET_FRACTION_OF_SOTA, the vendor-shingle ceiling, the anchor
semantic-divergence check and every negative control's bound-gate behaviour are
INHERITED placeholders that a GPU calibration pass (measure/golden/controls) must
re-derive and confirm; see spec.yaml.

Distribution invariance (frontier-defeat-analysis §7.1): q, k are L2-normalized
along K (the operator's own contraction convention, FLA carries
use_qk_l2norm_in_kernel for exactly this), so the recurrence stays bounded at any
hidden scale; the per-invocation scale grade_core draws multiplies v (the one
magnitude carrier), and the tensors that are timed ARE the tensors that are
checked. NC-DP plants the §5.4 magnitude-branch exploit on the oracle under
FORGE_ORACLE=1 and must zero at exactly G2.
"""

import importlib.util
import re
from pathlib import Path

import torch
import torch.nn.functional as F

import taskdef_api  # noqa: F401  (protocol module; defaults resolved via cfg)

ENTRY_NAME = "chunked_gated_delta"
REF_NAME = "chunked_gated_delta_ref"
SIGNATURE = ("chunked_gated_delta(q[B,T,H,K], k[B,T,H,K], v[B,T,HV,V], "
             "g[B,T,HV], beta[B,T,HV]) -> o[B,T,HV,V]")
SURFACE = "fwdbwd"
TASK_NAME = "hephaestus/chunked_gated_delta"

# INHERITED from the calibrated KDA/gdn family (bf16/fp16/fp32 knees). For fwdbwd
# this tolerance applies to the OUTPUT and to ALL FIVE gradients (dq, dk, dv, dg,
# dbeta). PLACEHOLDER -- a GPU tol-calibration pass must re-measure the smallest
# atol/rtol that passes the oracle AND the anchor on every graded cell, gradients
# included (the backward of a recurrence this long typically needs a wider knee
# than the forward; carry >= 3x margin over the measured worst honest drift).
TOL = {
    "bfloat16": (2.0e-2, 3.125e-2),
    "float16": (1.0e-2, 3.90625e-3),
    "float32": (2.0e-3, 2.0e-3),
}

# G1 name half: calling/importing the anchor library (or any library implementation
# of this operator family) is not writing a kernel. Reading the installed source is
# allowed; word-boundary scan over comment-stripped candidate source, strings and
# docstrings count.
FORBIDDEN = [
    "fla.ops", "import fla", "from fla",
    "chunk_gated_delta_rule", "chunk_gdn", "chunk_kda", "chunk_delta_rule",
    "chunk_gla", "chunk_simple_gla", "chunk_retention", "chunk_delta_h",
    "fused_recurrent_gated_delta_rule", "fused_recurrent_gdn",
    "fused_recurrent_kda", "fused_recurrent_delta_rule", "fused_recurrent_gla",
    "fused_recurrent_simple_gla", "fused_recurrent_retention", "fused_recurrent",
    "naive_recurrent_gated_delta_rule", "naive_chunk_gated_delta_rule",
]

# G1 structure half: 32-token rename-normalized shingles vs the installed source.
VENDOR_LIB_PACKAGE = "fla"
VENDOR_LIB_SUBDIRS = ("ops/gated_delta_rule", "ops/common", "ops/utils", "ops/kda")

ADOPTION_FLOOR = 0.60

# PLACEHOLDER (hard task -> lowest reachable fraction). Seed value only; the
# authoritative target is measured by the golden stage and set BELOW the oracle's
# repeated worst case. NEEDS GPU CALIBRATION.
TARGET_FRACTION_OF_SOTA = 0.70

# The from-scratch oracle routes its chunk GEMMs / WY inverse through torch (cuBLAS)
# and only the cumulative-gate through its own Triton kernel, so it cannot clear the
# 0.60 adoption floor honestly until those are kernelized -> the golden stage uses
# the ORACLE_ENV exemption (the same lever KDA uses). Flip to False once the oracle
# is fully kernelized. NEEDS GPU CALIBRATION.
ORACLE_ENV_FOR_GOLDEN = True


# ---------------------------------------------------------------------------
# Input draws (fwdbwd: grad_names carry every float input)
# ---------------------------------------------------------------------------

def _dims(shape):
    H = int(shape["H"]); HV = int(shape.get("HV", H))
    return int(shape["B"]), int(shape["T"]), H, HV, int(shape["K"]), int(shape["V"])


def _draw(shape, seed, scale, device):
    """One nominal draw. q, k L2-normalized along K (scale-free); the hidden scale
    multiplies v (the magnitude carrier); g is log-space decay in (-inf, 0]; beta
    in (0, 1]. NaN-free by construction."""
    B, T, H, HV, K, V = _dims(shape)
    gen = torch.Generator(device=device); gen.manual_seed(seed & 0x7FFFFFFF)
    rn = lambda *s: torch.randn(*s, device=device, dtype=torch.float32, generator=gen)  # noqa: E731
    qn = F.normalize(rn(B, T, H, K), dim=-1)
    kn = F.normalize(rn(B, T, H, K), dim=-1)
    vv = rn(B, T, HV, V) * float(scale)
    gg = F.logsigmoid(rn(B, T, HV))
    bb = torch.rand(B, T, HV, device=device, dtype=torch.float32, generator=gen) * 0.9 + 0.05
    return qn, kn, vv, gg, bb


def _pack(qn, kn, vv, gg, bb, dtype):
    args = {
        "q": qn.to(dtype).detach().contiguous().requires_grad_(True),
        "k": kn.to(dtype).detach().contiguous().requires_grad_(True),
        "v": vv.to(dtype).detach().contiguous().requires_grad_(True),
        "g": gg.detach().contiguous().requires_grad_(True),        # log-decay stays float32
        "beta": bb.to(dtype).detach().contiguous().requires_grad_(True),
    }
    return args, ("q", "k", "v", "g", "beta")


def make_inputs(shape, seed, scale, dtype, device):
    return _pack(*_draw(shape, seed, scale, device), dtype)


def sota(args):
    """Production anchor: FLA's chunk_gated_delta_rule (timing denominator ONLY;
    reference.py defines correctness). KEYWORDS; scale=None => K^-0.5."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    out = chunk_gated_delta_rule(q=args["q"], k=args["k"], v=args["v"],
                                 g=args["g"], beta=args["beta"], scale=None,
                                 use_qk_l2norm_in_kernel=False)
    return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------
# Stability modes (finiteness-and-not-raising gate; drift recorded only)
# ---------------------------------------------------------------------------

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
            gen = torch.Generator(device=device); gen.manual_seed((seed ^ 0x5A5A) & 0x7FFFFFFF)
            vv = vv * torch.pow(10.0, torch.empty_like(vv).uniform_(-3, 3, generator=gen))
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

SMOKE_SHAPE = {"B": 2, "T": 128, "H": 4, "HV": 4, "K": 64, "V": 64}
DETERMINISM_SHAPE = {"B": 4, "T": 512, "H": 4, "HV": 8, "K": 128, "V": 128}
STABILITY_SHAPE = {"B": 2, "T": 512, "H": 8, "HV": 8, "K": 128, "V": 128}

# fwdbwd surface -> default anti-cheat id set (includes
# timing_covers_forward_and_backward). Must match rubrics.jsonl at freeze.
ANTI_CHEAT_RUBRIC_IDS = list(taskdef_api.DEFAULT_ANTI_CHEAT_RUBRIC_IDS)

# Hidden graded / correctness-only shape sets (authoritative copy in spec.yaml;
# a scaffold pass emits hidden_shapes.json from here). GVA (HV != H) is exercised.
HIDDEN_SHAPES = {
    "graded": {
        "g1": {"B": 1, "T": 4096, "H": 8, "HV": 8, "K": 128, "V": 128, "dtypes": ["bfloat16"]},
        "g2": {"B": 4, "T": 1024, "H": 8, "HV": 8, "K": 128, "V": 128, "dtypes": ["bfloat16", "float16"]},
        "g3": {"B": 2, "T": 2048, "H": 4, "HV": 16, "K": 128, "V": 128, "dtypes": ["bfloat16"]},
        "g4": {"B": 2, "T": 2048, "H": 8, "HV": 8, "K": 256, "V": 64, "dtypes": ["bfloat16"]},
        "g5": {"B": 16, "T": 512, "H": 8, "HV": 8, "K": 128, "V": 128, "dtypes": ["bfloat16"]},
    },
    "correctness_only": {
        "c1": {"B": 1, "T": 127, "H": 4, "HV": 4, "K": 128, "V": 128, "dtypes": ["bfloat16", "float16", "float32"]},
        "c2": {"B": 2, "T": 1000, "H": 8, "HV": 8, "K": 64, "V": 64, "dtypes": ["bfloat16", "float32"]},
        "c3": {"B": 2, "T": 65, "H": 4, "HV": 12, "K": 96, "V": 96, "dtypes": ["bfloat16", "float32"]},
    },
}

BENCH_PUBLISHED_SHAPES = {
    "p1": {"B": 8, "T": 1024, "H": 8, "HV": 8, "K": 128, "V": 128, "dtypes": ["bfloat16"]},
    "p2": {"B": 4, "T": 2048, "H": 4, "HV": 8, "K": 64, "V": 128, "dtypes": ["bfloat16"]},
}
BENCH_CHECK_SHAPE = {"B": 2, "T": 128, "H": 4, "HV": 4, "K": 64, "V": 64}
BENCH_CHECK_SHAPE_NPOT = {"B": 3, "T": 130, "H": 3, "HV": 6, "K": 64, "V": 64}


# ---------------------------------------------------------------------------
# Negative controls (Invariant 17, both halves). Every arm must zero by EXACTLY
# its bound gate. AUTHORED-DRAFT: bound-gate behaviour is UNVERIFIED until the
# controls stage runs on the GPU (the vendor-shingle containment in particular
# must be measured against the pinned tree).
# ---------------------------------------------------------------------------

_FALLBACK_SRC = '''\
"""Pure-torch fallback candidate (negative control): no Triton anywhere."""
import torch


class _Fb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        with torch.enable_grad():
            qd = q.detach().requires_grad_(); kd = k.detach().requires_grad_()
            vd = v.detach().requires_grad_(); gd = g.detach().requires_grad_()
            bd = beta.detach().requires_grad_()
            B, T, H, K = qd.shape; HV = vd.shape[2]
            q32 = qd.float() * (K ** -0.5); k32 = kd.float(); v32 = vd.float()
            if HV != H:
                rep = HV // H
                q32 = q32.repeat_interleave(rep, dim=2); k32 = k32.repeat_interleave(rep, dim=2); H = HV
            g32 = gd.float(); b32 = bd.float()
            S = torch.zeros(B, H, K, vd.shape[-1], dtype=torch.float32, device=qd.device)
            outs = []
            for t in range(T):
                S = S * g32[:, t].exp()[:, :, None, None]
                kt = k32[:, t]
                pred = torch.einsum("bhk,bhkv->bhv", kt, S)
                delta = (v32[:, t] - pred) * b32[:, t][:, :, None]
                S = S + torch.einsum("bhk,bhv->bhkv", kt, delta)
                outs.append(torch.einsum("bhk,bhkv->bhv", q32[:, t], S))
            out = torch.stack(outs, 1).to(vd.dtype)
        ctx.save_for_backward(qd, kd, vd, gd, bd, out)
        return out.detach()

    @staticmethod
    def backward(ctx, dout):
        qd, kd, vd, gd, bd, out = ctx.saved_tensors
        return torch.autograd.grad(out, [qd, kd, vd, gd, bd], dout, allow_unused=True)


def chunked_gated_delta(q, k, v, g, beta):
    return _Fb.apply(q, k, v, g, beta)
'''


def _mut_vendor(src):
    """Replace the oracle with a RENAMED paste of the library's own naive source
    for this operator (structural-half G1 target: no forbidden name survives the
    renames, so only the rename-normalized shingle scan can see it). Carved from
    the pinned install at mutation time."""
    spec = importlib.util.find_spec("fla")
    root = Path(spec.origin).resolve().parent
    text = (root / "ops" / "gated_delta_rule" / "naive.py").read_text()
    for old, new in (("naive_recurrent_gated_delta_rule", "_vnd_scan"),
                     ("naive_chunk_gated_delta_rule", "_vnd_chunk_scan")):
        text = text.replace(old, new)
    wrapper = (
        "\n\ndef chunked_gated_delta(q, k, v, g, beta):\n"
        "    o = _vnd_scan(q, k, v, beta, g, scale=q.shape[-1] ** -0.5,\n"
        "                  output_final_state=False)\n"
        "    return o[0] if isinstance(o, tuple) else o\n"
    )
    out = text + wrapper
    stripped = "\n".join(l.split("#")[0] for l in out.splitlines())
    for sym in FORBIDDEN:
        if re.search(r"\b" + re.escape(sym) + r"\b", stripped):
            raise RuntimeError(f"INERT: forbidden name {sym!r} survived the rename")
    return out


NEGATIVE_CONTROLS = [
    {
        "name": "nc_import", "base": "oracle", "target_file": "kernel.py",
        "mutate": ("import triton.language as tl\n",
                   "import triton.language as tl\n"
                   "from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _lib\n"),
        "bound_gate": "G1", "oracle_env": False,
        "expectation": "the forbidden-symbol scan names the library import and zeroes at "
                       "G1 before the candidate is ever imported",
    },
    {
        "name": "nc_vendor", "base": "oracle", "target_file": "kernel.py",
        "mutate": _mut_vendor, "bound_gate": "G1", "oracle_env": False,
        "expectation": "a renamed paste of FLA's own naive source carries no forbidden "
                       "name, but its 32-token rename-normalized shingle containment "
                       "exceeds the ceiling: G1's structural half zeroes it (containment "
                       "value UNVERIFIED until the controls stage measures it)",
    },
    {
        "name": "nc_bad_scale", "base": "oracle", "target_file": "kernel.py",
        "mutate": ("scale = K ** -0.5", "scale = K ** -0.25"),
        "bound_gate": "G2", "oracle_env": False,
        "expectation": "the q-scaling is wrong by a factor K^0.25, a uniform output and "
                       "gradient error far outside tolerance: G2 fails at the smoke stage",
    },
    {
        "name": "nc_nondet", "base": "oracle", "target_file": "kernel.py",
        "mutate": lambda src: src + '''

_INNER = chunked_gated_delta
_N = [0]


def chunked_gated_delta(q, k, v, g, beta):
    _N[0] += 1
    if _N[0] % 2 == 0:
        return _INNER(q.flip(-1).contiguous(), k.flip(-1).contiguous(), v, g, beta)
    return _INNER(q, k, v, g, beta)
''',
        "bound_gate": "G2", "oracle_env": False,
        "expectation": "the K-lane reduction order alternates across invocations (flip q, k "
                       "along K -- identical math, different float32 order); every answer is "
                       "tolerance-valid so the single smoke run passes, but the three-run "
                       "bitwise determinism stage catches it at G2",
    },
    {
        "name": "nc_dp", "base": "oracle", "target_file": "kernel.py",
        "mutate": ("    return _GatedDeltaOracle.apply(q, k, v, g, beta)",
                   "    if float(k.float().abs().amax()) <= 0.55:\n"
                   "        v = v * 0.0\n"
                   "    return _GatedDeltaOracle.apply(q, k, v, g, beta)"),
        "bound_gate": "G2", "oracle_env": True,
        "expectation": "the canonical §5.4 differential path: detect the graded distribution "
                       "by key magnitude and run truncated math there (zero v). Timed tensors "
                       "ARE the checked tensors, so the cheap path runs on graded draws and G2 "
                       "fails; FORGE_ORACLE=1 keeps G1/G4 exempt so the zero can only come "
                       "from G2",
    },
    {
        "name": "nc_torch_fallback", "base": "oracle", "target_file": "kernel.py",
        "mutate": lambda src: _FALLBACK_SRC, "bound_gate": "G4", "oracle_env": False,
        "expectation": "correct and deterministic pure-torch scan, 100% of device time "
                       "outside candidate Triton kernels: G2 passes and the written-kernel "
                       "adoption floor zeroes at G4",
    },
]
