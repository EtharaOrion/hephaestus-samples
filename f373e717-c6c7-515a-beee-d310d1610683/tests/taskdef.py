"""Taskdef: chunk-parallel RWKV-6 (a.k.a. WKV-6), forward+backward (fwdbwd).

Member of the chunked_linear_attn family. HARD variant: distinct from every other
member because RWKV-6 carries a data-independent per-(head, channel) BONUS `u`
that boosts the CURRENT-step outer product ONLY in the output -- the state
carried to t+1 does not see u. That asymmetry breaks the clean chunked
lower-triangular attention factorization at the diagonal: the intra-chunk kernel
must compute two closely-related terms per position (u-boosted on the diagonal,
plain elsewhere) instead of one.

Operator (reference.py is the definition; this is a summary):
    chunked_rwkv6(r[B,T,H,K], k[B,T,H,K], v[B,T,H,V], w[B,T,H,K], u[H,K])
        -> o[B,T,H,V]
    S_t = diag(exp(w_t)) S_{t-1} + k_t v_t^T;
    o_t = r_t^T (S_{t-1} + diag(u) . k_t v_t^T).
    Decay AFTER readout. Per-key-channel log-decay `w`. `u` shared across (B,T).
    SURFACE fwdbwd: gradients of r, k, v, w, u are graded, timed and
    determinism-checked.

Anchor: fla.ops.rwkv6.chunk_rwkv6 -- the production chunk kernel of the pinned
library. sota() calls it with KEYWORDS and scale=None (=> K^-0.5). Returns
(o, final_state); sota() unwraps the tuple.

STATUS: authored-draft -- NEEDS GPU CALIBRATION. TOL, TARGET_FRACTION_OF_SOTA, the
vendor-shingle ceiling, the anchor semantic-divergence check and every negative
control's bound-gate behaviour are INHERITED placeholders; see spec.yaml.

Distribution invariance (frontier-defeat-analysis §7.1): r, k are L2-normalized
along K; w log-space in (-inf, 0]; the hidden per-invocation scale multiplies v.
u is drawn small (Uniform(-0.2, 0.2)) so its contribution stays comparable to
the recurrent term. Timed tensors ARE checked tensors; NC-DP plants the §5.4
magnitude-branch exploit on the oracle under FORGE_ORACLE=1 and must zero at
exactly G2.
"""

import importlib.util
import re
from pathlib import Path

import torch
import torch.nn.functional as F

import taskdef_api  # noqa: F401

ENTRY_NAME = "chunked_rwkv6"
REF_NAME = "chunked_rwkv6_ref"
SIGNATURE = (
    "chunked_rwkv6(r[B,T,H,K], k[B,T,H,K], v[B,T,H,V], "
    "w[B,T,H,K], u[H,K]) -> o[B,T,H,V]"
)
SURFACE = "fwdbwd"
TASK_NAME = "hephaestus/chunked_rwkv6"

# INHERITED KDA-family knees; applies to OUTPUT and to ALL FIVE gradients
# (dr, dk, dv, dw, du). Bf16 recurrences with a rank-1 u-bonus term accumulate
# both the state error AND the u-boost quadratic in the chunk; the upstream
# reference uses 0.05, and this variant carries the same relative slack for
# du/dw pending GPU calibration.
TOL = {
    "bfloat16": (5.0e-2, 5.0e-2),
    "float16": (1.0e-2, 3.90625e-3),
    "float32": (2.0e-3, 2.0e-3),
}

FORBIDDEN = [
    "fla.ops",
    "import fla",
    "from fla",
    "chunk_rwkv6",
    "fused_recurrent_rwkv6",
    "naive_recurrent_rwkv6",
    "naive_chunk_rwkv6",
    "chunk_rwkv7",
    "fused_recurrent_rwkv7",
    "chunk_gla",
    "chunk_simple_gla",
    "chunk_gated_delta_rule",
    "chunk_delta_rule",
    "chunk_retention",
    "chunk_gdn",
    "chunk_kda",
    "chunk_comba",
    "chunk_gdn2",
    "chunk_dplr_delta_rule",
    "chunk_gated_delta_product",
    "fused_recurrent_gla",
    "fused_recurrent_simple_gla",
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_delta_rule",
    "fused_recurrent_retention",
    "fused_recurrent",
]

VENDOR_LIB_PACKAGE = "fla"
VENDOR_LIB_SUBDIRS = ("ops/rwkv6", "ops/common", "ops/utils")

ADOPTION_FLOOR = 0.60

# PLACEHOLDER seed only; measured below the oracle's repeated worst case at
# golden. RWKV-6's u-bonus term costs an extra intra-chunk matmul per tile, so
# the reachable fraction against the fully fused anchor is not the highest of
# the family.
TARGET_FRACTION_OF_SOTA = 0.72

# From-scratch oracle: chunk GEMMs / u-bonus diagonal in torch; only the
# cumulative-decay kernel is real Triton -> golden uses the ORACLE_ENV exemption.
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
    rn_norm = F.normalize(rn(B, T, H, K), dim=-1)
    kn = F.normalize(rn(B, T, H, K), dim=-1)
    vv = rn(B, T, H, V) * float(scale)
    ww = F.logsigmoid(rn(B, T, H, K))  # per-channel log-decay <= 0
    uu = torch.empty(H, K, device=device, dtype=torch.float32).uniform_(
        -0.2, 0.2, generator=gen
    )
    return rn_norm, kn, vv, ww, uu


def _pack(rn, kn, vv, ww, uu, dtype):
    args = {
        "r": rn.to(dtype).detach().contiguous().requires_grad_(True),
        "k": kn.to(dtype).detach().contiguous().requires_grad_(True),
        "v": vv.to(dtype).detach().contiguous().requires_grad_(True),
        "w": ww.detach().contiguous().requires_grad_(True),  # log-decay stays float32
        "u": uu.detach().contiguous().requires_grad_(True),  # bonus stays float32
    }
    return args, ("r", "k", "v", "w", "u")


def make_inputs(shape, seed, scale, dtype, device):
    return _pack(*_draw(shape, seed, scale, device), dtype)


def sota(args):
    """Production anchor: FLA's chunk_rwkv6 (timing denominator ONLY;
    reference.py defines correctness). KEYWORDS; scale=None => K^-0.5.
    Returns (o, final_state); unwrap to the output tensor."""
    from fla.ops.rwkv6 import chunk_rwkv6

    out = chunk_rwkv6(
        r=args["r"],
        k=args["k"],
        v=args["v"],
        w=args["w"],
        u=args["u"],
        scale=None,
        output_final_state=False,
    )
    return out[0] if isinstance(out, tuple) else out


def _mode(
    no_decay=False,
    max_decay=False,
    mixed=False,
    u_zero=False,
    v_dyn=False,
    v_under=False,
):
    def maker(shape, seed, scale, dtype, device):
        rn, kn, vv, ww, uu = _draw(shape, seed, scale, device)
        if no_decay:
            ww = torch.zeros_like(ww)
        if max_decay:
            ww = torch.full_like(ww, -15.0)
        if mixed:
            half = ww.shape[-1] // 2
            ww = ww.clone()
            ww[..., :half] = 0.0
            ww[..., half:] = -15.0
        if u_zero:
            uu = torch.zeros_like(uu)
        if v_dyn:
            gen = torch.Generator(device=device)
            gen.manual_seed((seed ^ 0x5A5A) & 0x7FFFFFFF)
            vv = vv * torch.pow(
                10.0, torch.empty_like(vv).uniform_(-3, 3, generator=gen)
            )
        if v_under:
            vv = vv * 1e-30
        return _pack(rn, kn, vv, ww, uu, dtype)

    return maker


STABILITY_MODES = {
    "no_decay": _mode(no_decay=True),
    "max_decay": _mode(max_decay=True),
    "mixed_gate": _mode(mixed=True),
    "u_zero": _mode(u_zero=True),
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
        "g4": {"B": 2, "T": 2048, "H": 8, "K": 128, "V": 256, "dtypes": ["bfloat16"]},
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
    def forward(ctx, r, k, v, w, u):
        with torch.enable_grad():
            rd = r.detach().requires_grad_(); kd = k.detach().requires_grad_()
            vd = v.detach().requires_grad_(); wd = w.detach().requires_grad_()
            ud = u.detach().requires_grad_()
            B, T, H, K = rd.shape
            r32 = rd.float() * (K ** -0.5); k32 = kd.float(); v32 = vd.float()
            w32 = wd.float(); u32 = ud.float()
            S = torch.zeros(B, H, K, vd.shape[-1], dtype=torch.float32, device=rd.device)
            outs = []
            for t in range(T):
                kt = k32[:, t]; vt = v32[:, t]
                kv = kt[:, :, :, None] * vt[:, :, None, :]
                boosted = S + u32[None, :, :, None] * kv
                outs.append((r32[:, t][:, :, :, None] * boosted).sum(2))
                S = S * w32[:, t].exp()[:, :, :, None] + kv
            out = torch.stack(outs, 1).to(vd.dtype)
        ctx.save_for_backward(rd, kd, vd, wd, ud, out)
        return out.detach()

    @staticmethod
    def backward(ctx, dout):
        rd, kd, vd, wd, ud, out = ctx.saved_tensors
        return torch.autograd.grad(out, [rd, kd, vd, wd, ud], dout, allow_unused=True)


def chunked_rwkv6(r, k, v, w, u):
    return _Fb.apply(r, k, v, w, u)
'''


def _mut_vendor(src):
    """Renamed paste of the library's own naive RWKV-6 source (structural-half
    G1 target). Carved from the pinned install at mutation time."""
    spec = importlib.util.find_spec("fla")
    root = Path(spec.origin).resolve().parent
    text = (root / "ops" / "rwkv6" / "recurrent_naive.py").read_text()
    text = text.replace("naive_recurrent_rwkv6", "_vnd_scan")
    wrapper = (
        "\n\ndef chunked_rwkv6(r, k, v, w, u):\n"
        "    # naive expects [B, H, T, K] layout; adapt.\n"
        "    r2 = r.transpose(1, 2).contiguous(); k2 = k.transpose(1, 2).contiguous()\n"
        "    v2 = v.transpose(1, 2).contiguous(); w2 = w.transpose(1, 2).contiguous()\n"
        "    o, _ = _vnd_scan(r2, k2, v2, w2, u, scale=r.shape[-1] ** -0.5,\n"
        "                     output_final_state=False)\n"
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
            "from fla.ops.rwkv6 import chunk_rwkv6 as _lib\n",
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
        "expectation": "a renamed paste of FLA's own naive RWKV-6 source carries no "
        "forbidden name, but its rename-normalized shingle containment "
        "exceeds the ceiling: G1's structural half zeroes it (value "
        "UNVERIFIED until the controls stage measures it)",
    },
    {
        "name": "nc_bad_scale",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": ("r32 = r.float() * (K ** -0.5)", "r32 = r.float() * (K ** -0.25)"),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the r-scaling is wrong by K^0.25, a uniform output and gradient "
        "error far outside tolerance: G2 fails at the smoke stage",
    },
    {
        "name": "nc_nondet",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": lambda src: (
            src
            + """

_INNER = chunked_rwkv6
_N = [0]


def chunked_rwkv6(r, k, v, w, u):
    _N[0] += 1
    if _N[0] % 2 == 0:
        return _INNER(r.flip(-1).contiguous(), k.flip(-1).contiguous(), v,
                       w.flip(-1).contiguous(), u.flip(-1).contiguous())
    return _INNER(r, k, v, w, u)
"""
        ),
        "bound_gate": "G2",
        "oracle_env": False,
        "expectation": "the K-channel reduction order alternates across invocations (flip "
        "r, k, w, u consistently along K -- identical math, different "
        "float32 order); every answer is tolerance-valid so the single smoke "
        "run passes, but the three-run bitwise determinism stage catches it "
        "at G2",
    },
    {
        "name": "nc_dp",
        "base": "oracle",
        "target_file": "kernel.py",
        "mutate": (
            "    return _RWKV6.apply(r, k, v, w, u)",
            "    if float(k.float().abs().amax()) <= 0.55:\n"
            "        v = v * 0.0\n"
            "    return _RWKV6.apply(r, k, v, w, u)",
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
