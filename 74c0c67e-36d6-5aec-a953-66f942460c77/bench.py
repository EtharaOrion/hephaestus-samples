#!/usr/bin/env python3
"""Published-shape benchmark. This is a steering tool, not the grader: the
shapes here are not the graded shapes, and the number it prints is not your
score.

It gates BEFORE it times, in the grader's order, because a timing number for
a kernel that is about to score zero is the most misleading thing this script
could print: forbidden-name scan, tolerance correctness on output AND all
two gradients (including a non-power-of-two and a collapsed-routing shape),
bitwise determinism, then the written-kernel share, and only then the timed
forward+backward pairs against the cuBLAS grouped-GEMM composition. This
file may call `torch._grouped_mm`; your `kernel.py` may not -- the scan below
is the same word-boundary rule the grader applies to your source.
"""

import argparse
import json
import re
import statistics
import sys

import torch

import os  # noqa: E402

_HERE = os.path.dirname(
    os.path.abspath(__file__)
)  # bundle dir in the container, or the agent workspace on host
sys.path.insert(0, _HERE)
from kernel import fp8_e4m3_swiglu_moe  # noqa: E402
from reference import fp8_e4m3_swiglu_moe_ref  # noqa: E402

# Correctness is checked on these shapes, not the timed ones: reference is a
# python (slot, expert) loop, so keeping the check shapes small means you run
# the check on every edit, which is the point.
CHECK_SHAPE = {"T": 256, "D": 512, "F": 256, "E": 8, "A": 4}
# Non-power-of-two everything. Triton code that assumes power-of-two D, F, E
# or A can pass the check above and die on the graded set with a mask bug or
# a CompilationError you never saw locally.
CHECK_SHAPE_NPOT = {"T": 173, "D": 500, "F": 176, "E": 11, "A": 3}
# Every slot of every token routed to expert 0: duplicate experts per token
# are legal, one segment holds all T*A rows, and E-1 experts must come back
# with exactly zero weight gradients.
CHECK_SHAPE_COLLAPSE = {"T": 256, "D": 256, "F": 176, "E": 16, "A": 4, "collapse": True}
# Determinism probe: A=8 puts eight weighted contributions on every output
# element (and eight slot gradients on every dx element), so a combine whose
# accumulation order depends on scheduling cannot repeat bitwise.
CHECK_SHAPE_DET = {"T": 512, "D": 512, "F": 256, "E": 16, "A": 8}

SHAPES = {
    "p1": {"T": 4096, "D": 1536, "F": 1024, "E": 32, "A": 4},
    "p2": {"T": 2048, "D": 2048, "F": 704, "E": 96, "A": 8},
}

# The grader's G1 scan, applied locally so it can never be the thing you
# learn about only from a zero: word-boundary, comments stripped, strings and
# docstrings count.
FORBIDDEN = [
    "torch._grouped_mm",
    "_grouped_mm",
    "grouped_mm",
    "fused_moe",
    "vllm",
    "sglang",
    "grouped_gemm",
    "transformer_engine",
    "cutlass",
    "flashinfer",
]

# Disclosed tolerances (the grader's, verbatim). The gradient atol is sized
# for dw1/dw2, whose entries accumulate over expert segments thousands of
# rows deep; the output tolerance is much tighter.
OUT_TOL = {"bfloat16": (0.35, 0.15), "float32": (0.20, 0.08)}
GRAD_TOL = {"bfloat16": (6.0, 0.10), "float32": (2.0, 0.02)}

GRADS = ("x", "topk_w")


def _snap_e4m3(w):
    if hasattr(torch, "float8_e4m3fn"):
        return w.to(torch.float8_e4m3fn).to(w.dtype)
    return torch.clamp(w, -448.0, 448.0)


def _quant(w_raw, dtype):
    """Per-out-channel symmetric quant to the e4m3 grid (amax / 448)."""
    ch_amax = w_raw.abs().amax(dim=1).clamp_min(1e-6)
    scale = (ch_amax / 448.0).to(dtype)
    w_q = _snap_e4m3(w_raw / scale[:, None, :].to(w_raw.dtype)).to(dtype)
    return w_q, scale


# H100 SXM dense bf16 tensor-core peak. A constant rather than a query
# because torch exposes no peak; this task is graded on H100 only.
PEAK_TFLOPS = 989.0


def make(shape, seed=0, dtype=torch.bfloat16):
    """Fixed seed, fixed unit scale, fixed zipf-like routing bias, so a rerun
    is comparable to the last run. The graded draws also randomize the input
    magnitude AND the routing popularity (strength and assignment) from
    hidden ranges per invocation; instruction.md discloses that, and a kernel
    whose policy is uniform (as required) cannot tell the difference."""
    T, D, F, E, A = (shape[k] for k in ("T", "D", "F", "E", "A"))
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    x = torch.randn(T, D, generator=g, device="cuda")
    w1_raw = torch.randn(E, D, 2 * F, generator=g, device="cuda") * (D**-0.5)
    w2_raw = torch.randn(E, F, D, generator=g, device="cuda") * (F**-0.5)
    w1_q, w1_s = _quant(w1_raw, dtype)
    w2_q, w2_s = _quant(w2_raw, dtype)
    if shape.get("collapse"):
        ti = torch.zeros(T, A, dtype=torch.int64, device="cuda")
        tw = torch.full((T, A), 1.0 / A, device="cuda")
    else:
        bias = -1.4 * torch.log1p(torch.arange(E, device="cuda", dtype=torch.float32))
        logits = torch.randn(T, E, generator=g, device="cuda") + bias[None, :]
        vals, ti = torch.topk(logits, A, dim=-1)
        tw = torch.softmax(vals, dim=-1)
    args = {
        "x": x.to(dtype).requires_grad_(True),
        "w1_q": w1_q,
        "w1_s": w1_s,
        "w2_q": w2_q,
        "w2_s": w2_s,
        "topk_idx": ti,
        "topk_w": tw.to(dtype).detach().requires_grad_(True),
    }
    return args


def fwdbwd(fn, args):
    for n in GRADS:
        args[n].grad = None
    y = fn(
        args["x"],
        args["w1_q"],
        args["w1_s"],
        args["w2_q"],
        args["w2_s"],
        args["topk_idx"],
        args["topk_w"],
    )
    y.sum().backward()
    return y


def anchor(x, w1_q, w1_s, w2_q, w2_s, topk_idx, topk_w):
    """The grader's denominator: eager per-expert dequant of w1_q/w2_q to
    x.dtype, then cuBLAS grouped GEMM on expert-sorted tokens."""
    T, D = x.shape
    E = w1_q.shape[0]
    F = w1_q.shape[2] // 2
    A = topk_idx.shape[1]
    w1_dq = w1_q.to(x.dtype) * w1_s.to(x.dtype).unsqueeze(1)
    w2_dq = w2_q.to(x.dtype) * w2_s.to(x.dtype).unsqueeze(1)
    flat = topk_idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E)
    offs = counts.cumsum(0).to(torch.int32)
    src = order // A
    xg = x.index_select(0, src)
    h = torch._grouped_mm(xg, w1_dq, offs=offs)
    act = torch.nn.functional.silu(h[:, :F]) * h[:, F:]
    yp = torch._grouped_mm(act, w2_dq, offs=offs)
    w_slot = topk_w.reshape(-1).index_select(0, order)
    return x.new_zeros(T, D).index_add(0, src, yp * w_slot.unsqueeze(1))


def scan_forbidden():
    src = open(os.path.join(_HERE, "kernel.py")).read()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    return [s for s in FORBIDDEN if re.search(r"\b" + re.escape(s) + r"\b", code)]


def _close(got, want, atol, rtol):
    if not torch.isfinite(got).all():
        return "non-finite values"
    d = (got.float() - want.float()).abs()
    lim = atol + rtol * want.float().abs()
    bad = int((d > lim).sum())
    if bad:
        return f"{bad} elements outside tolerance, max abs {float(d.max()):.3e}"
    return ""


def tol_check(shape, dt_name):
    dtype = getattr(torch, dt_name)
    args = make(shape, 0, dtype)
    try:
        y = fwdbwd(fp8_e4m3_swiglu_moe, args)
    except Exception as e:  # noqa: BLE001
        return [f"candidate raised {type(e).__name__}: {e}"]
    got_g = {n: args[n].grad.clone() for n in GRADS}
    yr = fwdbwd(fp8_e4m3_swiglu_moe_ref, args)
    fails = []
    if y.dtype != args["x"].dtype:
        fails.append(f"output dtype {y.dtype}, must be {args['x'].dtype}")
    if tuple(y.shape) != tuple(yr.shape):
        return fails + [f"output shape {tuple(y.shape)} != {tuple(yr.shape)}"]
    msg = _close(y, yr, *OUT_TOL[dt_name])
    if msg:
        fails.append(f"y: {msg}")
    for n in GRADS:
        if got_g[n] is None:
            fails.append(f"d{n}: missing gradient")
            continue
        msg = _close(got_g[n], args[n].grad, *GRAD_TOL[dt_name])
        if msg:
            fails.append(f"d{n}: {msg}")
    return fails


def check_determinism():
    """Three runs on one A=8 input after a warmup, bitwise identical output
    AND gradients. The candidate against ITSELF -- no tolerance applies."""
    args = make(CHECK_SHAPE_DET, 0, torch.bfloat16)
    runs = []
    for _ in range(4):
        y = fwdbwd(fp8_e4m3_swiglu_moe, args)
        runs.append((y.detach().clone(), {n: args[n].grad.clone() for n in GRADS}))
    runs = runs[1:]
    fails = []
    y0, g0 = runs[0]
    for i, (y, g) in enumerate(runs[1:], start=2):
        if not torch.equal(y, y0):
            fails.append(f"run {i}: y differs from run 1 at the bit level")
        for n in GRADS:
            if not torch.equal(g[n], g0[n]):
                fails.append(f"run {i}: d{n} differs from run 1 at the bit level")
    return fails


def _kernel_is_mine(key, names):
    if "::" in key or key.startswith("void ") or "<" in key:
        return False
    if re.match(r"^(ampere|turing|volta|hopper|maxwell|pascal|sm\d+)_", key):
        return False
    if re.search(r"(cutlass|cublas|cudnn|nvjet)", key, re.I):
        return False
    return any(key == n or key.startswith(n + "_") for n in names)


def check_adoption():
    """Share of forward+backward device time in Triton kernels THIS module
    declares, the same attribution the grader applies. Below 60% the grader
    zeroes before timing anything."""
    import ast
    from torch.profiler import ProfilerActivity, profile

    try:
        tree = ast.parse(open(os.path.join(_HERE, "kernel.py")).read())
    except (OSError, SyntaxError):
        return None
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                d = dec.func if isinstance(dec, ast.Call) else dec
                if ast.unparse(d).replace(" ", "").endswith("triton.jit"):
                    names.add(node.name)
    if not names:
        return {"adoption": 0.0, "declared": [], "total_us": 0.0, "mine_us": 0.0}
    args = make(SHAPES["p1"], 0, torch.bfloat16)
    for _ in range(3):
        fwdbwd(fp8_e4m3_swiglu_moe, args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fwdbwd(fp8_e4m3_swiglu_moe, args)
        torch.cuda.synchronize()
    total = mine = 0.0
    for e in prof.key_averages():
        dev = float(getattr(e, "self_device_time_total", 0) or 0)
        if dev <= 0:
            continue
        total += dev
        if _kernel_is_mine(e.key, names):
            mine += dev
    return {
        "adoption": (mine / total if total else 0.0),
        "declared": sorted(names),
        "total_us": round(total, 1),
        "mine_us": round(mine, 1),
    }


ADOPTION_FLOOR = 0.60


def sample(fn, args):
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    fwdbwd(fn, args)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b)


def timed_pair(args, warmup=3, trials=7, budget_s=60.0):
    """The graded quantity is a ratio, so it is measured as a ratio: samples
    interleaved so clock and thermal drift divide out, order alternating so
    any first-versus-second effect changes sign each trial. The grader
    measures the same way (and adds an L2 flush and hidden scales)."""
    import time

    t0 = time.time()
    fwdbwd(fp8_e4m3_swiglu_moe, args)
    torch.cuda.synchronize()
    one = time.time() - t0
    if one * 2 * (warmup + trials) > budget_s:
        warmup, trials = (
            1,
            (3 if int(budget_s / (2 * one)) < 3 else int(budget_s / (2 * one))),
        )
    for _ in range(warmup):
        fwdbwd(fp8_e4m3_swiglu_moe, args)
        fwdbwd(anchor, args)
    torch.cuda.synchronize()
    ms, ps = [], []
    for i in range(trials):
        if i % 2 == 0:
            m = sample(fp8_e4m3_swiglu_moe, args)
            p = sample(anchor, args)
        else:
            p = sample(anchor, args)
            m = sample(fp8_e4m3_swiglu_moe, args)
        ms.append(m)
        ps.append(p)
    return statistics.median(ms), statistics.median(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="")
    args_cli = ap.parse_args()

    def emit(payload, rc):
        if args_cli.json:
            json.dump(payload, open(args_cli.json, "w"), indent=2)
        return rc

    hits = scan_forbidden()
    if hits:
        print("FORBIDDEN NAMES in kernel.py:", ", ".join(hits))
        print()
        print("The grader scans your comment-stripped source with the same rule and")
        print("scores zero at its first gate, before your code is even imported.")
        print("Strings and docstrings count; # comments do not. Nothing was run.")
        return emit({"rows": [], "geomean": None, "forbidden": hits}, 1)

    for label, s in (
        ("check", CHECK_SHAPE),
        ("non-power-of-two", CHECK_SHAPE_NPOT),
        ("collapsed-routing", CHECK_SHAPE_COLLAPSE),
    ):
        for dt_name in ("bfloat16", "float32"):
            fails = tol_check(s, dt_name)
            if fails:
                print(f"CORRECTNESS FAILED on the {label} shape {s} {dt_name}:")
                for f in fails:
                    print(f"    {f}")
                print()
                print("Nothing was timed. Output and BOTH gradients (dx,")
                print("dtopk_w) are compared against reference.py under the disclosed")
                print("tolerances. A fast wrong MoE scores zero, not partial credit.")
                return emit(
                    {
                        "rows": [],
                        "geomean": None,
                        "correctness_failed": fails,
                        "shape": s,
                        "dtype": dt_name,
                    },
                    1,
                )
        print(
            f"correctness ok ({label}: {tuple(s[k] for k in ('T', 'D', 'F', 'E', 'A'))}, "
            f"both dtypes, output + both gradients)"
        )

    det = check_determinism()
    if det:
        print()
        print("NOT DETERMINISTIC on an A=8 input:")
        for f in det[:8]:
            print(f"    {f}")
        print()
        print("Nothing was timed. With eight contributions landing on every output")
        print("element (and eight slot gradients on every dx element), an atomic or")
        print("otherwise unordered accumulation changes between runs. The grader runs")
        print("the same three-run bitwise check -- output AND gradients -- and it")
        print("gates. This is your kernel against itself, so no tolerance applies.")
        return emit({"rows": [], "geomean": None, "nondeterministic": det}, 1)
    print(
        "determinism ok: three runs after a warmup, y and both gradients "
        "bitwise identical"
    )

    ad = check_adoption()
    if ad is not None:
        pct = 100.0 * ad["adoption"]
        print(
            f"written-kernel share: {pct:.1f}% of fwd+bwd device time in your own "
            f"triton kernels ({', '.join(ad['declared']) or 'none declared'})"
        )
        if ad["adoption"] < ADOPTION_FLOOR:
            print()
            print(f"BELOW THE WRITTEN-KERNEL FLOOR of {ADOPTION_FLOOR:.0%}.")
            print(
                f"{ad['mine_us']:.0f} us of {ad['total_us']:.0f} us ran in kernels this"
            )
            print("file declares; the rest ran in framework operators. The grader")
            print(
                "attributes device time the same way over forward+backward and scores"
            )
            print("zero however fast the composition is. Move the expert GEMMs and the")
            print("backward into your kernels.")
            return emit({"rows": [], "geomean": None, "adoption": ad}, 1)

    rows = []
    for label, s in SHAPES.items():
        args = make(s, 0, torch.bfloat16)
        mine, prod = timed_pair(args)
        # The operator is GEMM-bound: 6*S*D*F flops forward, 12*S*D*F more in
        # backward. Achieved TFLOP/s against dense bf16 peak says which wall
        # you are against -- production near peak means the remaining gap is
        # scheduling you can fix; both far below peak means launch overhead,
        # imbalance or memory traffic, not arithmetic.
        S = s["T"] * s["A"]
        tf = 18.0 * S * s["D"] * s["F"]
        mine_tf = tf / (mine * 1e-3) / 1e12
        prod_tf = tf / (prod * 1e-3) / 1e12
        rows.append(
            {
                "shape": label,
                **{k: s[k] for k in ("T", "D", "F", "E", "A")},
                "yours_ms": round(mine, 4),
                "production_ms": round(prod, 4),
                "fraction_of_production": round(
                    prod / mine if prod / mine < 1.0 else 1.0, 5
                ),
                "yours_tflops": round(mine_tf, 1),
                "production_tflops": round(prod_tf, 1),
                "yours_pct_peak": round(100 * mine_tf / PEAK_TFLOPS, 2),
                "production_pct_peak": round(100 * prod_tf / PEAK_TFLOPS, 2),
            }
        )
        frac = prod / mine if prod / mine < 1.0 else 1.0
        print(
            f"{label:4s} T={s['T']:6d} D={s['D']:5d} F={s['F']:5d} E={s['E']:4d} "
            f"A={s['A']}  yours {mine:9.3f} ms   production {prod:9.3f} ms   "
            f"fraction {frac:.4f}"
        )
        print(
            f"     fwd+bwd arithmetic: yours {mine_tf:6.1f} TFLOP/s "
            f"({100 * mine_tf / PEAK_TFLOPS:5.2f}% of dense bf16 peak)   "
            f"production {prod_tf:6.1f} TFLOP/s "
            f"({100 * prod_tf / PEAK_TFLOPS:5.2f}%)"
        )

    geo = statistics.geometric_mean([r["fraction_of_production"] for r in rows])
    print(f"published-shape geomean fraction of production: {geo:.4f}")
    print("(not your score: the graded shapes, scales and routing draws are hidden)")
    return emit({"rows": rows, "geomean": geo, "adoption": ad}, 0)


if __name__ == "__main__":
    raise SystemExit(main())
