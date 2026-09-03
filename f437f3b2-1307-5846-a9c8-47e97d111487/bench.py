#!/usr/bin/env python3
"""Published-shape benchmark. This is a steering tool, not the grader: the
shapes here are not the graded shapes, and the number it prints is not your
score.

It gates BEFORE it times, in the grader's order, because a timing number for
a kernel that is about to score zero is the most misleading thing this script
could print: forbidden-name scan, correctness against the reference on a
small shape, a non-power-of-two shape and a tie-forcing shape (outputs AND
all four graded gradients), bitwise determinism, then the written-kernel
share, and only then the timed forward+backward pairs against the production
composition. This file may call `torch._grouped_mm`; your `kernel.py` may
not -- the scan below is the same word-boundary rule the grader applies to
your source.
"""

import argparse
import json
import re
import statistics
import sys

import torch

import os  # noqa: E402
_HERE = os.path.dirname(os.path.abspath(__file__))  # bundle dir in the container, or the agent workspace on host
sys.path.insert(0, _HERE)
from kernel import fused_moe  # noqa: E402
from reference import fused_moe_ref  # noqa: E402

# Correctness is checked on these shapes, not the timed ones: the reference
# is an expert-by-expert loop, so keeping the check shapes small means you
# run the check on every edit, which is the point.
CHECK_SHAPE = {"T": 512, "D": 512, "F": 256, "E": 16, "A": 4}
# Non-power-of-two everything. Triton code that assumes power-of-two D, F or
# E can pass the check above and die on the graded set with a compile or
# addressing error you never saw locally.
CHECK_SHAPE_NPOT = {"T": 301, "D": 200, "F": 96, "E": 11, "A": 3}
# Router columns drawn from a 4-column pool and a two-level bias: biased
# scores carry massive EXACT ties in every dtype, so the lower-index
# tie-break and scheduling-independent selection are actually exercised.
# Distinct-score draws test neither.
CHECK_SHAPE_TIES = {"T": 256, "D": 256, "F": 128, "E": 16, "A": 4,
                    "tie_pool": 4}

SHAPES = {
    "p1": {"T": 4096, "D": 1024, "F": 704, "E": 32, "A": 4},
    "p2": {"T": 2048, "D": 2048, "F": 1408, "E": 64, "A": 8},
}

# The grader's G1 scan, applied locally so it can never be the thing you
# learn about only from a zero: word-boundary, comments stripped, strings
# and docstrings count.
FORBIDDEN = [
    "torch._grouped_mm", "_grouped_mm",
    "vllm", "sglang", "grouped_gemm", "transformer_engine", "flashinfer",
]

# Disclosed tolerances (bfloat16 -- this script checks the bf16 surface;
# the grader also checks float32 at its disclosed budgets).
OUT_ATOL = OUT_RTOL = 4.5e-2
GRAD_ATOL = GRAD_RTOL = 4.5e-2

# H100 SXM peak dense tensor throughput for bfloat16 inputs. A constant
# rather than a query because torch exposes no peak; this task is graded on
# H100 only, so it is not a guess.
PEAK_TFLOPS_BF16 = 989.5


def _zipfish_bias(E, g):
    ranks = torch.empty(E, dtype=torch.float32, device="cuda")
    ranks[torch.randperm(E, generator=g, device="cuda")] = torch.arange(
        E, dtype=torch.float32, device="cuda")
    logpop = -1.1 * torch.log1p(ranks)
    return 0.6 * (logpop - logpop.mean())


def make(shape, seed):
    """Fixed seed, fixed unit scale, margin-enforced -- so a rerun is
    comparable to the last run. The graded draws also randomize the input
    magnitude and the expert-popularity prior from hidden ranges;
    instruction.md discloses that, and a kernel whose policy is uniform (as
    required) cannot tell the difference. The margin loop below mirrors the
    grader: rows whose selection boundary gap is positive but below 1e-4
    are redrawn, so agreement is never decided by float noise; exact ties
    (the tie_pool shape) are kept and decided by the tie-break rule."""
    T, D = shape["T"], shape["D"]
    F, E, A = shape["F"], shape["E"], shape["A"]
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    pool = shape.get("tie_pool", 0)
    if pool:
        cols = torch.randn(D, pool, generator=g, device="cuda") * D ** -0.5
        assign = torch.randint(0, pool, (E,), generator=g, device="cuda")
        rw = cols.index_select(1, assign).contiguous()
        rb = (torch.randint(0, 2, (E,), generator=g, device="cuda")
              .float() * 0.25)
    else:
        rw = torch.randn(D, E, generator=g, device="cuda") * D ** -0.5
        rb = _zipfish_bias(E, g)
    w1 = torch.randn(E, D, 2 * F, generator=g, device="cuda") * D ** -0.5
    w2 = torch.randn(E, F, D, generator=g, device="cuda") * F ** -0.5
    x = torch.randn(T, D, generator=g, device="cuda")
    rwq = rw.to(torch.bfloat16)
    xq = x.to(torch.bfloat16)
    if A < E:
        rw64 = rwq.double()
        logits = xq.double() @ rw64
        for _ in range(64):
            biased = torch.sigmoid(logits) + rb.double()
            v = torch.topk(biased, A + 1, dim=-1).values
            gap = v[:, A - 1] - v[:, A]
            bad = (gap > 0) & (gap < 1e-4)
            if not int(bad.sum()):
                break
            fresh = torch.randn(int(bad.sum()), D, generator=g, device="cuda")
            xq[bad] = fresh.to(torch.bfloat16)
            logits[bad] = xq[bad].double() @ rw64
    return {"x": xq.contiguous().requires_grad_(True),
            "router_w": rwq.contiguous().requires_grad_(True),
            "router_b": rb.float().contiguous(),
            "w1": w1.to(torch.bfloat16).contiguous().requires_grad_(True),
            "w2": w2.to(torch.bfloat16).contiguous().requires_grad_(True),
            "A": A}


def cotangent(shape_td):
    """Same convention as the grader: a fixed pseudorandom +-uniform
    cotangent per output shape, so the backward is graded like a training
    step, not like an all-ones reduction."""
    g = torch.Generator(device="cuda")
    g.manual_seed((0x5EEDC0DE ^ (shape_td[0] * 31 + shape_td[1])) & 0x7FFFFFFF)
    return (torch.rand(shape_td, generator=g, device="cuda") * 2.0
            - 1.0).to(torch.bfloat16)


GRADS = ("x", "router_w", "w1", "w2")


def run_fwdbwd(fn, args):
    for n in GRADS:
        if args[n].grad is not None:
            args[n].grad = None
    y = fn(args["x"], args["router_w"], args["router_b"],
           args["w1"], args["w2"], args["A"])
    v = cotangent(tuple(y.shape))
    (y.float() * v.float()).sum().backward()
    return y, {n: args[n].grad for n in GRADS}


def scan_forbidden():
    src = open(os.path.join(_HERE, "kernel.py")).read()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    return [s for s in FORBIDDEN if re.search(r"\b" + re.escape(s) + r"\b", code)]


def _close(got, want, atol, rtol, tag):
    if got is None:
        return [f"{tag}: missing"]
    if not torch.isfinite(got.float()).all():
        return [f"{tag}: non-finite values"]
    d = (got.float() - want.float()).abs()
    lim = atol + rtol * want.float().abs()
    bad = int((d > lim).sum())
    if bad:
        return [f"{tag}: {bad} elements outside tolerance, "
                f"max abs {float(d.max()):.3e}"]
    return []


def exact_check(shape, label):
    args = make(shape, 0)
    try:
        y, g = run_fwdbwd(fused_moe, args)
    except Exception as e:  # noqa: BLE001
        return [f"candidate raised {type(e).__name__}: {e}"]
    gc = {n: g[n].detach().clone() for n in GRADS}
    yr, gr = run_fwdbwd(
        lambda x, rw, rb, w1, w2, A: fused_moe_ref(x, rw, rb, w1, w2, A), args)
    fails = []
    if y.dtype != args["x"].dtype:
        fails.append(f"output dtype {y.dtype}, must be {args['x'].dtype}")
    fails += _close(y, yr, OUT_ATOL, OUT_RTOL, "y")
    for n in GRADS:
        fails += _close(gc[n], gr[n], GRAD_ATOL, GRAD_RTOL, f"d{n}")
    return fails


def check_determinism():
    """Three runs on the tie-forcing input, bitwise identical outputs AND
    gradients. The candidate against ITSELF -- no tolerance applies, and
    the exact score ties mean a scheduling-dependent tie winner or a racy
    float combine cannot hide."""
    args = make(CHECK_SHAPE_TIES, 0)
    run_fwdbwd(fused_moe, args)  # warmup (autotune must not read as a race)
    runs = []
    for _ in range(3):
        y, g = run_fwdbwd(fused_moe, args)
        runs.append((y.detach().clone(),
                     {n: g[n].detach().clone() for n in GRADS}))
    fails = []
    for i, (y, g) in enumerate(runs[1:], start=2):
        if not torch.equal(y.view(torch.int16), runs[0][0].view(torch.int16)):
            fails.append(f"run {i}: output differs from run 1 at the bit level")
        for n in GRADS:
            if not torch.equal(g[n], runs[0][1][n]):
                fails.append(f"run {i}: d{n} differs from run 1")
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
    zeroes however fast the composition is."""
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
    args = make(SHAPES["p1"], 0)
    for _ in range(3):
        run_fwdbwd(fused_moe, args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run_fwdbwd(fused_moe, args)
        torch.cuda.synchronize()
    total = mine = 0.0
    for e in prof.key_averages():
        dev = float(getattr(e, "self_device_time_total", 0) or 0)
        if dev <= 0:
            continue
        total += dev
        if _kernel_is_mine(e.key, names):
            mine += dev
    return {"adoption": (mine / total if total else 0.0),
            "declared": sorted(names),
            "total_us": round(total, 1), "mine_us": round(mine, 1)}


ADOPTION_FLOOR = 0.60


def production(x, rw, rb, w1, w2, A):
    """The anchor composition the grader times against: router matmul plus
    integer-key top-A in torch, then a torch._grouped_mm expert pipeline
    over tokens sorted by expert."""
    T, D = x.shape
    E = rw.shape[1]
    F = w1.shape[2] // 2
    s = torch.sigmoid(x.float() @ rw.float())
    with torch.no_grad():
        b = s.detach() + rb
        b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
        ib = b.view(torch.int32).to(torch.int64)
        u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
        e_idx = torch.arange(E, device=x.device, dtype=torch.int64)
        sel = torch.topk((u << 10) | (E - 1 - e_idx), A, dim=-1).indices
    s_sel = torch.gather(s, 1, sel)
    w = s_sel / s_sel.sum(dim=-1, keepdim=True)
    flat = sel.reshape(-1)
    order = torch.argsort(flat)
    tok = torch.div(order, A, rounding_mode="floor")
    offs = torch.bincount(flat, minlength=E).cumsum(0).to(torch.int32)
    xg = x.index_select(0, tok)
    h = torch._grouped_mm(xg, w1, offs=offs)
    act = torch.nn.functional.silu(h[:, :F]) * h[:, F:]
    yp = torch._grouped_mm(act, w2, offs=offs)
    ws = w.reshape(-1).index_select(0, order).to(x.dtype).unsqueeze(1)
    return x.new_zeros(T, D).index_add(0, tok, yp * ws)


def sample(fn, args):
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    run_fwdbwd(fn, args)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b)


def timed_pair(mine_fn, prod_fn, args, warmup=3, trials=7, budget_s=40.0):
    """The graded quantity is a ratio of forward+backward times, so it is
    measured as a ratio: samples interleaved so clock and thermal drift
    divide out, order alternating so any first-versus-second effect changes
    sign each trial. The grader measures the same way (and adds an L2 flush
    and hidden scales)."""
    import time
    t0 = time.time()
    run_fwdbwd(mine_fn, args)
    torch.cuda.synchronize()
    one = time.time() - t0
    if one * 2 * (warmup + trials) > budget_s:
        warmup, trials = 1, max(3, int(budget_s / (2 * one)) or 3)
    for _ in range(warmup):
        run_fwdbwd(mine_fn, args)
        run_fwdbwd(prod_fn, args)
    torch.cuda.synchronize()
    ms, ps = [], []
    for i in range(trials):
        if i % 2 == 0:
            m = sample(mine_fn, args)
            p = sample(prod_fn, args)
        else:
            p = sample(prod_fn, args)
            m = sample(mine_fn, args)
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

    for label, s in (("check", CHECK_SHAPE),
                     ("non-power-of-two", CHECK_SHAPE_NPOT),
                     ("tie-forcing", CHECK_SHAPE_TIES)):
        fails = exact_check(s, label)
        if fails:
            print(f"CORRECTNESS FAILED on the {label} shape "
                  f"T={s['T']} D={s['D']} F={s['F']} E={s['E']} A={s['A']} bfloat16:")
            for f in fails[:8]:
                print(f"    {f}")
            print()
            print("Nothing was timed. The output and ALL FOUR gradients (x, router_w,")
            print("w1, w2) are compared against reference.py under the disclosed")
            print("tolerances. Selection is graded through the output: a wrong expert")
            print("set or a wrong tie winner moves whole rows far outside tolerance.")
            return emit({"rows": [], "geomean": None,
                         "correctness_failed": fails, "shape": s}, 1)
        print(f"correctness ok ({label}: T={s['T']} D={s['D']} F={s['F']} "
              f"E={s['E']} A={s['A']}, output + 4 gradients)")

    det = check_determinism()
    if det:
        print()
        print("NOT DETERMINISTIC on the tie-forcing input:")
        for f in det[:8]:
            print(f"    {f}")
        print()
        print("Nothing was timed. With exact score ties, a selection or combine that")
        print("depends on scheduling order -- an atomic float add, a racy tie winner")
        print("-- changes between runs. The grader runs the same three-run bitwise")
        print("check on outputs AND gradients, and it gates. This is your kernel")
        print("against itself, so no tolerance applies.")
        return emit({"rows": [], "geomean": None, "nondeterministic": det}, 1)
    print("determinism ok: three forward+backward runs on the tie-forcing input, "
          "bitwise identical")

    ad = check_adoption()
    if ad is not None:
        pct = 100.0 * ad["adoption"]
        print(f"written-kernel share: {pct:.1f}% of forward+backward device time "
              f"in your own triton kernels")
        if ad["adoption"] < ADOPTION_FLOOR:
            print()
            print(f"BELOW THE WRITTEN-KERNEL FLOOR of {ADOPTION_FLOOR:.0%}.")
            print(f"{ad['mine_us']:.0f} us of {ad['total_us']:.0f} us ran in kernels this")
            print("file declares; the rest ran in framework operators. The grader")
            print("attributes device time the same way and scores zero however fast")
            print("the composition is. The permitted torch routing calls are cheap;")
            print("move the expert compute and combine into your kernels.")
            return emit({"rows": [], "geomean": None, "adoption": ad}, 1)

    rows = []
    for label, s in SHAPES.items():
        args = make(s, 0)
        mine, prod = timed_pair(fused_moe, production, args)
        # fwd+bwd flop model: 6*T*A*D*F forward, ~2x that backward. Against
        # peak, this says which wall you are on: production near a large
        # fraction of peak means the remaining gap is dispatch overhead and
        # memory traffic you can fuse away; both far below peak means launch
        # count or occupancy, not arithmetic.
        tflop = 18.0 * s["T"] * s["A"] * s["D"] * s["F"] / 1e12
        mine_tf = tflop / (mine * 1e-3)
        prod_tf = tflop / (prod * 1e-3)
        rows.append({"shape": label, **{k: s[k] for k in ("T", "D", "F", "E", "A")},
                     "yours_ms": round(mine, 4), "production_ms": round(prod, 4),
                     "fraction_of_production": round(min(prod / mine, 1.0), 5),
                     "yours_tflops": round(mine_tf, 1),
                     "production_tflops": round(prod_tf, 1),
                     "yours_pct_peak": round(100 * mine_tf / PEAK_TFLOPS_BF16, 2),
                     "production_pct_peak": round(100 * prod_tf / PEAK_TFLOPS_BF16, 2)})
        print(f"{label}  T={s['T']:5d} E={s['E']:3d} A={s['A']}  "
              f"yours {mine:9.3f} ms   production {prod:9.3f} ms   "
              f"fraction {min(prod / mine, 1.0):.4f}")
        print(f"     model-flop throughput: yours {mine_tf:6.1f} TF/s "
              f"({100 * mine_tf / PEAK_TFLOPS_BF16:5.2f}% of bf16 peak)   "
              f"production {prod_tf:6.1f} TF/s "
              f"({100 * prod_tf / PEAK_TFLOPS_BF16:5.2f}%)")

    geo = statistics.geometric_mean([r["fraction_of_production"] for r in rows])
    print(f"published-shape geomean fraction of production: {geo:.4f}")
    print("(not your score: the graded shapes, scales, popularity priors and both")
    print(" dtypes are hidden; float32 rows are graded too)")
    return emit({"rows": rows, "geomean": geo, "adoption": ad}, 0)


if __name__ == "__main__":
    raise SystemExit(main())
