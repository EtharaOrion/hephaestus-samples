#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml

Published-shape benchmark. This is a steering tool, not the grader: the shapes
here are not the graded shapes, and the number it prints is not your score.

It checks correctness BEFORE it times anything, because a timing number for a
wrong kernel is not a slow result you can improve on -- it is noise, and usually
a flattering one, since the cheapest way to be fast is to compute the wrong
thing. The check compares your output and all five gradients against
`reference.py`, at the bfloat16 tolerance `instruction.md` discloses. Nothing
hidden is revealed by it: the shapes are the published ones and the tolerance is
in the thresholds table.
"""

import argparse
import json
import statistics
import sys

import torch

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from kernel import gated_delta_rule  # noqa: E402
from reference import gated_delta_rule_ref  # noqa: E402
from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: E402

GRAD_NAMES = ("q", "k", "v", "g", "beta")
BF16_ATOL, BF16_RTOL = 2.0e-2, 3.125e-2

# Correctness is checked on THIS shape, not on the timed ones. `reference.py` is
# a sequential loop over T, so comparing against it at T=512 costs more than the
# whole benchmark and you would stop running it. Small and cheap means you run it
# on every edit, which is the point. The grader's first stage does the same thing
# for the same reason.
CHECK_SHAPE = {"B": 1, "T": 128, "H": 2, "K": 64, "V": 64}

# A SECOND preflight shape whose dimensions are NOT powers of two. Every dimension
# in CHECK_SHAPE above is a power of two, and Triton code that indexes with
# tl.arange(0, N) compiles only for power-of-two N -- so a kernel can pass the
# local check and then fail the graded set with a CompilationError the author never
# saw. Measured on this task: an attempt died at a non-power-of-two correctness
# shape after passing every local check it had.
#
# It is a CORRECTNESS shape only and is never timed, so it discloses nothing about
# the graded set beyond what instruction.md already states -- that non-power-of-two
# sequence lengths are checked.
CHECK_SHAPE_NPOT = {"B": 1, "T": 127, "H": 3, "K": 96, "V": 96}

SHAPES = {
    "p1": {
        "B": 1,
        "T": 512,
        "H": 4,
        "K": 128,
        "V": 128,
        "dtypes": [
            "bfloat16"
        ]
    },
    "p2": {
        "B": 2,
        "T": 512,
        "H": 8,
        "K": 128,
        "V": 128,
        "dtypes": [
            "bfloat16"
        ]
    }
}


def make(B, T, H, K, V, dtype, seed):
    """One draw, matching the shape of the graded distribution.

    q and k are L2-normalised along the head dimension. This is not cosmetic. The
    delta rule is a contraction only while `beta * ||k||^2` stays bounded, and
    unnormalised keys at this scale with K=128 diverge to inf -- that is the
    operator's own behaviour, not a bug in your kernel. Feeding it divergent
    input would have you debugging an explosion the grader never sees, and would
    time your kernel in a regime it is never graded in. Real callers normalise,
    and the production kernel carries `use_qk_l2norm_in_kernel` for this reason.

    The graded draw also picks a random scale from a hidden range. This one uses
    a fixed scale so a rerun is comparable to the previous run; the normalisation
    above is what makes that safe.
    """
    torch.manual_seed(seed)
    f = lambda *s: torch.randn(*s, device="cuda", dtype=torch.float32)  # noqa: E731
    qn = torch.nn.functional.normalize(f(B, T, H, K), dim=-1)
    kn = torch.nn.functional.normalize(f(B, T, H, K), dim=-1)
    g = (-torch.rand(B, T, H, device="cuda", dtype=torch.float32) * 0.5).requires_grad_()
    beta = (torch.rand(B, T, H, device="cuda", dtype=torch.float32) * 0.9 + 0.05).to(dtype).requires_grad_()
    return dict(q=qn.to(dtype).requires_grad_(), k=kn.to(dtype).requires_grad_(),
                v=f(B, T, H, V).to(dtype).requires_grad_(), g=g, beta=beta)


def step(fn, inp):
    for t in inp.values():
        if t.grad is not None:
            t.grad = None
    out = fn(**inp)
    out = out[0] if isinstance(out, tuple) else out
    out.sum().backward()
    return out


def grads_of(inp):
    return {n: (inp[n].grad.clone() if inp[n].grad is not None else None) for n in GRAD_NAMES}


def close(got, want):
    """Mirror of the grader's comparison, at the disclosed bfloat16 tolerance.

    Finiteness is tested separately and first. An inf or a nan does not fail a
    tolerance test in a way that reads clearly -- the subtraction propagates and
    every downstream number becomes nan -- so it is reported as what it is.
    """
    if got is None or want is None:
        return False, "missing tensor"
    if not torch.isfinite(got).all():
        n = int((~torch.isfinite(got)).sum())
        return False, f"NON-FINITE: {n} of {got.numel()} elements are inf or nan"
    d = (got.float() - want.float()).abs()
    lim = BF16_ATOL + BF16_RTOL * want.float().abs()
    bad = int((d > lim).sum())
    if bad:
        return False, f"{bad} of {got.numel()} elements outside tolerance, max abs {float(d.max()):.3e}"
    return True, "ok"


def check_determinism():
    """E1. Same input, three runs, bitwise identical. Paper Figure 2 stage 4.

    The grader runs this and `bench.py` did not, so a kernel with a race in a
    parallel reduction had no local instrument that would show it. Measured on
    this task: an attempt shipped after asserting "deterministic tiled reductions;
    no atomics" and the grader found four of five gradients differing across three
    runs on identical input. Nothing it could run would have told it otherwise.

    This is the candidate against ITSELF -- the reference is not involved -- so no
    tolerance applies. Either it repeats exactly or it does not.
    """
    s = CHECK_SHAPE
    inp = make(s["B"], s["T"], s["H"], s["K"], s["V"], torch.bfloat16, 0)
    runs = []
    for _ in range(3):
        out = step(gated_delta_rule, inp)
        runs.append((out.detach().clone(), grads_of(inp)))
    fails = []
    base_o, base_g = runs[0]
    for i, (o, g) in enumerate(runs[1:], start=2):
        if not torch.equal(o.detach(), base_o):
            fails.append(f"run {i}: out differs from run 1")
        for n in GRAD_NAMES:
            a, b = g.get(n), base_g.get(n)
            if a is None or b is None or not torch.equal(a, b):
                fails.append(f"run {i}: d{n} differs from run 1")
    return fails


def check_fp32_state():
    """E2. Every persisted recurrent-state buffer must be float32.

    The rubric that forbids narrowing the accumulator has now rejected two
    attempts on this task, the second costing a measured 0.167. Both kept the
    IN-REGISTER accumulator in float32 -- the visible half -- and both leaked the
    precision at the storage boundary, allocating the checkpoint buffer in the
    input dtype so every per-timestep state was rounded to bf16 on the way out and
    read back degraded in the backward pass.

    A static read catches exactly that encoding, which is the one that keeps
    happening, and says so in the terms the rubric uses.
    """
    import pathlib as _p
    import re as _re
    src = _p.Path(_HERE, "kernel.py").read_text()
    bad = []
    for m in _re.finditer(r"torch\.(?:empty|zeros|ones|empty_like|zeros_like)\s*\(([^)]*)\)", src):
        args = m.group(1)
        if "dtype" not in args:
            continue
        if _re.search(r"dtype\s*=\s*torch\.float32|dtype\s*=\s*tl\.float32", args):
            continue
        if _re.search(r"dtype\s*=\s*\w+\.dtype", args):
            bad.append(m.group(0)[:110])
    return bad


def _kernel_is_mine(key: str, names: set) -> bool:
    """Is this profiler entry a launch of a Triton kernel THIS module declares?

    Substring containment was the rule here, and it was exploitable. Measured
    2026-08-14 against real profiler keys: a module declaring `@triton.jit def
    gemm(...)` matched `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x3_nn`
    -- cuBLAS -- and one declaring `reduce` matched
    `at::native::reduce_kernel<512, 1, ReduceOp<float>>`. So a candidate could
    declare one trivial kernel, route the arithmetic through torch, and have the
    library's device time counted as its own. `gemm` and `reduce` are not
    adversarial names; they are what you would call those kernels anyway. The gate
    that exists to stop delegation was defeated by naming.

    Two conditions now, both required. The entry must not look like a vendor or
    ATen symbol, and the declared name must match the key exactly or as its
    leading component -- Triton appends specialisation suffixes after an
    underscore, it does not embed the name mid-string.
    """
    import re as _re
    if "::" in key or key.startswith("void ") or "<" in key:
        return False                      # C++/ATen mangling
    if _re.match(r"^(ampere|turing|volta|hopper|maxwell|pascal|sm\d+)_", key):
        return False                      # cuBLAS/cuDNN generated names
    if _re.search(r"(cutlass|cublas|cudnn|nvjet)", key, _re.I):
        return False
    return any(key == n or key.startswith(n + "_") for n in names)



# H100 SXM HBM3. A constant rather than a query because torch exposes no peak
# bandwidth; task.toml pins gpu_types = ["H100"], so this is the only device the
# task is graded on and the number is not a guess about your hardware.
PEAK_GBPS = 3350.0


def compulsory_bytes(B, T, H, K, V, itemsize):
    """HBM traffic this operator cannot avoid, forward plus backward.

    Every input read at least once and every output written at least once, on
    both passes. It is a LOWER bound on real traffic -- a kernel that spills
    state to HBM, or recomputes by re-reading, moves more -- so the achieved
    bandwidth derived from it is a lower bound too. That is the useful direction:
    if this already sits near peak, the kernel is memory-bound and no amount of
    tile tuning will help.
    """
    n = B * T * H
    fwd = n * (2 * K + V) * itemsize + n * 4 + n * itemsize   # q,k,v in; g fp32; beta
    fwd += n * V * itemsize                                   # o out
    bwd = n * V * itemsize                                    # do in
    bwd += n * (2 * K + V) * itemsize + n * 4 + n * itemsize  # q,k,v,g,beta re-read
    bwd += n * (2 * K + V) * itemsize + n * 4 + n * itemsize  # dq,dk,dv,dg,dbeta out
    return fwd + bwd


def check_adoption():
    """Share of device time in Triton kernels THIS module declares.

    The grader stops before timing anything when this is under the floor, and
    until now bench.py had no way to say so. Measured on the first graded
    iteration of this bundle: a correct kernel that declared `_bmm_kernel` and
    `_cast_kernel` and left the rest to torch ran 7.5% of its device time in its
    own kernels, scored zero on the written-kernel gate, and never learned it from
    any local instrument -- it ran bench.py twenty-six times and every one of them
    reported success. An instrument that stays silent about the thing that will
    reject you is worse than no instrument, because it is believed.
    """
    import ast as _ast
    import pathlib as _p
    from torch.profiler import ProfilerActivity, profile
    src = _p.Path(__file__).resolve().parent / "kernel.py"
    names = set()
    try:
        tree = _ast.parse(src.read_text())
    except (OSError, SyntaxError):
        return None
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef):
            for dec in node.decorator_list:
                d = dec.func if isinstance(dec, _ast.Call) else dec
                if _ast.unparse(d).replace(" ", "").endswith("triton.jit"):
                    names.add(node.name)
    if not names:
        return {"adoption": 0.0, "declared": [], "total_us": 0.0, "mine_us": 0.0}
    s = CHECK_SHAPE
    inp = make(s["B"], s["T"], s["H"], s["K"], s["V"], torch.bfloat16, 0)
    for _ in range(3):
        step(gated_delta_rule, inp)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step(gated_delta_rule, inp)
        torch.cuda.synchronize()
    total = mine = 0.0
    for e in prof.key_averages():
        dev = float(getattr(e, "self_device_time_total", 0) or 0)
        if dev <= 0:
            continue
        total += dev
        if _kernel_is_mine(e.key, names):
            mine += dev
    return {"adoption": (mine / total if total else 0.0), "declared": sorted(names),
            "total_us": round(total, 1), "mine_us": round(mine, 1)}


ADOPTION_FLOOR = 0.60


def check(shape=None):
    """Output plus all five gradients against reference.py. Returns a failure list.

    This runs once, before any timing. The grader's first stage does the same
    comparison on a shape you cannot see; if this passes and that one fails, the
    difference is the shape, which is a far smaller thing to hunt than a silent
    wrong answer. Passing here is necessary, not sufficient.
    """
    s = shape or CHECK_SHAPE
    inp = make(s["B"], s["T"], s["H"], s["K"], s["V"], torch.bfloat16, 0)
    fails = []
    try:
        got = step(gated_delta_rule, inp)
    except Exception as e:  # noqa: BLE001
        return [f"candidate raised {type(e).__name__}: {e}"]
    got_g = grads_of(inp)
    want = step(gated_delta_rule_ref, inp)
    want_g = grads_of(inp)
    ok, detail = close(got, want)
    if not ok:
        fails.append(f"out  {detail}")
    for n in GRAD_NAMES:
        ok, detail = close(got_g.get(n), want_g.get(n))
        if not ok:
            fails.append(f"d{n:<4} {detail}")
    return fails


def sample(fn, inp):
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record(); step(fn, inp); b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b)


def timed_pair(mine_fn, prod_fn, inp, warmup=3, trials=5, budget_s=30.0):
    """Measure the RATIO by interleaving, not by timing each side to completion.

    You are comparing two kernels on one GPU whose clocks and temperature move
    while you measure. Timing yours to completion and then production's puts the
    two measurements at different points on that curve, and the whole drift lands
    in the ratio you are trying to read.

    This is not a theoretical worry. The grader made the same mistake and, on the
    ORACLE - where the candidate IS the production kernel, so every ratio is
    known to be exactly 1.0 - one shape read 0.79, a 21 percent error against a
    known truth. Samples here are interleaved and the order alternates, so drift
    divides out and any first-versus-second effect changes sign each trial.

    So a difference this reports is much more likely to be your edit than the
    weather. Keep or revert on it accordingly.
    """
    import time as _t
    _s = _t.time(); step(mine_fn, inp); torch.cuda.synchronize(); one = _t.time() - _s
    if one * 2 * (warmup + trials) > budget_s:
        warmup, trials = 1, max(3, int(budget_s / (2 * one)) or 3)
    for _ in range(warmup):
        step(mine_fn, inp); step(prod_fn, inp)
    torch.cuda.synchronize()
    mine_xs, prod_xs = [], []
    for i in range(trials):
        if i % 2 == 0:
            m = sample(mine_fn, inp); p = sample(prod_fn, inp)
        else:
            p = sample(prod_fn, inp); m = sample(mine_fn, inp)
        mine_xs.append(m); prod_xs.append(p)
    return statistics.median(mine_xs), statistics.median(prod_xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()
    s = CHECK_SHAPE
    fails = check()
    if not fails:
        # non-power-of-two dimensions, correctness only, never timed
        n = CHECK_SHAPE_NPOT
        npot = check(n)
        if npot:
            s, fails = n, [f"(non-power-of-two shape) {f}" for f in npot]
    if fails:
        # Nothing is timed. A speed number for a wrong kernel is the single most
        # misleading thing this script could print: it looks like progress, it is
        # quotable, and reporting it is a rubric violation as well as a wrong
        # belief about your own code.
        print(f"CORRECTNESS FAILED at B={s['B']} T={s['T']} H={s['H']} "
              f"K={s['K']} V={s['V']} bfloat16")
        for f in fails:
            print(f"    {f}")
        print()
        print("Nothing was timed. Your forward pass can be right while your backward")
        print("is wrong -- all five gradients are compared, and the grader checks them")
        print("on shapes you cannot see before it times anything.")
        print("Fix correctness first. A fast wrong kernel scores zero, not partial credit.")
        if args.json:
            json.dump({"rows": [], "geomean": None, "correctness_failed": fails,
                       "check_shape": s}, open(args.json, "w"), indent=2)
        return 1
    print(f"correctness ok at B={s['B']} T={s['T']} H={s['H']} K={s['K']} V={s['V']} "
          f"bfloat16 (output and all five gradients)")

    det = check_determinism()
    if det:
        print()
        print("NOT DETERMINISTIC. The same input gave different answers across three runs:")
        for f in det[:12]:
            print(f"    {f}")
        print()
        print("Nothing was timed. A kernel that does not repeat cannot be measured -- the")
        print("speed number would belong to one of several answers, and gradients that")
        print("change between runs cannot be trained on. This is your kernel against")
        print("itself, so no tolerance applies. Look for a reduction whose order is not")
        print("fixed: atomics, or threads accumulating into shared memory without a")
        print("deterministic combine order.")
        if args.json:
            json.dump({"rows": [], "geomean": None, "nondeterministic": det,
                       "check_shape": s}, open(args.json, "w"), indent=2)
        return 1
    print("determinism ok: three runs on one input, bitwise identical")

    narrowed = check_fp32_state()
    if narrowed:
        print()
        print("RECURRENT STATE NARROWED BELOW FLOAT32:")
        for b in narrowed[:6]:
            print(f"    {b}")
        print()
        print("A persisted state or checkpoint buffer is allocated in the INPUT dtype, so")
        print("every value written to it is rounded to bfloat16 and read back degraded.")
        print("Keeping the in-register accumulator in float32 is not enough if the state")
        print("leaves the kernel through a narrower buffer. This is graded as narrowing")
        print("the accumulator and it scores zero however fast the kernel is.")
        if args.json:
            json.dump({"rows": [], "geomean": None, "narrowed_state": narrowed,
                       "check_shape": s}, open(args.json, "w"), indent=2)
        return 1
    print("recurrent state kept in float32")

    ad = check_adoption()
    if ad is not None:
        pct = 100.0 * ad["adoption"]
        print(f"written-kernel share: {pct:.1f}% of device time in your own "
              f"triton kernels ({', '.join(ad['declared']) or 'none declared'})")
        if ad["adoption"] < ADOPTION_FLOOR:
            print()
            print(f"BELOW THE WRITTEN-KERNEL FLOOR of {ADOPTION_FLOOR:.0%}.")
            print(f"{ad['mine_us']:.0f} us of {ad['total_us']:.0f} us ran in kernels this file")
            print("declares; the rest ran in framework operators. The grader measures this")
            print("the same way and stops BEFORE timing anything, so a kernel in this state")
            print("scores zero however fast it is. Move the arithmetic into your kernels.")
            if args.json:
                json.dump({"rows": [], "geomean": None, "adoption": ad,
                           "check_shape": s}, open(args.json, "w"), indent=2)
            return 1

    rows = []
    for label, s in SHAPES.items():
        inp = make(s["B"], s["T"], s["H"], s["K"], s["V"], torch.bfloat16, 0)
        mine, prod = timed_pair(
            gated_delta_rule,
            lambda **kw: chunk_gated_delta_rule(use_qk_l2norm_in_kernel=False, **kw), inp)
        # ROOFLINE, reported per shape. AutoKernel's design rationale (S10) is that
        # the benchmark reports bound-classification and percent of peak after every
        # experiment, and the agent picks its optimisation tier from that. Without
        # it a solver is choosing between tile tuning, coalescing and epilogue
        # fusion blind to which wall it is against. The production column is the
        # honest reference: it is the same operator on the same shape, so the gap
        # between the two bandwidths is the headroom that actually exists.
        nbytes = compulsory_bytes(s["B"], s["T"], s["H"], s["K"], s["V"], 2)
        mine_gbps = nbytes / (mine * 1e-3) / 1e9
        prod_gbps = nbytes / (prod * 1e-3) / 1e9
        rows.append({"shape": label, "yours_ms": round(mine, 5),
                     "production_ms": round(prod, 5),
                     "fraction_of_production": round(min(prod / mine, 1.0), 5),
                     "yours_gbps": round(mine_gbps, 1),
                     "production_gbps": round(prod_gbps, 1),
                     "yours_pct_peak": round(100 * mine_gbps / PEAK_GBPS, 2),
                     "production_pct_peak": round(100 * prod_gbps / PEAK_GBPS, 2)})
        print(f"{label:4s} yours {mine:9.4f} ms   production {prod:9.4f} ms   "
              f"fraction {min(prod / mine, 1.0):.4f}")
        print(f"     bandwidth: yours {mine_gbps:7.1f} GB/s ({100 * mine_gbps / PEAK_GBPS:5.2f}% of "
              f"HBM peak)   production {prod_gbps:7.1f} GB/s "
              f"({100 * prod_gbps / PEAK_GBPS:5.2f}%)")
        if prod_gbps > 0:
            print(f"     production moves the same bytes {prod_gbps / mine_gbps:.2f}x faster; "
                  f"{'both are far below peak, so this is not bandwidth-limited -- look at occupancy, launch count and compute' if max(mine_gbps, prod_gbps) < 0.35 * PEAK_GBPS else 'production is near the bandwidth wall, so the remaining gap is traffic you can avoid'}")

    geo = statistics.geometric_mean([r["fraction_of_production"] for r in rows])
    print(f"published-shape geomean fraction of production: {geo:.4f}")
    if args.json:
        json.dump({"rows": rows, "geomean": geo, "correctness_failed": [],
                   "check_shape": CHECK_SHAPE}, open(args.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
