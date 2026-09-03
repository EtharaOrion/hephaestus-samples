#!/usr/bin/env python3
"""Published-shape benchmark. This is a steering tool, not the grader: the
shapes here are not the graded shapes, and the number it prints is not your
score.

It gates BEFORE it times, in the grader's order, because a timing number for
a kernel that is about to score zero is the most misleading thing this
script could print: forbidden-name scan, exact correctness (including a
Mersenne-prime and a duplicate-saturated shape), determinism, then the
written-kernel share, and only then the timed pairs against `torch.topk`.
This file may call `torch.topk`; your `kernel.py` may not -- the scan below
is the same word-boundary rule the grader applies to your source.
"""

import argparse
import json
import re
import statistics
import sys

import torch

import os  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from kernel import topk_prime_row  # noqa: E402
from reference import topk_prime_row_ref  # noqa: E402

# All CHECK/SHAPES below use PRIME N -- the operator's precondition.
CHECK_SHAPE = {"R": 16, "N": 8191, "k": 32}  # Mersenne M13
CHECK_SHAPE_NPOT = {"R": 5, "N": 10007, "k": 23}  # prime
CHECK_SHAPE_TIES = {"R": 8, "N": 8191, "k": 96, "quantize": 16}  # prime + heavy ties

SHAPES = {
    "p1": {"R": 128, "N": 65537, "k": 128},  # Fermat prime F4
    "p2": {"R": 32, "N": 262147, "k": 512},  # prime > 2**18
}

FORBIDDEN = [
    "torch.topk",
    "torch.sort",
    "torch.argsort",
    "torch.msort",
    "torch.kthvalue",
    "torch.unique",
    "torch.median",
    "torch.nanmedian",
    "torch.quantile",
    "torch.nanquantile",
    ".topk",
    ".sort",
    ".argsort",
    ".msort",
    ".kthvalue",
    ".unique",
    ".median",
    ".quantile",
]

PEAK_GBPS = 3350.0


def make(R, N, k, seed, quantize=0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    if quantize:
        lev = torch.floor(torch.rand(R, N, generator=g, device="cuda") * quantize)
        x = (lev - quantize // 2) * 1.0
    else:
        x = torch.randn(R, N, generator=g, device="cuda")
    return x.to(torch.bfloat16).contiguous()


def scan_forbidden():
    src = open(os.path.join(_HERE, "kernel.py")).read()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    return [s for s in FORBIDDEN if re.search(r"\b" + re.escape(s) + r"\b", code)]


def exact_check(shape):
    x = make(shape["R"], shape["N"], shape["k"], 0, shape.get("quantize", 0))
    k = shape["k"]
    try:
        got = topk_prime_row(x, k)
    except Exception as e:  # noqa: BLE001
        return [f"candidate raised {type(e).__name__}: {e}"]
    want_v, want_i = topk_prime_row_ref(x, k)
    fails = []
    if not (isinstance(got, tuple) and len(got) == 2):
        return [f"output must be (values, indices), got {type(got).__name__}"]
    gv, gi = got
    if gi.dtype != torch.int64:
        fails.append(f"indices dtype {gi.dtype}, must be int64")
    elif not torch.equal(gi, want_i):
        n = int((gi != want_i).sum())
        fails.append(
            f"indices: {n} positions differ from the exact "
            f"(value desc, index asc) order"
        )
    if gv.dtype != x.dtype:
        fails.append(f"values dtype {gv.dtype}, must be {x.dtype}")
    elif not torch.equal(
        gv.contiguous().view(torch.int16), want_v.contiguous().view(torch.int16)
    ):
        n = int(
            (
                gv.contiguous().view(torch.int16)
                != want_v.contiguous().view(torch.int16)
            ).sum()
        )
        fails.append(f"values: {n} elements are not bit-exact gathers of the input")
    return fails


def check_determinism():
    s = CHECK_SHAPE_TIES
    x = make(s["R"], s["N"], s["k"], 0, s["quantize"])
    runs = [topk_prime_row(x, s["k"]) for _ in range(4)][1:]
    fails = []
    for i, (v, ix) in enumerate(runs[1:], start=2):
        if not torch.equal(ix, runs[0][1]):
            fails.append(f"run {i}: indices differ from run 1")
        if not torch.equal(v.view(torch.int16), runs[0][0].view(torch.int16)):
            fails.append(f"run {i}: values differ from run 1 at the bit level")
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
    s = SHAPES["p1"]
    x = make(s["R"], s["N"], s["k"], 0)
    for _ in range(3):
        topk_prime_row(x, s["k"])
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        topk_prime_row(x, s["k"])
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


def sample(fn, x, k):
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    fn(x, k)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b)


def timed_pair(mine_fn, prod_fn, x, k, warmup=3, trials=7, budget_s=30.0):
    import time

    t0 = time.time()
    mine_fn(x, k)
    torch.cuda.synchronize()
    one = time.time() - t0
    if one * 2 * (warmup + trials) > budget_s:
        warmup, trials = 1, max(3, int(budget_s / (2 * one)) or 3)
    for _ in range(warmup):
        mine_fn(x, k)
        prod_fn(x, k)
    torch.cuda.synchronize()
    ms, ps = [], []
    for i in range(trials):
        if i % 2 == 0:
            m = sample(mine_fn, x, k)
            p = sample(prod_fn, x, k)
        else:
            p = sample(prod_fn, x, k)
            m = sample(mine_fn, x, k)
        ms.append(m)
        ps.append(p)
    return statistics.median(ms), statistics.median(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    def emit(payload, rc):
        if args.json:
            json.dump(payload, open(args.json, "w"), indent=2)
        return rc

    hits = scan_forbidden()
    if hits:
        print("FORBIDDEN NAMES in kernel.py:", ", ".join(hits))
        print()
        print("The grader scans your comment-stripped source with the same rule and")
        print("scores zero at its first gate, before your code is even imported.")
        return emit({"rows": [], "geomean": None, "forbidden": hits}, 1)

    for label, s in (
        ("check (Mersenne M13)", CHECK_SHAPE),
        ("other prime", CHECK_SHAPE_NPOT),
        ("duplicate-saturated Mersenne", CHECK_SHAPE_TIES),
    ):
        fails = exact_check(s)
        if fails:
            print(
                f"CORRECTNESS FAILED on the {label} shape "
                f"R={s['R']} N={s['N']} k={s['k']} bfloat16:"
            )
            for f in fails:
                print(f"    {f}")
            print()
            print("Nothing was timed. Correctness here is EXACT -- indices by integer")
            print("equality, values bit-identical to the gathered input elements.")
            return emit(
                {"rows": [], "geomean": None, "correctness_failed": fails, "shape": s},
                1,
            )
        print(f"correctness ok ({label}: R={s['R']} N={s['N']} k={s['k']}, exact)")

    det = check_determinism()
    if det:
        print()
        print("NOT DETERMINISTIC on a duplicate-saturated Mersenne input:")
        for f in det[:8]:
            print(f"    {f}")
        return emit({"rows": [], "geomean": None, "nondeterministic": det}, 1)
    print(
        "determinism ok: three runs on a duplicate-saturated Mersenne input, bitwise identical"
    )

    ad = check_adoption()
    if ad is not None:
        pct = 100.0 * ad["adoption"]
        print(
            f"written-kernel share: {pct:.1f}% of device time in your own triton "
            f"kernels ({', '.join(ad['declared']) or 'none declared'})"
        )
        if ad["adoption"] < ADOPTION_FLOOR:
            print()
            print(f"BELOW THE WRITTEN-KERNEL FLOOR of {ADOPTION_FLOOR:.0%}.")
            return emit({"rows": [], "geomean": None, "adoption": ad}, 1)

    rows = []
    for label, s in SHAPES.items():
        R, N, k = s["R"], s["N"], s["k"]
        x = make(R, N, k, 0)
        mine, prod = timed_pair(
            topk_prime_row,
            lambda t, kk: torch.topk(t, kk, dim=-1, largest=True, sorted=True),
            x,
            k,
        )
        nbytes = R * N * x.element_size() + R * k * (x.element_size() + 8)
        mine_gbps = nbytes / (mine * 1e-3) / 1e9
        prod_gbps = nbytes / (prod * 1e-3) / 1e9
        rows.append(
            {
                "shape": label,
                "R": R,
                "N": N,
                "k": k,
                "yours_ms": round(mine, 5),
                "production_ms": round(prod, 5),
                "fraction_of_production": round(min(prod / mine, 1.0), 5),
                "yours_gbps": round(mine_gbps, 1),
                "production_gbps": round(prod_gbps, 1),
                "yours_pct_peak": round(100 * mine_gbps / PEAK_GBPS, 2),
                "production_pct_peak": round(100 * prod_gbps / PEAK_GBPS, 2),
            }
        )
        print(
            f"{label:4s} R={R:5d} N={N:7d} k={k:4d}  yours {mine:9.4f} ms   "
            f"production {prod:9.4f} ms   fraction {min(prod / mine, 1.0):.4f}"
        )
        print(
            f"     compulsory-traffic bandwidth: yours {mine_gbps:7.1f} GB/s "
            f"({100 * mine_gbps / PEAK_GBPS:5.2f}% of peak)   production "
            f"{prod_gbps:7.1f} GB/s ({100 * prod_gbps / PEAK_GBPS:5.2f}%)"
        )

    geo = statistics.geometric_mean([r["fraction_of_production"] for r in rows])
    print(f"published-shape geomean fraction of production: {geo:.4f}")
    print("(not your score: the graded shapes, scales and distributions are hidden)")
    return emit({"rows": rows, "geomean": geo, "adoption": ad}, 0)


if __name__ == "__main__":
    raise SystemExit(main())
