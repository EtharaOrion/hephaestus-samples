#!/usr/bin/env python3
"""Published-shape benchmark for argmax_mixed_dtype. Steering tool, not the
grader: the shapes here are not the graded shapes, and the number it prints
is not your score.

It gates BEFORE it times: forbidden-name scan, exact correctness (across
all three dtypes plus a non-power-of-two and a duplicate-saturated shape),
determinism, written-kernel share, then the timed pairs against
torch.argmax. This file may call torch.argmax; your kernel.py may not.
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
from kernel import argmax_mixed_dtype  # noqa: E402
from reference import argmax_mixed_dtype_ref  # noqa: E402

CHECK_SHAPE = {"R": 16, "N": 8192}
CHECK_SHAPE_NPOT = {"R": 5, "N": 10007}
CHECK_SHAPE_TIES = {"R": 8, "N": 8192, "quantize": 16}

SHAPES = {
    "p1": {"R": 512, "N": 65536},
    "p2": {"R": 128, "N": 524288},
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
    "torch.argmax",
    ".argmax",
]

PEAK_GBPS = 3350.0
DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def make(R, N, seed, dtype, quantize=0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    if quantize:
        lev = torch.floor(torch.rand(R, N, generator=g, device="cuda") * quantize)
        x = (lev - quantize // 2) * 1.0
    else:
        x = torch.randn(R, N, generator=g, device="cuda")
    if dtype == torch.float16:
        x = torch.clamp(x, min=-32768.0, max=32768.0)
    return x.to(dtype).contiguous()


def scan_forbidden():
    src = open(os.path.join(_HERE, "kernel.py")).read()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    return [s for s in FORBIDDEN if re.search(r"\b" + re.escape(s) + r"\b", code)]


def exact_check(shape, dtype):
    x = make(shape["R"], shape["N"], 0, dtype, shape.get("quantize", 0))
    try:
        got = argmax_mixed_dtype(x)
    except Exception as e:  # noqa: BLE001
        return [f"candidate raised {type(e).__name__}: {e}"]
    want = argmax_mixed_dtype_ref(x)
    if not isinstance(got, torch.Tensor):
        return [f"output must be a tensor, got {type(got).__name__}"]
    fails = []
    if got.dtype != torch.int64:
        fails.append(f"indices dtype {got.dtype}, must be int64")
    elif tuple(got.shape) != tuple(want.shape):
        fails.append(f"indices shape {tuple(got.shape)} != {tuple(want.shape)}")
    elif not torch.equal(got, want):
        n = int((got != want).sum())
        fails.append(
            f"indices: {n} rows differ from the exact (value desc, index asc) argmax"
        )
    return fails


def check_determinism():
    x = make(
        CHECK_SHAPE_TIES["R"],
        CHECK_SHAPE_TIES["N"],
        0,
        torch.bfloat16,
        CHECK_SHAPE_TIES["quantize"],
    )
    runs = [argmax_mixed_dtype(x) for _ in range(4)][1:]
    fails = []
    for i, ix in enumerate(runs[1:], start=2):
        if not torch.equal(ix, runs[0]):
            fails.append(f"run {i}: indices differ from run 1")
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
    x = make(s["R"], s["N"], 0, torch.bfloat16)
    for _ in range(3):
        argmax_mixed_dtype(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        argmax_mixed_dtype(x)
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


def sample(fn, x):
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    fn(x)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b)


def timed_pair(mine_fn, prod_fn, x, warmup=3, trials=7, budget_s=30.0):
    import time

    t0 = time.time()
    mine_fn(x)
    torch.cuda.synchronize()
    one = time.time() - t0
    if one * 2 * (warmup + trials) > budget_s:
        warmup, trials = 1, max(3, int(budget_s / (2 * one)) or 3)
    for _ in range(warmup):
        mine_fn(x)
        prod_fn(x)
    torch.cuda.synchronize()
    ms, ps = [], []
    for i in range(trials):
        if i % 2 == 0:
            m = sample(mine_fn, x)
            p = sample(prod_fn, x)
        else:
            p = sample(prod_fn, x)
            m = sample(mine_fn, x)
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
        return emit({"rows": [], "geomean": None, "forbidden": hits}, 1)

    # Every CHECK shape run under all three dtypes.
    for label, s in (
        ("check", CHECK_SHAPE),
        ("non-power-of-two", CHECK_SHAPE_NPOT),
        ("duplicate-saturated", CHECK_SHAPE_TIES),
    ):
        for dtype in DTYPES:
            fails = exact_check(s, dtype)
            if fails:
                print(
                    f"CORRECTNESS FAILED on the {label} shape "
                    f"R={s['R']} N={s['N']} {str(dtype).split('.')[-1]}:"
                )
                for f in fails:
                    print(f"    {f}")
                return emit(
                    {
                        "rows": [],
                        "geomean": None,
                        "correctness_failed": fails,
                        "shape": s,
                        "dtype": str(dtype),
                    },
                    1,
                )
            print(
                f"correctness ok ({label}: R={s['R']} N={s['N']} "
                f"{str(dtype).split('.')[-1]}, exact)"
            )

    det = check_determinism()
    if det:
        print("NOT DETERMINISTIC on a duplicate-saturated input:")
        for f in det[:8]:
            print(f"    {f}")
        return emit({"rows": [], "geomean": None, "nondeterministic": det}, 1)
    print(
        "determinism ok: three runs on a duplicate-saturated input, bitwise identical"
    )

    ad = check_adoption()
    if ad is not None:
        pct = 100.0 * ad["adoption"]
        print(
            f"written-kernel share: {pct:.1f}% of device time in your own triton "
            f"kernels ({', '.join(ad['declared']) or 'none declared'})"
        )
        if ad["adoption"] < ADOPTION_FLOOR:
            print(f"BELOW THE WRITTEN-KERNEL FLOOR of {ADOPTION_FLOOR:.0%}.")
            return emit({"rows": [], "geomean": None, "adoption": ad}, 1)

    rows = []
    for label, s in SHAPES.items():
        R, N = s["R"], s["N"]
        x = make(R, N, 0, torch.bfloat16)
        mine, prod = timed_pair(
            argmax_mixed_dtype, lambda t: torch.argmax(t, dim=-1), x
        )
        nbytes = R * N * x.element_size() + R * 8
        mine_gbps = nbytes / (mine * 1e-3) / 1e9
        prod_gbps = nbytes / (prod * 1e-3) / 1e9
        rows.append(
            {
                "shape": label,
                "R": R,
                "N": N,
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
            f"{label:4s} R={R:5d} N={N:7d}  yours {mine:9.4f} ms   "
            f"production {prod:9.4f} ms   fraction {min(prod / mine, 1.0):.4f}"
        )
        print(
            f"     compulsory-traffic bandwidth: yours {mine_gbps:7.1f} GB/s "
            f"({100 * mine_gbps / PEAK_GBPS:5.2f}% of peak)   production "
            f"{prod_gbps:7.1f} GB/s ({100 * prod_gbps / PEAK_GBPS:5.2f}%)"
        )

    geo = statistics.geometric_mean([r["fraction_of_production"] for r in rows])
    print(f"published-shape geomean fraction of production: {geo:.4f}")
    print("(not your score: the graded shapes, dtypes and scales are hidden)")
    return emit({"rows": rows, "geomean": geo, "adoption": ad}, 0)


if __name__ == "__main__":
    raise SystemExit(main())
