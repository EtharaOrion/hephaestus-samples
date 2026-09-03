#!/usr/bin/env python3
"""Published-shape benchmark for topk_segmented. Steering tool, not the grader:
the shapes here are not the graded shapes, and the number it prints is not
your score.

It gates BEFORE it times, in the grader's order: forbidden-name scan, exact
correctness (including a non-power-of-two-segments and a duplicate-saturated
shape, in both graded dtypes), determinism, then the written-kernel share, and
only then the timed pairs against the [R, G, S] segment-view torch.topk anchor.
This file may call torch.topk; your kernel.py may not.
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
from kernel import topk_segmented  # noqa: E402
from reference import topk_segmented_ref  # noqa: E402

CHECK_SHAPE = {"R": 16, "N": 8192, "k": 8, "segment": 128}
CHECK_SHAPE_NPOT = {"R": 5, "N": 10281, "k": 7, "segment": 3427}  # 10281 = 3 * 3427
CHECK_SHAPE_TIES = {"R": 8, "N": 8192, "k": 24, "segment": 256, "quantize": 16}

SHAPES = {
    "g1": {
        "R": 128,
        "N": 16384,
        "k": 32,
        "segment": 512,
        "dtypes": ["float32", "bfloat16"],
    },
    "g2": {
        "R": 256,
        "N": 65536,
        "k": 64,
        "segment": 1024,
        "dtypes": ["float32", "bfloat16"],
    },
    "g3": {
        "R": 64,
        "N": 262144,
        "k": 16,
        "segment": 256,
        "dtypes": ["float32", "bfloat16"],
    },
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

_BITVIEW = {torch.float32: torch.int32, torch.bfloat16: torch.int16}
_DTYPE_BY_NAME = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def make(shape, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    R, N = int(shape["R"]), int(shape["N"])
    k = int(shape["k"])
    S = int(shape["segment"])
    if N % S != 0:
        raise ValueError(f"segment={S} must divide N={N}")
    if not (1 <= k <= S):
        raise ValueError(f"k={k} out of range for segment length S={S}")
    if shape.get("quantize"):
        q = int(shape["quantize"])
        lev = torch.floor(torch.rand(R, N, generator=g, device="cuda") * q)
        x = (lev - q // 2) * 1.0
    else:
        x = torch.randn(R, N, generator=g, device="cuda")
    return x.to(dtype).contiguous(), k, S


def scan_forbidden():
    src = open(os.path.join(_HERE, "kernel.py")).read()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    return [s for s in FORBIDDEN if re.search(r"\b" + re.escape(s) + r"\b", code)]


def exact_check(shape, dtype=torch.bfloat16):
    x, k, S = make(shape, 0, dtype)
    try:
        got = topk_segmented(x, k, S)
    except Exception as e:  # noqa: BLE001
        return [f"candidate raised {type(e).__name__}: {e}"]
    want_v, want_i = topk_segmented_ref(x, k, S)
    if not (isinstance(got, tuple) and len(got) == 2):
        return [f"output must be (values, indices), got {type(got).__name__}"]
    gv, gi = got
    fails = []
    if gi.dtype != torch.int64:
        fails.append(f"indices dtype {gi.dtype}, must be int64")
    elif not torch.equal(gi, want_i):
        n = int((gi != want_i).sum())
        segs = int((gi != want_i).any(dim=-1).sum())
        fails.append(
            f"global indices: {n} positions across {segs} segments differ from "
            f"the exact per-segment (value desc, index asc) order"
        )
    if gv.dtype != x.dtype:
        fails.append(f"values dtype {gv.dtype}, must be {x.dtype}")
    elif not torch.equal(
        gv.contiguous().view(_BITVIEW[x.dtype]),
        want_v.contiguous().view(_BITVIEW[x.dtype]),
    ):
        n = int(
            (
                gv.contiguous().view(_BITVIEW[x.dtype])
                != want_v.contiguous().view(_BITVIEW[x.dtype])
            ).sum()
        )
        fails.append(f"values: {n} elements are not bit-exact gathers of the input")
    return fails


def check_determinism():
    x, k, S = make(CHECK_SHAPE_TIES, 0)
    runs = [topk_segmented(x, k, S) for _ in range(4)][1:]
    fails = []
    bv = _BITVIEW[runs[0][0].dtype]
    for i, (v, ix) in enumerate(runs[1:], start=2):
        if not torch.equal(ix, runs[0][1]):
            fails.append(f"run {i}: indices differ from run 1")
        if not torch.equal(v.view(bv), runs[0][0].view(bv)):
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
    x, k, S = make(SHAPES["g1"], 0)
    for _ in range(3):
        topk_segmented(x, k, S)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        topk_segmented(x, k, S)
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


def _seg_topk_anchor(x, k, segment):
    S = int(segment)
    R, N = x.shape
    G = N // S
    xr = x.view(R, G, S)
    v, i_local = torch.topk(xr, k, dim=-1, largest=True, sorted=True)
    seg_off = (torch.arange(G, device=x.device, dtype=torch.int64) * S).view(1, G, 1)
    return v.contiguous(), (i_local.to(torch.int64) + seg_off).contiguous()


def sample(fn, x, k, S):
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    fn(x, k, S)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b)


def timed_pair(mine_fn, prod_fn, x, k, S, warmup=3, trials=7, budget_s=30.0):
    import time

    t0 = time.time()
    mine_fn(x, k, S)
    torch.cuda.synchronize()
    one = time.time() - t0
    if one * 2 * (warmup + trials) > budget_s:
        warmup, trials = 1, max(3, int(budget_s / (2 * one)) or 3)
    for _ in range(warmup):
        mine_fn(x, k, S)
        prod_fn(x, k, S)
    torch.cuda.synchronize()
    ms, ps = [], []
    for i in range(trials):
        if i % 2 == 0:
            m = sample(mine_fn, x, k, S)
            p = sample(prod_fn, x, k, S)
        else:
            p = sample(prod_fn, x, k, S)
            m = sample(mine_fn, x, k, S)
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

    for label, s in (
        ("check", CHECK_SHAPE),
        ("non-power-of-two segments", CHECK_SHAPE_NPOT),
        ("duplicate-saturated", CHECK_SHAPE_TIES),
    ):
        for dt_name in ("float32", "bfloat16"):
            dt = _DTYPE_BY_NAME[dt_name]
            fails = exact_check(s, dt)
            if fails:
                print(
                    f"CORRECTNESS FAILED on the {label} shape "
                    f"R={s['R']} N={s['N']} k={s['k']} segment={s['segment']} "
                    f"{dt_name}:"
                )
                for f in fails:
                    print(f"    {f}")
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
                f"correctness ok ({label}: R={s['R']} N={s['N']} "
                f"k={s['k']} segment={s['segment']} {dt_name}, exact)"
            )

    det = check_determinism()
    if det:
        print()
        print("NOT DETERMINISTIC on a duplicate-saturated segmented input:")
        for f in det[:8]:
            print(f"    {f}")
        return emit({"rows": [], "geomean": None, "nondeterministic": det}, 1)
    print(
        "determinism ok: three runs on a duplicate-saturated segmented input, "
        "bitwise identical"
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
        for dt_name in s.get("dtypes", ["bfloat16"]):
            dt = _DTYPE_BY_NAME[dt_name]
            x, k, S = make(s, 0, dt)
            R, N = x.shape
            G = N // S
            mine, prod = timed_pair(topk_segmented, _seg_topk_anchor, x, k, S)
            elt = x.element_size()
            nbytes = R * N * elt + R * G * k * (elt + 8)
            mine_gbps = nbytes / (mine * 1e-3) / 1e9
            prod_gbps = nbytes / (prod * 1e-3) / 1e9
            rows.append(
                {
                    "shape": label,
                    "dtype": dt_name,
                    "R": R,
                    "N": N,
                    "segment": S,
                    "G": G,
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
                f"{label:4s} {dt_name:8s} R={R:4d} N={N:7d} S={S:5d} G={G:5d} "
                f"k={k:3d}  yours {mine:9.4f} ms   production {prod:9.4f} ms   "
                f"fraction {min(prod / mine, 1.0):.4f}"
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
