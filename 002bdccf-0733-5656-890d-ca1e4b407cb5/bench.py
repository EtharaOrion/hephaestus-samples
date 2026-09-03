#!/usr/bin/env python3
"""Published-shape benchmark for topk_ragged. Steering tool, not the grader:
the shapes here are not the graded shapes, and the number it prints is not
your score.

It gates BEFORE it times, in the grader's order: forbidden-name scan, exact
correctness (including a non-power-of-two-lengths and a duplicate-saturated
shape), determinism, then the written-kernel share, and only then the
timed pairs against the pad-to-max + torch.topk anchor. This file may call
torch.topk; your kernel.py may not.
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
from kernel import topk_ragged  # noqa: E402
from reference import topk_ragged_ref  # noqa: E402

CHECK_SHAPE = {"R": 16, "Lmin": 128, "Lmax": 512, "k": 16}
CHECK_SHAPE_NPOT = {"R": 5, "Lmin": 251, "Lmax": 2003, "k": 23}
CHECK_SHAPE_TIES = {"R": 8, "Lmin": 256, "Lmax": 1024, "k": 32, "quantize": 16}

SHAPES = {
    "p1": {"R": 128, "Lmin": 256, "Lmax": 2048, "k": 32},
    "p2": {"R": 64, "Lmin": 1024, "Lmax": 8192, "k": 64},
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


def make(shape, seed=0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    R = int(shape["R"])
    Lmin, Lmax = int(shape["Lmin"]), int(shape["Lmax"])
    k = int(shape["k"])
    lens = torch.randint(
        Lmin, Lmax + 1, (R,), generator=g, device="cuda", dtype=torch.int64
    )
    seg = torch.zeros(R + 1, dtype=torch.int64, device="cuda")
    seg[1:] = torch.cumsum(lens, dim=0)
    NNZ = int(seg[-1].item())
    if shape.get("quantize"):
        q = int(shape["quantize"])
        lev = torch.floor(torch.rand(NNZ, generator=g, device="cuda") * q)
        x = (lev - q // 2) * 1.0
    else:
        x = torch.randn(NNZ, generator=g, device="cuda")
    return x.to(torch.bfloat16).contiguous(), seg, k


def scan_forbidden():
    src = open(os.path.join(_HERE, "kernel.py")).read()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    return [s for s in FORBIDDEN if re.search(r"\b" + re.escape(s) + r"\b", code)]


def exact_check(shape):
    x, seg, k = make(shape, 0)
    try:
        got = topk_ragged(x, seg, k)
    except Exception as e:  # noqa: BLE001
        return [f"candidate raised {type(e).__name__}: {e}"]
    want_v, want_i = topk_ragged_ref(x, seg, k)
    if not (isinstance(got, tuple) and len(got) == 2):
        return [f"output must be (values, indices), got {type(got).__name__}"]
    gv, gi = got
    fails = []
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
    x, seg, k = make(CHECK_SHAPE_TIES, 0)
    runs = [topk_ragged(x, seg, k) for _ in range(4)][1:]
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
    x, seg, k = make(SHAPES["p1"], 0)
    for _ in range(3):
        topk_ragged(x, seg, k)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        topk_ragged(x, seg, k)
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


def _pad_topk_anchor(x, seg, k):
    R = int(seg.shape[0]) - 1
    lens = seg[1:] - seg[:-1]
    Lmax = int(lens.max().item())
    padded = torch.full((R, Lmax), float("-inf"), device=x.device, dtype=x.dtype)
    off = seg.tolist()
    for r in range(R):
        lo, hi = off[r], off[r + 1]
        padded[r, : hi - lo] = x[lo:hi]
    v, i_local = torch.topk(padded, k, dim=-1, largest=True, sorted=True)
    return v, i_local.to(torch.int64) + seg[:-1].view(R, 1)


def sample(fn, x, seg, k):
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    fn(x, seg, k)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b)


def timed_pair(mine_fn, prod_fn, x, seg, k, warmup=3, trials=7, budget_s=30.0):
    import time

    t0 = time.time()
    mine_fn(x, seg, k)
    torch.cuda.synchronize()
    one = time.time() - t0
    if one * 2 * (warmup + trials) > budget_s:
        warmup, trials = 1, max(3, int(budget_s / (2 * one)) or 3)
    for _ in range(warmup):
        mine_fn(x, seg, k)
        prod_fn(x, seg, k)
    torch.cuda.synchronize()
    ms, ps = [], []
    for i in range(trials):
        if i % 2 == 0:
            m = sample(mine_fn, x, seg, k)
            p = sample(prod_fn, x, seg, k)
        else:
            p = sample(prod_fn, x, seg, k)
            m = sample(mine_fn, x, seg, k)
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
        ("non-power-of-two lengths", CHECK_SHAPE_NPOT),
        ("duplicate-saturated", CHECK_SHAPE_TIES),
    ):
        fails = exact_check(s)
        if fails:
            print(
                f"CORRECTNESS FAILED on the {label} shape "
                f"R={s['R']} Lmin={s['Lmin']} Lmax={s['Lmax']} k={s['k']} bfloat16:"
            )
            for f in fails:
                print(f"    {f}")
            return emit(
                {"rows": [], "geomean": None, "correctness_failed": fails, "shape": s},
                1,
            )
        print(
            f"correctness ok ({label}: R={s['R']} Lmin={s['Lmin']} "
            f"Lmax={s['Lmax']} k={s['k']}, exact)"
        )

    det = check_determinism()
    if det:
        print()
        print("NOT DETERMINISTIC on a duplicate-saturated ragged input:")
        for f in det[:8]:
            print(f"    {f}")
        return emit({"rows": [], "geomean": None, "nondeterministic": det}, 1)
    print(
        "determinism ok: three runs on a duplicate-saturated ragged input, bitwise identical"
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
        x, seg, k = make(s, 0)
        R = int(seg.shape[0]) - 1
        NNZ = int(x.shape[0])
        Lmax = int((seg[1:] - seg[:-1]).max().item())
        mine, prod = timed_pair(topk_ragged, _pad_topk_anchor, x, seg, k)
        nbytes = NNZ * x.element_size() + R * k * (x.element_size() + 8)
        mine_gbps = nbytes / (mine * 1e-3) / 1e9
        prod_gbps = nbytes / (prod * 1e-3) / 1e9
        rows.append(
            {
                "shape": label,
                "R": R,
                "Lmax": Lmax,
                "NNZ": NNZ,
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
            f"{label:4s} R={R:4d} Lmax={Lmax:6d} NNZ={NNZ:9d} k={k:3d}  "
            f"yours {mine:9.4f} ms   production {prod:9.4f} ms   "
            f"fraction {min(prod / mine, 1.0):.4f}"
        )
        print(
            f"     compulsory-traffic bandwidth: yours {mine_gbps:7.1f} GB/s "
            f"({100 * mine_gbps / PEAK_GBPS:5.2f}% of peak)   production "
            f"{prod_gbps:7.1f} GB/s ({100 * prod_gbps / PEAK_GBPS:5.2f}%)"
        )

    geo = statistics.geometric_mean([r["fraction_of_production"] for r in rows])
    print(f"published-shape geomean fraction of production: {geo:.4f}")
    print(
        "(not your score: the graded shapes, scales and length distributions are hidden)"
    )
    return emit({"rows": rows, "geomean": geo, "adoption": ad}, 0)


if __name__ == "__main__":
    raise SystemExit(main())
