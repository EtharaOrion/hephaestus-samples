#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml

Published-shape benchmark for chunked_dplr_delta. Steering tool, not the grader.
"""

import argparse
import json
import statistics
import sys

import torch

import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from kernel import chunked_dplr_delta  # noqa: E402
from reference import chunked_dplr_delta_ref  # noqa: E402
from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule as _prod  # noqa: E402

GRAD_NAMES = ("q", "k", "v", "alpha", "beta", "gk")
BF16_ATOL, BF16_RTOL = 5.0e-2, 5.0e-2

CHECK_SHAPE = {"B": 1, "T": 128, "H": 2, "K": 64, "V": 64}
CHECK_SHAPE_NPOT = {"B": 1, "T": 127, "H": 3, "K": 96, "V": 96}

SHAPES = {
    "p1": {"B": 1, "T": 512, "H": 4, "K": 128, "V": 128, "dtypes": ["bfloat16"]},
    "p2": {"B": 2, "T": 512, "H": 8, "K": 128, "V": 128, "dtypes": ["bfloat16"]},
}


def make(B, T, H, K, V, dtype, seed):
    torch.manual_seed(seed)
    f = lambda *s: torch.randn(*s, device="cuda", dtype=torch.float32)  # noqa: E731
    qn = torch.nn.functional.normalize(f(B, T, H, K), dim=-1)
    kn = torch.nn.functional.normalize(f(B, T, H, K), dim=-1)
    v = f(B, T, H, V)
    alpha = (f(B, T, H, K) * 0.5).to(dtype).requires_grad_()
    beta = (f(B, T, H, K) * 0.5).to(dtype).requires_grad_()
    gk = torch.nn.functional.logsigmoid(f(B, T, H, K)).requires_grad_()
    return dict(
        q=qn.to(dtype).requires_grad_(),
        k=kn.to(dtype).requires_grad_(),
        v=v.to(dtype).requires_grad_(),
        alpha=alpha,
        beta=beta,
        gk=gk,
    )


def step(fn, inp):
    for t in inp.values():
        if t.grad is not None:
            t.grad = None
    out = fn(**inp)
    out = out[0] if isinstance(out, tuple) else out
    out.sum().backward()
    return out


def grads_of(inp):
    return {
        n: (inp[n].grad.clone() if inp[n].grad is not None else None)
        for n in GRAD_NAMES
    }


def close(got, want):
    if got is None or want is None:
        return False, "missing tensor"
    if not torch.isfinite(got).all():
        n = int((~torch.isfinite(got)).sum())
        return False, f"NON-FINITE: {n} of {got.numel()} elements are inf or nan"
    d = (got.float() - want.float()).abs()
    lim = BF16_ATOL + BF16_RTOL * want.float().abs()
    bad = int((d > lim).sum())
    if bad:
        return (
            False,
            f"{bad} of {got.numel()} elements outside tolerance, max abs {float(d.max()):.3e}",
        )
    return True, "ok"


def check_determinism():
    s = CHECK_SHAPE
    inp = make(s["B"], s["T"], s["H"], s["K"], s["V"], torch.bfloat16, 0)
    runs = []
    for _ in range(3):
        out = step(chunked_dplr_delta, inp)
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
    import pathlib as _p
    import re as _re

    src = _p.Path(_HERE, "kernel.py").read_text()
    bad = []
    for m in _re.finditer(
        r"torch\.(?:empty|zeros|ones|empty_like|zeros_like)\s*\(([^)]*)\)", src
    ):
        args = m.group(1)
        if "dtype" not in args:
            continue
        if _re.search(r"dtype\s*=\s*torch\.float32|dtype\s*=\s*tl\.float32", args):
            continue
        if _re.search(r"dtype\s*=\s*\w+\.dtype", args):
            bad.append(m.group(0)[:110])
    return bad


def _kernel_is_mine(key: str, names: set) -> bool:
    import re as _re

    if "::" in key or key.startswith("void ") or "<" in key:
        return False
    if _re.match(r"^(ampere|turing|volta|hopper|maxwell|pascal|sm\d+)_", key):
        return False
    if _re.search(r"(cutlass|cublas|cudnn|nvjet)", key, _re.I):
        return False
    return any(key == n or key.startswith(n + "_") for n in names)


PEAK_GBPS = 3350.0


def compulsory_bytes(B, T, H, K, V, itemsize):
    """Lower-bound HBM traffic; q,k,alpha,beta [B,T,H,K] + v [B,T,H,V] + gk fp32 [B,T,H,K]."""
    n = B * T * H
    fwd = n * (4 * K + V) * itemsize + n * K * 4  # q,k,alpha,beta,v + gk fp32
    fwd += n * V * itemsize  # o out
    bwd = n * V * itemsize
    bwd += n * (4 * K + V) * itemsize + n * K * 4
    bwd += n * (4 * K + V) * itemsize + n * K * 4
    return fwd + bwd


def check_adoption():
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
        step(chunked_dplr_delta, inp)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step(chunked_dplr_delta, inp)
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


def _prod_call(**kw):
    # Anchor takes a, b, gk as keyword arg names.
    o = _prod(
        q=kw["q"],
        k=kw["k"],
        v=kw["v"],
        a=kw["alpha"],
        b=kw["beta"],
        gk=kw["gk"],
        scale=None,
        output_final_state=False,
    )
    return o[0] if isinstance(o, tuple) else o


def check(shape=None):
    s = shape or CHECK_SHAPE
    inp = make(s["B"], s["T"], s["H"], s["K"], s["V"], torch.bfloat16, 0)
    fails = []
    try:
        got = step(chunked_dplr_delta, inp)
    except Exception as e:  # noqa: BLE001
        return [f"candidate raised {type(e).__name__}: {e}"]
    got_g = grads_of(inp)
    want = step(chunked_dplr_delta_ref, inp)
    want_g = grads_of(inp)
    ok, detail = close(got, want)
    if not ok:
        fails.append(f"out  {detail}")
    for n in GRAD_NAMES:
        ok, detail = close(got_g.get(n), want_g.get(n))
        if not ok:
            fails.append(f"d{n:<6} {detail}")
    return fails


def sample(fn, inp):
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    step(fn, inp)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b)


def timed_pair(mine_fn, prod_fn, inp, warmup=3, trials=5, budget_s=30.0):
    import time as _t

    _s = _t.time()
    step(mine_fn, inp)
    torch.cuda.synchronize()
    one = _t.time() - _s
    if one * 2 * (warmup + trials) > budget_s:
        warmup, trials = 1, max(3, int(budget_s / (2 * one)) or 3)
    for _ in range(warmup):
        step(mine_fn, inp)
        step(prod_fn, inp)
    torch.cuda.synchronize()
    mine_xs, prod_xs = [], []
    for i in range(trials):
        if i % 2 == 0:
            m = sample(mine_fn, inp)
            p = sample(prod_fn, inp)
        else:
            p = sample(prod_fn, inp)
            m = sample(mine_fn, inp)
        mine_xs.append(m)
        prod_xs.append(p)
    return statistics.median(mine_xs), statistics.median(prod_xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()
    s = CHECK_SHAPE
    fails = check()
    if not fails:
        n = CHECK_SHAPE_NPOT
        npot = check(n)
        if npot:
            s, fails = n, [f"(non-power-of-two shape) {f}" for f in npot]
    if fails:
        print(
            f"CORRECTNESS FAILED at B={s['B']} T={s['T']} H={s['H']} K={s['K']} V={s['V']} bfloat16"
        )
        for f in fails:
            print(f"    {f}")
        if args.json:
            json.dump(
                {
                    "rows": [],
                    "geomean": None,
                    "correctness_failed": fails,
                    "check_shape": s,
                },
                open(args.json, "w"),
                indent=2,
            )
        return 1
    print(
        f"correctness ok at B={s['B']} T={s['T']} H={s['H']} K={s['K']} V={s['V']} bfloat16"
    )

    det = check_determinism()
    if det:
        print("NOT DETERMINISTIC.")
        for f in det[:12]:
            print(f"    {f}")
        if args.json:
            json.dump(
                {
                    "rows": [],
                    "geomean": None,
                    "nondeterministic": det,
                    "check_shape": s,
                },
                open(args.json, "w"),
                indent=2,
            )
        return 1
    print("determinism ok: three runs on one input, bitwise identical")

    narrowed = check_fp32_state()
    if narrowed:
        print("RECURRENT STATE NARROWED BELOW FLOAT32:")
        for b in narrowed[:6]:
            print(f"    {b}")
        if args.json:
            json.dump(
                {
                    "rows": [],
                    "geomean": None,
                    "narrowed_state": narrowed,
                    "check_shape": s,
                },
                open(args.json, "w"),
                indent=2,
            )
        return 1
    print("recurrent state kept in float32")

    ad = check_adoption()
    if ad is not None:
        pct = 100.0 * ad["adoption"]
        print(
            f"written-kernel share: {pct:.1f}% of device time in your own "
            f"triton kernels ({', '.join(ad['declared']) or 'none declared'})"
        )
        if ad["adoption"] < ADOPTION_FLOOR:
            print(f"BELOW THE WRITTEN-KERNEL FLOOR of {ADOPTION_FLOOR:.0%}.")
            if args.json:
                json.dump(
                    {"rows": [], "geomean": None, "adoption": ad, "check_shape": s},
                    open(args.json, "w"),
                    indent=2,
                )
            return 1

    rows = []
    for label, s in SHAPES.items():
        inp = make(s["B"], s["T"], s["H"], s["K"], s["V"], torch.bfloat16, 0)
        mine, prod = timed_pair(chunked_dplr_delta, _prod_call, inp)
        nbytes = compulsory_bytes(s["B"], s["T"], s["H"], s["K"], s["V"], 2)
        mine_gbps = nbytes / (mine * 1e-3) / 1e9
        prod_gbps = nbytes / (prod * 1e-3) / 1e9
        rows.append(
            {
                "shape": label,
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
            f"{label:4s} yours {mine:9.4f} ms   production {prod:9.4f} ms   "
            f"fraction {min(prod / mine, 1.0):.4f}"
        )
        print(
            f"     bandwidth: yours {mine_gbps:7.1f} GB/s ({100 * mine_gbps / PEAK_GBPS:5.2f}% of "
            f"HBM peak)   production {prod_gbps:7.1f} GB/s "
            f"({100 * prod_gbps / PEAK_GBPS:5.2f}%)"
        )

    geo = statistics.geometric_mean([r["fraction_of_production"] for r in rows])
    print(f"published-shape geomean fraction of production: {geo:.4f}")
    if args.json:
        json.dump(
            {
                "rows": rows,
                "geomean": geo,
                "correctness_failed": [],
                "check_shape": CHECK_SHAPE,
            },
            open(args.json, "w"),
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
