#!/usr/bin/env python3
"""Published-shape benchmark. Steering tool, not the grader."""

import argparse
import json
import os
import re
import statistics
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from kernel import fused_moe  # noqa: E402
from reference import fused_moe_ref  # noqa: E402

CHECK_SHAPE = {"T": 512, "D": 512, "F": 256, "E": 64, "K": 4}
CHECK_SHAPE_NPOT = {"T": 301, "D": 200, "F": 96, "E": 67, "K": 5}
CHECK_SHAPE_TIES = {"T": 256, "D": 256, "F": 128, "E": 64, "K": 4, "tie_pool": 4}
SHAPES = {
    "p1": {"T": 4096, "D": 1024, "F": 704, "E": 256, "K": 8},
    "p2": {"T": 2048, "D": 2048, "F": 1408, "E": 128, "K": 4},
}

FORBIDDEN = [
    "torch._grouped_mm",
    "_grouped_mm",
    "vllm",
    "sglang",
    "grouped_gemm",
    "transformer_engine",
    "flashinfer",
]

OUT_ATOL = OUT_RTOL = 4.5e-2
GRAD_ATOL = GRAD_RTOL = 4.5e-2
PEAK_TFLOPS_BF16 = 989.5


def _sel_topK(scores, K):
    E = scores.shape[-1]
    b = torch.where(scores == 0.0, torch.zeros_like(scores), scores).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=scores.device, dtype=torch.int64)
    ebits = max((E - 1).bit_length(), 1)
    keys = (u << ebits) | (E - 1 - e_idx)
    return torch.topk(keys, K, dim=-1, largest=True, sorted=True).indices


def make(shape, seed):
    T, D = shape["T"], shape["D"]
    F, E, K = shape["F"], shape["E"], shape["K"]
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    pool = shape.get("tie_pool", 0)
    if pool:
        cols = torch.randn(D, pool, generator=g, device="cuda") * D**-0.5
        assign = torch.randint(0, pool, (E,), generator=g, device="cuda")
        rw = cols.index_select(1, assign).contiguous()
    else:
        rw = torch.randn(D, E, generator=g, device="cuda") * D**-0.5
    w1 = torch.randn(E, D, 2 * F, generator=g, device="cuda") * D**-0.5
    w2 = torch.randn(E, F, D, generator=g, device="cuda") * F**-0.5
    x = torch.randn(T, D, generator=g, device="cuda")
    rwq = rw.to(torch.bfloat16)
    xq = x.to(torch.bfloat16)
    if K < E and not pool:
        rw64 = rwq.double()
        for _ in range(64):
            p = torch.softmax(xq.double() @ rw64, dim=-1)
            v = torch.topk(p, K + 1, dim=-1).values
            gap = v[:, K - 1] - v[:, K]
            bad = (gap > 0) & (gap < 1e-4)
            if not int(bad.sum()):
                break
            fresh = torch.randn(int(bad.sum()), D, generator=g, device="cuda")
            xq[bad] = fresh.to(torch.bfloat16)
    return {
        "x": xq.contiguous().requires_grad_(True),
        "router_w": rwq.contiguous().requires_grad_(True),
        "w1": w1.to(torch.bfloat16).contiguous().requires_grad_(True),
        "w2": w2.to(torch.bfloat16).contiguous().requires_grad_(True),
        "K": K,
    }


def cotangent(shape_td):
    g = torch.Generator(device="cuda")
    g.manual_seed((0x5EEDC0DE ^ (shape_td[0] * 31 + shape_td[1])) & 0x7FFFFFFF)
    return (torch.rand(shape_td, generator=g, device="cuda") * 2.0 - 1.0).to(
        torch.bfloat16
    )


GRADS = ("x", "router_w", "w1", "w2")


def run_fwdbwd(fn, args):
    for n in GRADS:
        if args[n].grad is not None:
            args[n].grad = None
    y = fn(args["x"], args["router_w"], args["w1"], args["w2"], args["K"])
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
        return [
            f"{tag}: {bad} elements outside tolerance, max abs {float(d.max()):.3e}"
        ]
    return []


def exact_check(shape, label):
    args = make(shape, 0)
    try:
        y, g = run_fwdbwd(fused_moe, args)
    except Exception as e:  # noqa: BLE001
        return [f"candidate raised {type(e).__name__}: {e}"]
    gc = {n: g[n].detach().clone() for n in GRADS}
    yr, gr = run_fwdbwd(lambda x, rw, w1, w2, K: fused_moe_ref(x, rw, w1, w2, K), args)
    fails = []
    if y.dtype != args["x"].dtype:
        fails.append(f"output dtype {y.dtype}, must be {args['x'].dtype}")
    fails += _close(y, yr, OUT_ATOL, OUT_RTOL, "y")
    for n in GRADS:
        fails += _close(gc[n], gr[n], GRAD_ATOL, GRAD_RTOL, f"d{n}")
    return fails


def check_determinism():
    args = make(CHECK_SHAPE_TIES, 0)
    run_fwdbwd(fused_moe, args)
    runs = []
    for _ in range(3):
        y, g = run_fwdbwd(fused_moe, args)
        runs.append((y.detach().clone(), {n: g[n].detach().clone() for n in GRADS}))
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
    return {
        "adoption": (mine / total if total else 0.0),
        "declared": sorted(names),
        "total_us": round(total, 1),
        "mine_us": round(mine, 1),
    }


ADOPTION_FLOOR = 0.60


def production(x, rw, w1, w2, K):
    """Anchor: router matmul + softmax + int-key top-K + gather + softmax over
    selected logits, then torch._grouped_mm expert pipeline over token-sorted
    kept pairs."""
    T, D = x.shape
    E = rw.shape[1]
    F = w1.shape[2] // 2
    logits = x.float() @ rw.float()
    p = torch.softmax(logits, dim=-1)
    with torch.no_grad():
        sel = _sel_topK(p.detach(), K)
    sel_logits = torch.gather(logits, 1, sel)
    w = torch.softmax(sel_logits, dim=-1)
    flat = sel.reshape(-1)
    order = torch.argsort(flat)
    tok = torch.div(order, K, rounding_mode="floor")
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
        return emit({"rows": [], "geomean": None, "forbidden": hits}, 1)

    for label, s in (
        ("check", CHECK_SHAPE),
        ("non-power-of-two", CHECK_SHAPE_NPOT),
        ("tie-forcing", CHECK_SHAPE_TIES),
    ):
        fails = exact_check(s, label)
        if fails:
            print(
                f"CORRECTNESS FAILED on the {label} shape "
                f"T={s['T']} D={s['D']} F={s['F']} E={s['E']} K={s['K']} bfloat16:"
            )
            for f in fails[:8]:
                print(f"    {f}")
            return emit(
                {"rows": [], "geomean": None, "correctness_failed": fails, "shape": s},
                1,
            )
        print(
            f"correctness ok ({label}: T={s['T']} D={s['D']} F={s['F']} "
            f"E={s['E']} K={s['K']}, output + 4 gradients)"
        )

    det = check_determinism()
    if det:
        print("NOT DETERMINISTIC on the tie-forcing input:")
        for f in det[:8]:
            print(f"    {f}")
        return emit({"rows": [], "geomean": None, "nondeterministic": det}, 1)
    print("determinism ok")

    ad = check_adoption()
    if ad is not None:
        pct = 100.0 * ad["adoption"]
        print(f"written-kernel share: {pct:.1f}% of forward+backward device time")
        if ad["adoption"] < ADOPTION_FLOOR:
            print(f"BELOW THE WRITTEN-KERNEL FLOOR of {ADOPTION_FLOOR:.0%}.")
            return emit({"rows": [], "geomean": None, "adoption": ad}, 1)

    rows = []
    for label, s in SHAPES.items():
        args = make(s, 0)
        mine, prod = timed_pair(fused_moe, production, args)
        tflop = 18.0 * s["T"] * s["K"] * s["D"] * s["F"] / 1e12
        mine_tf = tflop / (mine * 1e-3)
        prod_tf = tflop / (prod * 1e-3)
        rows.append(
            {
                "shape": label,
                **{k: s[k] for k in ("T", "D", "F", "E", "K")},
                "yours_ms": round(mine, 4),
                "production_ms": round(prod, 4),
                "fraction_of_production": round(min(prod / mine, 1.0), 5),
                "yours_tflops": round(mine_tf, 1),
                "production_tflops": round(prod_tf, 1),
                "yours_pct_peak": round(100 * mine_tf / PEAK_TFLOPS_BF16, 2),
                "production_pct_peak": round(100 * prod_tf / PEAK_TFLOPS_BF16, 2),
            }
        )
        print(
            f"{label}  T={s['T']:5d} E={s['E']:3d} K={s['K']}  "
            f"yours {mine:9.3f} ms   production {prod:9.3f} ms   "
            f"fraction {min(prod / mine, 1.0):.4f}"
        )

    geo = statistics.geometric_mean([r["fraction_of_production"] for r in rows])
    print(f"published-shape geomean fraction of production: {geo:.4f}")
    return emit({"rows": rows, "geomean": geo, "adoption": ad}, 0)


if __name__ == "__main__":
    raise SystemExit(main())
