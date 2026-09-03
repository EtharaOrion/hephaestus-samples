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

CHECK_SHAPE = {"T": 512, "D": 512, "F": 256, "E": 16, "G": 4, "K": 2}
CHECK_SHAPE_NPOT = {"T": 301, "D": 200, "F": 96, "E": 12, "G": 4, "K": 2}
CHECK_SHAPE_TIES = {
    "T": 256,
    "D": 256,
    "F": 128,
    "E": 16,
    "G": 4,
    "K": 2,
    "tie_pool": 4,
}
SHAPES = {
    "p1": {"T": 4096, "D": 1024, "F": 704, "E": 32, "G": 8, "K": 2},
    "p2": {"T": 2048, "D": 2048, "F": 1408, "E": 64, "G": 8, "K": 4},
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


def _sel_lowest_index(scores, k, ibits):
    b = torch.where(scores == 0.0, torch.zeros_like(scores), scores).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    N = scores.shape[-1]
    idx = torch.arange(N, device=scores.device, dtype=torch.int64)
    keys = (u << ibits) | (N - 1 - idx)
    return torch.topk(keys, k, dim=-1, largest=True, sorted=True).indices


def make(shape, seed):
    T, D = shape["T"], shape["D"]
    F, E = shape["F"], shape["E"]
    G, K = shape["G"], shape["K"]
    assert E % G == 0
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
    # margin fix (both boundaries) on non-tie draws
    if not pool:
        S = E // G
        rw64 = rwq.double()
        for _ in range(64):
            p = torch.softmax(xq.double() @ rw64, dim=-1)
            pg = p.view(T, G, S)
            gs = (
                pg.squeeze(-1)
                if S == 1
                else torch.topk(pg, min(2, S), dim=-1).values.sum(dim=-1)
            )
            if G > 1:
                gv = torch.topk(gs, 2, dim=-1).values
                bad_g = ((gv[:, 0] - gv[:, 1]) > 0) & ((gv[:, 0] - gv[:, 1]) < 1e-4)
            else:
                bad_g = torch.zeros(T, dtype=torch.bool, device="cuda")
            grp = torch.argmax(gs, dim=-1)
            row = torch.arange(T, device="cuda", dtype=torch.int64)
            chosen = pg[row, grp]
            if K < S:
                v = torch.topk(chosen, K + 1, dim=-1).values
                bad_k = ((v[:, K - 1] - v[:, K]) > 0) & ((v[:, K - 1] - v[:, K]) < 1e-4)
            else:
                bad_k = torch.zeros(T, dtype=torch.bool, device="cuda")
            bad = bad_g | bad_k
            if not int(bad.sum()):
                break
            fresh = torch.randn(int(bad.sum()), D, generator=g, device="cuda")
            xq[bad] = fresh.to(torch.bfloat16)
    return {
        "x": xq.contiguous().requires_grad_(True),
        "router_w": rwq.contiguous().requires_grad_(True),
        "w1": w1.to(torch.bfloat16).contiguous().requires_grad_(True),
        "w2": w2.to(torch.bfloat16).contiguous().requires_grad_(True),
        "G": G,
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
    y = fn(args["x"], args["router_w"], args["w1"], args["w2"], args["G"], args["K"])
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
    yr, gr = run_fwdbwd(
        lambda x, rw, w1, w2, G, K: fused_moe_ref(x, rw, w1, w2, G, K), args
    )
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


def production(x, rw, w1, w2, G, K):
    """Anchor: two-stage int-key selection + p_sel/sum + torch._grouped_mm
    expert pipeline over token-sorted kept pairs."""
    T, D = x.shape
    E = rw.shape[1]
    F = w1.shape[2] // 2
    S = E // G
    p = torch.softmax(x.float() @ rw.float(), dim=-1)
    pg_det = p.detach().view(T, G, S)
    gbits = max((G - 1).bit_length(), 1)
    sbits = max((S - 1).bit_length(), 1)
    with torch.no_grad():
        if S == 1:
            gs = pg_det.squeeze(-1)
        else:
            gs = torch.topk(pg_det, min(2, S), dim=-1).values.sum(dim=-1)
        grp = _sel_lowest_index(gs, 1, gbits).squeeze(-1)
        row = torch.arange(T, device=x.device, dtype=torch.int64)
        chosen = pg_det[row, grp]
        sel_local = _sel_lowest_index(chosen, K, sbits)
        sel = grp.unsqueeze(-1) * S + sel_local
    p_sel = torch.gather(p, 1, sel)
    w = p_sel / p_sel.sum(dim=-1, keepdim=True)
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
                f"CORRECTNESS FAILED on the {label} shape T={s['T']} "
                f"D={s['D']} F={s['F']} E={s['E']} G={s['G']} K={s['K']}:"
            )
            for f in fails[:8]:
                print(f"    {f}")
            return emit(
                {"rows": [], "geomean": None, "correctness_failed": fails, "shape": s},
                1,
            )
        print(f"correctness ok ({label})")

    det = check_determinism()
    if det:
        print("NOT DETERMINISTIC:")
        for f in det[:8]:
            print(f"    {f}")
        return emit({"rows": [], "geomean": None, "nondeterministic": det}, 1)
    print("determinism ok")

    ad = check_adoption()
    if ad is not None:
        pct = 100.0 * ad["adoption"]
        print(f"written-kernel share: {pct:.1f}%")
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
                **{k: s[k] for k in ("T", "D", "F", "E", "G", "K")},
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
            f"{label}  T={s['T']} E={s['E']} G={s['G']} K={s['K']}  "
            f"yours {mine:9.3f} ms  production {prod:9.3f} ms  "
            f"fraction {min(prod / mine, 1.0):.4f}"
        )

    geo = statistics.geometric_mean([r["fraction_of_production"] for r in rows])
    print(f"published-shape geomean fraction of production: {geo:.4f}")
    return emit({"rows": rows, "geomean": geo, "adoption": ad}, 0)


if __name__ == "__main__":
    raise SystemExit(main())
