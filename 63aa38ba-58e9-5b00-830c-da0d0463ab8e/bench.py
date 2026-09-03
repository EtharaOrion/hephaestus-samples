#!/usr/bin/env python3
"""Published-shape benchmark. Steering tool, not the grader: the shapes here
are not the graded shapes, and the number it prints is not your score.

Gates BEFORE it times, in the grader's order: forbidden-name scan, correctness
on both outputs (including a non-power-of-two shape and the single-step S=1
case), three-run bitwise determinism, written-kernel share, and only then the
timed pairs against the torch.compile baseline. This file may call torch.compile
directly; your `kernel.py` may not import any linear-attention or paged-attention
library implementation -- the scan below is the same word-boundary rule the
grader applies to your source.
"""

import argparse
import functools
import json
import os
import re
import statistics
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from kernel import paged_gla_decode  # noqa: E402
from reference import paged_gla_decode_ref  # noqa: E402

CHECK_SHAPE = {"B": 4, "S": 64, "H": 4, "K": 64, "V": 64, "P": 32}
CHECK_SHAPE_NPOT = {"B": 5, "S": 57, "H": 3, "K": 64, "V": 64, "P": 41}
CHECK_SHAPE_SINGLE = {"B": 16, "S": 1, "H": 4, "K": 128, "V": 128, "P": 128}
DET_SHAPE = {"B": 32, "S": 128, "H": 8, "K": 128, "V": 128, "P": 256}

SHAPES = {
    "p1": {"B": 96, "S": 96, "H": 8, "K": 128, "V": 128, "P": 512},
    "p2": {"B": 48, "S": 320, "H": 8, "K": 64, "V": 128, "P": 256},
}

FORBIDDEN = [
    "fla.ops",
    "import fla",
    "from fla",
    "fused_recurrent_gla",
    "chunk_gla",
    "fused_recurrent",
    "paged_attention",
    "vllm.attention",
    "flash_attn_with_kvcache",
]

TOL = {"bfloat16": (2.0e-2, 3.125e-2), "float32": (2.0e-3, 2.0e-3)}
STATE_TOL = (2.0e-3, 2.0e-3)

# H100 SXM HBM3. Constant rather than a query; this task is graded on H100 only.
PEAK_GBPS = 3350.0


def make(shape, seed, dtype=torch.bfloat16):
    """Fixed seed and fixed unit scale so a rerun is comparable. Graded draws
    also randomize the magnitude of v/paged_state from a hidden range; a
    uniform-policy kernel cannot tell the difference."""
    B, S, H, K, V, P = (shape[x] for x in ("B", "S", "H", "K", "V", "P"))
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    f = lambda *s: torch.randn(*s, device="cuda", generator=gen)  # noqa: E731
    q = F.normalize(f(B, S, H, K), dim=-1).to(dtype).contiguous()
    k = F.normalize(f(B, S, H, K), dim=-1).to(dtype).contiguous()
    v = f(B, S, H, V).to(dtype).contiguous()
    g = F.logsigmoid(f(B, S, H, K)).contiguous()
    pool = (f(P, H, K, V) * 0.5).contiguous()
    pt = (
        torch.randperm(P, device="cuda", generator=gen)[:B].to(torch.int32).contiguous()
    )
    return {"q": q, "k": k, "v": v, "g": g, "page_table": pt, "paged_state": pool}


def scan_forbidden():
    src = open(os.path.join(_HERE, "kernel.py")).read()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    return [s for s in FORBIDDEN if re.search(r"\b" + re.escape(s) + r"\b", code)]


def _close(got, want, atol, rtol, tag):
    if not torch.isfinite(got).all():
        return [f"{tag}: non-finite values"]
    d = (got.float() - want.float()).abs()
    lim = atol + rtol * want.float().abs()
    bad = int((d > lim).sum())
    if bad:
        return [
            f"{tag}: {bad} elements outside tolerance (max abs {float(d.max()):.3e})"
        ]
    return []


def correctness_check(shape, dtype=torch.bfloat16):
    args = make(shape, 0, dtype)
    ref_args = {
        k: (t.clone() if isinstance(t, torch.Tensor) else t) for k, t in args.items()
    }
    try:
        got = paged_gla_decode(**args)
    except Exception as e:  # noqa: BLE001
        return [f"candidate raised {type(e).__name__}: {e}"]
    if not (isinstance(got, tuple) and len(got) == 2):
        return [f"output must be (o, state_out), got {type(got).__name__}"]
    go, gs = got
    want_o, want_s = paged_gla_decode_ref(**ref_args)
    fails = []
    for name, t in args.items():
        if isinstance(t, torch.Tensor) and not torch.equal(t, ref_args[name]):
            fails.append(f"input {name} was mutated in place by the call")
    if go.dtype != want_o.dtype:
        fails.append(f"o dtype {go.dtype}, must be {want_o.dtype} (v's dtype)")
    if gs.dtype != torch.float32:
        fails.append(f"state_out dtype {gs.dtype}, must be float32")
    if fails:
        return fails
    dt_name = "bfloat16" if dtype == torch.bfloat16 else "float32"
    fails += _close(go, want_o, *TOL[dt_name], "o")
    fails += _close(gs, want_s, *STATE_TOL, "state_out")
    return fails


def check_determinism():
    """Four runs on one input, first discarded, bitwise identical o AND
    state_out. The candidate against itself; no tolerance applies."""
    args = make(DET_SHAPE, 0)
    runs = [paged_gla_decode(**args) for _ in range(4)][1:]
    fails = []
    for i, (o, s) in enumerate(runs[1:], start=2):
        if not torch.equal(o, runs[0][0]):
            fails.append(f"run {i}: o differs from run 1 at the bit level")
        if not torch.equal(s, runs[0][1]):
            fails.append(f"run {i}: state_out differs from run 1 at the bit level")
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
        paged_gla_decode(**args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        paged_gla_decode(**args)
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


@functools.lru_cache(maxsize=1)
def _compiled_step():
    def _step(state, qt, kt, vt, gt, scale):
        state = state * gt.exp()[:, :, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", kt, vt)
        y = torch.einsum("bhk,bhkv->bhv", qt * scale, state)
        return state, y

    return torch.compile(_step)


def _production(args):
    q, k, v, g = args["q"], args["k"], args["v"], args["g"]
    idx = args["page_table"].long()
    state = args["paged_state"][idx].float().clone()
    K = q.shape[-1]
    scale = K**-0.5
    step = _compiled_step()
    S = q.shape[1]
    q32, k32, v32, g32 = q.float(), k.float(), v.float(), g.float()
    ys = []
    for t in range(S):
        state, y = step(state, q32[:, t], k32[:, t], v32[:, t], g32[:, t], scale)
        ys.append(y)
    return torch.stack(ys, dim=1).to(v.dtype), state


def sample(fn, args):
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    fn(args)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b)


def timed_pair(mine_fn, prod_fn, args, warmup=3, trials=7, budget_s=30.0):
    import time

    t0 = time.time()
    mine_fn(args)
    torch.cuda.synchronize()
    one = time.time() - t0
    if one * 2 * (warmup + trials) > budget_s:
        warmup, trials = 1, max(3, int(budget_s / (2 * one)) or 3)
    for _ in range(warmup):
        mine_fn(args)
        prod_fn(args)
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
        return emit({"rows": [], "geomean": None, "forbidden": hits}, 1)

    for label, s in (
        ("check", CHECK_SHAPE),
        ("non-power-of-two", CHECK_SHAPE_NPOT),
        ("single-step", CHECK_SHAPE_SINGLE),
    ):
        fails = correctness_check(s)
        if fails:
            print(
                f"CORRECTNESS FAILED on the {label} shape "
                f"B={s['B']} S={s['S']} H={s['H']} K={s['K']} V={s['V']} P={s['P']} bfloat16:"
            )
            for f in fails:
                print(f"    {f}")
            print()
            print(
                "Nothing was timed. Both outputs are graded -- o at the input dtype's"
            )
            print("tolerance AND state_out at the float32 tolerance. A fast wrong")
            print("recurrence scores zero, not partial credit.")
            return emit(
                {"rows": [], "geomean": None, "correctness_failed": fails, "shape": s},
                1,
            )
        print(
            f"correctness ok ({label}: B={s['B']} S={s['S']} H={s['H']} "
            f"K={s['K']} V={s['V']} P={s['P']}, both outputs)"
        )

    det = check_determinism()
    if det:
        print()
        print("NOT DETERMINISTIC:")
        for f in det[:8]:
            print(f"    {f}")
        return emit({"rows": [], "geomean": None, "nondeterministic": det}, 1)
    print(
        "determinism ok: three runs after a warmup, o and state_out bitwise identical"
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
            print(f"{ad['mine_us']:.0f} us of {ad['total_us']:.0f} us ran in kernels")
            print("this file declares; the rest ran in framework operators. Move the")
            print("recurrence AND the page-table gather itself into your kernels.")
            return emit({"rows": [], "geomean": None, "adoption": ad}, 1)

    rows = []
    for label, s in SHAPES.items():
        B, S, H, K, V, P = (s[x] for x in ("B", "S", "H", "K", "V", "P"))
        args = make(s, 0)
        mine, prod = timed_pair(lambda a: paged_gla_decode(**a), _production, args)
        # Compulsory-traffic accounting: q/k/v/g stream read once, o written once,
        # per-batch initial state gathered once from the pool and final state
        # written once. page_table is B int32s (negligible).
        e = args["v"].element_size()
        nbytes = (
            2 * B * S * H * K * e
            + B * S * H * V * e
            + B * S * H * K * 4
            + B * S * H * V * e
            + 2 * B * H * K * V * 4
        )
        mine_gbps = nbytes / (mine * 1e-3) / 1e9
        prod_gbps = nbytes / (prod * 1e-3) / 1e9
        rows.append(
            {
                "shape": label,
                **{x: s[x] for x in ("B", "S", "H", "K", "V", "P")},
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
            f"{label:4s} B={B:4d} S={S:4d} H={H:3d} K={K:4d} V={V:4d} P={P:5d}  "
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
    print("(not your score: graded shapes, scales and page permutations are hidden)")
    return emit({"rows": rows, "geomean": geo, "adoption": ad}, 0)


if __name__ == "__main__":
    raise SystemExit(main())
