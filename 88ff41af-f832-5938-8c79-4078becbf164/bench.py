#!/usr/bin/env python3
"""Published-shape benchmark for mamba2_ssd_decode. Steering tool, not the grader."""

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
from kernel import mamba2_ssd_decode  # noqa: E402
from reference import mamba2_ssd_decode_ref  # noqa: E402

CHECK_SHAPE = {"B": 4, "S": 64, "H": 4, "P": 64, "N": 64}
CHECK_SHAPE_NPOT = {"B": 5, "S": 57, "H": 3, "P": 64, "N": 64}
CHECK_SHAPE_SINGLE = {"B": 16, "S": 1, "H": 4, "P": 64, "N": 128}
DET_SHAPE = {"B": 64, "S": 128, "H": 8, "P": 64, "N": 128}

SHAPES = {
    "g1": {"B": 128, "S": 128, "H": 32, "P": 64, "N": 128},
    "g2": {"B": 256, "S": 64, "H": 16, "P": 64, "N": 64},
    "g3": {"B": 64, "S": 256, "H": 8, "P": 64, "N": 128},
}

FORBIDDEN = [
    "fla.ops",
    "import fla",
    "from fla",
    "selective_state_update",
    "selective_scan",
    "mamba_chunk_scan",
    "mamba_ssm",
    "mamba_split_conv1d_scan",
]

TOL = {"bfloat16": (2.0e-2, 3.125e-2), "float32": (2.0e-3, 2.0e-3)}
STATE_TOL = (2.0e-3, 2.0e-3)

PEAK_GBPS = 3350.0


def make(shape, seed, dtype=torch.bfloat16):
    Bb, S, H, P, Nn = (shape[_k] for _k in ("B", "S", "H", "P", "N"))
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", dtype=torch.float32, generator=gen)  # noqa: E731
    x = r(Bb, S, H, P).to(dtype).contiguous()
    dt = (r(Bb, S, H) * 0.5).contiguous()
    A_log = (r(H) * 0.5).contiguous()
    B = r(Bb, S, H, Nn).to(dtype).contiguous()
    C = r(Bb, S, H, Nn).to(dtype).contiguous()
    D = r(H).contiguous()
    dt_bias = (r(H) * 0.1).contiguous()
    state0 = (r(Bb, H, P, Nn) * 0.5).contiguous()
    return {
        "x": x,
        "dt": dt,
        "A_log": A_log,
        "B": B,
        "C": C,
        "D": D,
        "dt_bias": dt_bias,
        "state0": state0,
    }


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
        got = mamba2_ssd_decode(**args)
    except Exception as e:  # noqa: BLE001
        return [f"candidate raised {type(e).__name__}: {e}"]
    if not (isinstance(got, tuple) and len(got) == 2):
        return [f"output must be (o, state_out), got {type(got).__name__}"]
    go, gs = got
    want_o, want_s = mamba2_ssd_decode_ref(**ref_args)
    fails = []
    for name, t in args.items():
        if isinstance(t, torch.Tensor) and not torch.equal(t, ref_args[name]):
            fails.append(f"input {name} was mutated in place by the call")
    if go.dtype != want_o.dtype:
        fails.append(f"o dtype {go.dtype}, must be {want_o.dtype} (x's dtype)")
    if gs.dtype != torch.float32:
        fails.append(f"state_out dtype {gs.dtype}, must be float32")
    if fails:
        return fails
    dt_name = "bfloat16" if dtype == torch.bfloat16 else "float32"
    fails += _close(go, want_o, *TOL[dt_name], "o")
    fails += _close(gs, want_s, *STATE_TOL, "state_out")
    return fails


def check_determinism():
    args = make(DET_SHAPE, 0)
    runs = [mamba2_ssd_decode(**args) for _ in range(4)][1:]
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
    args = make(SHAPES["g1"], 0)
    for _ in range(3):
        mamba2_ssd_decode(**args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        mamba2_ssd_decode(**args)
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
    def _step(state, xt, dt_pre, A, B_t, C_t, D, dt_bias):
        dt_t = F.softplus(dt_pre + dt_bias[None, :])
        dA = (dt_t * A[None, :]).exp()
        dtx = dt_t[:, :, None] * xt
        state = state * dA[:, :, None, None] + torch.einsum("bhp,bhn->bhpn", dtx, B_t)
        y = torch.einsum("bhpn,bhn->bhp", state, C_t) + D[None, :, None] * xt
        return state, y

    return torch.compile(_step)


def _production(args):
    x, dt = args["x"], args["dt"]
    A = -args["A_log"].float().exp()
    D, dt_bias = args["D"].float(), args["dt_bias"].float()
    Bs, Cs = args["B"].float(), args["C"].float()
    x32 = x.float()
    state = args["state0"].float().clone()
    step = _compiled_step()
    S = x.shape[1]
    ys = []
    for t in range(S):
        state, y = step(
            state, x32[:, t], dt[:, t].float(), A, Bs[:, t], Cs[:, t], D, dt_bias
        )
        ys.append(y)
    o = torch.stack(ys, dim=1).to(x.dtype)
    return o, state


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
                f"B={s['B']} S={s['S']} H={s['H']} P={s['P']} N={s['N']}:"
            )
            for f in fails:
                print(f"    {f}")
            return emit(
                {"rows": [], "geomean": None, "correctness_failed": fails, "shape": s},
                1,
            )
        print(
            f"correctness ok ({label}: B={s['B']} S={s['S']} H={s['H']} "
            f"P={s['P']} N={s['N']}, both outputs)"
        )

    det = check_determinism()
    if det:
        print("\nNOT DETERMINISTIC:")
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
            print(f"\nBELOW THE WRITTEN-KERNEL FLOOR of {ADOPTION_FLOOR:.0%}.")
            return emit({"rows": [], "geomean": None, "adoption": ad}, 1)

    rows = []
    for label, s in SHAPES.items():
        Bb, S, H, P, Nn = (s[_k] for _k in ("B", "S", "H", "P", "N"))
        args = make(s, 0)
        mine, prod = timed_pair(lambda a: mamba2_ssd_decode(**a), _production, args)
        e = args["x"].element_size()
        # Compulsory traffic: x/B/C/o per (batch, step, head) in x's dtype, dt as
        # fp32 scalar-per-(batch,step,head), state read+written once as fp32.
        # Head-only tensors (A_log, D, dt_bias) are O(H) and negligible.
        nbytes = (
            Bb * S * H * P * e
            + Bb * S * H * Nn * e
            + Bb * S * H * Nn * e
            + Bb * S * H * 4
            + Bb * S * H * P * e
            + 2 * Bb * H * P * Nn * 4
        )
        mine_gbps = nbytes / (mine * 1e-3) / 1e9
        prod_gbps = nbytes / (prod * 1e-3) / 1e9
        rows.append(
            {
                "shape": label,
                **{_k: s[_k] for _k in ("B", "S", "H", "P", "N")},
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
            f"{label:4s} B={Bb:4d} S={S:4d} H={H:3d} P={P:3d} N={Nn:4d}  "
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
    print("(not your score: graded shapes/scales/dtypes are hidden)")
    return emit({"rows": rows, "geomean": geo, "adoption": ad}, 0)


if __name__ == "__main__":
    raise SystemExit(main())
