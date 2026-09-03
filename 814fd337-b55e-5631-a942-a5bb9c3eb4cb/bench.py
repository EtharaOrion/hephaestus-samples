#!/usr/bin/env python3
"""Published-shape benchmark. Steering tool, not the grader."""

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
from kernel import int8_state_delta_decode  # noqa: E402
from reference import int8_state_delta_decode_ref, _quantize_int8_state  # noqa: E402

CHECK_SHAPE = {"B": 4, "S": 64, "H": 4, "K": 64, "V": 64}
CHECK_SHAPE_NPOT = {"B": 5, "S": 57, "H": 3, "K": 64, "V": 64}
CHECK_SHAPE_SINGLE = {"B": 16, "S": 1, "H": 4, "K": 128, "V": 128}
DET_SHAPE = {"B": 32, "S": 128, "H": 8, "K": 128, "V": 128}

SHAPES = {
    "p1": {"B": 96, "S": 96, "H": 8, "K": 128, "V": 128},
    "p2": {"B": 48, "S": 320, "H": 8, "K": 64, "V": 128},
}

FORBIDDEN = [
    "fla.ops",
    "import fla",
    "from fla",
    "fused_recurrent_delta_rule",
    "chunk_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "chunk_gated_delta_rule",
    "fused_recurrent",
    "torch.quantization",
    "quantize_per_tensor",
    "quantize_per_channel",
    "bitsandbytes",
    "tensorrt",
    "torchao.quantization",
]

TOL = {"bfloat16": (2.0e-2, 3.125e-2), "float32": (2.0e-3, 2.0e-3)}
STATE_TOL = (2.0e-3, 2.0e-3)

PEAK_GBPS = 3350.0


def make(shape, seed, dtype=torch.bfloat16):
    B, S, H, K, V = (shape[x] for x in ("B", "S", "H", "K", "V"))
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    f = lambda *s: torch.randn(*s, device="cuda", generator=gen)  # noqa: E731
    q = F.normalize(f(B, S, H, K), dim=-1).to(dtype).contiguous()
    k = F.normalize(f(B, S, H, K), dim=-1).to(dtype).contiguous()
    v = f(B, S, H, V).to(dtype).contiguous()
    beta = (
        (torch.rand(B, S, H, device="cuda", generator=gen) * 0.9 + 0.05)
        .to(dtype)
        .contiguous()
    )
    st_fp32 = (f(B, H, K, V) * 0.5).contiguous()
    st_q, st_scale = _quantize_int8_state(st_fp32)
    return {
        "q": q,
        "k": k,
        "v": v,
        "beta": beta,
        "state0_q": st_q.contiguous(),
        "state0_scale": st_scale.contiguous(),
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
        got = int8_state_delta_decode(**args)
    except Exception as e:  # noqa: BLE001
        return [f"candidate raised {type(e).__name__}: {e}"]
    if not (isinstance(got, tuple) and len(got) == 3):
        return [
            f"output must be (o, state_out_q, state_out_scale), got "
            f"{type(got).__name__}"
        ]
    go, gq, gs = got
    want_o, want_q, want_scale = int8_state_delta_decode_ref(**ref_args)
    fails = []
    for name, t in args.items():
        if isinstance(t, torch.Tensor) and not torch.equal(t, ref_args[name]):
            fails.append(f"input {name} was mutated in place by the call")
    if go.dtype != want_o.dtype:
        fails.append(f"o dtype {go.dtype}, must be {want_o.dtype} (v's dtype)")
    if gq.dtype != torch.int8:
        fails.append(f"state_out_q dtype {gq.dtype}, must be int8")
    if gs.dtype != torch.float32:
        fails.append(f"state_out_scale dtype {gs.dtype}, must be float32")
    if fails:
        return fails
    dt_name = "bfloat16" if dtype == torch.bfloat16 else "float32"
    fails += _close(go, want_o, *TOL[dt_name], "o")
    # State: compare on the dequantized value with a per-(b,h) quantization slack.
    got_state = gq.float() * gs[:, :, None, None]
    want_state = want_q.float() * want_scale[:, :, None, None]
    s_atol, s_rtol = STATE_TOL
    q_slack = (args["state0_scale"] / 2.0)[:, :, None, None].expand_as(want_state)
    diff = (got_state.float() - want_state.float()).abs()
    lim = s_atol + s_rtol * want_state.float().abs() + q_slack
    bad = int((diff > lim).sum())
    if bad:
        fails.append(
            f"state_out (dequantized): {bad} elements outside tolerance "
            f"(max abs {float(diff.max()):.3e})"
        )
    fails += _close(gs, want_scale, *STATE_TOL, "state_out_scale")
    return fails


def check_determinism():
    args = make(DET_SHAPE, 0)
    runs = [int8_state_delta_decode(**args) for _ in range(4)][1:]
    fails = []
    for i, (o, sq, ss) in enumerate(runs[1:], start=2):
        if not torch.equal(o, runs[0][0]):
            fails.append(f"run {i}: o differs from run 1 at the bit level")
        if not torch.equal(sq, runs[0][1]):
            fails.append(f"run {i}: state_out_q differs from run 1 at the bit level")
        if not torch.equal(ss, runs[0][2]):
            fails.append(
                f"run {i}: state_out_scale differs from run 1 at the bit level"
            )
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
        int8_state_delta_decode(**args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        int8_state_delta_decode(**args)
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
    def _step(state, qt, kt, vt, bt, scale):
        pred = torch.einsum("bhk,bhkv->bhv", kt, state)
        delta = (vt - pred) * bt[:, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", kt, delta)
        y = torch.einsum("bhk,bhkv->bhv", qt * scale, state)
        return state, y

    return torch.compile(_step, dynamic=False)


def _production(args):
    q, k, v, beta = args["q"], args["k"], args["v"], args["beta"]
    K = q.shape[-1]
    scale = K**-0.5
    state = args["state0_q"].float() * args["state0_scale"][:, :, None, None]
    state = state.contiguous()
    step = _compiled_step()
    S = q.shape[1]
    q32, k32, v32, b32 = q.float(), k.float(), v.float(), beta.float()
    ys = []
    for t in range(S):
        state, y = step(state, q32[:, t], k32[:, t], v32[:, t], b32[:, t], scale)
        ys.append(y)
    o = torch.stack(ys, dim=1).to(v.dtype)
    state_q, state_scale = _quantize_int8_state(state)
    return o, state_q, state_scale


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
                f"B={s['B']} S={s['S']} H={s['H']} K={s['K']} V={s['V']}:"
            )
            for f in fails:
                print(f"    {f}")
            return emit(
                {"rows": [], "geomean": None, "correctness_failed": fails, "shape": s},
                1,
            )
        print(
            f"correctness ok ({label}: B={s['B']} S={s['S']} H={s['H']} "
            f"K={s['K']} V={s['V']}, all three outputs)"
        )

    det = check_determinism()
    if det:
        print("\nNOT DETERMINISTIC:")
        for f in det[:8]:
            print(f"    {f}")
        return emit({"rows": [], "geomean": None, "nondeterministic": det}, 1)
    print(
        "determinism ok: three runs after a warmup, all three outputs bitwise identical"
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
        B, S, H, K, V = (s[x] for x in ("B", "S", "H", "K", "V"))
        args = make(s, 0)
        mine, prod = timed_pair(
            lambda a: int8_state_delta_decode(**a), _production, args
        )
        e = args["v"].element_size()
        # Compulsory traffic: q/k/v/beta stream, o written, int8 state (1B/elem)
        # in + out, per-(b,h) scale (fp32) in + out.
        nbytes = (
            2 * B * S * H * K * e
            + 2 * B * S * H * V * e
            + B * S * H * e
            + 2 * B * H * K * V * 1
            + 2 * B * H * 4
        )
        mine_gbps = nbytes / (mine * 1e-3) / 1e9
        prod_gbps = nbytes / (prod * 1e-3) / 1e9
        rows.append(
            {
                "shape": label,
                **{x: s[x] for x in ("B", "S", "H", "K", "V")},
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
            f"{label:4s} B={B:4d} S={S:4d} H={H:3d} K={K:4d} V={V:4d}  "
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
