"""Combine results/eval/<clip>/ into one table: every run relative to per-frame drawing (run A), with bootstrap CIs,
paired differences between flow propagation (C_*) and propagation without flow (BL_*), drift slopes, and the
selection rule.

  python scripts/aggregate.py --clips dog --out results/summary/dev
  python scripts/aggregate.py --clips blackswan camel horsejump-high --out results/summary/test

Unit = one (clip, seed) run. Ratios divide by A's mean over its seeds on the same clip. CIs are 95% percentile
bootstrap over units (empty when there is only one unit).
Drift = OLS slope of the per-frame LPIPS gap to A (A's mean over seeds at the same frame) against steps since the
last keyframe, frames 1..T-1, fitted per unit.
Selection rule (configs/settings.yaml): lowest flicker ratio (--flicker, default chamfer_eval) among runs with lpips ratio <= 1.10 and
clip_q ratio >= 0.95, chosen separately for C_* and BL_*.
Main flicker metric: chamfer_eval. On the dev clip, E_warp (pixel L1) turned out to track the amount of ink: with thin
lines and large motion almost no ink overlaps between pages, so drawing less looks smoother (A stopped at 150
iterations: E_warp 0.84 x A but chamfer 2.0 x A). E_warp and E_warp per unit of ink are still reported.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
METRICS = ["chamfer_eval", "ewarp_eval", "ewarp_per_ink", "ewarp_raft", "lpips", "clip_q", "ink", "iters_per_frame",
           "gpu_s_per_frame"]
LOWER_BETTER = {"chamfer_eval", "ewarp_eval", "ewarp_per_ink", "ewarp_raft", "lpips", "iters_per_frame",
                "gpu_s_per_frame"}
SELECTION = {"lpips": 1.10, "clip_q": 0.95}  # limits of the selection rule
RNG = np.random.default_rng(0)


def boot_ci(x, n=10000):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) < 2:
        return [float("nan"), float("nan")]
    m = x[RNG.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def stat(x):
    x = [v for v in x if np.isfinite(v)]
    return {"mean": float(np.mean(x)) if x else float("nan"), "ci": boot_ci(x), "n": len(x)}


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", nargs="+", required=True)
    p.add_argument("--eval_root", default=str(REPO / "results/eval"))
    p.add_argument("--out", required=True)
    p.add_argument("--flicker", default="chamfer_eval", choices=["chamfer_eval", "ewarp_eval", "ewarp_per_ink"])
    p.add_argument("--pairs", nargs="*", default=[], help="extra paired comparisons 'runX:runY' (X relative to Y)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ratios = defaultdict(lambda: defaultdict(list))   # run -> metric -> [ratio per unit]
    raw = defaultdict(lambda: defaultdict(list))      # run -> metric -> [value per unit]
    unit_ratio = defaultdict(dict)                    # run -> (clip, seed) -> {metric: ratio}
    slopes = defaultdict(list)
    seed_noise = {}
    for clip in a.clips:
        ev = Path(a.eval_root) / clip
        rj = json.loads((ev / "runs.json").read_text())
        seed_noise[clip] = rj.get("A_seed_to_seed")
        runs = rj["runs"]
        for seeds in runs.values():
            for v in seeds.values():
                v["ewarp_per_ink"] = v["ewarp_eval"] / v["ink"]
        A = {m: np.mean([runs["A"][s][m] for s in runs["A"]]) for m in METRICS}
        for r, seeds in runs.items():
            for s, v in seeds.items():
                for m in METRICS:
                    raw[r][m].append(v[m])
                    ratios[r][m].append(v[m] / A[m] if A[m] else float("nan"))
                unit_ratio[r][(clip, s)] = {m: v[m] / A[m] if A[m] else float("nan") for m in METRICS}
        # drift
        pf = read_csv(ev / "per_frame.csv")
        a_lp = defaultdict(list)
        for row in pf:
            if row["run"] == "A":
                a_lp[int(row["frame"])].append(float(row["lpips"]))
        a_lp = {t: np.mean(v) for t, v in a_lp.items()}
        by_unit = defaultdict(list)
        for row in pf:
            t = int(row["frame"])
            if t > 0 and not row["run"].startswith("A"):
                by_unit[(row["run"], row["seed"])].append((int(row["steps_since_anchor"]), float(row["lpips"]) - a_lp[t]))
        for (r, s), pts in by_unit.items():
            x, y = np.array(pts).T
            if np.ptp(x) > 0:
                slopes[r].append(float(np.polyfit(x, y, 1)[0]))

    table = {}
    for r in sorted(ratios):
        table[r] = {"rel_A": {m: stat(ratios[r][m]) for m in METRICS},
                    "abs": {m: stat(raw[r][m]) for m in METRICS},
                    "drift_slope_lpips_gap_per_step": stat(slopes.get(r, []))}

    # paired flow propagation (C_*) vs propagation without flow (BL_*) with the same settings, plus any --pairs
    paired = {}
    todo = [(r, "BL_" + r[2:]) for r in table if r.startswith("C_") and ("BL_" + r[2:]) in table]
    todo += [tuple(x.split(":")) for x in a.pairs if all(y in table for y in x.split(":"))]
    for r, bl in todo:
        common = sorted(set(unit_ratio[r]) & set(unit_ratio[bl]))
        rel = lambda m: [unit_ratio[r][u][m] / unit_ratio[bl][u][m] - 1 for u in common]
        dl = [unit_ratio[r][u]["lpips"] - unit_ratio[bl][u]["lpips"] for u in common]
        paired[f"{r} vs {bl}"] = {"chamfer_rel_change": stat(rel("chamfer_eval")),
                                  "ewarp_rel_change": stat(rel("ewarp_eval")), "lpips_ratio_diff": stat(dl)}

    # selection rule
    choice = {}
    for prefix in ("C_", "BL_"):
        ok = [(table[r]["rel_A"][a.flicker]["mean"], r) for r in table if r.startswith(prefix)
              and table[r]["rel_A"]["lpips"]["mean"] <= SELECTION["lpips"]
              and table[r]["rel_A"]["clip_q"]["mean"] >= SELECTION["clip_q"]]
        choice[prefix.rstrip("_")] = min(ok)[1] if ok else None

    res = {"clips": a.clips, "flicker_metric": a.flicker, "selection_limits": SELECTION, "selection_choice": choice, "A_seed_to_seed": seed_noise,
           "runs": table, "paired_C_vs_BL": paired}
    (out / "summary.json").write_text(json.dumps(res, indent=2))

    def fmt(s, d=3):
        m, (lo, hi) = s["mean"], s["ci"]
        return f"{m:.{d}f}" + ("" if not np.isfinite(lo) else f" [{lo:.{d}f}, {hi:.{d}f}]")

    order = sorted(table, key=lambda r: (not r.startswith("A"), r.startswith("A@"), r))
    lines = [f"# Summary: {', '.join(a.clips)}", "",
             "Run names: A = per-frame (A@<n> = stopped after n iterations), B = warm start, BL_lam<l>_K<K> = "
             "propagation without flow, C_lam<l>_K<K> = flow propagation, BF = warm start + flow (exploratory).", "",
             "Ratios to per-frame drawing (A = 1). Flicker: lower is smoother. LPIPS lower and CLIP higher is better. "
             "Mean over (clip, seed) units, 95% bootstrap CI in brackets when there are 2 or more units.", "",
             f"Main flicker metric: {a.flicker}.", "",
             "| Run | n | Flicker: chamfer | E_warp | E_warp per ink | E_warp with RAFT | LPIPS | CLIP | Iters/frame | "
             "Drift slope (LPIPS gap per frame since keyframe) | Fidelity limits met |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in order:
        t = table[r]["rel_A"]
        passes = t["lpips"]["mean"] <= SELECTION["lpips"] and t["clip_q"]["mean"] >= SELECTION["clip_q"]
        lines.append(f"| {r} | {t['chamfer_eval']['n']} | {fmt(t['chamfer_eval'])} | {fmt(t['ewarp_eval'])} | "
                     f"{fmt(t['ewarp_per_ink'])} | {fmt(t['ewarp_raft'])} | {fmt(t['lpips'])} | {fmt(t['clip_q'])} | "
                     f"{table[r]['abs']['iters_per_frame']['mean']:.0f} | "
                     f"{fmt(table[r]['drift_slope_lpips_gap_per_step'], 4)} | {'fidelity ok' if passes else ''} |")
    lines += ["", "## Paired comparisons (by clip and seed): flow propagation vs propagation without flow at equal settings, "
              "then other pairs", "",
              "| Pair (X vs Y) | Chamfer flicker change (X / Y - 1) | E_warp change | LPIPS ratio difference (X - Y) |",
              "| --- | --- | --- | --- |"]
    for k, v in paired.items():
        lines.append(f"| {k} | {fmt(v['chamfer_rel_change'])} | {fmt(v['ewarp_rel_change'])} | {fmt(v['lpips_ratio_diff'])} |")
    lines += ["", f"Selection rule applied to these clips (used only on the development clip; the settings are "
              f"in configs/settings.yaml): flow propagation = `{choice['C']}`, without flow = `{choice['BL']}`", "",
              f"Per-frame seed-to-seed chamfer (px), the same frame drawn twice: {json.dumps(seed_noise)}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
