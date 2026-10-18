#!/usr/bin/env python3
"""Comparison figures and summary tables for pipeline-v2 runs (first --max_cycle generations).

Finds <runs_root>/out_<prefix>_<config>_s<seed>/nngpt/pref_v2 and reads every generation's
metrics.json and candidates.jsonl. Per generation:
  valid           share of samples passing the generator's forward check
  good            not a duplicate and accuracy >= threshold
  novel good      good, and its WL topology was not known before (not in LEMUR, not generated
                  earlier in the run). Logged the same way in every run, gate on or off.
  repeat rate     traced samples whose WL topology was already known
  new topologies  distinct topologies first seen in the run
  niches          descriptor cells (log2 depth, log2 params, residual add, concat,
                  normalization, >= 2 linear/matmul ops) covered by all generated samples
                  (unaffected by which samples get evaluated) and by good architectures.
                  No gate or selection rule optimizes these.
Figures are small multiples, one panel per diversity-mechanism cell; colour = update rule.

  python -m ab.gpt.kto_pipeline.plot_v2 --runs_root . --prefix v2 --max_cycle 21 --out plots_v2_c21
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
NAN = float("nan")
# (rule, selection) -> (label, colour). Slots 1-3 of the reference palette validate all-pairs;
# the untrained baseline is neutral gray. The last two only appear in Tier-2 runs.
SERIES = {
    ("kto", "binary"): ("KTO", "#2a78d6"),
    ("dpo", "binary"): ("DPO", "#eb6834"),
    ("dpo", "graded"): ("DPO + graded selection", "#1baf7a"),
    ("frozen", "binary"): ("Untrained (no training)", MUTED),
    ("kto", "graded"): ("KTO + graded selection", "#6250d6"),
    ("sft", "binary"): ("SFT", "#eda100"),
}
CELLS = [((False, False), "Lexical filter only"), ((True, False), "+ structural archive"),
         ((False, True), "+ LEMUR anchors"), ((True, True), "+ archive + anchors")]
CELL_SHORT = ["none", "archive", "anchors", "both"]


def discover(root: Path, prefix: str, seed: int) -> List[Dict[str, Any]]:
    pat = re.compile(rf"^out_{re.escape(prefix)}_(.+)_s{seed}$")
    runs = []
    for d in sorted(root.glob(f"out_{prefix}_*_s{seed}")):
        m, run_dir = pat.match(d.name), d / "nngpt" / "pref_v2"
        if not m or not (run_dir / "v2_config.json").exists():
            continue
        cfg = json.loads((run_dir / "v2_config.json").read_text(encoding="utf-8"))
        runs.append({
            "name": m.group(1), "dir": run_dir, "cfg": cfg,
            "series": (cfg.get("rule"), cfg.get("selection", "binary")),
            "cell": (bool(cfg.get("struct_archive")), int(cfg.get("anchors_per_generation", 0)) > 0),
            "beta": float(cfg.get("beta", 0.1)),
        })
    return runs


def _niche(d: Dict[str, Any]) -> Tuple:
    def lg(v):
        return int(math.log2(v)) if v and v > 0 else -1
    return (lg(d.get("depth")), lg(d.get("params")), d.get("add", 0) > 0, d.get("cat", 0) > 0,
            d.get("norm", 0) > 0, d.get("linear", 0) >= 2)


def _mean(xs):
    return sum(xs) / len(xs) if xs else NAN


def load_run(run: Dict[str, Any], max_cycle: int) -> None:
    gens, niches_all, niches_good, good_accs = [], set(), set(), []
    cum_good = cum_novel = cum_new = 0
    best_so_far = NAN
    for g in range(1, max_cycle + 1):
        cdir = run["dir"] / f"cycle_{g}"
        try:
            m = json.loads((cdir / "metrics.json").read_text(encoding="utf-8"))
            rows = [json.loads(l) for l in (cdir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
                    if l.strip()]
        except (OSError, ValueError):
            break
        n = int(m.get("generated", 0)) or len(rows) or 1
        accs = [r["acc"] for r in rows if r.get("acc") is not None]
        evaluated = [r for r in rows if "jaccard" in r and not r.get("dup")]
        good = [r for r in rows if r.get("reason") == "good"]
        novel = [r for r in good if not r.get("struct_seen")]
        traced = [r for r in rows if r.get("wl_hash")]
        repeats = sum(1 for r in traced if r.get("struct_seen"))
        cum_good, cum_novel, cum_new = cum_good + len(good), cum_novel + len(novel), cum_new + len(traced) - repeats
        niches_all.update(_niche(r["desc"]) for r in traced if r.get("desc"))
        niches_good.update(_niche(r["desc"]) for r in good if r.get("desc"))
        good_accs += [r["acc"] for r in good]
        if accs:
            best_so_far = max(accs) if math.isnan(best_so_far) else max(best_so_far, max(accs))
        tr = m.get("training", {})
        gens.append({
            "generation": g, "samples": n,
            "valid_pct": 100 * m.get("extractable", 0) / n,
            "train_success_pct": 100 * len(accs) / len(evaluated) if evaluated else NAN,
            "good": len(good), "novel_good": len(novel),
            "cum_good": cum_good, "cum_novel_good": cum_novel,
            "repeat_pct": 100 * repeats / len(traced) if traced else NAN,
            "lemur_match_pct": 100 * sum(1 for r in traced if r.get("struct_seen_lemur")) / len(traced) if traced else NAN,
            "lex_dup_pct": 100 * sum(1 for r in rows if r.get("lex_dup")) / n,
            "new_topologies": len(traced) - repeats, "cum_new_topologies": cum_new,
            "cum_niches_generated": len(niches_all), "cum_niches_good": len(niches_good),
            "best_acc": 100 * max(accs) if accs else NAN,
            "best_so_far": 100 * best_so_far,
            "mean_good_acc": 100 * _mean([r["acc"] for r in good]),
            "median_acc": 100 * statistics.median(accs) if accs else NAN,
            "training_failed": not tr.get("success") and not tr.get("skipped"),
        })
    run["gens"], run["good_accs"] = gens, good_accs


# ── figures ───────────────────────────────────────────────────────────────────

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=7.5, length=2.5)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _label(r: Dict[str, Any]) -> Tuple[str, str]:
    label, colour = SERIES.get(r["series"], (r["name"], INK2))
    if abs(r["beta"] - 0.1) > 1e-9:
        label += f" (beta {r['beta']:g})"
    return label, colour


def _direct_labels(ax, ends: List[Tuple[float, float, str]], x: float) -> None:
    """End-of-line labels, nudged apart vertically so they never collide."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    gap = 0.075 * (hi - lo)
    ends = sorted(ends, key=lambda e: e[0])
    placed: List[float] = []
    for y, _, _ in ends:
        placed.append(y if not placed else max(y, placed[-1] + gap))
    for (y0, text, _), y in zip(ends, placed):
        ax.annotate(text, (x, y0), xytext=(x + 0.35, y), textcoords="data", fontsize=6.8, color=INK2,
                    va="center", annotation_clip=False)


def _header(fig, title: str, handles: Dict[str, Any], height: float) -> float:
    """Title and legend in a fixed band at the top; returns the top of the plotting area."""
    fig.text(0.01, 1 - 0.05 / height, title, fontsize=10, color=INK, ha="left", va="top")
    leg = fig.legend(list(handles.values()), list(handles.keys()), loc="upper left", ncol=len(handles),
                     frameon=False, fontsize=7.5, bbox_to_anchor=(0.005, 1 - 0.24 / height), handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(INK2)
    return 1 - 0.44 / height


def small_multiples(runs, key: str, ylabel: str, title: str, path: Path, max_cycle: int,
                    steps: bool = False) -> None:
    cells = [c for c in CELLS if any(r["cell"] == c[0] for r in runs)]
    ncol = 2 if len(cells) > 1 else 1
    nrow = math.ceil(len(cells) / ncol)
    height = 2.3 * nrow + 0.6
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, height), sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor("white")
    handles, panel_ends = {}, []
    for ax, (cell, ctitle) in zip(axes.flat, cells):
        _style(ax)
        ends = []
        for r in runs:
            if r["cell"] != cell or not r["gens"]:
                continue
            label, colour = _label(r)
            xs = [g["generation"] for g in r["gens"]]
            ys = [g[key] for g in r["gens"]]
            (line,) = ax.plot(xs, ys, color=colour, linewidth=1.8, solid_capstyle="round",
                              drawstyle="steps-post" if steps else "default",
                              linestyle="--" if abs(r["beta"] - 0.1) > 1e-9 else "-")
            handles.setdefault(label, line)
            last = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
            if last:
                ends.append((last[-1][1], label.split(" (")[0].replace(" + graded selection", " graded"), colour))
        ax.set_title(ctitle, fontsize=8.5, color=INK, loc="left")
        ax.set_xlim(1, max_cycle + 4.2)
        ax.set_xticks([t for t in (1, 5, 10, 15, 20, 25) if t <= max_cycle])
        panel_ends.append(ends)
    for ax in axes.flat[len(cells):]:
        ax.set_visible(False)
    for ax, ends in zip(axes.flat, panel_ends):   # after the shared y-limits are final
        _direct_labels(ax, ends, max_cycle)
    for ax in axes[-1]:
        ax.set_xlabel("Generation", fontsize=8, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    top = _header(fig, title, handles, height)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def endpoint_plot(runs, path: Path, max_cycle: int) -> None:
    """Value after the last generation for each mechanism cell: the rule x mechanism view."""
    metrics = [("cum_novel_good", "Novel good architectures"), ("cum_new_topologies", "New topologies"),
               ("cum_niches_generated", "Descriptor niches covered (all samples)"),
               ("best_so_far", "Best accuracy (%)")]
    present = [i for i, (cell, _) in enumerate(CELLS) if any(r["cell"] == cell for r in runs)]
    height = 4.9
    fig, axes = plt.subplots(2, 2, figsize=(7.2, height))
    axes = axes.flat
    fig.patch.set_facecolor("white")
    handles = {}
    for ax, (key, mtitle) in zip(axes, metrics):
        _style(ax)
        by_series: Dict[Tuple, List[Tuple[int, float]]] = {}
        for r in runs:
            if r["gens"]:
                i = next(i for i, (cell, _) in enumerate(CELLS) if cell == r["cell"])
                by_series.setdefault((r["series"], r["beta"]), []).append((present.index(i), r["gens"][-1][key]))
        for (series, beta), pts in by_series.items():
            label, colour = _label({"series": series, "beta": beta, "name": str(series)})
            pts.sort()
            (line,) = ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, linewidth=1.6,
                              marker="o", markersize=5, markeredgecolor="white", markeredgewidth=1.0)
            handles.setdefault(label, line)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([CELL_SHORT[i] for i in present], fontsize=7.5)
        ax.set_xlim(-0.4, len(present) - 0.6)
        ax.set_title(mtitle, fontsize=8, color=INK, loc="left")
    top = _header(fig, f"After generation {max_cycle}, by diversity mechanism", handles, height)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# ── tables ────────────────────────────────────────────────────────────────────

def summarize(r: Dict[str, Any]) -> Dict[str, Any]:
    gs = r["gens"]
    best = max(gs, key=lambda g: -1 if math.isnan(g["best_acc"]) else g["best_acc"])
    c = r["cfg"]
    return {
        "config": r["name"], "rule": c.get("rule"), "selection": c.get("selection"),
        "archive": int(bool(c.get("struct_archive"))), "anchors": int(c.get("anchors_per_generation", 0)),
        "beta": r["beta"], "generations": len(gs),
        "valid_pct": round(_mean([g["valid_pct"] for g in gs]), 1),
        "train_success_pct": round(_mean([g["train_success_pct"] for g in gs if not math.isnan(g["train_success_pct"])]), 1),
        "good": gs[-1]["cum_good"], "novel_good": gs[-1]["cum_novel_good"],
        "new_topologies": gs[-1]["cum_new_topologies"],
        "niches_generated": gs[-1]["cum_niches_generated"], "niches_good": gs[-1]["cum_niches_good"],
        "repeat_pct": round(_mean([g["repeat_pct"] for g in gs if not math.isnan(g["repeat_pct"])]), 1),
        "lemur_match_pct": round(_mean([g["lemur_match_pct"] for g in gs if not math.isnan(g["lemur_match_pct"])]), 1),
        "lex_dup_pct": round(_mean([g["lex_dup_pct"] for g in gs]), 1),
        "best_acc": round(best["best_acc"], 2), "best_gen": best["generation"],
        "mean_good_acc": round(100 * _mean(r["good_accs"]), 2),
        "training_failures": sum(1 for g in gs if g["training_failed"]),
    }


def write_tables(runs, out: Path) -> List[Dict[str, Any]]:
    rows = [summarize(r) for r in runs if r["gens"]]
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    long_cols = ["config"] + list(runs[0]["gens"][0])
    with open(out / "per_generation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=long_cols)
        w.writeheader()
        for r in runs:
            for g in r["gens"]:
                w.writerow({"config": r["name"], **g})
    head = ["config", "generations", "valid_pct", "good", "novel_good", "new_topologies",
            "niches_generated", "niches_good",
            "repeat_pct", "lemur_match_pct", "best_acc", "best_gen", "mean_good_acc", "training_failures"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(r[h]) for h in head) + " |" for r in rows]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline-v2 comparison plots")
    ap.add_argument("--runs_root", default=".")
    ap.add_argument("--prefix", default="v2", help="Run folders out_<prefix>_<config>_s<seed>")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--max_cycle", type=int, default=21)
    ap.add_argument("--out", default=None, help="Default: plots_<prefix>_c<max_cycle>")
    a = ap.parse_args()

    out = Path(a.out or f"plots_{a.prefix}_c{a.max_cycle}")
    out.mkdir(parents=True, exist_ok=True)
    order = {k: i for i, k in enumerate(SERIES)}
    runs = discover(Path(a.runs_root), a.prefix, a.seed)
    if not runs:
        raise SystemExit(f"no runs found as {a.runs_root}/out_{a.prefix}_*_s{a.seed}/nngpt/pref_v2")
    runs.sort(key=lambda r: ([c for c, _ in CELLS].index(r["cell"]), order.get(r["series"], 99), r["beta"]))
    for r in runs:
        load_run(r, a.max_cycle)
        if len(r["gens"]) < a.max_cycle:
            print(f"[plot_v2] WARNING {r['name']}: only {len(r['gens'])}/{a.max_cycle} generations")
    runs = [r for r in runs if r["gens"]]
    dataset = runs[0]["cfg"].get("dataset", "cifar-10").upper()

    figs = [
        ("cum_novel_good", "Cumulative count", "Novel good architectures (topology new, accuracy >= threshold)", "01_novel_good_cumulative"),
        ("good", "Count per generation", "Good architectures per generation (repeats count when the archive is off)", "02_good_per_generation"),
        ("repeat_pct", "% of traced samples", "Structural repeats: topology already in LEMUR or generated earlier", "03_structural_repeat_rate"),
        ("cum_new_topologies", "Cumulative count", "Distinct new topologies generated", "04_new_topologies_cumulative"),
        ("cum_niches_generated", "Cumulative count", "Descriptor niches covered by all generated samples (no gate optimizes these)", "05_niches_generated_cumulative"),
        ("cum_niches_good", "Cumulative count", "Descriptor niches filled by good architectures", "06_niches_good_cumulative"),
        ("best_so_far", "Accuracy (%)", f"Best accuracy so far (1 epoch, {dataset})", "07_best_accuracy_so_far"),
        ("mean_good_acc", "Accuracy (%)", "Mean accuracy of good architectures per generation", "08_mean_good_accuracy"),
        ("valid_pct", "% of samples", "Valid samples (pass the generator's forward check)", "09_valid_rate"),
        ("lex_dup_pct", "% of samples", "Lexical duplicates (Jaccard >= 0.85 to LEMUR or earlier samples)", "10_lexical_duplicate_rate"),
    ]
    for key, ylabel, title, fname in figs:
        small_multiples(runs, key, ylabel, title, out / fname, a.max_cycle, steps=key == "best_so_far")
    endpoint_plot(runs, out / "11_endpoints_by_mechanism", a.max_cycle)
    write_tables(runs, out)

    print(f"[plot_v2] {len(runs)} runs, generations 1-{a.max_cycle} -> {out}/")
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"figures: {', '.join(f[3] for f in figs)}, 11_endpoints_by_mechanism (.png + .pdf)")
    print("tables : summary.csv, summary.md, per_generation.csv")


if __name__ == "__main__":
    main()
