"""Analysis artifacts for the system-description paper.

Subcommands (each appends a section under docs/analysis_notes.md figures live
in docs/figures/):
    truncation   per-LP over-limit rates and chunk-count distribution
    aggs         MCC by aggregation strategy (min vs mean vs wmean)
    distributions  score histograms by gold label, per LP
    compare      side-by-side eval MCC of two calibrations (e.g. two models,
                 or chunked vs truncate)

Usage examples:
    python -m src.analyze truncation --run runs/dev --model cometkiwi --mode chunked
    python -m src.analyze compare --a runs/dev/calibration/cometkiwi_chunked_min \
        --b runs/dev/calibration/cometkiwi_truncate_min --label-a chunked --label-b truncate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.calibrate import best_threshold, evaluate, load_scores, split_lp
from src.common import DEFAULT_SEED

FIG_DIR = Path("docs/figures")


def cmd_truncation(args: argparse.Namespace) -> None:
    df = load_scores(Path(args.run), args.model, args.mode)
    g = (
        df.groupby("lp")
        .agg(
            n=("segment_id", "size"),
            over_limit_rate=("over_limit", "mean"),
            mean_chunks=("n_chunks", "mean"),
            max_chunks=("n_chunks", "max"),
            oversized=("oversized_chunk", "sum"),
            median_joint_tokens=("joint_tok_len", "median"),
        )
        .sort_values("over_limit_rate", ascending=False)
    )
    print(g.round(3).to_markdown())
    out = Path(f"docs/truncation_{args.model}_{args.mode}.md")
    out.write_text(
        f"# Truncation exposure ({args.model}, {args.mode})\n\n" + g.round(3).to_markdown() + "\n"
    )
    print(f"[analyze] -> {out}")


def cmd_aggs(args: argparse.Namespace) -> None:
    """Which chunk aggregation calibrates best? (expects chunked scores)"""
    df = load_scores(Path(args.run), args.model, "chunked")
    rows = []
    for agg in ("min", "mean", "wmean"):
        col = f"score_{agg}"
        fits, evals = {}, {}
        for lp, df_lp in df.groupby("lp"):
            fits[lp], evals[lp] = split_lp(df_lp, 0.2, DEFAULT_SEED)
        fit = pd.concat(fits.values(), ignore_index=True)
        ev = pd.concat([e for e in evals.values() if len(e)], ignore_index=True)
        thr, _ = best_threshold(fit[col].to_numpy(), fit["error_free"].to_numpy())
        m = evaluate(ev[col].to_numpy(), ev["error_free"].to_numpy(), thr)
        rows.append(
            {
                "agg": agg,
                "global_thr": round(thr, 4),
                "eval_mcc": round(m["mcc"], 4),
                "pred_pos_rate": round(m["pred_pos_rate"], 3),
            }
        )
    table = pd.DataFrame(rows)
    print(table.to_markdown(index=False))


def cmd_distributions(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_scores(Path(args.run), args.model, args.mode)
    col = f"score_{args.agg}"
    lps = sorted(df["lp"].unique())
    ncols = 4
    nrows = (len(lps) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), squeeze=False)
    for ax, lp in zip(axes.flat, lps, strict=False):
        sub = df[df["lp"] == lp]
        for label, color in ((1, "tab:green"), (0, "tab:red")):
            vals = sub.loc[sub["error_free"] == label, col]
            if len(vals):
                ax.hist(vals, bins=30, alpha=0.55, color=color, label=f"free={label}")
        ax.set_title(f"{lp} (n={len(sub)})", fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(lps) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"dist_{args.model}_{args.mode}_{args.agg}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[analyze] -> {out}")


def cmd_compare(args: argparse.Namespace) -> None:
    a = json.loads((Path(args.a) / "thresholds.json").read_text())
    b = json.loads((Path(args.b) / "thresholds.json").read_text())
    rows = []
    for name, blob in ((args.label_a, a), (args.label_b, b)):
        rows.append(
            {
                "system": name,
                "model": blob["model"],
                "mode": blob["mode"],
                "agg": blob["agg"],
                "global_thr": round(blob["global"]["threshold"], 4),
                "pooled_eval_mcc": blob["global"]["eval"].get("mcc"),
                "macro_mcc_global_thr": blob.get("eval_macro_mcc_global"),
                "macro_mcc_per_lp_thr": blob.get("eval_macro_mcc_per_lp"),
                "spearman_vs_esa": (blob.get("esa_correlation") or {}).get("spearman"),
            }
        )
    print(pd.DataFrame(rows).to_markdown(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(prog="src.analyze")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("truncation")
    p.add_argument("--run", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--mode", default="chunked")
    p.set_defaults(func=cmd_truncation)

    p = sub.add_parser("aggs")
    p.add_argument("--run", required=True)
    p.add_argument("--model", required=True)
    p.set_defaults(func=cmd_aggs)

    p = sub.add_parser("distributions")
    p.add_argument("--run", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--mode", default="chunked")
    p.add_argument("--agg", default="min")
    p.set_defaults(func=cmd_distributions)

    p = sub.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
