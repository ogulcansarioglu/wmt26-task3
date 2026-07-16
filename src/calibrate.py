"""Threshold calibration for Task 3: per-LP + global MCC-optimal thresholds.

Protocol (docs/task_facts.md):
  * per LP: stratified 80/20 fit/eval split (no official split exists)
  * sweep score thresholds, maximize MCC on the fit split
  * one global threshold fitted on all pooled fit splits
  * guard rails: LPs with tiny data, a missing class, or a degenerate
    (single-class) prediction on the eval split fall back to the global
    threshold — MCC is zero/undefined on single-class predictions
  * outputs: thresholds.json + results table (md/csv) with per-LP MCC and
    prediction balance for global vs per-LP thresholds

Usage:
    python -m src.calibrate --run runs/dev --model cometkiwi --mode chunked --agg min
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import DEFAULT_SEED, RunManifest


def mcc_sweep(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized MCC for every candidate threshold (midpoints between sorted
    unique scores, plus outer bounds). Prediction rule: score > t -> 1."""
    order = np.argsort(scores, kind="mergesort")
    s, y = scores[order], labels[order].astype(np.int64)
    uniq = np.unique(s)
    if len(uniq) == 1:
        cands = np.array([uniq[0] - 1e-9, uniq[0] + 1e-9])
    else:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        cands = np.concatenate(([uniq[0] - 1e-9], mids, [uniq[-1] + 1e-9]))

    P = int(y.sum())
    N = len(y) - P
    # For threshold t: predictions are 1 for scores > t.
    # cum_pos[i] = gold-positives among the i smallest scores (predicted 0).
    cum_pos = np.concatenate(([0], np.cumsum(y)))
    idx = np.searchsorted(s, cands, side="right")
    fn = cum_pos[idx].astype(np.float64)  # gold 1, predicted 0
    tn = idx - fn  # gold 0, predicted 0
    tp = P - fn
    fp = N - tn
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        mcc = np.where(denom > 0, (tp * tn - fp * fn) / denom, 0.0)
    return cands, mcc


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    cands, mcc = mcc_sweep(scores, labels)
    best = mcc.max()
    winners = np.flatnonzero(mcc >= best - 1e-12)
    # middle of the widest winning plateau -> stable under resampling
    pick = winners[len(winners) // 2]
    return float(cands[pick]), float(best)


def evaluate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import matthews_corrcoef, precision_score, recall_score

    preds = (scores > threshold).astype(int)
    single_class = len(np.unique(preds)) == 1
    return {
        "mcc": 0.0 if single_class else float(matthews_corrcoef(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "pred_pos_rate": float(preds.mean()),
        "gold_pos_rate": float(labels.mean()),
        "single_class_pred": bool(single_class),
        "n": int(len(labels)),
    }


def split_lp(df_lp: pd.DataFrame, holdout: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    if df_lp["error_free"].nunique() < 2 or len(df_lp) < 10:
        return df_lp, df_lp.iloc[0:0]
    fit, ev = train_test_split(
        df_lp, test_size=holdout, random_state=seed, stratify=df_lp["error_free"]
    )
    return fit, ev


def load_scores(run: Path, model: str, mode: str) -> pd.DataFrame:
    score_dir = run / "scores" / model / mode
    files = sorted(score_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no score parquets under {score_dir}; run src.score first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "error_free" not in df.columns or df["error_free"].isna().any():
        raise SystemExit("scores lack gold error_free labels; calibrate needs dev data")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(prog="src.calibrate")
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="chunked")
    parser.add_argument("--agg", default="min", choices=["min", "mean", "wmean"])
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-n", type=int, default=50, help="min fit rows for a per-LP threshold")
    parser.add_argument(
        "--min-class", type=int, default=5, help="min minority-class rows in fit split"
    )
    args = parser.parse_args()

    run = Path(args.run)
    df = load_scores(run, args.model, args.mode)
    score_col = f"score_{args.agg}"
    if score_col not in df.columns:
        raise SystemExit(f"missing column {score_col}")

    fits, evals = {}, {}
    for lp, df_lp in df.groupby("lp"):
        fits[lp], evals[lp] = split_lp(df_lp, args.holdout, args.seed)

    pooled_fit = pd.concat(fits.values(), ignore_index=True)
    non_empty_evals = [e for e in evals.values() if len(e)]
    pooled_eval = (
        pd.concat(non_empty_evals, ignore_index=True) if non_empty_evals else pooled_fit.iloc[0:0]
    )

    g_thr, g_fit_mcc = best_threshold(
        pooled_fit[score_col].to_numpy(), pooled_fit["error_free"].to_numpy()
    )
    global_eval = (
        evaluate(pooled_eval[score_col].to_numpy(), pooled_eval["error_free"].to_numpy(), g_thr)
        if len(pooled_eval)
        else {"mcc": None, "note": "no eval split"}
    )

    per_lp: dict[str, dict] = {}
    table_rows = []
    for lp in sorted(fits):
        fit, ev = fits[lp], evals[lp]
        y_fit = fit["error_free"].to_numpy()
        entry: dict = {"threshold": g_thr, "source": "global_fallback"}
        reason = None
        if len(fit) < args.min_n:
            reason = f"fit split too small ({len(fit)} < {args.min_n})"
        elif min((y_fit == 1).sum(), (y_fit == 0).sum()) < args.min_class:
            reason = "minority class below --min-class in fit split"
        else:
            thr, _fit_mcc = best_threshold(fit[score_col].to_numpy(), y_fit)
            ev_scores = ev[score_col].to_numpy()
            ev_labels = ev["error_free"].to_numpy()
            check = evaluate(ev_scores, ev_labels, thr) if len(ev) else None
            if check and check["single_class_pred"]:
                reason = "per-LP threshold degenerate (single-class predictions on eval)"
            else:
                entry = {"threshold": thr, "source": "per_lp"}
        if reason:
            entry["reason"] = reason
        per_lp[lp] = entry

        if len(ev):
            ev_scores = ev[score_col].to_numpy()
            ev_labels = ev["error_free"].to_numpy()
            m_global = evaluate(ev_scores, ev_labels, g_thr)
            m_lp = evaluate(ev_scores, ev_labels, entry["threshold"])
            table_rows.append(
                {
                    "lp": lp,
                    "n_eval": m_lp["n"],
                    "gold_pos_rate": round(m_lp["gold_pos_rate"], 3),
                    "thr_source": entry["source"],
                    "mcc_global": round(m_global["mcc"], 4),
                    "mcc_per_lp": round(m_lp["mcc"], 4),
                    "pred_pos_global": round(m_global["pred_pos_rate"], 3),
                    "pred_pos_per_lp": round(m_lp["pred_pos_rate"], 3),
                }
            )
        else:
            table_rows.append(
                {
                    "lp": lp,
                    "n_eval": 0,
                    "gold_pos_rate": None,
                    "thr_source": entry["source"],
                    "mcc_global": None,
                    "mcc_per_lp": None,
                    "pred_pos_global": None,
                    "pred_pos_per_lp": None,
                }
            )

    # Task 2 sanity: correlation of the continuous score with gold ESA.
    corr = {}
    if "esa_gold" in df.columns and df["esa_gold"].notna().any():
        from scipy.stats import pearsonr, spearmanr  # scipy ships with scikit-learn

        corr = {
            "pearson": float(pearsonr(df[score_col], df["esa_gold"])[0]),
            "spearman": float(spearmanr(df[score_col], df["esa_gold"])[0]),
        }

    out_dir = run / "calibration" / f"{args.model}_{args.mode}_{args.agg}"
    out_dir.mkdir(parents=True, exist_ok=True)

    mcc_col = [r["mcc_per_lp"] for r in table_rows if r["mcc_per_lp"] is not None]
    mcc_glob = [r["mcc_global"] for r in table_rows if r["mcc_global"] is not None]
    summary = {
        "model": args.model,
        "mode": args.mode,
        "agg": args.agg,
        "seed": args.seed,
        "holdout": args.holdout,
        "prediction_rule": "score > threshold  =>  error-free (1)",
        "global": {"threshold": g_thr, "fit_mcc": g_fit_mcc, "eval": global_eval},
        "per_lp": per_lp,
        "eval_macro_mcc_global": float(np.mean(mcc_glob)) if mcc_glob else None,
        "eval_macro_mcc_per_lp": float(np.mean(mcc_col)) if mcc_col else None,
        "esa_correlation": corr,
    }
    (out_dir / "thresholds.json").write_text(json.dumps(summary, indent=2))

    table = pd.DataFrame(table_rows)
    table.to_csv(out_dir / "results.csv", index=False)
    md = [
        "# Calibration results",
        "",
        f"model=`{args.model}` mode=`{args.mode}` agg=`{args.agg}` seed={args.seed}",
        "",
        f"Global threshold **{g_thr:.4f}** | pooled eval MCC **{global_eval.get('mcc')}** | "
        f"macro eval MCC global **{summary['eval_macro_mcc_global']}** vs per-LP "
        f"**{summary['eval_macro_mcc_per_lp']}**",
        "",
        table.to_markdown(index=False),
    ]
    (out_dir / "results.md").write_text("\n".join(md))

    RunManifest(run).record(
        "calibrate",
        f"{args.model}/{args.mode}/{args.agg}",
        {
            "out_dir": str(out_dir),
            "global_threshold": g_thr,
            "macro_mcc_global": summary["eval_macro_mcc_global"],
            "macro_mcc_per_lp": summary["eval_macro_mcc_per_lp"],
        },
    )

    print(table.to_markdown(index=False))
    print(f"\n[calibrate] global thr {g_thr:.4f}; results -> {out_dir}")


if __name__ == "__main__":
    main()
