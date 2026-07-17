"""Stage C ensemble: logistic regression over both models' signals.

Features: CometKiwi chunked min-score, xCOMET-XL chunked min-score,
severity/confidence-weighted predicted-span mass (xCOMET), log segment
length, and language-pair one-hots (unseen LPs at test time get all-zero
one-hots, i.e. the base operating point).

Kept per the Stage C gate: on held-out dev it beats both single models
(pooled MCC ~0.32 vs 0.17; per-LP macro ~0.22 vs 0.17; stable over 4 split
seeds — see docs/analysis_notes.md §4).

Subcommands:
    train    fit on ALL dev rows, persist transparent coefficients JSON,
             and report a per-seed holdout table (the honest estimate)
    predict  apply a trained ensemble to score parquets, emitting standard
             scores/ensemble/chunked/<lp>.parquet with the probability in
             score_min/mean/wmean — downstream calibrate/submit work as-is

Usage:
    python -m src.ensemble train   --run runs/dev --out runs/dev/ensemble
    python -m src.ensemble predict --run runs/<test> --model-json runs/dev/ensemble/model.json
    python -m src.calibrate --run runs/dev --model ensemble --mode chunked --agg min
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibrate import best_threshold, evaluate, split_lp
from src.common import DEFAULT_SEED, RunManifest

SEVERITY_WEIGHTS = {"minor": 1.0, "major": 3.0, "critical": 7.0}
FEATURES = ["score_min_kiwi", "score_min_xl", "wsum", "log_len"]


def span_mass(payload: str | None) -> float:
    if payload is None:
        return 0.0
    return sum(
        s.get("confidence", 0.0) * SEVERITY_WEIGHTS.get(s.get("severity"), 3.0)
        for chunk in json.loads(payload)
        for s in (chunk or [])
    )


def load_joined(run: Path) -> pd.DataFrame:
    def load(model: str) -> pd.DataFrame:
        files = sorted((run / "scores" / model / "chunked").glob("*.parquet"))
        if not files:
            raise SystemExit(f"no chunked scores for {model} under {run}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    kiwi, xl = load("cometkiwi"), load("xcomet-xl")
    xl = xl.copy()
    xl["wsum"] = xl["error_spans"].map(span_mass)
    keep_kiwi = ["segment_id", "lp", "doc_id", "system", "score_min", "joint_tok_len"]
    gold = [c for c in ("error_free", "esa_gold", "n_errors_total") if c in kiwi.columns]
    joined = kiwi[keep_kiwi + gold].merge(
        xl[["segment_id", "score_min", "wsum"]],
        on="segment_id",
        suffixes=("_kiwi", "_xl"),
    )
    joined["log_len"] = np.log1p(joined["joint_tok_len"])
    return joined


def design_matrix(df: pd.DataFrame, lp_columns: list[str] | None = None):
    X = pd.get_dummies(df[FEATURES + ["lp"]], columns=["lp"], dtype=float)
    if lp_columns is None:
        lp_columns = sorted(X.columns)
    X = X.reindex(columns=lp_columns, fill_value=0.0)
    return X, lp_columns


def fit_lr(X: pd.DataFrame, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)


def cmd_train(args: argparse.Namespace) -> None:
    run = Path(args.run)
    df = load_joined(run)
    if "error_free" not in df.columns or df["error_free"].isna().all():
        raise SystemExit("training requires gold error_free labels (dev data)")
    df = df[df["error_free"].notna()].copy()
    df["error_free"] = df["error_free"].astype(int)

    # Honest performance estimate: repeated holdout, never reported from train fit
    report_rows = []
    for seed in (DEFAULT_SEED, 29, 47, 101):
        fits, evals = {}, {}
        for lp, g in df.groupby("lp"):
            fits[lp], evals[lp] = split_lp(g, 0.2, seed)
        fit_all = pd.concat(fits.values())
        ev_all = pd.concat([e for e in evals.values() if len(e)])
        X_fit, lp_cols = design_matrix(fit_all)
        lr = fit_lr(X_fit, fit_all["error_free"].to_numpy())
        p_fit = lr.predict_proba(X_fit)[:, 1]
        p_ev = lr.predict_proba(design_matrix(ev_all, lp_cols)[0])[:, 1]
        gthr, _ = best_threshold(p_fit, fit_all["error_free"].to_numpy())
        pooled = evaluate(p_ev, ev_all["error_free"].to_numpy(), gthr)
        per = []
        fit_all = fit_all.assign(p=p_fit)
        ev_all = ev_all.assign(p=p_ev)
        for lp, e_ in ev_all.groupby("lp"):
            f_ = fit_all[fit_all["lp"] == lp]
            if not len(e_) or f_["error_free"].nunique() < 2:
                continue
            thr, _ = best_threshold(f_["p"].to_numpy(), f_["error_free"].to_numpy())
            per.append(evaluate(e_["p"].to_numpy(), e_["error_free"].to_numpy(), thr)["mcc"])
        report_rows.append(
            {
                "seed": seed,
                "pooled_eval_mcc": round(pooled["mcc"], 4),
                "macro_mcc_per_lp": round(float(np.mean(per)), 4),
            }
        )
    report = pd.DataFrame(report_rows)
    print(report.to_markdown(index=False))

    # Final artifact: fit on ALL dev rows
    X_all, lp_cols = design_matrix(df)
    lr = fit_lr(X_all, df["error_free"].to_numpy())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "features": FEATURES,
        "lp_columns": lp_cols,
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "severity_weights": SEVERITY_WEIGHTS,
        "class_weight": "balanced",
        "trained_on": str(run),
        "n_rows": int(len(df)),
        "holdout_report": report_rows,
    }
    (out_dir / "model.json").write_text(json.dumps(model, indent=2))
    report.to_csv(out_dir / "holdout_report.csv", index=False)
    RunManifest(run).record(
        "ensemble", "train", {"out": str(out_dir), "n_rows": len(df), "report": report_rows}
    )
    print(f"[ensemble] trained on {len(df):,} rows -> {out_dir}/model.json")


def predict_proba(df: pd.DataFrame, model: dict) -> np.ndarray:
    X, _ = design_matrix(df, model["lp_columns"])
    z = X.to_numpy() @ np.asarray(model["coef"]) + model["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def cmd_predict(args: argparse.Namespace) -> None:
    run = Path(args.run)
    model = json.loads(Path(args.model_json).read_text())
    df = load_joined(run)
    df["p"] = predict_proba(df, model)
    unseen = sorted(
        set(df["lp"]) - {c.removeprefix("lp_") for c in model["lp_columns"] if c.startswith("lp_")}
    )
    if unseen:
        print(
            f"[ensemble] NOTE: {len(unseen)} LPs unseen in training (base operating point): {unseen}"
        )
    out_dir = run / "scores" / "ensemble" / "chunked"
    out_dir.mkdir(parents=True, exist_ok=True)
    passthrough = [
        c
        for c in (
            "lp",
            "doc_id",
            "system",
            "segment_id",
            "error_free",
            "esa_gold",
            "n_errors_total",
            "joint_tok_len",
        )
        if c in df.columns
    ]
    for lp, g in df.groupby("lp"):
        out = g[passthrough].copy()
        out["score_min"] = g["p"]
        out["score_mean"] = g["p"]
        out["score_wmean"] = g["p"]
        out["n_chunks"] = 1
        out["over_limit"] = False
        out["oversized_chunk"] = False
        out.to_parquet(out_dir / f"{lp}.parquet", index=False)
    RunManifest(run).record("ensemble", "predict", {"model": args.model_json, "n_rows": len(df)})
    print(f"[ensemble] wrote probability parquets for {df['lp'].nunique()} LPs -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="src.ensemble")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--run", required=True)
    p.add_argument("--model-json", required=True)
    p.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
