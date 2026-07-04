import json
import os

import numpy as np
import pandas as pd

import cv as CV

TARGET = "Exited"
ID = "id"


def main():
    with open(CV.REGISTRY) as f:
        reg = json.load(f)
    with open(f"{CV.ROOT}/experiments/reports/ensemble.json") as f:
        ens = json.load(f)

    y = np.load(f"{CV.PRED_DIR}/_y.npy")
    test_id = np.load(f"{CV.PRED_DIR}/_test_id.npy")

    singles = {n: m["cv_roc_auc"] for n, m in reg.items()}
    best_single = max(singles, key=singles.get)
    ensemble_score = ens["ensemble_cv_roc_auc"]

    use_ensemble = ensemble_score >= singles[best_single]
    if use_ensemble:
        winner = "ensemble"
        winner_score = ensemble_score
        test_pred = np.load(f"{CV.PRED_DIR}/ensemble_test.npy")
    else:
        winner = best_single
        winner_score = singles[best_single]
        test_pred = np.load(f"{CV.PRED_DIR}/{best_single}_test.npy")

    sub = pd.DataFrame({ID: test_id.astype(int), TARGET: test_pred})
    sub.to_csv(f"{CV.ROOT}/submission.csv", index=False)

    classes = {}
    for n, m in reg.items():
        classes.setdefault(m["arch_class"], []).append((n, m["cv_roc_auc"]))

    lines = []
    lines.append("# Auto-Train Final Report — S4E1 Bank Churn\n")
    lines.append(f"- Metric: ROC AUC (maximize), 5-fold StratifiedKFold (seed {CV.SEED})")
    lines.append(f"- Baseline floor (exp_000 logreg): {singles.get('exp_000_logreg', float('nan')):.6f}")
    lines.append(f"- Architecture classes explored: {sorted(classes)}")
    lines.append(f"- Experiments run: {len(reg)}\n")

    lines.append("## Leaderboard (CV ROC AUC)\n")
    lines.append("| rank | experiment | arch class | cv_roc_auc | vs baseline |")
    lines.append("|---|---|---|---|---|")
    base = singles.get("exp_000_logreg", 0.0)
    for rank, (n, s) in enumerate(sorted(singles.items(), key=lambda kv: -kv[1]), 1):
        lines.append(f"| {rank} | {n} | {reg[n]['arch_class']} | {s:.6f} | {s - base:+.6f} |")

    lines.append("\n## Ensemble (greedy Caruana)\n")
    lines.append(f"- Ensemble CV ROC AUC: {ensemble_score:.6f}")
    lines.append(f"- Best single ({best_single}): {singles[best_single]:.6f}")
    lines.append(f"- Uplift vs best single: {ensemble_score - singles[best_single]:+.6f}")
    lines.append("- Weights:")
    for k, v in sorted(ens["weights"].items(), key=lambda kv: -kv[1]):
        lines.append(f"    - {k}: {v:.3f}")

    lines.append("\n## Winner & Submission\n")
    lines.append(f"- Winner: **{winner}** (CV ROC AUC {winner_score:.6f})")
    lines.append(f"- submission.csv rows: {len(sub)}")
    lines.append(f"- prediction range: [{test_pred.min():.5f}, {test_pred.max():.5f}], mean {test_pred.mean():.5f}")

    report = "\n".join(lines) + "\n"
    with open(f"{CV.ROOT}/experiments/reports/FINAL_REPORT.md", "w") as f:
        f.write(report)
    print(report)
    print(f"submission.csv written with {len(sub)} rows; winner={winner} cv={winner_score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
