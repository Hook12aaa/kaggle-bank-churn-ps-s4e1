import json
import sys

import cv as CV

MAX_ITERATIONS = 15
ARCH_CLASSES_MIN = 3
TIER1_MIN_GAIN = 1e-4


def check():
    with open(CV.REGISTRY) as f:
        reg = json.load(f)

    n_exps = len(reg)
    classes = sorted({m["arch_class"] for m in reg.values()})
    scores = sorted((m["cv_roc_auc"] for m in reg.values()), reverse=True)
    best = scores[0] if scores else float("nan")
    runner_up = scores[1] if len(scores) > 1 else best

    tier2_ok = len(classes) >= ARCH_CLASSES_MIN
    tier1_ok = (best - runner_up) < TIER1_MIN_GAIN and n_exps >= ARCH_CLASSES_MIN + 1
    budget_hit = n_exps >= MAX_ITERATIONS

    if tier2_ok and (tier1_ok or budget_hit):
        verdict = "CONVERGED"
    else:
        verdict = "EXPLORING"

    result = {
        "verdict": verdict,
        "experiments": n_exps,
        "arch_classes": classes,
        "tier2_cross_class_ok": tier2_ok,
        "tier1_within_class_exhausted": tier1_ok,
        "budget_hit": budget_hit,
        "best_cv_roc_auc": best,
        "runner_up_cv_roc_auc": runner_up,
        "top_minus_runner_up": best - runner_up,
    }
    return result


def main():
    result = check()
    with open(f"{CV.ROOT}/experiments/reports/convergence.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
