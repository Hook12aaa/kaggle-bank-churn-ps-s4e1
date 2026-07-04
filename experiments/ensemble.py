import json
import os

import numpy as np

import cv as CV


def load_pool():
    with open(CV.REGISTRY) as f:
        reg = json.load(f)
    y = np.load(f"{CV.PRED_DIR}/_y.npy")
    test_id = np.load(f"{CV.PRED_DIR}/_test_id.npy")
    names, oofs, tests = [], [], []
    for name, meta in reg.items():
        if not meta.get("accepted", True):
            continue
        oof = np.load(f"{CV.PRED_DIR}/{name}_oof.npy")
        tst = np.load(f"{CV.PRED_DIR}/{name}_test.npy")
        names.append(name)
        oofs.append(oof)
        tests.append(tst)
    return reg, y, test_id, names, np.array(oofs), np.array(tests)


def caruana(y, oofs, names, n_iter=60):
    """Greedy forward selection with replacement (Caruana 2004).

    Selection weights come only from OOF AUC, so the blend is chosen without
    ever touching the test set.
    """
    n_models = oofs.shape[0]
    selected = []
    current = np.zeros_like(oofs[0])
    history = []
    for step in range(n_iter):
        best_score, best_j = -1.0, -1
        for j in range(n_models):
            cand = (current * len(selected) + oofs[j]) / (len(selected) + 1)
            s = CV.score(y, cand)
            if s > best_score:
                best_score, best_j = s, j
        selected.append(best_j)
        current = (current * (len(selected) - 1) + oofs[best_j]) / len(selected)
        history.append((names[best_j], best_score))
    counts = {names[j]: selected.count(j) for j in set(selected)}
    weights = {k: v / len(selected) for k, v in counts.items()}
    return weights, history, CV.score(y, current)


def main():
    reg, y, test_id, names, oofs, tests = load_pool()
    weights, history, blend_score = caruana(y, oofs, names)

    name_to_idx = {n: i for i, n in enumerate(names)}
    blend_test = np.zeros(len(test_id))
    blend_oof = np.zeros(len(y))
    for n, w in weights.items():
        blend_test += w * tests[name_to_idx[n]]
        blend_oof += w * oofs[name_to_idx[n]]

    np.save(f"{CV.PRED_DIR}/ensemble_oof.npy", blend_oof)
    np.save(f"{CV.PRED_DIR}/ensemble_test.npy", blend_test)

    singles = {n: reg[n]["cv_roc_auc"] for n in names}
    best_single = max(singles, key=singles.get)
    result = {
        "weights": weights,
        "ensemble_cv_roc_auc": blend_score,
        "best_single": best_single,
        "best_single_cv_roc_auc": singles[best_single],
        "uplift_vs_best_single": blend_score - singles[best_single],
        "singles": singles,
    }
    with open(f"{CV.ROOT}/experiments/reports/ensemble.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
