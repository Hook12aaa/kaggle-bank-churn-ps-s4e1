import sys
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

import cv as CV

SEED = CV.SEED


def cv_train(factory, x_tr, x_te, y, folds, scale=False):
    oof = np.zeros(len(x_tr))
    test_pred = np.zeros(len(x_te))
    xtr_v = x_tr.values
    xte_v = x_te.values
    for tr_idx, va_idx in folds:
        Xtr, Xva, Xte = xtr_v[tr_idx], xtr_v[va_idx], xte_v
        if scale:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xva = sc.transform(Xva)
            Xte = sc.transform(Xte)
        model = factory()
        model.fit(Xtr, y[tr_idx])
        oof[va_idx] = model.predict_proba(Xva)[:, 1]
        test_pred += model.predict_proba(Xte)[:, 1] / len(folds)
    return oof, test_pred


def make_experiments():
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier

    spw = 0.788401 / 0.211599

    return {
        "exp_000_logreg": dict(
            arch="linear", scale=True,
            factory=lambda: LogisticRegression(max_iter=2000, C=1.0, random_state=SEED),
        ),
        "exp_001_lgbm": dict(
            arch="gbdt", scale=False,
            factory=lambda: LGBMClassifier(
                n_estimators=900, learning_rate=0.03, num_leaves=63, max_depth=-1,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=2.0, reg_alpha=0.5, min_child_samples=80,
                random_state=SEED, n_jobs=-1, verbosity=-1,
            ),
        ),
        "exp_002_xgb": dict(
            arch="gbdt", scale=False,
            factory=lambda: XGBClassifier(
                n_estimators=900, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, reg_alpha=0.5,
                min_child_weight=5, gamma=0.1, tree_method="hist",
                eval_metric="auc", random_state=SEED, n_jobs=-1,
            ),
        ),
        "exp_003_cat": dict(
            arch="gbdt", scale=False,
            factory=lambda: CatBoostClassifier(
                iterations=1200, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                random_seed=SEED, verbose=0, eval_metric="AUC",
            ),
        ),
        "exp_004_hist": dict(
            arch="gbdt", scale=False,
            factory=lambda: HistGradientBoostingClassifier(
                max_iter=700, learning_rate=0.04, max_leaf_nodes=48,
                l2_regularization=1.0, min_samples_leaf=60,
                random_state=SEED,
            ),
        ),
        "exp_005_extratrees": dict(
            arch="bagging", scale=False,
            factory=lambda: ExtraTreesClassifier(
                n_estimators=600, max_features=0.6, min_samples_leaf=10,
                n_jobs=-1, random_state=SEED,
            ),
        ),
        "exp_006_rf": dict(
            arch="bagging", scale=False,
            factory=lambda: RandomForestClassifier(
                n_estimators=500, max_features=0.5, min_samples_leaf=15,
                n_jobs=-1, random_state=SEED,
            ),
        ),
        "exp_007_mlp": dict(
            arch="neural", scale=True,
            factory=lambda: MLPClassifier(
                hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=512,
                learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                n_iter_no_change=8, random_state=SEED,
            ),
        ),
        "exp_008_lgbm_deep": dict(
            arch="gbdt", scale=False,
            factory=lambda: LGBMClassifier(
                n_estimators=1500, learning_rate=0.02, num_leaves=127, max_depth=9,
                subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
                reg_lambda=5.0, reg_alpha=1.0, min_child_samples=120,
                scale_pos_weight=spw, random_state=SEED, n_jobs=-1, verbosity=-1,
            ),
        ),
    }


def main():
    targets = sys.argv[1:]
    x_tr, x_te, y, test_id, raw_tr, raw_te = CV.load_dataset()
    folds = CV.get_folds(y)
    x_tr, x_te = CV.add_target_encodings(x_tr, x_te, y, raw_tr, raw_te, folds)
    np.save(f"{CV.PRED_DIR}/_test_id.npy", test_id)
    np.save(f"{CV.PRED_DIR}/_y.npy", y)

    exps = make_experiments()
    if targets:
        exps = {k: v for k, v in exps.items() if k in targets}

    for name, spec in exps.items():
        t0 = time.time()
        oof, test_pred = cv_train(spec["factory"], x_tr, x_te, y, folds, scale=spec["scale"])
        s = CV.score(y, oof)
        CV.register(name, spec["arch"], oof, test_pred, s, {"scale": spec["scale"]})
        print(f"{name:24s} arch={spec['arch']:8s} cv_roc_auc={s:.6f}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
