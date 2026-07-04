import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import features as F

SEED = 42
N_SPLITS = 5
ROOT = F.ROOT
PRED_DIR = os.path.join(ROOT, "experiments/preds")
REGISTRY = os.path.join(ROOT, "experiments/registry.json")
TARGET_ENCODE_COLS = ["Surname", "CustomerId", "GeoGender_raw", "AgeBin"]


def get_folds(y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def score(y, pred):
    return float(roc_auc_score(y, pred))


def fold_target_encode(train_keys, valid_keys, test_keys, y_train, global_mean, smoothing=20.0):
    df = pd.DataFrame({"k": train_keys, "y": y_train})
    agg = df.groupby("k")["y"].agg(["mean", "count"])
    smooth = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
    mapping = smooth.to_dict()
    enc_valid = pd.Series(valid_keys).map(mapping).fillna(global_mean).values
    enc_test = pd.Series(test_keys).map(mapping).fillna(global_mean).values
    return enc_valid, enc_test


def add_target_encodings(x_tr, x_te, y, raw_tr, raw_te, folds):
    """Fold-safe mean target encoding for high-cardinality keys.

    Returns OOF-encoded train columns (no leakage) and test columns averaged
    over folds. raw_tr/raw_te carry the original (pre-codes) key values.
    """
    global_mean = float(np.mean(y))
    keys = ["Surname", "CustomerId", "GeoGender", "AgeBin"]
    te_train = {k: np.zeros(len(x_tr)) for k in keys}
    te_test = {k: np.zeros(len(x_te)) for k in keys}

    for tr_idx, va_idx in folds:
        for k in keys:
            enc_va, enc_te = fold_target_encode(
                raw_tr[k].values[tr_idx],
                raw_tr[k].values[va_idx],
                raw_te[k].values,
                y[tr_idx],
                global_mean,
            )
            te_train[k][va_idx] = enc_va
            te_test[k] += enc_te / len(folds)

    for k in keys:
        x_tr[f"{k}_te"] = te_train[k]
        x_te[f"{k}_te"] = te_test[k]
    return x_tr, x_te


def load_dataset():
    """Return modeling matrices plus a raw-key frame (used only by add_target_encodings)."""
    x_tr, x_te, y, test_id = F.build_features()
    tr_raw = pd.read_csv("train.csv")
    te_raw = pd.read_csv("test.csv")
    raw_tr = pd.DataFrame(
        {
            "Surname": tr_raw["Surname"].astype(str).values,
            "CustomerId": tr_raw["CustomerId"].values,
            "GeoGender": (tr_raw["Geography"].astype(str) + "_" + tr_raw["Gender"].astype(str)).values,
            "AgeBin": pd.cut(tr_raw["Age"], bins=[0, 30, 40, 50, 60, 200], labels=False).values,
        }
    )
    raw_te = pd.DataFrame(
        {
            "Surname": te_raw["Surname"].astype(str).values,
            "CustomerId": te_raw["CustomerId"].values,
            "GeoGender": (te_raw["Geography"].astype(str) + "_" + te_raw["Gender"].astype(str)).values,
            "AgeBin": pd.cut(te_raw["Age"], bins=[0, 30, 40, 50, 60, 200], labels=False).values,
        }
    )
    return x_tr, x_te, y, test_id, raw_tr, raw_te


def register(name, arch_class, oof, test_pred, cv, params, accepted=True):
    os.makedirs(PRED_DIR, exist_ok=True)
    np.save(f"{PRED_DIR}/{name}_oof.npy", oof)
    np.save(f"{PRED_DIR}/{name}_test.npy", test_pred)
    reg = {}
    if os.path.exists(REGISTRY):
        with open(REGISTRY) as f:
            reg = json.load(f)
    reg[name] = {
        "arch_class": arch_class,
        "cv_roc_auc": float(cv),
        "params": params,
        "accepted": bool(accepted),
    }
    with open(REGISTRY, "w") as f:
        json.dump(reg, f, indent=2)
    return reg
