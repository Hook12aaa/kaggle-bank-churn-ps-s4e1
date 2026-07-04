import json
import sys

import numpy as np
import pandas as pd

TRAIN = "train.csv"
TEST = "test.csv"
TARGET = "Exited"
ID = "id"


def main():
    tr = pd.read_csv(TRAIN)
    te = pd.read_csv(TEST)
    report = {"checks": {}, "status": "PASS", "concerns": []}

    feature_cols = [c for c in tr.columns if c not in (TARGET,)]

    miss_tr = tr.isna().sum()
    miss_te = te.isna().sum()
    report["checks"]["missing_values"] = {
        "train_cols_with_na": {k: int(v) for k, v in miss_tr[miss_tr > 0].items()},
        "test_cols_with_na": {k: int(v) for k, v in miss_te[miss_te > 0].items()},
    }

    report["checks"]["duplicate_rows_excluding_id"] = int(
        tr.drop(columns=[ID]).duplicated().sum()
    )

    pos_rate = float(tr[TARGET].mean())
    report["checks"]["target"] = {
        "present": TARGET in tr.columns,
        "n_classes": int(tr[TARGET].nunique()),
        "positive_rate": pos_rate,
        "imbalanced": pos_rate < 0.1 or pos_rate > 0.9,
    }
    if report["checks"]["target"]["imbalanced"]:
        report["concerns"].append("target class imbalance")

    const_cols = [c for c in feature_cols if tr[c].nunique(dropna=False) <= 1]
    report["checks"]["constant_columns"] = const_cols

    only_train = sorted(set(tr.columns) - set(te.columns) - {TARGET})
    only_test = sorted(set(te.columns) - set(tr.columns))
    report["checks"]["column_consistency"] = {
        "only_in_train": only_train,
        "only_in_test": only_test,
    }
    if only_train or only_test:
        report["concerns"].append("train/test feature mismatch")

    report["checks"]["id_uniqueness"] = {
        "train_unique": int(tr[ID].is_unique),
        "test_unique": int(te[ID].is_unique),
        "train_test_overlap": int(len(set(tr[ID]) & set(te[ID]))),
    }

    cat_cols = [c for c in feature_cols if tr[c].dtype == object or str(tr[c].dtype) == "str"]
    high_card = {c: int(tr[c].nunique()) for c in cat_cols if tr[c].nunique() > 50}
    report["checks"]["categoricals"] = {
        "categorical_columns": cat_cols,
        "high_cardinality": high_card,
    }

    dtype_mismatch = {
        c: [str(tr[c].dtype), str(te[c].dtype)]
        for c in te.columns
        if c in tr.columns and str(tr[c].dtype) != str(te[c].dtype)
    }
    report["checks"]["dtype_consistency"] = dtype_mismatch

    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(tr[c]) and c != ID]
    corrs = {}
    for c in num_cols:
        corr = np.corrcoef(tr[c].fillna(tr[c].median()), tr[TARGET])[0, 1]
        corrs[c] = float(corr)
    report["checks"]["target_correlations"] = corrs
    # corr ~1.0 with the target means the feature encodes the label (leakage), not signal
    leaky = {c: v for c, v in corrs.items() if abs(v) > 0.98}
    if leaky:
        report["concerns"].append(f"possible leakage: {leaky}")

    report["shapes"] = {"train": list(tr.shape), "test": list(te.shape)}
    if report["concerns"]:
        report["status"] = "PASS_WITH_CONCERNS"

    with open("experiments/reports/data_validation.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
