import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = "Exited"
ID = "id"

RAW_CATEGORICAL = ["Geography", "Gender"]
FREQ_ENCODE = ["Surname", "CustomerId", "Geography", "Gender"]
TARGET_ENCODE = ["Surname", "CustomerId", "Geography", "Gender", "GeoGender", "AgeBin"]


def load_raw():
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test.csv"))
    return tr, te


def _engineer(df):
    df = df.copy()
    df["GeoGender"] = df["Geography"].astype(str) + "_" + df["Gender"].astype(str)
    df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1.0)
    df["BalancePerProduct"] = df["Balance"] / df["NumOfProducts"]
    df["AgePerTenure"] = df["Age"] / (df["Tenure"] + 1.0)
    df["CreditPerAge"] = df["CreditScore"] / df["Age"]
    df["TenureByAge"] = df["Tenure"] / df["Age"]
    df["ProductsXActive"] = df["NumOfProducts"] * df["IsActiveMember"]
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 30, 40, 50, 60, 200], labels=False).astype(int)
    df["IsSenior"] = (df["Age"] >= 60).astype(int)
    df["SurnameLen"] = df["Surname"].astype(str).str.len()
    df["HasBalanceAndCard"] = ((df["Balance"] > 0) & (df["HasCrCard"] == 1)).astype(int)
    return df


def build_features():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).values
    n_tr = len(tr)

    full = pd.concat([tr.drop(columns=[TARGET]), te], axis=0, ignore_index=True)
    full = _engineer(full)

    for col in FREQ_ENCODE:
        counts = full[col].map(full[col].value_counts())
        full[f"{col}_freq"] = counts.astype(float)

    for col in RAW_CATEGORICAL + ["GeoGender"]:
        full[col] = full[col].astype("category").cat.codes

    drop_cols = [ID, "Surname"]
    feat = full.drop(columns=drop_cols)

    x_tr = feat.iloc[:n_tr].reset_index(drop=True)
    x_te = feat.iloc[n_tr:].reset_index(drop=True)

    test_id = te[ID].values
    return x_tr, x_te, y, test_id


def categorical_columns():
    return [c for c in RAW_CATEGORICAL] + ["GeoGender", "AgeBin"]
