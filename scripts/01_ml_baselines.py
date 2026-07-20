#!/usr/bin/env python3
"""Run conventional ML baselines under two vaccine-history policies.

For exact reproduction of the archived 67-variable/75-variable results, pass the
same cleaned table and feature manifest used in the original run. Without a
manifest, this script provides a transparent generic baseline and will not be
numerically identical to the archived metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

OTHER_VACCINE_COLUMNS = {
    "SHTCVD191_A", "SHTCVD19NM2_A", "SHTPNUEV_A", "SHTPNEUNB_A",
    "SHTSHINGL1_A", "SHINGRIX3_A", "SHTHEPA_A",
}
FLU_LEAKAGE_COLUMNS = {"SHTFLU12M_A", "SHTFLUM_A", "SHTFLUY_A", "vaccinated"}
DEFAULT_DROP = {"WTFA_A", "HHX", "PSTRAT", "PPSU", "SRVY_YR"}


def recode_target(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if set(x.dropna().unique()).issubset({0, 1}):
        return x
    return x.map({1: 1.0, 2: 0.0})


def load_manifest(path: str, scenario: str) -> list[str] | None:
    if not path:
        return None
    obj = json.loads(Path(path).read_text())
    value = obj.get(scenario)
    if not isinstance(value, list):
        raise ValueError(f"Manifest must contain a list at key {scenario!r}")
    return [str(x) for x in value]


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def models(seed: int):
    out = {
        "logistic": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "svm": SVC(probability=True, random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed),
        "knn": KNeighborsClassifier(n_neighbors=15),
    }
    try:
        from xgboost import XGBClassifier
        out["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
            random_state=seed, n_jobs=-1,
        )
    except ImportError:
        pass
    return out


def run_scenario(df: pd.DataFrame, y: pd.Series, features: list[str], scenario: str,
                 seed: int, test_size: float, model_names: set[str] | None = None) -> list[dict]:
    X = df[features].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    rows = []
    available = models(seed)
    if model_names is not None:
        unknown = sorted(model_names - set(available))
        if unknown:
            raise ValueError(f"Unknown or unavailable model names: {unknown}")
        available = {k: v for k, v in available.items() if k in model_names}
    for name, model in available.items():
        pipe = Pipeline([("prep", make_preprocessor(X)), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "scenario": scenario,
            "n_features": len(features),
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, prob),
            "f1_positive": f1_score(y_test, pred),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "random_seed": seed,
        })
        print(rows[-1])
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--target", default="SHTFLU12M_A")
    p.add_argument("--feature_manifest", default="")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--sample_size", type=int, default=0,
                   help="Optional stratified sample size for smoke tests; 0 uses all valid rows.")
    p.add_argument("--models", default="",
                   help="Optional comma-separated subset, e.g. logistic,xgboost.")
    args = p.parse_args()

    df = pd.read_csv(args.data_path, low_memory=False)
    if args.target not in df:
        raise KeyError(f"Target {args.target!r} not found")
    y = recode_target(df[args.target])
    valid = y.isin([0, 1])
    df, y = df.loc[valid].reset_index(drop=True), y.loc[valid].astype(int).reset_index(drop=True)
    if args.sample_size and args.sample_size < len(df):
        selected, _ = train_test_split(
            np.arange(len(df)), train_size=args.sample_size, random_state=args.random_seed, stratify=y
        )
        df = df.iloc[selected].reset_index(drop=True)
        y = y.iloc[selected].reset_index(drop=True)

    requested_models = {x.strip() for x in args.models.split(",") if x.strip()} or None

    all_candidates = [c for c in df.columns if c not in DEFAULT_DROP | FLU_LEAKAGE_COLUMNS | {args.target}]
    rows = []
    for scenario, allow_other in [
        ("without_other_vaccine_history", False),
        ("with_other_vaccine_history", True),
    ]:
        manifest = load_manifest(args.feature_manifest, scenario)
        if manifest is not None:
            features = [c for c in manifest if c in df.columns]
        else:
            features = [c for c in all_candidates if allow_other or c not in OTHER_VACCINE_COLUMNS]
        rows.extend(run_scenario(
            df, y, features, scenario, args.random_seed, args.test_size, requested_models
        ))

    out = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
