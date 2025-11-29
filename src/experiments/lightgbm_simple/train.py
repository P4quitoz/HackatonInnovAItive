from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.config import (
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from src.common.evaluation import mae, rmse, nde, sae

# =====================================================
# EXPERIMENT CONFIG
# =====================================================

EXP_NAME = "lightgbm_simple"
EXPORT_DIR = PROJECT_ROOT / "exports" / EXP_NAME
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = EXPORT_DIR / "model_final.txt"
ARTIFACT_PATH = EXPORT_DIR / "artifact.json"

# A bit stronger / more flexible model than before
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 256,
    "max_depth": -1,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "n_estimators": 3000,  # fixed number of trees, no early stopping
    "n_jobs": -1,
}


def main():
    print(f"Loading train features from: {TRAIN_FEATURES_PATH}")
    X_full = pd.read_parquet(TRAIN_FEATURES_PATH)
    print(f"Loading train target from:   {TRAIN_TARGET_PATH}")
    y_full = pd.read_parquet(TRAIN_TARGET_PATH)

    # target as 1D float array
    y = y_full.values.reshape(-1).astype(float)

    # Remove meta columns from features
    drop_cols = [c for c in ["home_id", "datetime"] if c in X_full.columns]
    feature_cols = [c for c in X_full.columns if c not in drop_cols]
    X = X_full[feature_cols]

    print("Total samples :", X.shape[0])
    print("Num features  :", X.shape[1])
    if "home_id" in X_full.columns:
        print("Homes in data :", X_full["home_id"].nunique())

    # =================================================
    # Train/validation split (simple, leaderboard-style)
    # =================================================
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    print("\nTrain samples:", X_train.shape[0])
    print("Valid samples:", X_valid.shape[0])

    # =================================================
    # Train LightGBM
    # =================================================
    print("\nTraining LightGBM model...")
    print("Params:")
    for k, v in LGBM_PARAMS.items():
        print(f"  {k}: {v}")

    model = lgb.LGBMRegressor(
        **LGBM_PARAMS,
        random_state=RANDOM_SEED,
    )

    model.fit(X_train, y_train)

    # =================================================
    # Validation metrics
    # =================================================
    print("\nEvaluating on validation split...")
    y_pred = model.predict(X_valid)
    # clip to non-negative (fridge power cannot be < 0)
    y_pred = np.clip(y_pred, 0.0, None)

    m_mae = mae(y_valid, y_pred)
    m_rmse = rmse(y_valid, y_pred)
    m_nde = nde(y_valid, y_pred)
    m_sae = sae(y_valid, y_pred)

    print("\n=== Simple split metrics (validation) ===")
    print(f"MAE : {m_mae:.6f}")
    print(f"RMSE: {m_rmse:.6f}")
    print(f"NDE : {m_nde:.6f}")
    print(f"SAE : {m_sae:.6f}")

    # =================================================
    # Save model
    # =================================================
    print(f"\nSaving model to: {MODEL_PATH}")
    model.booster_.save_model(str(MODEL_PATH))

    # =================================================
    # Save artifact (for reproducibility)
    # =================================================
    artifact = {
        "experiment": EXP_NAME,
        "model_type": "LightGBMRegressor",
        "params": LGBM_PARAMS,
        "random_state": RANDOM_SEED,
        "feature_cols": feature_cols,
        "metrics": {
            "val_mae": float(m_mae),
            "val_rmse": float(m_rmse),
            "val_nde": float(m_nde),
            "val_sae": float(m_sae),
        },
        "data_shape": {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        },
        "paths": {
            "model_final": str(MODEL_PATH),
        },
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Saved artifact to: {ARTIFACT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
