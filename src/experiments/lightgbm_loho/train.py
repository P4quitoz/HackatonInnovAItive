from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common.config import (
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
    LGBM_DEFAULT_PARAMS,
)
from src.common.evaluation import mae, rmse, nde, sae, loho_split

# =====================================================
# EXPERIMENT CONFIG
# =====================================================

EXP_NAME = "lightgbm_loho"
EXPORT_DIR = PROJECT_ROOT / "exports" / EXP_NAME
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

PER_HOME_METRICS_CSV = EXPORT_DIR / "loho_metrics_per_home.csv"
SUMMARY_JSON = EXPORT_DIR / "loho_summary.json"
FINAL_MODEL_PATH = EXPORT_DIR / "model_all_homes.txt"


def make_model_params() -> dict:
    """
    Start from LGBM_DEFAULT_PARAMS and add LOHO-specific settings.
    """
    params = dict(LGBM_DEFAULT_PARAMS)
    # fixed number of trees (no early stopping here)
    params.setdefault("n_estimators", 2000)
    params.setdefault("n_jobs", -1)
    return params


def main():
    print(f"[LOHO] Loading train features from: {TRAIN_FEATURES_PATH}")
    X_full = pd.read_parquet(TRAIN_FEATURES_PATH)

    print(f"[LOHO] Loading train target from:   {TRAIN_TARGET_PATH}")
    y_full = pd.read_parquet(TRAIN_TARGET_PATH)

    # Target as flat float array
    y = y_full.values.reshape(-1).astype(float)

    if "home_id" not in X_full.columns:
        raise ValueError(
            "LOHO requires 'home_id' column in train features. "
            "Make sure your feature builder keeps home_id."
        )

    print("\n[LOHO] Dataset overview")
    print("------------------------")
    print("Total samples :", X_full.shape[0])
    print("Num features  :", X_full.shape[1])
    print("Num homes     :", X_full["home_id"].nunique())

    # =================================================
    # LOHO CROSS-VALIDATION
    # =================================================
    per_home_records = []
    y_true_all = []
    y_pred_all = []

    params = make_model_params()
    print("\n[LOHO] Using LightGBM params:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    for home, X_train, X_valid, y_train, y_valid in loho_split(
        X_full, y, home_col="home_id"
    ):
        print(f"\n[LOHO] Home {home}:")
        print("  train samples:", X_train.shape[0])
        print("  valid samples:", X_valid.shape[0])

        # Drop meta columns for model input
        drop_cols = [c for c in ["home_id", "datetime"] if c in X_train.columns]
        feature_cols = [c for c in X_train.columns if c not in drop_cols]

        X_train_model = X_train[feature_cols]
        X_valid_model = X_valid[feature_cols]

        model = lgb.LGBMRegressor(
            **params,
            random_state=RANDOM_SEED,
        )

        # Fit on all homes except the left-out one
        model.fit(X_train_model, y_train)

        # Predict on left-out home
        y_pred = model.predict(X_valid_model)
        # Fridge power cannot be negative
        y_pred = np.clip(y_pred, 0.0, None)

        # Collect for global metrics
        y_true_all.append(np.asarray(y_valid))
        y_pred_all.append(np.asarray(y_pred))

        # Per-home metrics
        m_mae = mae(y_valid, y_pred)
        m_rmse = rmse(y_valid, y_pred)
        m_nde = nde(y_valid, y_pred)
        m_sae = sae(y_valid, y_pred)

        print("  MAE :", m_mae)
        print("  RMSE:", m_rmse)
        print("  NDE :", m_nde)
        print("  SAE :", m_sae)

        per_home_records.append(
            {
                "home_id": home,
                "n_train": int(X_train.shape[0]),
                "n_valid": int(X_valid.shape[0]),
                "mae": float(m_mae),
                "rmse": float(m_rmse),
                "nde": float(m_nde),
                "sae": float(m_sae),
            }
        )

    # -------------------------------------------------
    # Aggregate global metrics over all homes
    # -------------------------------------------------
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    global_mae = mae(y_true_all, y_pred_all)
    global_rmse = rmse(y_true_all, y_pred_all)
    global_nde = nde(y_true_all, y_pred_all)
    global_sae = sae(y_true_all, y_pred_all)

    print("\n[LOHO] === Global metrics over all homes (concatenated) ===")
    print(f"MAE : {global_mae:.6f}")
    print(f"RMSE: {global_rmse:.6f}")
    print(f"NDE : {global_nde:.6f}")
    print(f"SAE : {global_sae:.6f}")

    # Save per-home metrics
    df_metrics = pd.DataFrame(per_home_records).sort_values("home_id")
    df_metrics.to_csv(PER_HOME_METRICS_CSV, index=False)
    print(f"\n[LOHO] Saved per-home metrics to: {PER_HOME_METRICS_CSV}")

    # Save summary JSON
    summary = {
        "experiment": EXP_NAME,
        "params": params,
        "random_state": RANDOM_SEED,
        "n_samples": int(X_full.shape[0]),
        "n_features": int(X_full.shape[1]),
        "n_homes": int(X_full["home_id"].nunique()),
        "global_metrics": {
            "mae": float(global_mae),
            "rmse": float(global_rmse),
            "nde": float(global_nde),
            "sae": float(global_sae),
        },
        "per_home_csv": str(PER_HOME_METRICS_CSV),
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[LOHO] Saved summary JSON to: {SUMMARY_JSON}")

    # =================================================
    # Train final model on ALL homes (research only)
    # =================================================
    print("\n[LOHO] Training final model on ALL homes (research model)...")

    drop_cols_all = [c for c in ["home_id", "datetime"] if c in X_full.columns]
    feature_cols_all = [c for c in X_full.columns if c not in drop_cols_all]

    X_all_model = X_full[feature_cols_all]

    final_model = lgb.LGBMRegressor(
        **params,
        random_state=RANDOM_SEED,
    )

    final_model.fit(X_all_model, y)

    booster = final_model.booster_
    booster.save_model(str(FINAL_MODEL_PATH))

    print(f"[LOHO] Saved final all-homes model to: {FINAL_MODEL_PATH}")
    print("[LOHO] Done.")


if __name__ == "__main__":
    main()
