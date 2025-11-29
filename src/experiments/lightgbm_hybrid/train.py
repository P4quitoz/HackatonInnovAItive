# src/experiments/lightgbm_hybrid/train.py

from __future__ import annotations

import json

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import (
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
    LGBM_DEFAULT_PARAMS,
    RANDOM_SEED,
    PROJECT_ROOT,
)
from src.common.evaluation import mae, rmse, nde, sae, loho_split


# =====================================================
# EXPERIMENT CONFIG
# =====================================================

EXP_NAME = "lightgbm_hybrid"
EXPORT_DIR = PROJECT_ROOT / "exports" / EXP_NAME
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

ON_THRESHOLD = 15.0  # W: fridge considered ON above this power
PROBA_ON_THRESHOLD = 0.4  # classifier probability cutoff for ON


# Classifier params (LightGBM binary)
LGBM_CLF_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "max_depth": -1,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "n_estimators": 500,
    "n_jobs": -1,
}


# =====================================================
# LOAD DATA
# =====================================================

print(f"Loading train features from: {TRAIN_FEATURES_PATH}")
X_full = pd.read_parquet(TRAIN_FEATURES_PATH)
y_full = pd.read_parquet(TRAIN_TARGET_PATH)

if "home_id" not in X_full.columns:
    raise ValueError("TRAIN_FEATURES does not contain 'home_id'.")

feature_cols = [c for c in X_full.columns if c not in ["home_id", "datetime"]]

y_values = y_full.values.reshape(-1).astype(float)

# Binary ON/OFF labels
y_onoff = (y_values > ON_THRESHOLD).astype(int)

print("Total samples :", X_full.shape[0])
print("Num features  :", len(feature_cols))
print("Homes in train:", X_full["home_id"].unique())


# =====================================================
# LOHO CROSS-VALIDATION (HYBRID)
# =====================================================

per_home_metrics: dict[str, dict] = {}
all_errors: list[np.ndarray] = []
all_importances_reg: list[np.ndarray] = []
all_importances_clf: list[np.ndarray] = []

for home, X_train, X_valid, y_train_df, y_valid_df in loho_split(
    X_full, y_full, home_col="home_id"
):
    print(f"\n=== LOHO fold (hybrid): validate on {home} ===")

    # Masks for this home
    valid_mask = X_full["home_id"] == home
    train_mask = ~valid_mask

    # Extract 1D arrays for targets
    y_train = y_values[train_mask]
    y_valid = y_values[valid_mask]
    y_train_onoff = y_onoff[train_mask]

    X_train_model = X_train[feature_cols]
    X_valid_model = X_valid[feature_cols]

    # -------------------------------------------------
    # 1) Train ON/OFF classifier on ALL training samples
    # -------------------------------------------------
    clf_params = LGBM_CLF_PARAMS.copy()
    clf_params["random_state"] = RANDOM_SEED

    clf = lgb.LGBMClassifier(**clf_params)
    clf.fit(X_train_model, y_train_onoff)

    # -------------------------------------------------
    # 2) Train regressor ONLY on ON samples
    # -------------------------------------------------
    reg_params = LGBM_DEFAULT_PARAMS.copy()
    reg_params["random_state"] = RANDOM_SEED

    on_train_mask = y_train > ON_THRESHOLD
    n_on = int(on_train_mask.sum())
    if n_on < 50:
        # not enough ON samples for this home; fall back to all
        on_train_mask = np.ones_like(y_train, dtype=bool)
        print(
            f"  [WARN] home {home}: only {n_on} ON samples, using all samples for regressor."
        )

    X_train_reg = X_train_model[on_train_mask]
    y_train_reg = y_train[on_train_mask]

    reg = lgb.LGBMRegressor(**reg_params)
    reg.fit(X_train_reg, y_train_reg)

    # -------------------------------------------------
    # 3) Predict: ON/OFF + regression combined
    # -------------------------------------------------
    proba_on = clf.predict_proba(X_valid_model)[:, 1]
    on_mask_pred = proba_on >= PROBA_ON_THRESHOLD

    y_pred_reg = reg.predict(X_valid_model)
    y_pred_reg = np.clip(y_pred_reg, 0, None)

    y_pred = np.where(on_mask_pred, y_pred_reg, 0.0)

    # -------------------------------------------------
    # 4) Metrics
    # -------------------------------------------------
    yv = y_valid.reshape(-1)
    yp = y_pred.reshape(-1)

    m_mae = mae(yv, yp)
    m_rmse = rmse(yv, yp)
    m_nde = nde(yv, yp)
    m_sae = sae(yv, yp)

    per_home_metrics[str(home)] = {
        "mae": m_mae,
        "rmse": m_rmse,
        "nde": m_nde,
        "sae": m_sae,
        "n_samples": int(len(yv)),
    }

    print(f"  MAE : {m_mae:.6f}")
    print(f"  RMSE: {m_rmse:.6f}")
    print(f"  NDE : {m_nde:.6f}")
    print(f"  SAE : {m_sae:.6f}")

    all_errors.append(yv - yp)
    all_importances_reg.append(reg.feature_importances_)
    all_importances_clf.append(clf.feature_importances_)


# =====================================================
# AGGREGATE METRICS
# =====================================================

mae_vals = [v["mae"] for v in per_home_metrics.values()]
rmse_vals = [v["rmse"] for v in per_home_metrics.values()]
nde_vals = [v["nde"] for v in per_home_metrics.values()]
sae_vals = [v["sae"] for v in per_home_metrics.values()]

global_metrics = {
    "mae_mean": float(np.mean(mae_vals)),
    "mae_std": float(np.std(mae_vals)),
    "rmse_mean": float(np.mean(rmse_vals)),
    "rmse_std": float(np.std(rmse_vals)),
    "nde_mean": float(np.mean(nde_vals)),
    "nde_std": float(np.std(nde_vals)),
    "sae_mean": float(np.mean(sae_vals)),
    "sae_std": float(np.std(sae_vals)),
}

print("\n=== LOHO summary (hybrid) ===")
print(global_metrics)


# =====================================================
# PLOTS
# =====================================================

homes_sorted = list(per_home_metrics.keys())
mae_list = [per_home_metrics[h]["mae"] for h in homes_sorted]

plt.figure(figsize=(8, 5))
plt.bar(homes_sorted, mae_list)
plt.xlabel("Home")
plt.ylabel("MAE [W]")
plt.title("Hybrid LightGBM – LOHO MAE per Home")
plt.tight_layout()
mae_plot = EXPORT_DIR / "mae_per_home.png"
plt.savefig(mae_plot)
plt.close()

# Error histogram
errors_concat = np.concatenate(all_errors)
plt.figure(figsize=(8, 5))
plt.hist(errors_concat, bins=80)
plt.xlabel("Error [W] (y_true - y_pred)")
plt.ylabel("Count")
plt.title("Hybrid LightGBM – Global Error Histogram (LOHO)")
plt.tight_layout()
err_hist = EXPORT_DIR / "error_hist_global.png"
plt.savefig(err_hist)
plt.close()

# Feature importance (regressor)
importances_reg_mean = np.mean(np.vstack(all_importances_reg), axis=0)
idx_reg = np.argsort(importances_reg_mean)[::-1][:30]

plt.figure(figsize=(8, 10))
plt.barh(
    np.arange(len(idx_reg)),
    importances_reg_mean[idx_reg][::-1],
)
plt.yticks(
    np.arange(len(idx_reg)),
    [feature_cols[i] for i in idx_reg][::-1],
)
plt.xlabel("Importance")
plt.title("Hybrid LightGBM – Regressor Feature Importance (avg)")
plt.tight_layout()
fi_reg_plot = EXPORT_DIR / "feature_importance_reg.png"
plt.savefig(fi_reg_plot)
plt.close()

# Feature importance (classifier)
importances_clf_mean = np.mean(np.vstack(all_importances_clf), axis=0)
idx_clf = np.argsort(importances_clf_mean)[::-1][:30]

plt.figure(figsize=(8, 10))
plt.barh(
    np.arange(len(idx_clf)),
    importances_clf_mean[idx_clf][::-1],
)
plt.yticks(
    np.arange(len(idx_clf)),
    [feature_cols[i] for i in idx_clf][::-1],
)
plt.xlabel("Importance")
plt.title("Hybrid LightGBM – Classifier Feature Importance (avg)")
plt.tight_layout()
fi_clf_plot = EXPORT_DIR / "feature_importance_clf.png"
plt.savefig(fi_clf_plot)
plt.close()


# =====================================================
# FINAL MODELS ON ALL HOMES
# =====================================================

print("\nTraining final hybrid models on ALL homes...")

X_all = X_full[feature_cols]
y_all = y_values
y_all_onoff = y_onoff

# Final classifier
final_clf_params = LGBM_CLF_PARAMS.copy()
final_clf_params["random_state"] = RANDOM_SEED
final_clf = lgb.LGBMClassifier(**final_clf_params)
final_clf.fit(X_all, y_all_onoff)

# Final regressor on ON samples
final_reg_params = LGBM_DEFAULT_PARAMS.copy()
final_reg_params["random_state"] = RANDOM_SEED

on_all_mask = y_all > ON_THRESHOLD
X_all_reg = X_all[on_all_mask]
y_all_reg = y_all[on_all_mask]

final_reg = lgb.LGBMRegressor(**final_reg_params)
final_reg.fit(X_all_reg, y_all_reg)

clf_model_path = EXPORT_DIR / "model_clf_final.txt"
reg_model_path = EXPORT_DIR / "model_reg_final.txt"
final_clf.booster_.save_model(str(clf_model_path))
final_reg.booster_.save_model(str(reg_model_path))


# =====================================================
# SAVE ARTIFACT JSON
# =====================================================

artifact = {
    "experiment": EXP_NAME,
    "type": "lightgbm_hybrid_onoff",
    "on_threshold": ON_THRESHOLD,
    "proba_on_threshold": PROBA_ON_THRESHOLD,
    "global_metrics": global_metrics,
    "per_home_metrics": per_home_metrics,
    "paths": {
        "clf_model": str(clf_model_path),
        "reg_model": str(reg_model_path),
        "mae_per_home_png": str(mae_plot),
        "error_hist_png": str(err_hist),
        "fi_reg_png": str(fi_reg_plot),
        "fi_clf_png": str(fi_clf_plot),
    },
}

artifact_path = EXPORT_DIR / "artifact.json"
with open(artifact_path, "w") as f:
    json.dump(artifact, f, indent=2)

print("Saved hybrid artifacts to:", EXPORT_DIR)
print("Done.")
