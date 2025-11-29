# src/common/features.py

from __future__ import annotations

import numpy as np
import pandas as pd


# =====================================================
# Helper: ensure datetime formatting & sorting
# =====================================================
def _ensure_datetime(df: pd.DataFrame, time_col="datetime"):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    if "home_id" in df.columns:
        return df.sort_values(["home_id", time_col])
    return df.sort_values(time_col)


# =====================================================
# Target cleaning (fridge power)
# =====================================================
def _clean_fridge(df: pd.DataFrame) -> pd.Series:
    f = df["fridge"].astype(float).clip(lower=0, upper=350)

    # 5-minute rolling median smooths out noise without destroying cycles
    f = f.rolling(5, center=True, min_periods=1).median()

    # Light smoothing
    f = f.rolling(3, min_periods=1).mean()

    # Force OFF threshold
    f[f < 10] = 0

    f = f.bfill().ffill()

    return f


# =====================================================
# Basic power smoothing
# =====================================================
def _smooth_power(df: pd.DataFrame, col="power"):
    p = df[col].astype(float)
    df["p_raw"] = p
    df["p_ewm_03"] = p.ewm(alpha=0.3, adjust=False).mean()
    df["p_roll3"] = p.rolling(3, min_periods=1).mean()
    df["p_roll5"] = p.rolling(5, min_periods=1).mean()
    df["p_roll15"] = p.rolling(15, min_periods=1).mean()
    return df


# =====================================================
# Lag features
# =====================================================
def _add_lags(df: pd.DataFrame, col="power"):
    p = df[col]
    for lag in [1, 2, 3, 5, 10, 15, 30, 45, 60, 90, 120, 180]:
        df[f"{col}_lag{lag}"] = p.shift(lag)


# =====================================================
# Rolling statistics
# =====================================================
def _add_rolling_stats(df: pd.DataFrame, col="power"):
    p = df[col]

    df["roll3_mean"] = p.rolling(3).mean()
    df["roll5_mean"] = p.rolling(5).mean()
    df["roll15_mean"] = p.rolling(15).mean()
    df["roll30_mean"] = p.rolling(30).mean()

    df["roll5_std"] = p.rolling(5).std()
    df["roll15_std"] = p.rolling(15).std()

    df["roll15_min"] = p.rolling(15).min()
    df["roll15_max"] = p.rolling(15).max()
    df["roll15_range"] = df["roll15_max"] - df["roll15_min"]

    df["roll15_q10"] = p.rolling(15).quantile(0.1)
    df["roll15_q90"] = p.rolling(15).quantile(0.9)
    df["roll15_iqr"] = df["roll15_q90"] - df["roll15_q10"]

    df["roll15_skew"] = p.rolling(15).skew()
    df["roll15_kurt"] = p.rolling(15).kurt()


# =====================================================
# Baseline subtraction & highpass
# =====================================================
def _add_baseline_features(df: pd.DataFrame, col="power"):
    p = df[col]

    df["base15"] = p.rolling(15).mean()
    df["base30"] = p.rolling(30).mean()

    df["p_minus_base15"] = p - df["base15"]
    df["p_minus_base30"] = p - df["base30"]

    # High-pass: remove very slow drift
    df["p_highpass"] = p - p.ewm(alpha=0.02, adjust=False).mean()


# =====================================================
# FFT features (improved)
# =====================================================
def _add_fft_features(df: pd.DataFrame, col="power"):
    p = df[col].astype(float)

    WINDOW = 60  # 60-minute FFT window
    fft_energy = []
    fft_domfreq = []
    fft_lowhigh_ratio = []

    for i in range(len(p)):
        if i < WINDOW:
            fft_energy.append(np.nan)
            fft_domfreq.append(np.nan)
            fft_lowhigh_ratio.append(np.nan)
            continue

        segment = p[i - WINDOW : i].values
        fft_vals = np.fft.rfft(segment)
        mag = np.abs(fft_vals)

        # Energy
        fft_energy.append(float(np.sum(mag)))

        # Dominant freq index
        fft_domfreq.append(float(np.argmax(mag[1:]) + 1))

        # Low/high energy ratio
        half = len(mag) // 2
        low = np.sum(mag[:half])
        high = np.sum(mag[half:])
        fft_lowhigh_ratio.append(float(low / (high + 1e-8)))

    df["fft60_energy"] = fft_energy
    df["fft60_domfreq"] = fft_domfreq
    df["fft60_lowhigh"] = fft_lowhigh_ratio


# =====================================================
# Time-of-day features
# =====================================================
def _add_time_features(df: pd.DataFrame):
    dt = df["datetime"].dt
    df["hour"] = dt.hour
    df["dow"] = dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)


# =====================================================
# Main builder
# =====================================================
def build_features(df: pd.DataFrame, is_train=True):
    df = _ensure_datetime(df)

    # ----------------------------------------------
    # TARGET (train only)
    # ----------------------------------------------
    y = None
    if is_train:
        y = _clean_fridge(df)

    # ----------------------------------------------
    # POWER FEATURES
    # ----------------------------------------------
    _smooth_power(df, col="power")
    _add_lags(df, col="power")
    _add_rolling_stats(df, col="power")
    _add_baseline_features(df, col="power")
    _add_fft_features(df, col="power")

    # ----------------------------------------------
    # TIME FEATURES
    # ----------------------------------------------
    _add_time_features(df)

    # ----------------------------------------------
    # Build feature matrix
    # ----------------------------------------------
    feature_cols = [
        # smoothed
        "p_raw",
        "p_ewm_03",
        "p_roll3",
        "p_roll5",
        "p_roll15",
        # rolling stats
        "roll3_mean",
        "roll5_mean",
        "roll15_mean",
        "roll30_mean",
        "roll5_std",
        "roll15_std",
        "roll15_min",
        "roll15_max",
        "roll15_range",
        "roll15_q10",
        "roll15_q90",
        "roll15_iqr",
        "roll15_skew",
        "roll15_kurt",
        # baseline
        "base15",
        "base30",
        "p_minus_base15",
        "p_minus_base30",
        # highpass
        "p_highpass",
        # FFT
        "fft60_energy",
        "fft60_domfreq",
        "fft60_lowhigh",
        # time
        "hour",
        "dow",
        "is_weekend",
        "hour_sin",
        "hour_cos",
    ]

    # lags
    for lag in [1, 2, 3, 5, 10, 15, 30, 45, 60, 90, 120, 180]:
        feature_cols.append(f"power_lag{lag}")

    X = df[feature_cols].copy()

    # meta columns for LOHO
    if "home_id" in df.columns:
        X["home_id"] = df["home_id"]
    X["datetime"] = df["datetime"]

    # fill nans
    X = X.bfill().ffill()

    if is_train:
        return X, y
    return X, None
