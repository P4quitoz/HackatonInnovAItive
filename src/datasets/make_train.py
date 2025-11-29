import pandas as pd

from src.common.config import (
    TRAIN_1MIN_PATH,
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
)
from src.common.features import build_features


def main():
    print(f"Loading resampled train from: {TRAIN_1MIN_PATH}")
    df_train = pd.read_csv(TRAIN_1MIN_PATH)

    print("Building train features...")
    X_train, y_train = build_features(df_train, is_train=True)

    print(f"Saving train features to: {TRAIN_FEATURES_PATH}")
    X_train.to_parquet(TRAIN_FEATURES_PATH)

    print(f"Saving train target to: {TRAIN_TARGET_PATH}")
    pd.DataFrame({"fridge_clean": y_train}).to_parquet(TRAIN_TARGET_PATH)

    print("Done.")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)


if __name__ == "__main__":
    main()
