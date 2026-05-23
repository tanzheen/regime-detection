## script to prepare data and save into /data/inputs
## Split into training and validation set using walk forward validation 

from tiingo import TiingoClient
import os
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd 

def download_data(): 
    client = TiingoClient()  # reads TIINGO_API_KEY from env

    # --- SPY full history ---
    print("\n=== SPY Full History (daily) ===")
    spy = client.get_spy_history()
    return spy 

def train_val_split(spy: pd.DataFrame, n_splits: int = 5) -> None:
    """
    Walk-forward split SPY data and save each fold as parquet.

    Saves to:
        data/inputs/spy500.parquet              ← full dataset
        data/inputs/folds/fold_1_train.parquet
        data/inputs/folds/fold_1_val.parquet
        data/inputs/folds/fold_2_train.parquet
        ...
    """
    spy = spy.sort_values("date").reset_index(drop=True)

    # --- Save full dataset ---
    os.makedirs("data/inputs/folds", exist_ok=True)
    spy.to_parquet("data/inputs/spy500.parquet", index=False)
    print(f"Saved full dataset: {len(spy)} rows")

    # --- Walk-forward splits ---
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(spy), start=1):
        train = spy.iloc[train_idx]
        val   = spy.iloc[val_idx]

        train_path = f"data/inputs/folds/fold_{fold}_train.parquet"
        val_path   = f"data/inputs/folds/fold_{fold}_val.parquet"

        train.to_parquet(train_path, index=False)
        val.to_parquet(val_path, index=False)

        print(
            f"Fold {fold} | "
            f"Train: {len(train):>5} rows ({train['date'].min().date()} → {train['date'].max().date()}) | "
            f"Val: {len(val):>5} rows ({val['date'].min().date()} → {val['date'].max().date()})"
        )

if __name__ == "__main__":
    spy = download_data()
    train_val_split(spy, n_splits=5)