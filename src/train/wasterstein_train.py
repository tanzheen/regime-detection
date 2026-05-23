import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.wasterstein import WKMeans

INPUT_FOLDS_DIR = PROJECT_ROOT / "data" / "inputs" / "folds"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "wasterstein"

# =============================================================================
# STEP 1 — Data preparation (Section 1.3)
# =============================================================================

def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Equation (2) from the paper: r_i = log(s_{i+1}) - log(s_i)"""
    return np.diff(np.log(prices))


def stream_lift(returns: np.ndarray, h1: int, h2: int) -> np.ndarray:
    """
    Definition 1.2 — Stream lift.
    Partition returns into overlapping windows of length h1 with overlap h2.
    
    Step size = h1 - h2  (the non-overlapping part)
    
    Paper used (h1=35, h2=28) on hourly data → step=7 (1 day).
    For daily data: (h1=20, h2=15) → step=5 (1 week) is a good starting point.

    Args:
        returns: 1D array of log returns
        h1:      window length
        h2:      overlap between consecutive windows (h2 < h1)

    Returns:
        (M, h1) array of return windows
    """
    assert h1 > h2, "h1 must be greater than h2"
    step = h1 - h2
    M = (len(returns) - h1) // step + 1
    segments = np.array([returns[i * step: i * step + h1] for i in range(M)])
    return segments


# =============================================================================
# STEP 5 — Map segment labels back onto the price path
# =============================================================================

def map_labels_to_prices(
    df: pd.DataFrame,
    labels: np.ndarray,
    h1: int,
    h2: int,
) -> pd.Series:
    """
    Map WK-means segment labels to the price row where each window ends.

    This keeps evaluation causal: the label for a 10-day window is only
    available after the 10th return in that window has been observed.
    """
    step = h1 - h2
    asof_labels = np.full(len(df), np.nan)

    for i, label in enumerate(labels):
        start = i * step
        end = start + h1
        if end < len(asof_labels):
            asof_labels[end] = label

    return pd.Series(asof_labels, index=df.index, name="regime")


# =============================================================================
# STEP 6 — Plotting
# =============================================================================

def plot_regimes(
    df: pd.DataFrame,
    label_series: pd.Series,
    output_dir: Path,
    fold: int = None,
) -> None:
    """Colour the price path by regime — replicates Figure 2 from the paper."""
    fig, ax = plt.subplots(figsize=(14, 5))

    title = f"SPY — WK-means Market Regimes (k=2)"
    if fold is not None:
        title += f" | Fold {fold}"
    ax.set_title(title)

    # Colour each segment by average regime membership
    dates  = df["date"].values
    prices = df["adjClose"].values
    labels = label_series.values

    for i in range(len(dates) - 1):
        if np.isnan(labels[i]):
            continue
        color = plt.cm.RdYlGn(1 - labels[i])   # green = bull (0), red = bear (1)
        ax.plot(dates[i:i+2], prices[i:i+2], color=color, linewidth=0.8)

    bull_patch = mpatches.Patch(color="green", label="Bull (low vol)")
    bear_patch = mpatches.Patch(color="red",   label="Bear (high vol)")
    ax.legend(handles=[bull_patch, bear_patch])
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"regimes_fold_{fold}.png", dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN — Run on walk-forward folds
# =============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Hyperparameters (adapt from paper's hourly (35,28) to daily data)
    # h1=5 ≈ 1 week of daily bars, step = 5-4 = 1 ≈ 1 day
    H1 = 5
    H2 = 4
    K  = 2    # bull / bear

    # --- Run on each fold's training set ---
    for fold in range(1, 6):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")

        train = pd.read_parquet(INPUT_FOLDS_DIR / f"fold_{fold}_train.parquet")
        val   = pd.read_parquet(INPUT_FOLDS_DIR / f"fold_{fold}_val.parquet")

        # --- Fit on training data ---
        train_returns  = compute_log_returns(train["adjClose"].values)
        train_segments = stream_lift(train_returns, h1=H1, h2=H2)
        print(f"Train: {len(train)} rows → {len(train_segments)} segments")

        model = WKMeans(k=K, p=1, max_iter=100, tol=1e-7, random_state=42)
        model.fit(train_segments)

        train_labels = map_labels_to_prices(train, model.labels_, H1, H2)
        print(f"\nTrain regime distribution:\n{train_labels.value_counts(normalize=True, dropna=True).round(3)}")

        # --- Predict on validation data ---
        val_returns   = compute_log_returns(val["adjClose"].values)
        val_segments  = stream_lift(val_returns, h1=H1, h2=H2)
        val_preds     = model.predict(val_segments)
        val_labels    = map_labels_to_prices(val, val_preds, H1, H2)

        print(f"Val regime distribution:\n{val_labels.value_counts(normalize=True, dropna=True).round(3)}")
        print(f"\nCentroid variances: Bull={np.var(model.centroids_[0]):.6f} | Bear={np.var(model.centroids_[1]):.6f}")

        plot_regimes(val, val_labels, output_dir=OUTPUT_DIR, fold=fold)
