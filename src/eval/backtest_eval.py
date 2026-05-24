from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRICE_PATH = PROJECT_ROOT / "data" / "inputs" / "spy500.parquet"
DEFAULT_PREDICTIONS_ROOT = PROJECT_ROOT / "data" / "outputs" / "wasterstein"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "outputs" / "backtests" / "trend_following_wasterstein"
)


@dataclass(frozen=True)
class Experiment:
    name: str
    feature_set: str
    method: str
    path: Path


@dataclass(frozen=True)
class StrategySpec:
    name: str
    description: str
    allowed_regimes: frozenset[int] | None
    short_regimes: frozenset[int] | None = None
    signal_type: str = "ma_filter"


STRATEGIES = [
    StrategySpec(
        name="buy_hold",
        description="Buy-and-hold SPY over the same walk-forward validation bars.",
        allowed_regimes=None,
        signal_type="buy_hold",
    ),
    StrategySpec(
        name="ma_only",
        description="10/30 SMA crossover with no volatility-regime filter.",
        allowed_regimes=None,
    ),
    StrategySpec(
        name="allow_low_medium",
        description=(
            "Allow long exposure only when the Wasserstein regime is low or "
            "medium volatility."
        ),
        allowed_regimes=frozenset({0, 1}),
    ),
    StrategySpec(
        name="allow_low_only",
        description=(
            "Allow long exposure only when the Wasserstein regime is low "
            "volatility."
        ),
        allowed_regimes=frozenset({0}),
    ),
    StrategySpec(
        name="switch_low_vs_medium_high",
        description=(
            "Regime-switch strategy: long in low volatility and flat in "
            "medium or high volatility."
        ),
        allowed_regimes=frozenset({0}),
        signal_type="regime_switch",
    ),
    StrategySpec(
        name="switch_low_short_high",
        description=(
            "Regime-switch strategy: long in low volatility, flat in medium "
            "volatility, and short in high volatility."
        ),
        allowed_regimes=frozenset({0}),
        short_regimes=frozenset({2}),
        signal_type="regime_long_short",
    ),
    StrategySpec(
        name="switch_low_medium_vs_high",
        description=(
            "Regime-switch strategy: long in low or medium volatility and "
            "flat in high volatility."
        ),
        allowed_regimes=frozenset({0, 1}),
        signal_type="regime_switch",
    ),
    StrategySpec(
        name="switch_low_medium_short_high",
        description=(
            "Regime-switch strategy: long in low or medium volatility and "
            "short in high volatility."
        ),
        allowed_regimes=frozenset({0, 1}),
        short_regimes=frozenset({2}),
        signal_type="regime_long_short",
    ),
]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    price_features = load_price_features(
        Path(args.price_path),
        short_window=args.short_window,
        long_window=args.long_window,
    )
    experiments = discover_experiments(Path(args.predictions_root))
    if not experiments:
        raise FileNotFoundError(
            f"No validation_predictions.parquet files found under {args.predictions_root}"
        )

    strategy_specs = [
        spec for spec in STRATEGIES if args.include_baseline or spec.name != "ma_only"
    ]

    all_daily_results = []
    overall_rows = []
    fold_rows = []

    for experiment in experiments:
        validation = load_validation_predictions(experiment.path)
        validation = filter_validation_folds(
            validation,
            folds=args.folds,
            last_n_folds=args.last_n_folds,
        )
        backtest_frame = prepare_backtest_frame(validation, price_features)

        for strategy in strategy_specs:
            result = run_strategy(backtest_frame, strategy)
            result["experiment"] = experiment.name
            result["feature_set"] = experiment.feature_set
            result["method"] = experiment.method
            result["strategy"] = strategy.name
            result["strategy_description"] = strategy.description
            all_daily_results.append(result)

            overall_rows.append(
                summarize_returns(
                    result,
                    experiment=experiment,
                    strategy=strategy,
                    annualization=args.annualization,
                )
            )

            for fold, fold_frame in result.groupby("fold", sort=True):
                fold_rows.append(
                    summarize_returns(
                        fold_frame,
                        experiment=experiment,
                        strategy=strategy,
                        annualization=args.annualization,
                        fold=int(fold),
                    )
                )

    daily_results = pd.concat(all_daily_results, ignore_index=True)
    overall = pd.DataFrame(overall_rows).sort_values(
        ["strategy", "annual_avg_return"],
        ascending=[True, False],
    )
    fold_summary = pd.DataFrame(fold_rows).sort_values(
        ["strategy", "experiment", "fold"]
    )

    comparison_path = output_dir / "comparison.csv"
    fold_path = output_dir / "fold_comparison.csv"
    daily_path = output_dir / "daily_results.parquet"
    report_path = output_dir / "report.md"

    overall.to_csv(comparison_path, index=False)
    fold_summary.to_csv(fold_path, index=False)
    daily_results.to_parquet(daily_path, index=False)
    write_report(
        report_path,
        overall=overall,
        fold_summary=fold_summary,
        experiments=experiments,
        strategies=strategy_specs,
        args=args,
    )

    print("\nOverall comparison")
    print(
        format_console_table(
            overall[
                [
                    "experiment",
                    "strategy",
                    "annual_avg_return",
                    "sharpe",
                    "cagr",
                    "max_drawdown",
                    "gross_exposure",
                    "net_exposure",
                    "entries",
                ]
            ]
        )
    )
    print(f"\nSaved comparison: {comparison_path}")
    print(f"Saved fold comparison: {fold_path}")
    print(f"Saved daily results: {daily_path}")
    print(f"Saved report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest a 10/30 SMA trend-following strategy against "
            "walk-forward Wasserstein regime predictions."
        )
    )
    parser.add_argument(
        "--price-path",
        default=str(DEFAULT_PRICE_PATH),
        help="Full SPY price history parquet used to calculate returns and SMAs.",
    )
    parser.add_argument(
        "--predictions-root",
        default=str(DEFAULT_PREDICTIONS_ROOT),
        help=(
            "Root containing */*/validation_predictions.parquet files from "
            "Wasserstein walk-forward validation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where comparison CSV, daily results, and Markdown report are saved.",
    )
    parser.add_argument(
        "--short-window",
        type=int,
        default=10,
        help="Short SMA lookback in trading days.",
    )
    parser.add_argument(
        "--long-window",
        type=int,
        default=30,
        help="Long SMA lookback in trading days.",
    )
    parser.add_argument(
        "--annualization",
        type=int,
        default=252,
        help="Trading-day annualization factor for annual return and Sharpe.",
    )
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the unfiltered 10/30 SMA crossover baseline.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        help="Only include these validation fold numbers.",
    )
    parser.add_argument(
        "--last-n-folds",
        type=int,
        help="Only include the most recent N validation folds.",
    )

    args = parser.parse_args()
    if args.folds and args.last_n_folds:
        parser.error("--folds and --last-n-folds cannot be used together")
    if args.last_n_folds is not None and args.last_n_folds <= 0:
        parser.error("--last-n-folds must be positive")
    return args


def discover_experiments(predictions_root: Path) -> list[Experiment]:
    experiments = []
    for path in sorted(predictions_root.glob("*/*/validation_predictions.parquet")):
        method = path.parent.name
        feature_set = path.parent.parent.name
        experiments.append(
            Experiment(
                name=f"{feature_set}_{method}",
                feature_set=feature_set,
                method=method,
                path=path,
            )
        )
    return experiments


def load_price_features(
    price_path: Path,
    *,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    if short_window <= 0:
        raise ValueError("--short-window must be positive")
    if long_window <= short_window:
        raise ValueError("--long-window must be greater than --short-window")

    data = pd.read_parquet(price_path)
    required = {"date", "adjClose"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{price_path} is missing required columns: {sorted(missing)}")

    data = data.sort_values("date").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"])
    data["asset_return"] = data["adjClose"].pct_change()
    data["sma_short"] = data["adjClose"].rolling(short_window).mean()
    data["sma_long"] = data["adjClose"].rolling(long_window).mean()
    return data[["date", "adjClose", "asset_return", "sma_short", "sma_long"]]


def load_validation_predictions(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path)
    required = {"date", "fold", "regime"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["date", "fold"]).reset_index(drop=True)
    duplicate_dates = data["date"].duplicated(keep=False)
    if duplicate_dates.any():
        examples = data.loc[duplicate_dates, "date"].head(5).dt.date.tolist()
        raise ValueError(f"{path} has duplicate validation dates, examples: {examples}")
    return data[["date", "fold", "regime"]]


def filter_validation_folds(
    validation: pd.DataFrame,
    *,
    folds: list[int] | None,
    last_n_folds: int | None,
) -> pd.DataFrame:
    available_folds = sorted(validation["fold"].dropna().astype(int).unique())
    if not available_folds:
        raise ValueError("Validation predictions contain no fold values")

    if folds:
        selected_folds = sorted(set(folds))
        missing = sorted(set(selected_folds).difference(available_folds))
        if missing:
            raise ValueError(
                f"Requested folds are unavailable: {missing}. "
                f"Available folds: {available_folds}"
            )
    elif last_n_folds is not None:
        selected_folds = available_folds[-last_n_folds:]
    else:
        selected_folds = available_folds

    filtered = validation[validation["fold"].isin(selected_folds)].copy()
    if filtered.empty:
        raise ValueError(f"No validation rows remain for folds: {selected_folds}")
    return filtered.reset_index(drop=True)


def prepare_backtest_frame(
    validation: pd.DataFrame,
    price_features: pd.DataFrame,
) -> pd.DataFrame:
    frame = validation.merge(price_features, on="date", how="left", validate="one_to_one")
    missing_prices = frame["asset_return"].isna() & frame["date"].ne(price_features["date"].min())
    if missing_prices.any():
        examples = frame.loc[missing_prices, "date"].head(5).dt.date.tolist()
        raise ValueError(f"Missing price features for validation dates: {examples}")

    frame = frame.dropna(subset=["asset_return", "sma_short", "sma_long"])
    return frame.sort_values("date").reset_index(drop=True)


def run_strategy(frame: pd.DataFrame, strategy: StrategySpec) -> pd.DataFrame:
    if strategy.signal_type == "buy_hold":
        return run_buy_hold_strategy(frame)
    if strategy.signal_type == "ma_filter":
        return run_ma_filter_strategy(frame, strategy)
    if strategy.signal_type == "regime_switch":
        return run_regime_switch_strategy(frame, strategy)
    if strategy.signal_type == "regime_long_short":
        return run_regime_long_short_strategy(frame, strategy)
    raise ValueError(f"Unsupported strategy signal_type: {strategy.signal_type}")


def run_buy_hold_strategy(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["position_after_signal"] = 1
    result["position"] = 1
    result["strategy_return"] = result["asset_return"]
    result["equity"] = (1.0 + result["strategy_return"]).cumprod()
    result["entries"] = 1
    result["short_entries"] = 0
    result["ma_exits"] = 0
    result["regime_exits"] = 0
    result["blocked_entries"] = 0
    return result


def run_ma_filter_strategy(frame: pd.DataFrame, strategy: StrategySpec) -> pd.DataFrame:
    positions_after_signal: list[int] = []
    position = 0
    entries = 0
    ma_exits = 0
    regime_exits = 0
    blocked_entries = 0

    for row in frame.itertuples(index=False):
        bull_trend = row.sma_short > row.sma_long
        bear_trend = row.sma_long > row.sma_short
        has_regime = pd.notna(row.regime)

        if strategy.allowed_regimes is None:
            regime_allowed = True
            regime_blocks_position = False
        elif has_regime:
            regime_allowed = int(row.regime) in strategy.allowed_regimes
            regime_blocks_position = not regime_allowed
        else:
            regime_allowed = False
            regime_blocks_position = False

        if position == 1 and regime_blocks_position:
            position = 0
            regime_exits += 1

        if position == 1 and bear_trend:
            position = 0
            ma_exits += 1

        if position == 0 and bull_trend:
            if regime_allowed:
                position = 1
                entries += 1
            else:
                blocked_entries += 1

        positions_after_signal.append(position)

    result = frame.copy()
    result["position_after_signal"] = positions_after_signal
    result["position"] = result["position_after_signal"].shift(1).fillna(0).astype(int)
    result["strategy_return"] = result["position"] * result["asset_return"]
    result["equity"] = (1.0 + result["strategy_return"]).cumprod()
    result["entries"] = entries
    result["short_entries"] = 0
    result["ma_exits"] = ma_exits
    result["regime_exits"] = regime_exits
    result["blocked_entries"] = blocked_entries
    return result


def run_regime_switch_strategy(
    frame: pd.DataFrame,
    strategy: StrategySpec,
) -> pd.DataFrame:
    if strategy.allowed_regimes is None:
        raise ValueError("Regime-switch strategies require allowed_regimes")

    positions_after_signal: list[int] = []
    position = 0
    entries = 0
    regime_exits = 0

    for row in frame.itertuples(index=False):
        if pd.notna(row.regime):
            regime_allowed = int(row.regime) in strategy.allowed_regimes
            if position == 1 and not regime_allowed:
                position = 0
                regime_exits += 1
            elif position == 0 and regime_allowed:
                position = 1
                entries += 1

        positions_after_signal.append(position)

    result = frame.copy()
    result["position_after_signal"] = positions_after_signal
    result["position"] = result["position_after_signal"].shift(1).fillna(0).astype(int)
    result["strategy_return"] = result["position"] * result["asset_return"]
    result["equity"] = (1.0 + result["strategy_return"]).cumprod()
    result["entries"] = entries
    result["short_entries"] = 0
    result["ma_exits"] = 0
    result["regime_exits"] = regime_exits
    result["blocked_entries"] = 0
    return result


def run_regime_long_short_strategy(
    frame: pd.DataFrame,
    strategy: StrategySpec,
) -> pd.DataFrame:
    if strategy.allowed_regimes is None or strategy.short_regimes is None:
        raise ValueError("Long/short regime strategies require long and short regimes")

    positions_after_signal: list[int] = []
    position = 0
    entries = 0
    short_entries = 0
    regime_exits = 0

    for row in frame.itertuples(index=False):
        if pd.notna(row.regime):
            regime = int(row.regime)
            if regime in strategy.allowed_regimes:
                target_position = 1
            elif regime in strategy.short_regimes:
                target_position = -1
            else:
                target_position = 0

            if position != target_position:
                if position != 0:
                    regime_exits += 1
                if target_position == 1:
                    entries += 1
                elif target_position == -1:
                    short_entries += 1
                position = target_position

        positions_after_signal.append(position)

    result = frame.copy()
    result["position_after_signal"] = positions_after_signal
    result["position"] = result["position_after_signal"].shift(1).fillna(0).astype(int)
    result["strategy_return"] = result["position"] * result["asset_return"]
    result["equity"] = (1.0 + result["strategy_return"]).cumprod()
    result["entries"] = entries
    result["short_entries"] = short_entries
    result["ma_exits"] = 0
    result["regime_exits"] = regime_exits
    result["blocked_entries"] = 0
    return result


def summarize_returns(
    result: pd.DataFrame,
    *,
    experiment: Experiment,
    strategy: StrategySpec,
    annualization: int,
    fold: int | None = None,
) -> dict[str, object]:
    returns = result["strategy_return"].fillna(0.0)
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    years = max(len(returns) / annualization, 1.0 / annualization)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    annual_avg_return = float(returns.mean() * annualization)
    annual_volatility = float(returns.std(ddof=0) * np.sqrt(annualization))
    sharpe = (
        float(annual_avg_return / annual_volatility)
        if annual_volatility > 0
        else np.nan
    )
    drawdown = equity / equity.cummax() - 1.0

    return {
        "experiment": experiment.name,
        "feature_set": experiment.feature_set,
        "method": experiment.method,
        "strategy": strategy.name,
        "fold": fold if fold is not None else "all",
        "start": result["date"].min().date().isoformat(),
        "end": result["date"].max().date().isoformat(),
        "bars": int(len(result)),
        "annual_avg_return": annual_avg_return,
        "sharpe": sharpe,
        "cagr": cagr,
        "total_return": total_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": float(drawdown.min()),
        "exposure": float(result["position"].mean()),
        "net_exposure": float(result["position"].mean()),
        "gross_exposure": float(result["position"].abs().mean()),
        "entries": int(result["entries"].iloc[0]) if len(result) else 0,
        "short_entries": int(result["short_entries"].iloc[0]) if len(result) else 0,
        "ma_exits": int(result["ma_exits"].iloc[0]) if len(result) else 0,
        "regime_exits": int(result["regime_exits"].iloc[0]) if len(result) else 0,
        "blocked_entries": (
            int(result["blocked_entries"].iloc[0]) if len(result) else 0
        ),
    }


def write_report(
    path: Path,
    *,
    overall: pd.DataFrame,
    fold_summary: pd.DataFrame,
    experiments: Iterable[Experiment],
    strategies: Iterable[StrategySpec],
    args: argparse.Namespace,
) -> None:
    metric_columns = [
        "experiment",
        "strategy",
        "annual_avg_return",
        "sharpe",
        "cagr",
        "total_return",
        "max_drawdown",
        "gross_exposure",
        "net_exposure",
        "entries",
        "short_entries",
        "regime_exits",
        "blocked_entries",
    ]
    fold_columns = [
        "experiment",
        "strategy",
        "fold",
        "annual_avg_return",
        "sharpe",
        "cagr",
        "max_drawdown",
        "gross_exposure",
        "net_exposure",
    ]

    best_by_strategy = (
        overall.sort_values(["strategy", "annual_avg_return"], ascending=[True, False])
        .groupby("strategy", as_index=False)
        .head(1)
    )
    selected_folds = sorted(fold_summary["fold"].astype(int).unique())

    lines = [
        "# Trend-Following Backtest Report",
        "",
        "## Configuration",
        "",
        f"- Price path: `{args.price_path}`",
        f"- Prediction root: `{args.predictions_root}`",
        f"- Short SMA: `{args.short_window}` bars",
        f"- Long SMA: `{args.long_window}` bars",
        f"- Annualization: `{args.annualization}` trading days",
        f"- Validation folds included: `{', '.join(map(str, selected_folds))}`",
        "- Signals are calculated at the close and applied to the next bar's return.",
        "- A known disallowed volatility regime exits an open position and blocks new long entries.",
        "- Missing regime rows block new long entries but do not force-close an existing trade.",
        "- Regime-switch strategies ignore the moving average signal and trade only the selected volatility bucket.",
        "- Long/short regime strategies are long in selected risk-on regimes and short only in high volatility.",
        "- Annual average return is arithmetic mean daily strategy return multiplied by annualization.",
        "- Sharpe ratio uses zero risk-free rate.",
        "",
        "## Experiments",
        "",
    ]

    for experiment in experiments:
        lines.append(f"- `{experiment.name}`: `{experiment.path}`")

    lines.extend(["", "## Strategy Filters", ""])
    for strategy in strategies:
        lines.append(f"- `{strategy.name}`: {strategy.description}")

    lines.extend(
        [
            "",
            "## Best Version By Strategy",
            "",
            to_markdown_table(best_by_strategy[metric_columns]),
            "",
            "## Overall Comparison",
            "",
            to_markdown_table(overall[metric_columns]),
            "",
            "## Fold-Level Comparison",
            "",
            to_markdown_table(fold_summary[fold_columns]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def format_console_table(data: pd.DataFrame) -> str:
    return to_markdown_table(data, float_format="{:.4f}")


def to_markdown_table(
    data: pd.DataFrame,
    *,
    float_format: str = "{:.6f}",
) -> str:
    if data.empty:
        return "_No rows._"

    formatted = data.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else float_format.format(value)
            )
        else:
            formatted[column] = formatted[column].astype(str)

    headers = list(formatted.columns)
    rows = formatted.astype(str).values.tolist()
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    ) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    row_lines = [
        "| " + " | ".join(
            row[index].ljust(widths[index]) for index in range(len(headers))
        ) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *row_lines])


if __name__ == "__main__":
    main()
