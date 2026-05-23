import os
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)


def _find_dotenv() -> Optional[Path]:
    """Find .env from the current directory or this module's parent tree."""
    seen: set[Path] = set()
    search_roots = (
        Path.cwd().resolve(),
        *Path.cwd().resolve().parents,
        *Path(__file__).resolve().parents,
    )

    for root in search_roots:
        candidate = root / ".env"
        if candidate in seen:
            continue
        seen.add(candidate)

        if candidate.exists():
            return candidate

    return None


def _load_dotenv(path: Optional[Path] = None) -> None:
    """
    Load KEY=VALUE entries from a .env file into os.environ.

    Existing shell environment values take precedence over .env values.
    """
    env_path = path or _find_dotenv()
    if not env_path:
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key.startswith("export "):
            key = key.removeprefix("export ").strip()

        if key and key not in os.environ:
            os.environ[key] = value


class TiingoClient:
    """
    Tiingo REST API client for historical EOD price data.

    Usage
    -----
    # One-time instantiation, reuse across calls
    client = TiingoClient()                          # reads TIINGO_API_KEY from env
    client = TiingoClient(api_key="your_key_here")  # explicit

    spy    = client.get_stock_data("SPY")                         # full history
    aapl   = client.get_stock_data("AAPL", start_date="2015-01-01")
    batch  = client.get_multiple(["AAPL", "MSFT", "GOOGL"])       # dict of DataFrames
    meta   = client.get_metadata("SPY")
    """

    BASE_URL = "https://api.tiingo.com"
    SPY_INCEPTION = None 
    DEFAULT_START = None 
    VALID_FREQUENCIES = {"daily", "weekly", "monthly", "annually"}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        """
        Args:
            api_key: Tiingo API key. Falls back to TIINGO_API_KEY env var.
            timeout: Request timeout in seconds.
        """
        _load_dotenv()
        self.api_key = api_key or os.environ.get("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Tiingo API key is required. "
                "Pass api_key= or set TIINGO_API_KEY in your environment or .env file."
            )
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}",
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "daily",
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for any ticker.

        Args:
            symbol:     Ticker symbol, e.g. "SPY", "AAPL".
            start_date: "YYYY-MM-DD". Defaults to the earliest available data.
            end_date:   "YYYY-MM-DD". Defaults to today.
            frequency:  One of "daily" | "weekly" | "monthly" | "annually".
            adjusted:   If True, keep split/dividend-adjusted columns.
                        If False, drop adj* columns and return raw OHLCV only.

        Returns:
            pd.DataFrame sorted ascending by date.

        Columns (adjusted=True):
            date, open, high, low, close, volume,
            adjOpen, adjHigh, adjLow, adjClose, adjVolume,
            divCash, splitFactor
        """
        self._validate_frequency(frequency)
        start = start_date or self.DEFAULT_START
        end   = end_date   or date.today().isoformat()

        logger.info("Fetching %s  %s → %s  [%s]", symbol.upper(), start, end, frequency)

        raw = self._get(
            f"/tiingo/daily/{symbol.upper()}/prices",
            params={"startDate": start, "endDate": end, "resampleFreq": frequency},
        )

        df = self._to_dataframe(raw, symbol)

        if not adjusted:
            drop = [c for c in df.columns if c.startswith("adj") or c in ("divCash", "splitFactor")]
            df = df.drop(columns=drop, errors="ignore")

        return df

    def get_spy_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "daily",
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Convenience wrapper — fetches SPY from its inception date (1993-01-29).

        Identical signature to get_stock_data(); start_date defaults to
        SPY's first trading day instead of DEFAULT_START.
        """
        return self.get_stock_data(
            symbol="SPY",
            start_date=start_date or self.SPY_INCEPTION,
            end_date=end_date,
            frequency=frequency,
            adjusted=adjusted,
        )

    def get_multiple(
        self,
        symbols: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "daily",
        adjusted: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical data for a list of tickers.

        Returns:
            dict mapping ticker → DataFrame.
            Failed tickers are logged and omitted from the result.
        """
        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                result[sym.upper()] = self.get_stock_data(
                    sym, start_date, end_date, frequency, adjusted
                )
            except Exception as exc:
                logger.warning("Skipping %s — %s", sym.upper(), exc)
        return result

    def get_metadata(self, symbol: str) -> dict:
        """
        Return Tiingo's metadata for a ticker.

        Includes: name, description, exchange, startDate, endDate.
        """
        return self._get(f"/tiingo/daily/{symbol.upper()}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> list | dict:
        """Execute a GET request and return parsed JSON."""
        url = self.BASE_URL + path
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            raise RuntimeError(
                f"Tiingo HTTP {status} for {path!r}: {exc}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Tiingo request failed for {path!r}: {exc}") from exc

        return resp.json()

    def _to_dataframe(self, data: list, symbol: str) -> pd.DataFrame:
        """Convert raw JSON list to a clean, sorted DataFrame."""
        if not data:
            raise ValueError(f"Tiingo returned no data for '{symbol}'.")

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)

        # Canonical column order
        ordered = [
            "date",
            "open", "high", "low", "close", "volume",
            "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume",
            "divCash", "splitFactor",
        ]
        existing = [c for c in ordered if c in df.columns]
        extras   = [c for c in df.columns if c not in existing]
        return df[existing + extras]

    @staticmethod
    def _validate_frequency(frequency: str) -> None:
        if frequency not in TiingoClient.VALID_FREQUENCIES:
            raise ValueError(
                f"Invalid frequency '{frequency}'. "
                f"Choose from: {TiingoClient.VALID_FREQUENCIES}"
            )

if __name__ == "__main__":
    import pprint

    client = TiingoClient()  # reads TIINGO_API_KEY from env

    # --- Metadata ---
    print("=== SPY Metadata ===")
    meta = client.get_metadata("SPY")
    pprint.pprint(meta)

    # --- SPY full history ---
    print("\n=== SPY Full History (daily) ===")
    spy = client.get_spy_history()
    print(spy.head())
    print(spy.tail())
    print(f"\nRows: {len(spy)} | {spy['date'].min().date()} → {spy['date'].max().date()}")


    # --- Batch ---
    print("\n=== Batch Fetch ===")
    watchlist = ["SPY"]
    batch = client.get_multiple(watchlist)
    for ticker, df in batch.items():
        print(f"  {ticker}: {len(df)} rows | latest close = {df['adjClose'].iloc[-1]:.2f}")
