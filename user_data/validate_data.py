#!/usr/bin/env python3
"""
Validate HARMONIC data files before trading.

The live strategy depends on the long-history custom signal files produced by
DATA_MANAGER.py. Standard Freqtrade data files can be checked as an optional
extra, but custom signal freshness and metadata integrity are the critical path.
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

STRATEGY_DIR = Path(__file__).resolve().parent / "strategies"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from DATA_MANAGER import DataManager, _load_strategy_hyperparams  # noqa: E402

logger = logging.getLogger("validate_data")


def _standard_data_dir(user_data_dir: Path) -> Path:
    return user_data_dir / "data" / "binance" / "futures"


def _standard_filename(pair: str, timeframe: str) -> str:
    clean_pair = pair.replace("/USDT:USDT", "").replace("/BUSD:BUSD", "")
    return f"{clean_pair}_USDT_USDT-{timeframe}-futures.feather"


def _validate_standard_file(path: Path, timeframe: str, min_rows: int) -> bool:
    if not path.exists():
        logger.warning(f"Standard data file missing: {path}")
        return False

    try:
        df = pd.read_feather(path)
    except Exception as e:
        logger.warning(f"Could not read standard data file {path.name}: {e}")
        return False

    required_cols = {"date", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning(f"Standard data file {path.name} missing columns: {sorted(missing)}")
        return False

    if len(df) < min_rows:
        logger.warning(f"Standard data file {path.name} has too few rows: {len(df)} < {min_rows}")
        return False

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    if df["date"].isna().any():
        logger.warning(f"Standard data file {path.name} contains invalid dates")
        return False

    if df["date"].duplicated().any():
        logger.warning(f"Standard data file {path.name} contains duplicate candles")
        return False

    if not df["date"].is_monotonic_increasing:
        logger.warning(f"Standard data file {path.name} is not sorted")
        return False

    expected_delta = pd.Timedelta(timeframe)
    bad_diffs = df["date"].diff().dropna()
    bad_diffs = bad_diffs[bad_diffs != expected_delta]
    if not bad_diffs.empty:
        logger.warning(
            f"Standard data file {path.name} has non-{timeframe} spacing at row {bad_diffs.index[0]}"
        )
        return False

    logger.info(f"Standard data OK: {path.name}, rows={len(df)}, latest={df['date'].iloc[-1]}")
    return True


def validate_standard_data(dm: DataManager, min_rows: int) -> bool:
    data_dir = _standard_data_dir(Path(dm.user_data_dir))
    all_ok = True
    for pair in dm.pairs:
        path = data_dir / _standard_filename(pair, dm.timeframe)
        all_ok = _validate_standard_file(path, dm.timeframe, min_rows) and all_ok
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate HARMONIC custom signal data.")
    parser.add_argument(
        "--max-age-minutes",
        type=float,
        default=DataManager.DEFAULT_MAX_AGE_MINUTES,
        help="Maximum age of the latest closed custom candle.",
    )
    parser.add_argument(
        "--check-standard",
        action="store_true",
        help="Also validate standard Freqtrade OHLCV files if you use them for offline workflows.",
    )
    parser.add_argument("--min-standard-rows", type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    dm = DataManager()
    sigma, err_thresh, kept_patterns = _load_strategy_hyperparams(dm.user_data_dir)

    logger.info("Validating custom HARMONIC signal data")
    custom_ok = dm.validate_all_with_signals(
        sigma,
        err_thresh,
        kept_patterns,
        max_age_minutes=args.max_age_minutes,
    )

    standard_ok = True
    if args.check_standard:
        logger.info("Validating standard Freqtrade OHLCV data")
        standard_ok = validate_standard_data(dm, args.min_standard_rows)

    if custom_ok and standard_ok:
        logger.info("All requested validations passed")
        return 0

    logger.error("Validation failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
