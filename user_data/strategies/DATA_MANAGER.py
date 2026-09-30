import argparse
import math
import requests
import pandas as pd
import numpy as np
import os
import json
import time
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Union, Optional, List, Tuple
from math import log
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================
# Harmonic pattern detection (relocated from HARMONIC.py)
# ============================================================

@dataclass
class XABCD:
    XA_AB: Union[float, list, None]
    AB_BC: Union[float, list, None]
    BC_CD: Union[float, list, None]
    XA_AD: Union[float, list, None]
    name: str


@dataclass
class XABCDFound:
    X: int
    A: int
    B: int
    C: int
    D: int
    error: float
    name: str
    bull: bool


GARTLEY = XABCD(0.618, [0.382, 0.886], [1.13, 1.618], 0.786, "Gartley")
BAT = XABCD([0.382, 0.50], [0.382, 0.886], [1.618, 2.618], 0.886, "Bat")
BUTTERFLY = XABCD(0.786, [0.382, 0.886], [1.618, 2.24], [1.27, 1.41], "Butterfly")
CRAB = XABCD([0.382, 0.618], [0.382, 0.886], [2.618, 3.618], 1.618, "Crab")
DEEP_CRAB = XABCD(0.886, [0.382, 0.886], [2.0, 3.618], 1.618, "Deep Crab")
CYPHER = XABCD([0.382, 0.618], [1.13, 1.41], [1.27, 2.00], 0.786, "Cypher")
SHARK = XABCD(None, [1.13, 1.618], [1.618, 2.24], [0.886, 1.13], "Shark")

PATTERN_LIBRARY = {
    "Gartley": GARTLEY,
    "Bat": BAT,
    "Butterfly": BUTTERFLY,
    "Crab": CRAB,
    "Deep Crab": DEEP_CRAB,
    "Cypher": CYPHER,
    "Shark": SHARK,
}

ALL_PATTERNS = [GARTLEY, BAT, BUTTERFLY, CRAB, DEEP_CRAB, CYPHER, SHARK]

sequences_with_three_ones = ['{0:07b}'.format(i) for i in range(128) if bin(i).count('1') >= 3]


def filter_by_index_map(original_array, index_map_str):
    if len(original_array) != len(index_map_str):
        raise ValueError("Original array and index map must have the same length")
    return [original_array[i] for i in range(len(original_array)) if index_map_str[i] == '1']


def directional_change(close: pd.Series, high: pd.Series, low: pd.Series, sigma: float):
    # Hoist to NumPy once. Scalar .iloc[i] access inside a 600k+ row loop is
    # ~50x slower than indexing a raw ndarray; the state machine below is
    # inherently sequential, so only the per-element access is optimised.
    # Values and output are identical to the prior pandas implementation.
    close_a = np.asarray(close, dtype=np.float64)
    high_a = np.asarray(high, dtype=np.float64)
    low_a = np.asarray(low, dtype=np.float64)

    up_zig = True
    tmp_max = high_a[0]
    tmp_min = low_a[0]
    tmp_max_i = 0
    tmp_min_i = 0

    tops = []
    bottoms = []

    for i in range(len(close_a)):
        if up_zig:
            if high_a[i] > tmp_max:
                tmp_max = high_a[i]
                tmp_max_i = i
            elif close_a[i] < tmp_max - tmp_max * sigma:
                tops.append([i, tmp_max_i, tmp_max])
                up_zig = False
                tmp_min = low_a[i]
                tmp_min_i = i
        else:
            if low_a[i] < tmp_min:
                tmp_min = low_a[i]
                tmp_min_i = i
            elif close_a[i] > tmp_min + tmp_min * sigma:
                bottoms.append([i, tmp_min_i, tmp_min])
                up_zig = True
                tmp_max = high_a[i]
                tmp_max_i = i

    return tops, bottoms


def get_extremes(ohlc: pd.DataFrame, sigma: float):
    if ohlc.empty:
        return pd.DataFrame(columns=['conf_i', 'ext_i', 'ext_p', 'type']).set_index('conf_i')
    tops, bottoms = directional_change(ohlc['close'], ohlc['high'], ohlc['low'], sigma)
    rows = [(*top, 1) for top in tops] + [(*bottom, -1) for bottom in bottoms]
    return pd.DataFrame(rows, columns=['conf_i', 'ext_i', 'ext_p', 'type']).sort_values('conf_i').set_index('conf_i')


def get_error(actual_ratio: float, pattern_ratio: Union[float, list, None]) -> float:
    if pattern_ratio is None:
        return 0.0

    log_actual = log(actual_ratio)

    if isinstance(pattern_ratio, list):
        log_pat0 = log(pattern_ratio[0])
        log_pat1 = log(pattern_ratio[1])
        if log_pat0 > log_pat1:
            log_pat0, log_pat1 = log_pat1, log_pat0
        if log_pat0 <= log_actual <= log_pat1:
            return 0.0
        err = min(abs(log_actual - log_pat0), abs(log_actual - log_pat1))
        return err * 2.0
    elif isinstance(pattern_ratio, float):
        return abs(log_actual - log(pattern_ratio))
    else:
        raise TypeError("Invalid pattern ratio type")


def find_xabcd(ohlc: pd.DataFrame, extremes: pd.DataFrame, err_thresh: float, kept_patterns_map: str):
    used_patterns = filter_by_index_map(ALL_PATTERNS, kept_patterns_map)

    if extremes.empty:
        return {pat.name: {
            'bull_signal': np.zeros(len(ohlc)),
            'bull_patterns': [],
            'bear_signal': np.zeros(len(ohlc)),
            'bear_patterns': [],
        } for pat in used_patterns}

    extremes = extremes.copy()
    extremes['seg_height'] = extremes['ext_p'].diff().abs()
    extremes['retrace_ratio'] = extremes['seg_height'] / extremes['seg_height'].shift(1)

    output = {pat.name: {
        'bull_signal': np.zeros(len(ohlc)),
        'bull_patterns': [],
        'bear_signal': np.zeros(len(ohlc)),
        'bear_patterns': [],
    } for pat in used_patterns}

    if len(extremes) < 4:
        return output

    # Hoist everything the loop touches to NumPy. Positional access on raw
    # ndarrays replaces per-iteration .iloc[]/.index[] lookups (the dominant
    # cost over 600k+ candles). Values and control flow are unchanged; note the
    # ZeroDivisionError branch below is retained for parity even though NumPy
    # scalar division yields inf/nan (which get_error already rejects) rather
    # than raising.
    low_a = ohlc['low'].to_numpy()
    high_a = ohlc['high'].to_numpy()
    conf_idx = extremes.index.to_numpy()
    ext_type_a = extremes['type'].to_numpy()
    ext_p_a = extremes['ext_p'].to_numpy()
    seg_height_a = extremes['seg_height'].to_numpy()
    retrace_ratio_a = extremes['retrace_ratio'].to_numpy()
    ext_i_a = extremes['ext_i'].to_numpy()
    n_ext = len(conf_idx)
    n_ohlc = len(ohlc)

    first_conf = int(conf_idx[0])
    extreme_i = 0
    entry_taken = 0
    pattern_used = None

    for i in range(first_conf, n_ohlc):
        if extreme_i + 1 < n_ext and conf_idx[extreme_i + 1] == i:
            entry_taken = 0
            extreme_i += 1

        if entry_taken != 0:
            if entry_taken == 1:
                output[pattern_used]['bull_signal'][i] = 1
            else:
                output[pattern_used]['bear_signal'][i] = -1
            continue

        # X, A, B and C must be confirmed. No *next* extreme is needed: requiring
        # one would backfill historical entries that were unavailable in real time.
        if extreme_i < 3:
            continue

        ext_type = ext_type_a[extreme_i]
        last_conf_i = int(conf_idx[extreme_i])

        if ext_type > 0.0:
            D_price = low_a[i]
            if low_a[last_conf_i:i + 1].min() < D_price:
                continue
        else:
            D_price = high_a[i]
            if high_a[last_conf_i:i + 1].max() > D_price:
                continue

        try:
            dc_retrace = abs(D_price - ext_p_a[extreme_i]) / seg_height_a[extreme_i]
            xa_ad_retrace = abs(D_price - ext_p_a[extreme_i - 2]) / seg_height_a[extreme_i - 2]
        except (KeyError, IndexError, ZeroDivisionError):
            continue

        best_err = float('inf')
        best_pat = None

        for pat in used_patterns:
            try:
                err = 0.0
                err += get_error(retrace_ratio_a[extreme_i], pat.AB_BC)
                err += get_error(retrace_ratio_a[extreme_i - 1], pat.XA_AB)
                err += get_error(dc_retrace, pat.BC_CD)
                err += get_error(xa_ad_retrace, pat.XA_AD)
                if err < best_err:
                    best_err = err
                    best_pat = pat.name
            except Exception as e:
                logger.debug(f"Error calculating pattern error: {e}")
                continue

        if best_err <= err_thresh and best_pat is not None:
            pattern_data = XABCDFound(
                int(ext_i_a[extreme_i - 3]),
                int(ext_i_a[extreme_i - 2]),
                int(ext_i_a[extreme_i - 1]),
                int(ext_i_a[extreme_i]),
                i, best_err, best_pat, True,
            )
            pattern_used = best_pat
            if ext_type > 0.0:
                entry_taken = 1
                pattern_data.name = "Bull" + pattern_data.name
                pattern_data.bull = True
                output[pattern_used]['bull_signal'][i] = 1
                output[pattern_used]['bull_patterns'].append(pattern_data)
            else:
                entry_taken = -1
                pattern_data.name = "Bear" + pattern_data.name
                pattern_data.bull = False
                output[pattern_used]['bear_signal'][i] = -1
                output[pattern_used]['bear_patterns'].append(pattern_data)

    return output


# ============================================================
# DataManager
# ============================================================

class DataManager:
    BASE_URL = 'https://fapi.binance.com'

    # Bump when calculation semantics change: old precomputed signals must not
    # pass validation just because their numerical parameters still match.
    SIGNAL_VERSION = 3

    TAIL_REPAIR_CANDLES = 12
    # How far the sidecar meta may legitimately differ from the feather while
    # a persist cycle is mid-flight. One repair window plus slack; anything
    # larger is not a write gap and is treated as corruption.
    META_LAG_TOLERANCE_ROWS = 16
    CLOSED_CANDLE_SAFETY_MS = 1000
    FULL_RECOMPUTE_INTERVAL_SECONDS = 24 * 60 * 60
    LOOP_CLOSE_BUFFER_SECONDS = 10
    DEFAULT_MAX_AGE_MINUTES = 15
    VALID_SIGNAL_VALUES = {-1, 0, 1}

    def __init__(self, config_path=None, user_data_dir=None):
        self.root_dir = self._find_project_root()
        logger.info(f"Project root detected: {self.root_dir}")

        if config_path is None:
            self.config_path = os.path.join(self.root_dir, 'user_data', 'config.json')
            if not os.path.exists(self.config_path):
                self.config_path = os.path.join(self.root_dir, 'config.json')
        else:
            self.config_path = config_path

        if user_data_dir is None:
            self.user_data_dir = os.path.join(self.root_dir, 'user_data')
            if not os.path.exists(self.user_data_dir):
                self.user_data_dir = self.root_dir
        else:
            self.user_data_dir = user_data_dir

        self.config = self._load_config()
        self.data_dir = os.path.join(self.user_data_dir, 'data', 'binance', 'futures_custom_data')
        logger.info(f"Custom data directory: {self.data_dir}")
        os.makedirs(self.data_dir, exist_ok=True)

        self.timeframe = '5m'
        self.pairs = self.config.get('exchange', {}).get('pair_whitelist', [])

    def _find_project_root(self):
        current_path = Path(__file__).resolve()

        for _ in range(10):
            if (current_path / 'user_data').exists() and (current_path / 'user_data').is_dir():
                return str(current_path)

            if current_path.name == 'user_data':
                has_config = (current_path / 'config.json').exists()
                has_strategies = (current_path / 'strategies').exists()
                has_data = (current_path / 'data').exists()
                if has_config or has_strategies or has_data:
                    return str(current_path.parent)

            parent = current_path.parent
            if parent == current_path:
                break
            current_path = parent

        logger.warning("Could not detect project root from script location. Using CWD.")
        return os.getcwd()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found at {self.config_path}. Using default root: {self.root_dir}")
            return {}
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _get_feather_path(self, pair, timeframe):
        clean_pair = pair.replace('/USDT:USDT', '').replace('/BUSD:BUSD', '')
        filename = f"{clean_pair}_USDT_USDT-{timeframe}-futures-custom.feather"
        return os.path.join(self.data_dir, filename)

    def _parse_timeframe_to_ms(self, timeframe):
        amount = int(timeframe[:-1])
        unit = timeframe[-1]
        if unit == 'm':
            return amount * 60 * 1000
        elif unit == 'h':
            return amount * 60 * 60 * 1000
        elif unit == 'd':
            return amount * 24 * 60 * 60 * 1000
        elif unit == 'w':
            return amount * 7 * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"Unknown timeframe unit: {unit}")

    def _closed_candle_cutoff_ms(self):
        return int(time.time() * 1000) - self.CLOSED_CANDLE_SAFETY_MS

    def _filter_closed_klines(self, klines):
        cutoff_ms = self._closed_candle_cutoff_ms()
        closed = [k for k in klines if int(k[6]) <= cutoff_ms]
        skipped = len(klines) - len(closed)
        if skipped:
            logger.info(f"Skipped {skipped} incomplete candle(s) from Binance response.")
        return closed

    def _normalize_ohlcv(self, df, timeframe=None, drop_unclosed=True):
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
        if df['date'].isna().any():
            raise ValueError('date column contains invalid values')

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        errors = self._ohlcv_errors(df)
        if errors:
            raise ValueError('; '.join(errors))
        # Stable sort: for a duplicate date the later row (a fresh download) wins.
        df = df.sort_values('date', kind='stable').drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)

        if drop_unclosed and not df.empty:
            timeframe = timeframe or self.timeframe
            timeframe_ms = self._parse_timeframe_to_ms(timeframe)
            cutoff_ms = self._closed_candle_cutoff_ms()
            date_ms = df['date'].astype('int64') // 1_000_000
            closed_mask = (date_ms + timeframe_ms - 1) <= cutoff_ms
            dropped = int((~closed_mask).sum())
            if dropped:
                logger.info(f"Dropped {dropped} incomplete cached candle(s).")
                df = df.loc[closed_mask].reset_index(drop=True)

        return df

    @staticmethod
    def _ohlcv_errors(df):
        prices = df[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        volume = df['volume'].to_numpy(dtype=float)
        errors = []
        if not np.isfinite(prices).all() or not np.isfinite(volume).all():
            errors.append('OHLCV values must be finite')
        if (prices <= 0).any() or (volume < 0).any():
            errors.append('prices must be positive and volume nonnegative')
        if ((df['high'] < df[['open', 'low', 'close']].max(axis=1)).any()
                or (df['low'] > df[['open', 'high', 'close']].min(axis=1)).any()):
            errors.append('OHLC prices violate high/low bounds')
        return errors

    def _pair_to_symbol(self, pair):
        clean = pair.split(':')[0]
        return clean.replace('/', '')

    def _get_start_timestamp(self, date_str):
        if date_str.endswith('Z'):
            date_str = date_str[:-1]
        dt = datetime.fromisoformat(date_str)
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

    def _fetch_klines_chunk(self, symbol, interval, start_time, limit=1500):
        url = f"{self.BASE_URL}/fapi/v1/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'limit': limit,
        }

        retries = 0
        max_retries = 10
        backoff_factor = 2
        current_wait = 1

        while retries < max_retries:
            try:
                response = requests.get(url, params=params, timeout=10)

                if response.status_code in [429, 418]:
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        wait_time = int(retry_after) + 1
                    else:
                        wait_time = current_wait

                    logger.warning(f"Rate limit hit ({response.status_code}). Backing off for {wait_time}s...")
                    time.sleep(wait_time)

                    if not retry_after:
                        current_wait = min(current_wait * backoff_factor, 120)

                    retries += 1
                    continue

                response.raise_for_status()
                return response.json()

            except requests.RequestException as e:
                logger.error(f"Network error fetching klines for {symbol}: {e}. Retrying in {current_wait}s...")
                time.sleep(current_wait)
                current_wait = min(current_wait * backoff_factor, 60)
                retries += 1

        logger.error(f"Max retries exceeded for {symbol} at start_time {start_time}")
        return None

    def _temp_path_for(self, filepath):
        path = Path(filepath)
        for attempt in range(100):
            ts_ms = int(time.time() * 1000)
            suffix = f"{os.getpid()}.{ts_ms}" if attempt == 0 else f"{os.getpid()}.{ts_ms}.{attempt}"
            tmp_path = path.with_name(f".{path.name}.tmp.{suffix}")
            if not tmp_path.exists():
                return tmp_path
        raise RuntimeError(f"Could not allocate temp path for {path}")

    def _cleanup_temp_path(self, path):
        try:
            if path.exists():
                path.unlink()
        except Exception as e:
            logger.warning(f"Could not remove temp file {path}: {e}")

    def _atomic_write_feather(self, df, filepath, label):
        """
        Write a feather file without exposing partial files to readers.

        The temp file is created in the target directory so os.replace can
        atomically swap it into place after validation.
        """
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._temp_path_for(target)

        try:
            df.to_feather(tmp_path)
            pd.read_feather(tmp_path)
            os.replace(tmp_path, target)
            logger.info(f"{label}: Saved {len(df)} rows to {target}")
        except Exception:
            self._cleanup_temp_path(tmp_path)
            raise

    def _atomic_write_json(self, payload, filepath):
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._temp_path_for(target)

        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
                f.write('\n')
            with open(tmp_path, 'r', encoding='utf-8') as f:
                json.load(f)
            os.replace(tmp_path, target)
        except Exception:
            self._cleanup_temp_path(tmp_path)
            raise

    def _quarantine_corrupt_feather(self, filepath, error):
        path = Path(filepath)
        if not path.exists():
            return

        ts = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        corrupt_path = path.with_name(f"{path.stem}.corrupt.{ts}{path.suffix}")
        suffix = 1
        while corrupt_path.exists():
            corrupt_path = path.with_name(f"{path.stem}.corrupt.{ts}.{suffix}{path.suffix}")
            suffix += 1

        try:
            os.replace(path, corrupt_path)
            logger.warning(
                f"Quarantined unreadable feather file {path} -> {corrupt_path}. "
                f"Read error was: {error}"
            )
        except Exception as quarantine_error:
            logger.error(
                f"Could not quarantine unreadable feather file {path}: {quarantine_error}. "
                f"Original read error was: {error}"
            )

    def _should_quarantine_feather_error(self, error):
        error_type = type(error).__name__.lower()
        error_text = str(error).lower()
        if isinstance(error, (FileNotFoundError, PermissionError)):
            return False
        return (
            'arrow' in error_type
            or 'arrow' in error_text
            or 'feather' in error_text
            or 'not a feather' in error_text
        )

    def _read_feather(self, filepath, context, quarantine_on_error=False):
        try:
            return pd.read_feather(filepath)
        except Exception as e:
            if quarantine_on_error and self._should_quarantine_feather_error(e):
                self._quarantine_corrupt_feather(filepath, e)
            logger.error(f"Error reading {filepath} while {context}: {e}")
            raise

    def _save_data(self, df, filepath):
        """
        Write dataframe to feather with integrity checks. Preserves any 'signal'
        column already present in df; callers are responsible for making sure
        signals are aligned to date.
        """
        if df.empty:
            return

        df = self._normalize_ohlcv(df, self.timeframe, drop_unclosed=True)

        if df.empty:
            return

        if not df['date'].is_monotonic_increasing:
            logger.error("Data is NOT monotonic after sorting! Logic error.")
            df = df.sort_values('date').reset_index(drop=True)

        try:
            self._atomic_write_feather(df, filepath, "Checkpoint")
        except Exception as e:
            logger.error(f"Failed to save feather file: {e}")
            raise

    def download_data(self, pair, timeframe, start_date_str='2020-01-01T00:00:00Z', persist=True):
        filepath = self._get_feather_path(pair, timeframe)

        since = self._get_start_timestamp(start_date_str)
        current_data = None
        prefix_len = 0

        existing_data = None
        if os.path.exists(filepath):
            try:
                existing_data = self._read_feather(
                    filepath, f"loading existing data for {pair}", quarantine_on_error=True
                )
                existing_data = self._normalize_ohlcv(existing_data, timeframe, drop_unclosed=True)
                if not existing_data.empty:
                    repair_count = min(self.TAIL_REPAIR_CANDLES, len(existing_data))
                    repair_start = len(existing_data) - repair_count
                    repair_start_ts = existing_data['date'].iloc[repair_start]
                    since = int(repair_start_ts.timestamp() * 1000)
                    current_data = existing_data.iloc[:repair_start].copy()
                    prefix_len = repair_start
                    if current_data.empty:
                        current_data = None
                    logger.info(
                        f"Found existing data for {pair}. Repairing last {repair_count} candle(s) "
                        f"from {datetime.fromtimestamp(since/1000, tz=timezone.utc)}"
                    )
            except Exception as e:
                logger.error(f"Unable to use existing data for {pair}: {e}. Redownloading all.")
                existing_data = None

        now = int(time.time() * 1000)
        symbol = self._pair_to_symbol(pair)

        ohlcv_buffer = []
        SAVE_INTERVAL = 10
        request_count = 0

        while since < now:
            logger.info(f"Downloading {symbol} from {datetime.fromtimestamp(since/1000, tz=timezone.utc)}...")

            klines = self._fetch_klines_chunk(symbol, timeframe, since)

            if not klines:
                break

            closed_klines = self._filter_closed_klines(klines)
            last_kline_close_time = klines[-1][6]
            since = last_kline_close_time + 1

            if not closed_klines:
                logger.info(f"No closed candles returned for {symbol}; waiting for next update.")
                break

            processed_chunk = []
            for k in closed_klines:
                processed_chunk.append([
                    k[0],
                    float(k[1]),
                    float(k[2]),
                    float(k[3]),
                    float(k[4]),
                    float(k[5]),
                ])

            ohlcv_buffer.extend(processed_chunk)

            request_count += 1

            if request_count % SAVE_INTERVAL == 0:
                logger.info("Processing downloaded candle batch...")
                new_data = pd.DataFrame(ohlcv_buffer, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                new_data['date'] = pd.to_datetime(new_data['date'], unit='ms', utc=True)

                if current_data is not None:
                    current_data = pd.concat([current_data, new_data])
                else:
                    current_data = new_data

                if persist:
                    self._save_data(current_data, filepath)
                ohlcv_buffer = []

            time.sleep(0.2)

        if ohlcv_buffer:
            logger.info("Processing final downloaded candle batch...")
            new_data = pd.DataFrame(ohlcv_buffer, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            new_data['date'] = pd.to_datetime(new_data['date'], unit='ms', utc=True)

            if current_data is not None:
                current_data = pd.concat([current_data, new_data])
            else:
                current_data = new_data

            if persist:
                self._save_data(current_data, filepath)

        if not persist:
            # Publish only once signals are ready. A failed repair must not
            # discard the old tail or pass a partial download as a new snapshot.
            # Append only the downloaded rows: they replace old tail rows of the
            # same date (stable sort, keep='last') without copying the history.
            if current_data is None:
                current_data = existing_data
            elif existing_data is not None:
                current_data = pd.concat([existing_data, current_data.iloc[prefix_len:]], ignore_index=True)
            return self._normalize_ohlcv(current_data, timeframe)

        if os.path.exists(filepath):
            try:
                final_data = self._read_feather(
                    filepath, f"verifying downloaded data for {pair}", quarantine_on_error=True
                )
            except Exception:
                final_data = pd.DataFrame()
        else:
            final_data = pd.DataFrame()

        if len(final_data) > 1:
            time_diffs = final_data['date'].diff().dropna()
            expected_diff = self._parse_timeframe_to_ms(timeframe)
            if pd.api.types.is_timedelta64_dtype(time_diffs):
                time_diffs_ms = time_diffs.dt.total_seconds() * 1000
            else:
                time_diffs_ms = time_diffs

            gaps = time_diffs_ms[time_diffs_ms > (expected_diff * 1.5)]

            if not gaps.empty:
                logger.warning(f"Found {len(gaps)} gaps in data for {pair}. First gap at index {gaps.index[0]} of size {gaps.iloc[0]}ms.")
            else:
                logger.info(f"Data continuity verified for {pair}. No significant gaps found.")

        return final_data

    def ensure_data_up_to_date(self, pair, persist=True):
        filepath = self._get_feather_path(pair, self.timeframe)

        needs_update = False
        if not os.path.exists(filepath):
            needs_update = True
        else:
            try:
                df = self._read_feather(
                    filepath, f"checking freshness for {pair}", quarantine_on_error=True
                )
                raw_len = len(df)
                df = self._normalize_ohlcv(df, self.timeframe, drop_unclosed=True)
                if len(df) != raw_len and persist:
                    self._save_data(df, filepath)
                if df.empty:
                    needs_update = True
                else:
                    last_date = df['date'].iloc[-1]
                    now = datetime.now(timezone.utc)
                    expected_open = pd.Timestamp(now).floor('5min') - pd.Timedelta(minutes=5)
                    if last_date < expected_open:
                        needs_update = True
            except Exception as e:
                logger.error(f"Unable to use existing data for {pair}: {e}. Will update.")
                needs_update = True

        if needs_update:
            logger.info(f"Data for {pair} needs update.")
            return self.download_data(pair, self.timeframe, persist=persist)

        return df

    # ---------- signal-related helpers ----------

    def _meta_path(self, feather_path):
        return feather_path + '.signal_meta.json'

    def _fingerprint(self, sigma, err_thresh, kept_patterns_map):
        raw = f"{self.SIGNAL_VERSION}|{float(sigma):.8f}|{float(err_thresh):.8f}|{kept_patterns_map}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _read_signal_meta(self, meta_path):
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read signal meta {meta_path}: {e}")
            return None

    def _as_utc_timestamp(self, value):
        if value is None:
            return None
        try:
            ts = pd.Timestamp(value)
            if pd.isna(ts):
                return None
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
            else:
                ts = ts.tz_convert('UTC')
            return ts
        except Exception:
            return None

    def _format_utc_timestamp(self, value):
        ts = self._as_utc_timestamp(value)
        if ts is None:
            return None
        return ts.isoformat()

    def _timestamps_match(self, actual, expected):
        actual_ts = self._as_utc_timestamp(actual)
        expected_ts = self._as_utc_timestamp(expected)
        if actual_ts is None and expected_ts is None:
            return True
        if actual_ts is None or expected_ts is None:
            return False
        return actual_ts == expected_ts

    def _signal_counts(self, df):
        values = pd.to_numeric(df['signal'], errors='coerce').fillna(0).astype(int)
        counts = values.value_counts().sort_index()
        return {str(int(signal)): int(count) for signal, count in counts.items()}

    def _latest_signal_timestamp(self, df, signal_value=None):
        if 'signal' not in df.columns or df.empty:
            return None
        if signal_value is None:
            subset = df[df['signal'] != 0]
        else:
            subset = df[df['signal'] == signal_value]
        if subset.empty:
            return None
        return self._format_utc_timestamp(subset['date'].max())

    def _signal_age_text(self, timestamp_iso):
        ts = self._as_utc_timestamp(timestamp_iso)
        if ts is None:
            return "never"
        age = pd.Timestamp.now(tz='UTC') - ts
        minutes = age.total_seconds() / 60
        if minutes < 120:
            return f"{minutes:.1f} minutes ago"
        hours = minutes / 60
        if hours < 72:
            return f"{hours:.1f} hours ago"
        return f"{hours / 24:.1f} days ago"

    def _build_signal_metadata(
        self,
        pair,
        df,
        fingerprint,
        sigma,
        err_thresh,
        kept_patterns_map,
        compute_mode,
        previous_meta=None,
    ):
        now_iso = datetime.now(timezone.utc).isoformat()
        last_full_compute_at = None
        if compute_mode == 'full':
            last_full_compute_at = now_iso
        elif previous_meta is not None:
            last_full_compute_at = previous_meta.get('last_full_compute_at')
        if not last_full_compute_at:
            last_full_compute_at = now_iso

        return {
            'metadata_version': 2,
            'valid': True,
            'pair': pair,
            'timeframe': self.timeframe,
            'fingerprint': fingerprint,
            'sigma': float(sigma),
            'err_thresh': float(err_thresh),
            'kept_patterns_map': kept_patterns_map,
            'row_count': int(len(df)),
            'latest_candle': self._format_utc_timestamp(df['date'].iloc[-1]) if not df.empty else None,
            'latest_long_signal': self._latest_signal_timestamp(df, 1),
            'latest_nonzero_signal': self._latest_signal_timestamp(df, None),
            'signal_counts': self._signal_counts(df),
            'compute_mode': compute_mode,
            'last_update_at': now_iso,
            'last_full_compute_at': last_full_compute_at,
            'computed_at': now_iso,
        }

    def _full_recompute_due(self, meta):
        if not meta:
            return True
        last_full = self._as_utc_timestamp(meta.get('last_full_compute_at'))
        if last_full is None:
            return True
        age_seconds = (pd.Timestamp.now(tz='UTC') - last_full).total_seconds()
        return age_seconds >= self.FULL_RECOMPUTE_INTERVAL_SECONDS

    def _validate_signal_dataframe(
        self,
        pair,
        df,
        meta,
        sigma,
        err_thresh,
        kept_patterns_map,
        max_age_minutes,
    ):
        """
        Classify findings about the prepared signal dataframe.

        Returns (fatal, soft, fresh_msgs, df):
        - fatal: data is structurally unusable, or its signal params don't match
          the requested (sigma, err_thresh, kept_patterns_map). Never trade on it.
        - soft: data is usable but imperfect (e.g. a historical gap in 5m
          spacing). Logged as a warning; does NOT block trading.
        - fresh_msgs: the recent tail is stale or not yet closed. Used to gate
          new activity at startup and to flag degraded/hold mode at runtime.
          Always empty when max_age_minutes is None (backtest/hyperopt).
        """
        fatal: List[str] = []
        soft: List[str] = []
        fresh_msgs: List[str] = []
        required_cols = {'date', 'open', 'high', 'low', 'close', 'volume', 'signal'}

        if df is None or df.empty:
            return ["custom signal file is empty"], soft, fresh_msgs, pd.DataFrame()

        missing = required_cols - set(df.columns)
        if missing:
            fatal.append(f"missing columns: {sorted(missing)}")
            return fatal, soft, fresh_msgs, df

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
        if df['date'].isna().any():
            fatal.append("date column contains invalid values")

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        fatal.extend(self._ohlcv_errors(df))

        signal_numeric = pd.to_numeric(df['signal'], errors='coerce')

        # The feather has two writers. `_save_data` ("Checkpoint") appends newly
        # downloaded candles and only *preserves* an existing signal column, so
        # those rows carry NaN until `persist_signals` recomputes and rewrites,
        # then updates the meta. A reader landing in that window sees trailing
        # NaN signals AND a row_count one ahead of the meta. Treating either as
        # fatal makes the strategy refuse the file and hold - which is why this
        # bot never opened a single trade.
        #
        # A NaN block at the very END is that known window: drop it and use the
        # last fully-signalled history. Truncating to the last signalled row is
        # exactly the state the meta describes, so the row_count and
        # latest_candle checks below then line up. NaN anywhere else is real
        # corruption and stays fatal.
        dropped_trailing = 0
        na_signal = signal_numeric.isna()
        if na_signal.any():
            valid_positions = (~na_signal).to_numpy().nonzero()[0]
            trailing_only = (
                len(valid_positions) > 0
                and bool(na_signal.to_numpy()[valid_positions[-1] + 1:].all())
            )
            # `keep < len(df)` matters: with NaN only in the middle the trailing
            # slice is empty and .all() is vacuously True, so without this guard
            # a corrupt file would log a misleading "dropped 0 trailing".
            keep = valid_positions[-1] + 1 if len(valid_positions) else 0
            if trailing_only and keep < len(df):
                dropped_trailing = len(df) - keep
                soft.append(
                    f"dropped {dropped_trailing} trailing candle(s) awaiting signal computation"
                )
                df = df.iloc[:keep].copy()
                signal_numeric = signal_numeric.iloc[:keep]
                na_signal = na_signal.iloc[:keep]

        if na_signal.any():
            fatal.append("signal column contains invalid values")
        else:
            signal_int = signal_numeric.astype(int)
            if not (signal_numeric == signal_int).all():
                fatal.append("signal column contains non-integer values")
            invalid_signals = set(signal_int.unique()) - self.VALID_SIGNAL_VALUES
            if invalid_signals:
                fatal.append(f"signal column contains values outside [-1, 0, 1]: {sorted(invalid_signals)}")
            df['signal'] = signal_int

        if dropped_trailing > self.META_LAG_TOLERANCE_ROWS:
            fatal.append('unfinished signal tail exceeds the write-window tolerance')

        if df['date'].duplicated().any():
            fatal.append("date column contains duplicate candles")

        if not df['date'].is_monotonic_increasing:
            fatal.append("date column is not strictly sorted")

        # Historical gaps are tolerated: the exchange can have missing candles
        # over multi-year history. We warn (soft) instead of rejecting, so one
        # old missing candle does not disable trading.
        if len(df) > 1:
            expected_delta = pd.Timedelta(milliseconds=self._parse_timeframe_to_ms(self.timeframe))
            diffs = df['date'].diff().dropna()
            bad_diffs = diffs[diffs != expected_delta]
            if not bad_diffs.empty:
                first_bad = bad_diffs.index[0]
                soft.append(
                    f"date spacing is not strict {self.timeframe}: {len(bad_diffs)} gap(s); "
                    f"first bad row index={first_bad}, diff={bad_diffs.iloc[0]}"
                )

        # Tail freshness is a separate concern from structural integrity: stale
        # data is still usable for holding open positions; it just must not drive
        # new entries. Skipped entirely when max_age_minutes is None (backtest).
        if not df['date'].isna().all():
            timeframe_delta = pd.Timedelta(milliseconds=self._parse_timeframe_to_ms(self.timeframe))
            latest_open = df['date'].iloc[-1]
            latest_close = latest_open + timeframe_delta
            now = pd.Timestamp.now(tz='UTC')
            safety_delta = pd.Timedelta(milliseconds=self.CLOSED_CANDLE_SAFETY_MS)

            if latest_close > now - safety_delta:
                fresh_msgs.append(
                    f"latest candle is not closed yet: latest_open={latest_open.isoformat()}"
                )

            if max_age_minutes is not None:
                age_minutes = (now - latest_close).total_seconds() / 60
                if age_minutes > max_age_minutes:
                    fresh_msgs.append(
                        f"latest closed candle is stale: age={age_minutes:.1f}m, max={max_age_minutes}m"
                    )

        if meta is None:
            fatal.append("signal metadata file is missing or unreadable")
            return fatal, soft, fresh_msgs, df

        fingerprint = self._fingerprint(sigma, err_thresh, kept_patterns_map)
        if meta.get('valid') is not True:
            fatal.append("signal metadata is not marked valid")
        if meta.get('pair') != pair:
            fatal.append(f"metadata pair mismatch: expected {pair}, got {meta.get('pair')}")
        if meta.get('timeframe') != self.timeframe:
            fatal.append(f"metadata timeframe mismatch: expected {self.timeframe}, got {meta.get('timeframe')}")
        if meta.get('fingerprint') != fingerprint:
            fatal.append(f"metadata fingerprint mismatch: expected {fingerprint}, got {meta.get('fingerprint')}")
        try:
            if abs(float(meta.get('sigma')) - float(sigma)) > 1e-12:
                fatal.append("metadata sigma mismatch")
            if abs(float(meta.get('err_thresh')) - float(err_thresh)) > 1e-12:
                fatal.append("metadata err_thresh mismatch")
        except Exception:
            fatal.append("metadata signal parameters are invalid")
        if meta.get('kept_patterns_map') != kept_patterns_map:
            fatal.append("metadata kept_patterns_map mismatch")
        # A persist cycle updates the feather and its meta as two separate
        # atomic writes, so a reader can land between them in EITHER direction:
        #
        #   meta AHEAD  - checkpoint blanked the tail (TAIL_REPAIR_CANDLES rows
        #                 re-downloaded without signals) and we truncated above,
        #                 so the meta still describes the larger prior state.
        #   meta BEHIND - signals were just persisted to the feather and the
        #                 meta write has not landed yet (the verify re-read of
        #                 ~700k rows sits between them).
        #
        # Both are brief and benign. The bound keeps that narrow: a lag larger
        # than one repair window is not a write gap and stays fatal, as does any
        # mismatch when the data itself is suspect. The parameter checks above
        # (fingerprint/sigma/err_thresh/kept_patterns) remain the real guard
        # against trading on the wrong signals.
        meta_rows = int(meta.get('row_count', -1))
        meta_lag = len(df) - meta_rows
        in_write_window = (
            (0 < dropped_trailing <= self.META_LAG_TOLERANCE_ROWS
             and -self.META_LAG_TOLERANCE_ROWS <= meta_lag <= 0)
            or (not dropped_trailing and 0 < meta_lag <= self.META_LAG_TOLERANCE_ROWS)
        )
        if meta_rows != len(df):
            if in_write_window:
                direction = "ahead of" if meta_lag < 0 else "behind"
                soft.append(
                    f"metadata row_count {meta_rows} is {direction} the {len(df)} row(s) "
                    "held; a persist cycle is mid-flight"
                )
            else:
                fatal.append(
                    f"metadata row_count mismatch: expected {len(df)}, got {meta_rows}")

        latest_candle = self._format_utc_timestamp(df['date'].iloc[-1]) if not df.empty else None
        if not self._timestamps_match(meta.get('latest_candle'), latest_candle):
            if in_write_window:
                soft.append(
                    f"metadata latest_candle {meta.get('latest_candle')} differs from "
                    f"{latest_candle}; persist cycle mid-flight"
                )
            else:
                fatal.append(
                    f"metadata latest_candle mismatch: expected {latest_candle}, got {meta.get('latest_candle')}"
                )

        # Same write-window reasoning as the row_count/latest_candle checks: with
        # trailing rows dropped, every derived aggregate is computed over a
        # PREFIX of what the meta describes. A prefix can only have counts <=
        # the meta's, so that is tolerable; anything else is genuine corruption.
        actual_counts = self._signal_counts(df)
        meta_counts = {str(k): int(v) for k, v in (meta.get('signal_counts') or {}).items()}
        if meta_counts != actual_counts:
            # One side must be a prefix of the other, and which side depends on
            # the lag direction: with meta ahead our counts are the smaller
            # subset; with meta behind, theirs are. Counts moving the *wrong*
            # way for the observed lag means the two files disagree about
            # history, not about recency, so that stays fatal.
            keys = set(actual_counts) | set(meta_counts)
            if meta_lag < 0:
                consistent = all(actual_counts.get(k, 0) <= meta_counts.get(k, 0) for k in keys)
            else:
                consistent = all(meta_counts.get(k, 0) <= actual_counts.get(k, 0) for k in keys)
            if in_write_window and consistent:
                soft.append(
                    "metadata signal_counts cover a different candle span than the "
                    "rows held; persist cycle mid-flight"
                )
            else:
                fatal.append(
                    f"metadata signal_counts mismatch: expected {actual_counts}, got {meta_counts}")

        latest_long = self._latest_signal_timestamp(df, 1)
        latest_nonzero = self._latest_signal_timestamp(df, None)
        if not self._timestamps_match(meta.get('latest_long_signal'), latest_long):
            if in_write_window:
                soft.append("metadata latest_long_signal differs; persist cycle mid-flight")
            else:
                fatal.append("metadata latest_long_signal mismatch")
        if not self._timestamps_match(meta.get('latest_nonzero_signal'), latest_nonzero):
            if in_write_window:
                soft.append("metadata latest_nonzero_signal differs; persist cycle mid-flight")
            else:
                fatal.append("metadata latest_nonzero_signal mismatch")

        if self._as_utc_timestamp(meta.get('last_update_at')) is None:
            fatal.append("metadata last_update_at is missing or invalid")
        if self._as_utc_timestamp(meta.get('last_full_compute_at')) is None:
            fatal.append("metadata last_full_compute_at is missing or invalid")

        return fatal, soft, fresh_msgs, df

    def validate_signal_file(
        self,
        pair,
        sigma,
        err_thresh,
        kept_patterns_map,
        max_age_minutes: Optional[float] = DEFAULT_MAX_AGE_MINUTES,
        quarantine_on_error: bool = False,
    ) -> Tuple[bool, bool, List[str], pd.DataFrame]:
        """
        Returns (structurally_ok, fresh, messages, df).
        - structurally_ok: no fatal problems; the signals are safe to use.
        - fresh: the recent tail is current (always True when max_age_minutes is None).
        - messages: all findings (fatal + soft + freshness), for logging.
        Soft (historical-gap) findings are logged here as warnings and never
        affect structurally_ok.
        """
        filepath = self._get_feather_path(pair, self.timeframe)
        meta_path = self._meta_path(filepath)

        if not os.path.exists(filepath):
            return False, False, [f"custom signal file not found: {filepath}"], pd.DataFrame()

        try:
            df = self._read_feather(
                filepath,
                f"validating signal data for {pair}",
                quarantine_on_error=quarantine_on_error,
            )
        except Exception as e:
            return False, False, [f"could not read custom signal file: {e}"], pd.DataFrame()

        meta = self._read_signal_meta(meta_path)
        fatal, soft, fresh_msgs, normalized_df = self._validate_signal_dataframe(
            pair,
            df,
            meta,
            sigma,
            err_thresh,
            kept_patterns_map,
            max_age_minutes,
        )
        for msg in soft:
            logger.warning(f"Signal data for {pair} has a tolerable issue: {msg}")
        structurally_ok = len(fatal) == 0
        fresh = len(fresh_msgs) == 0
        messages = fatal + soft + fresh_msgs
        return structurally_ok, fresh, messages, normalized_df

    def load_precomputed_signals(
        self,
        pair,
        sigma,
        err_thresh,
        kept_patterns_map,
        max_age_minutes: Optional[float] = DEFAULT_MAX_AGE_MINUTES,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Returns (df, fresh). Raises ValueError only on fatal problems (structural
        corruption or wrong signal params). Staleness and historical gaps are
        non-fatal: they surface via fresh=False / warnings so the strategy can
        degrade to hold-mode instead of flattening open positions.
        """
        structurally_ok, fresh, messages, df = self.validate_signal_file(
            pair,
            sigma,
            err_thresh,
            kept_patterns_map,
            max_age_minutes=max_age_minutes,
            quarantine_on_error=False,
        )
        if not structurally_ok:
            raise ValueError("; ".join(messages))
        if not fresh:
            logger.warning(f"Signal data for {pair} is not fresh: {'; '.join(messages)}")
        return df, fresh

    def compute_signals(self, df: pd.DataFrame, sigma, err_thresh, kept_patterns_map) -> np.ndarray:
        """Full-history causal signal computation. Returns int64 array of length len(df)."""
        if not 0 < sigma < 1 or not np.isfinite(err_thresh) or err_thresh < 0:
            raise ValueError('sigma must be in (0, 1) and err_thresh finite and nonnegative')
        if len(kept_patterns_map) != len(ALL_PATTERNS) or set(kept_patterns_map) - {'0', '1'}:
            raise ValueError('kept_patterns_map must contain seven binary digits')
        errors = self._ohlcv_errors(df)
        if errors:
            raise ValueError('; '.join(errors))
        extremes = get_extremes(df, sigma)
        output = find_xabcd(df, extremes, err_thresh, kept_patterns_map)
        used_patterns = filter_by_index_map(ALL_PATTERNS, kept_patterns_map)
        sig = np.zeros(len(df))
        for pat in used_patterns:
            sig += output[pat.name]['bear_signal'] + output[pat.name]['bull_signal']
        return sig.astype(np.int64)

    def ensure_data_and_signals_up_to_date(
        self,
        pair: str,
        sigma: float,
        err_thresh: float,
        kept_patterns_map: str,
        update_ohlcv: bool = True,
        persist_signals: bool = True,
        force_full_recompute: bool = False,
    ) -> pd.DataFrame:
        """
        Guarantee the feather for `pair` has current OHLCV and a `signal` column
        matching `(sigma, err_thresh, kept_patterns_map)`.

        - `update_ohlcv=True` (live/dry-run): fetches missing candles first.
        - `update_ohlcv=False` (backtest/hyperopt): reads feather as-is; no network.
        - `persist_signals=True`: writes signal column + sidecar meta to disk.
        - `persist_signals=False`: returns df with in-memory signal only.
        - `force_full_recompute=True`: ignore reusable cached signals.
        """
        filepath = self._get_feather_path(pair, self.timeframe)
        meta_path = self._meta_path(filepath)

        if update_ohlcv:
            df = self.ensure_data_up_to_date(pair, persist=False)
        else:
            if not os.path.exists(filepath):
                logger.error(f"No feather found for {pair} at {filepath} (update_ohlcv=False).")
                return pd.DataFrame()
            try:
                df = self._read_feather(
                    filepath, f"loading signal data for {pair}", quarantine_on_error=True
                )
            except Exception:
                return pd.DataFrame()

        if df is None or df.empty:
            logger.warning(f"Empty dataframe for {pair}")
            return df if df is not None else pd.DataFrame()

        df = self._normalize_ohlcv(df, self.timeframe, drop_unclosed=True)
        if df.empty:
            logger.warning(f"Empty normalized dataframe for {pair}")
            return df

        fingerprint = self._fingerprint(sigma, err_thresh, kept_patterns_map)
        row_count = len(df)
        cached_meta = self._read_signal_meta(meta_path)
        force_full = force_full_recompute or self._full_recompute_due(cached_meta)

        has_signal_column = 'signal' in df.columns
        signal_col_complete = has_signal_column and not df['signal'].isna().any()
        cache_fp_match = cached_meta is not None and cached_meta.get('fingerprint') == fingerprint
        cache_rows_match = cached_meta is not None and cached_meta.get('row_count') == row_count

        if has_signal_column and signal_col_complete and cache_fp_match and cache_rows_match and not force_full:
            structurally_ok, fresh, messages, validated_df = self.validate_signal_file(
                pair,
                sigma,
                err_thresh,
                kept_patterns_map,
                max_age_minutes=self.DEFAULT_MAX_AGE_MINUTES if update_ohlcv else None,
                quarantine_on_error=False,
            )
            if structurally_ok and fresh:
                return validated_df
            logger.info(f"Cached signal data for {pair} failed validation; recomputing: {'; '.join(messages)}")

        t0 = time.time()
        # A slice resets directional-change state and erases signals at its
        # boundary. Full computation fits the 5m budget for two pairs. If this
        # stops fitting, carry exact detector state rather than guessing a warmup.
        new_sig = self.compute_signals(df, sigma, err_thresh, kept_patterns_map)
        mode = 'full'
        elapsed = time.time() - t0

        logger.info(
            f"Computed signals [{mode}] for {pair}: rows={row_count}, fingerprint={fingerprint}, "
            f"nonzero={int((new_sig != 0).sum())}, took {elapsed:.1f}s"
        )

        df = df.copy()
        df['signal'] = new_sig

        if persist_signals:
            try:
                self._atomic_write_feather(df, filepath, f"Persisted signals for {pair}")
            except Exception as e:
                logger.error(f"Failed to persist signals for {pair}: {e}")
                raise

            meta_payload = self._build_signal_metadata(
                pair,
                df,
                fingerprint,
                sigma,
                err_thresh,
                kept_patterns_map,
                mode,
                previous_meta=cached_meta,
            )
            try:
                written_df = self._read_feather(filepath, f"verifying persisted signals for {pair}")
                written_df = self._normalize_ohlcv(written_df, self.timeframe, drop_unclosed=False)
                if len(written_df) != row_count:
                    raise RuntimeError(f"row count changed after write: expected={row_count}, got={len(written_df)}")
                if self._format_utc_timestamp(written_df['date'].iloc[-1]) != meta_payload['latest_candle']:
                    raise RuntimeError("latest candle changed after write")
                self._atomic_write_json(meta_payload, meta_path)
                logger.info(f"Persisted signal meta for {pair} to {meta_path}")
            except Exception as e:
                logger.error(f"Persisted signals for {pair}, but failed to persist signal meta: {e}")
                raise

        return df

    def update_all(self):
        for pair in self.pairs:
            self.download_data(pair, self.timeframe)

    def update_all_with_signals(self, sigma, err_thresh, kept_patterns_map, force_full_recompute=False):
        all_ok = True
        for pair in self.pairs:
            try:
                self.ensure_data_and_signals_up_to_date(
                    pair, sigma, err_thresh, kept_patterns_map,
                    update_ohlcv=True,
                    persist_signals=True,
                    force_full_recompute=force_full_recompute,
                )
            except Exception as e:
                logger.error(f"update_all_with_signals failed for {pair}: {e}")
                all_ok = False
        return all_ok

    def validate_all_with_signals(
        self,
        sigma,
        err_thresh,
        kept_patterns_map,
        max_age_minutes: Optional[float] = DEFAULT_MAX_AGE_MINUTES,
    ):
        all_ok = True
        for pair in self.pairs:
            structurally_ok, fresh, messages, df = self.validate_signal_file(
                pair,
                sigma,
                err_thresh,
                kept_patterns_map,
                max_age_minutes=max_age_minutes,
                quarantine_on_error=False,
            )
            if structurally_ok and fresh:
                latest_candle = self._format_utc_timestamp(df['date'].iloc[-1]) if not df.empty else None
                latest_long = self._latest_signal_timestamp(df, 1)
                latest_nonzero = self._latest_signal_timestamp(df, None)
                logger.info(
                    f"Validation OK for {pair}: rows={len(df)}, latest_candle={latest_candle}, "
                    f"latest_long={latest_long or 'never'} ({self._signal_age_text(latest_long)}), "
                    f"latest_nonzero={latest_nonzero or 'never'} ({self._signal_age_text(latest_nonzero)})"
                )
            else:
                all_ok = False
                logger.error(f"Validation failed for {pair}: {'; '.join(messages)}")
        return all_ok

    def _seconds_until_next_candle_close(self):
        timeframe_seconds = self._parse_timeframe_to_ms(self.timeframe) / 1000
        now = time.time()
        next_boundary = (math.floor(now / timeframe_seconds) + 1) * timeframe_seconds
        next_run = next_boundary + self.LOOP_CLOSE_BUFFER_SECONDS
        return max(5.0, next_run - now)

    def run_loop_with_signals(
        self,
        sigma,
        err_thresh,
        kept_patterns_map,
        max_age_minutes: Optional[float] = DEFAULT_MAX_AGE_MINUTES,
        force_full_recompute: bool = False,
    ):
        force_full_next = force_full_recompute
        while True:
            try:
                update_ok = self.update_all_with_signals(
                    sigma,
                    err_thresh,
                    kept_patterns_map,
                    force_full_recompute=force_full_next,
                )
                force_full_next = False
                validation_ok = self.validate_all_with_signals(
                    sigma,
                    err_thresh,
                    kept_patterns_map,
                    max_age_minutes=max_age_minutes,
                )
                if not update_ok or not validation_ok:
                    logger.error("Data-manager cycle completed with errors; will retry next candle.")
            except Exception as e:
                logger.exception(f"Data-manager cycle failed unexpectedly: {e}")

            sleep_seconds = self._seconds_until_next_candle_close()
            logger.info(f"Sleeping {sleep_seconds:.1f}s until next closed-candle update cycle.")
            time.sleep(sleep_seconds)


# ============================================================
# __main__ — container startup entrypoint
# ============================================================

def _load_strategy_hyperparams(user_data_dir):
    """Read sigma / err_thresh / kept_patterns from HARMONIC.json (fallback to defaults)."""
    path = os.path.join(user_data_dir, 'strategies', 'HARMONIC.json')
    defaults = (0.035, 0.16, '1011000')
    if not os.path.exists(path):
        logger.warning(f"HARMONIC.json not found at {path}; using defaults {defaults}.")
        return defaults
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        buy = data.get('params', {}).get('buy', {})
        return (
            float(buy.get('sigma', defaults[0])),
            float(buy.get('err_thresh', defaults[1])),
            str(buy.get('kept_patterns', defaults[2])),
        )
    except Exception as e:
        logger.error(f"Failed to parse HARMONIC.json: {e}; using defaults {defaults}.")
        return defaults


def _setup_logging(log_file=None):
    handlers = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding='utf-8'))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Manage long-history HARMONIC custom signal data.")
    parser.add_argument("--loop", action="store_true", help="Run forever, updating after each 5m candle closes.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate prepared signal files and metadata.")
    parser.add_argument("--full-recompute", action="store_true", help="Force a full signal recompute on the first cycle.")
    parser.add_argument(
        "--max-age-minutes",
        type=float,
        default=DataManager.DEFAULT_MAX_AGE_MINUTES,
        help="Maximum age of the latest closed candle for validation.",
    )
    parser.add_argument("--log-file", default=None, help="Optional extra log file path.")
    args = parser.parse_args()

    _setup_logging(args.log_file)
    dm = DataManager()
    sigma, err_thresh, kept_patterns = _load_strategy_hyperparams(dm.user_data_dir)
    logger.info(f"Hyperparams loaded: sigma={sigma}, err_thresh={err_thresh}, kept_patterns={kept_patterns}")

    if args.validate_only:
        return 0 if dm.validate_all_with_signals(
            sigma,
            err_thresh,
            kept_patterns,
            max_age_minutes=args.max_age_minutes,
        ) else 1

    if args.loop:
        dm.run_loop_with_signals(
            sigma,
            err_thresh,
            kept_patterns,
            max_age_minutes=args.max_age_minutes,
            force_full_recompute=args.full_recompute,
        )
        return 0

    update_ok = dm.update_all_with_signals(
        sigma,
        err_thresh,
        kept_patterns,
        force_full_recompute=args.full_recompute,
    )
    validation_ok = dm.validate_all_with_signals(
        sigma,
        err_thresh,
        kept_patterns,
        max_age_minutes=args.max_age_minutes,
    )
    return 0 if update_ok and validation_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
