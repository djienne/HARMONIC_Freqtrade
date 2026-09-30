import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from freqtrade.enums import TRADE_MODES, RunMode
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IStrategy, IntParameter

# Import DataManager + the precomputed-signal plumbing.
try:
    from user_data.strategies.DATA_MANAGER import DataManager, sequences_with_three_ones
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    try:
        from DATA_MANAGER import DataManager, sequences_with_three_ones
    except ImportError:
        sys.path.append(str(Path(__file__).parent))
        from DATA_MANAGER import DataManager, sequences_with_three_ones

logger = logging.getLogger(__name__)

# Singleton DataManager so every populate_indicators call reuses config.
data_manager = DataManager()


def _merge_precomputed_signals(data_ft: pd.DataFrame, data: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Join by candle date; an absent signal is unavailable, never an exit."""
    if 'date' not in data.columns or 'signal' not in data.columns:
        raise ValueError(f"Precomputed data for {pair} missing 'date' or 'signal' column")

    if data['date'].dt.tz is None:
        logger.warning(f"Custom data for {pair} is timezone-naive. Assuming UTC.")
        data = data.copy()
        data['date'] = data['date'].dt.tz_localize('UTC')

    # Slice before joining: live calls need only the trader's recent candles.
    signals = data.loc[data['date'].between(data_ft['date'].min(), data_ft['date'].max()), ['date', 'signal']]
    merged = data_ft.drop(columns=['signal', 'data_ok'], errors='ignore').merge(
        signals, on='date', how='left', validate='one_to_one'
    )
    merged['data_ok'] = merged['signal'].notna()
    merged['signal'] = merged['signal'].fillna(0).astype(int)
    return merged


class HARMONIC(IStrategy):
    """
    HARMONIC pattern trading strategy.

    Live trading reads prepared signals. Offline analysis calls the same causal
    detector in memory using the requested parameters and historical candles.
    """

    # RISK NOTE (intentional design — see project review):
    #   * minimal_roi {"0": 5000} effectively disables ROI exits: the strategy
    #     exits only on a bearish/flat harmonic signal, never on a profit target.
    #   * stoploss -0.85 is deliberately very wide. Freqtrade applies the stop on
    #     the *leveraged* trade profit, so at LEV=5x isolated margin it triggers
    #     near a ~17% adverse price move — i.e. just shy of the liquidation point.
    #     A single trade can therefore lose most of its margin; this is accepted.
    #   * Sizing (unlimited stake, tradable_balance_ratio 0.99, max_open_trades 2)
    #     is likewise intentional. Tune via config.json, not here.
    minimal_roi = {"0": 5000}
    stoploss = -0.85
    timeframe = '5m'
    startup_candle_count: int = 500
    use_custom_stoploss: bool = False
    # The producer finishes AFTER the exchange candle arrives. Retry the same
    # candle when its signals arrive; unchanged files are cached below.
    process_only_new_candles: bool = False
    can_short: bool = False
    SIGNAL_MAX_AGE_MINUTES = 15

    # Hyperopt parameters. This strategy is intentionally long-only in live
    # trading; bearish harmonic signals are used as exits, not short entries.
    sigma = DecimalParameter(0.010, 0.050, decimals=3, default=0.035, space='buy', optimize=True)
    err_thresh = DecimalParameter(0.10, 0.30, decimals=2, default=0.16, space='buy', optimize=True)
    kept_patterns = CategoricalParameter(sequences_with_three_ones, default='1011000', space="buy", optimize=True)
    LEV = IntParameter(1, 5, default=5, space='buy', optimize=True)

    def _load_signals(self, pair: str) -> Tuple[pd.DataFrame, bool]:
        """Cache validated live files by revision, but reassess age on every call."""
        params = (self.sigma.value, self.err_thresh.value, self.kept_patterns.value)
        path = Path(data_manager._get_feather_path(pair, self.timeframe))
        paths = (path, Path(data_manager._meta_path(str(path))))
        revision = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in paths) + params
        if not hasattr(self, '_signal_cache'):
            self._signal_cache = {}
        cached = self._signal_cache.get(pair)
        if cached is None or cached[0] != revision:
            try:
                data, _ = data_manager.load_precomputed_signals(
                    pair, *params, max_age_minutes=self.SIGNAL_MAX_AGE_MINUTES
                )
            except Exception as e:
                # Cache failures too: unchanged bad files are not re-validated
                # every loop, which would delay stop-loss checks at 0.1 CPU.
                self._signal_cache[pair] = (revision, e)
                raise
            self._signal_cache[pair] = (revision, data)
        else:
            data = cached[1]
            if isinstance(data, Exception):
                # Fresh traceback: re-raising the same instance would grow it forever.
                raise data.with_traceback(None)
        age = pd.Timestamp.now(tz='UTC') - (data['date'].iloc[-1] + pd.Timedelta(self.timeframe))
        return data, pd.Timedelta(seconds=1) <= age <= pd.Timedelta(minutes=self.SIGNAL_MAX_AGE_MINUTES)

    def _offline_signals(self, dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
        # Use custom OHLCV only BEFORE the supplied candles, as detector warmup.
        # The native backtest candles take precedence; future custom rows and
        # cached signals never participate. This also works with no custom file.
        path = data_manager._get_feather_path(pair, self.timeframe)
        columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        data = dataframe[columns].copy()
        if os.path.exists(path) and not data.empty:
            history = pd.read_feather(path, columns=columns)
            history['date'] = pd.to_datetime(history['date'], utc=True)
            history = history.loc[history['date'] < data['date'].min()]
            data = pd.concat([history, data], ignore_index=True)
        data = data_manager._normalize_ohlcv(data, self.timeframe, drop_unclosed=False)
        data['signal'] = data_manager.compute_signals(
            data, self.sigma.value, self.err_thresh.value, self.kept_patterns.value
        )
        return _merge_precomputed_signals(dataframe, data, pair)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata['pair']
        data_ft = dataframe.copy()
        if data_ft.empty:
            return data_ft.assign(signal=0, data_ok=False)
        runmode = getattr(getattr(self, 'dp', None), 'runmode', None)
        if runmode == RunMode.HYPEROPT and not self.config.get('analyze_per_epoch', False):
            raise ValueError('HARMONIC hyperopt requires --analyze-per-epoch for signal parameters')
        if runmode not in TRADE_MODES:
            # Only live/dry-run read the producer's files. Every other mode
            # (backtest, hyperopt, plot, lookahead/recursive analysis) runs the
            # detector in memory. Offline data errors must fail the experiment,
            # not score a silent zero-trade strategy as if computation had succeeded.
            return self._offline_signals(data_ft, pair)

        # `data_ok` gates BOTH entries and exits. When it is False the bot holds
        # existing positions and opens nothing new — used whenever the custom
        # signal data is unavailable, stale, or has a gap, so a transient data
        # problem never force-flattens open trades.
        try:
            data, fresh = self._load_signals(pair)
        except Exception as e:
            logger.error(f"Error loading signals for {pair}: {e}. Holding (no entries/exits).")
            data_ft['signal'] = 0
            data_ft['data_ok'] = False
            return data_ft

        if data is None or data.empty:
            logger.warning(f"No signal data for {pair}. Holding (no entries/exits).")
            data_ft['signal'] = 0
            data_ft['data_ok'] = False
            return data_ft

        try:
            merged = _merge_precomputed_signals(data_ft, data, pair)
        except Exception as e:
            logger.error(f"Error merging signals for {pair}: {e}. Holding (no entries/exits).")
            data_ft['signal'] = 0
            data_ft['data_ok'] = False
            return data_ft

        merged['data_ok'] &= bool(fresh)
        if not fresh:
            logger.warning(
                f"{pair}: signal data is not fresh; holding open positions and blocking new entries."
            )
        return merged

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        dataframe.loc[(dataframe['signal'] == 1) & dataframe['data_ok'], 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Exit a long whenever the signal is no longer bullish (0 or -1), but only
        # when the data is fresh (data_ok). This also handles direct 1 -> -1 flips
        # on the same candle. On degraded data we hold rather than exit.
        dataframe['exit_long'] = 0
        dataframe.loc[(dataframe['signal'] <= 0) & dataframe['data_ok'], 'exit_long'] = 1
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        return min(self.LEV.value, max_leverage)
