"""Backtest vs live parity: inputs, signals, then trades.

Research only: never imported by the strategy or data manager. Read-only on the
producer's custom feathers, freqtrade's native candles and backtest results.
  1. Inputs: native (ccxt) OHLCV must equal the producer's (fapi) OHLCV.
  2. Signals: the strategy's BACKTEST path on native candles must equal the live
     file's signal on every shared candle from --start.
  3. Trades (latest backtest export): every entry is on the candle after a live +1,
     every exit_signal is on the candle after a live <= 0, and every live long start
     with no position open on that pair became a trade.
Exit code 1 on any mismatch.

Run (research/ is not in the compose volume, so mount it), after a backtest with
`--export trades` over the same range:
    docker compose run --rm --no-deps -v "$PWD/research:/freqtrade/research" \
        --entrypoint python freqtrade -B research/check_backtest_parity.py --start 2025-01-01
"""
import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from freqtrade.data.btanalysis import load_backtest_data
from freqtrade.enums import RunMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))
import DATA_MANAGER as dm  # noqa: E402
import HARMONIC as strategy_module  # noqa: E402

COLS = ['date', 'open', 'high', 'low', 'close', 'volume']
STEP = pd.Timedelta(minutes=5)


def show(label, dates, limit=5):
    dates = list(dates)
    print(f"  {label}: {len(dates)}" + (f" e.g. {[str(d) for d in dates[:limit]]}" if dates else ''))
    return len(dates)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--start', default='2025-01-01')
    args = parser.parse_args()
    logging.disable(logging.INFO)

    manager = dm.DataManager()
    user_data = Path(manager.user_data_dir)
    params = dm._load_strategy_hyperparams(manager.user_data_dir)
    strategy = strategy_module.HARMONIC({})
    strategy.dp = SimpleNamespace(runmode=RunMode.BACKTEST)
    strategy.sigma.value, strategy.err_thresh.value, strategy.kept_patterns.value = params
    start = pd.Timestamp(args.start, tz='UTC')
    print(f'params sigma={params[0]} err_thresh={params[1]} kept_patterns={params[2]}, start {start}')
    try:
        trades = load_backtest_data(user_data / 'backtest_results')
    except Exception as e:
        trades = None
        print(f'no backtest export ({e}); trade check skipped')
    failures = 0

    for pair in manager.pairs:
        live = pd.read_feather(manager._get_feather_path(pair, '5m'), columns=COLS + ['signal'])
        live['date'] = pd.to_datetime(live['date'], utc=True)
        native_file = user_data / 'data' / 'binance' / 'futures' / (
            pair.replace('/', '_').replace(':', '_') + '-5m-futures.feather')
        native = pd.read_feather(native_file, columns=COLS)
        native['date'] = pd.to_datetime(native['date'], utc=True)
        end = min(live['date'].iloc[-1], native['date'].iloc[-1])
        print(f'{pair}: live {live.date.iloc[0]}..{live.date.iloc[-1]}, native {native.date.iloc[0]}..'
              f'{native.date.iloc[-1]}, compared {start}..{end}')

        # 1. Inputs.
        both = live.merge(native, on='date', suffixes=('_live', '_native'))
        both = both[both['date'].between(start, end)]
        price_diff = (both[[c + '_live' for c in COLS[1:5]]].to_numpy()
                      != both[[c + '_native' for c in COLS[1:5]]].to_numpy()).any(axis=1)
        missing = set(live.loc[live.date.between(start, end), 'date']) ^ set(
            native.loc[native.date.between(start, end), 'date'])
        failures += show('candles with different OHLC', both.loc[price_diff, 'date'])
        failures += show('candles present in only one source', sorted(missing))
        vol_diff = (both['volume_live'] != both['volume_native']).sum()
        print(f'  volume differs on {vol_diff} candles (not a detector input)')

        # 2. Signals, through the same strategy method a backtest calls, with the
        # same 500-candle startup window freqtrade prepends to the timerange.
        window = native[native['date'] >= start - strategy.startup_candle_count * STEP]
        window = window[window['date'] <= end].reset_index(drop=True)
        bt = strategy.populate_indicators(window.copy(), {'pair': pair})
        bt = bt[bt['date'] >= start].merge(live[['date', 'signal']], on='date', suffixes=('_bt', '_live'))
        differ = bt['signal_bt'] != bt['signal_live']
        failures += show('signal rows differing (backtest path vs live file)', bt.loc[differ, 'date'])
        failures += show('backtest rows without data_ok', bt.loc[~bt['data_ok'], 'date'])
        sig = live.set_index('date')['signal']
        starts = sig[(sig == 1) & (sig.shift(1) != 1)].loc[start:end].index
        print(f'  live long starts in range: {len(starts)}')

        # 3. Trades.
        if trades is None:
            continue
        t = trades[trades['pair'] == pair]
        t_open = pd.to_datetime(t['open_date'], utc=True)
        t_close = pd.to_datetime(t['close_date'], utc=True)
        print(f'  backtest trades: {len(t)} ({t["exit_reason"].value_counts().to_dict()})')
        bad_entry = [d for d in t_open if sig.get(d - STEP) != 1]
        failures += show('entries not after a live +1 candle', bad_entry)
        signal_exits = t_close[t['exit_reason'] == 'exit_signal']
        failures += show('signal exits not after a live <= 0 candle',
                         [d for d in signal_exits if not sig.get(d - STEP, 1) <= 0])
        held = [(o, c) for o, c in zip(t_open, t_close)]
        missed = [s for s in starts
                  if s + STEP not in set(t_open)
                  and not any(o < s + STEP <= c for o, c in held)
                  and s + STEP <= end]
        failures += show('live long starts with no position that did not become a trade', missed)

    print('PASS' if failures == 0 else f'FAIL ({failures} mismatches)')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
