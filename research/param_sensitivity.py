"""Sensitivity of the causal detector's trades to sigma and err_thresh.

Research only: never imported by the strategy or data manager. Reads the
producer's custom feathers read-only and runs the production detector over a
sigma x err_thresh grid; kept_patterns, stop-loss and LEV come from
HARMONIC.json, the fee from config.json.

Trade walk (crude, for relative comparison only; NOT a freqtrade backtest):
enter at the next open when signal == 1 and flat; exit at the next open when
signal <= 0; stop at entry * (1 - |stoploss| / LEV) when a later low reaches it.
Returns are per trade at price level (1x), minus the fee on both sides. No
funding, slippage or intrabar ordering. Trades are split by entry date at
--split (default: the HARMONIC.json export date, i.e. in-sample vs after).

Run (research/ is not in the compose volume, so mount it):
    docker compose run --rm --no-deps -v "$PWD/research:/freqtrade/research" \
        --entrypoint python freqtrade -B research/param_sensitivity.py
30 full-history detector runs: about 15 minutes at the 0.1 CPU limit.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))
from DATA_MANAGER import DataManager  # noqa: E402

SIGMAS = (0.025, 0.030, 0.035, 0.040, 0.045)
ERR_THRESHOLDS = (0.12, 0.16, 0.20)


def walk(df, sig, stop, fee):
    """Return (entry row, net price return) for each trade."""
    opens, lows = df['open'].to_numpy(), df['low'].to_numpy()
    trades, entry, entry_i = [], None, None
    for i in range(len(df) - 1):
        if entry is not None and i > entry_i and lows[i] <= entry * (1 - stop):
            trades.append((entry_i, -stop - 2 * fee))
            entry = None
            continue
        if entry is not None and sig[i] <= 0:
            trades.append((entry_i, opens[i + 1] / entry - 1 - 2 * fee))
            entry = None
            continue
        if entry is None and sig[i] == 1:
            entry, entry_i = opens[i + 1], i + 1
    return trades


def summary(returns):
    if not returns:
        return '   0     n/a     n/a'
    r = np.array(returns)
    return f'{len(r):4d} {r.mean() * 100:+6.2f}% {(r > 0).mean() * 100:5.0f}%'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--split', default='2025-03-10', help='UTC date splitting trades by entry.')
    args = parser.parse_args()

    dm = DataManager()
    params = json.loads((Path(dm.user_data_dir) / 'strategies' / 'HARMONIC.json').read_text())['params']
    buy = params['buy']
    stop = abs(params['stoploss']['stoploss']) / buy['LEV']
    fee = float(dm.config.get('fee', 0.001))
    split = pd.Timestamp(args.split, tz='UTC')

    frames = []
    for pair in dm.pairs:
        path = dm._get_feather_path(pair, dm.timeframe)
        df = pd.read_feather(path, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df = dm._normalize_ohlcv(df, dm.timeframe, drop_unclosed=False)
        frames.append(df)
        print(f'{pair}: {len(df)} candles {df.date.iloc[0]:%Y-%m-%d}..{df.date.iloc[-1]:%Y-%m-%d}')
    print(f"kept_patterns={buy['kept_patterns']} stop={stop:.3f} (price) fee={fee}/side; * = live params")
    print(f'sigma  err    | before {split:%Y-%m-%d}: n   mean   win% | after: n   mean   win%')

    for sigma in SIGMAS:
        for err in ERR_THRESHOLDS:
            before, after = [], []
            for df in frames:
                sig = dm.compute_signals(df, sigma, err, buy['kept_patterns'])
                for entry_i, ret in walk(df, sig, stop, fee):
                    (before if df['date'].iloc[entry_i] < split else after).append(ret)
            live = '*' if (sigma, err) == (buy['sigma'], buy['err_thresh']) else ' '
            print(f'{sigma:.3f} {err:.2f} {live} |        {summary(before)} |     {summary(after)}')


if __name__ == '__main__':
    main()
