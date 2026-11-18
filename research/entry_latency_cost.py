"""Cost of live signal latency versus the backtest's next-open fill.

Research only. A backtest enters a long at the open of the candle after the
signal candle. Live, the producer publishes 40-160 s after that open (median
41 s BTC / 75 s ETH, measured 2026-11), so the dry-run fill is a later price.
For every historical long start in the producer's file, fetch the Binance 1m
candles opening at the backtest fill time and bracket the live price with the
1m closes at +1, +2 and +3 min. Cost = live price / backtest open - 1, positive =
worse for a long. Compared with the mean price move of the backtest's trades.

Ponytail: the 1m closes only bracket the true fill price. Upgrade path: aggTrades
at the exact latency.

Run:
    docker compose run --rm --no-deps -v "$PWD/research:/freqtrade/research" \
        --entrypoint python freqtrade -B research/entry_latency_cost.py
About 110 light REST requests, spaced 0.2 s.
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from freqtrade.data.btanalysis import load_backtest_data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))
import DATA_MANAGER as dm  # noqa: E402

OFFSETS = [1, 2, 3]  # minutes after the backtest fill time


def main():
    logging.disable(logging.WARNING)
    manager = dm.DataManager()
    try:
        trades = load_backtest_data(Path(manager.user_data_dir) / 'backtest_results')
    except Exception:
        trades = None

    for pair in manager.pairs:
        sig = pd.read_feather(manager._get_feather_path(pair, '5m'), columns=['date', 'signal'])
        sig['date'] = pd.to_datetime(sig['date'], utc=True)
        s = sig['signal'].to_numpy()
        starts = sig['date'][(s == 1) & (np.roll(s, 1) != 1)].iloc[:-1]
        costs = {m: [] for m in OFFSETS}
        for start in starts:
            fill = start + pd.Timedelta(minutes=5)  # backtest entry = this candle's open
            klines = manager._fetch_klines_chunk(manager._pair_to_symbol(pair), '1m',
                                                 int(fill.timestamp() * 1000), limit=len(OFFSETS))
            time.sleep(0.2)
            if not klines or len(klines) < len(OFFSETS) or klines[0][0] != int(fill.timestamp() * 1000):
                continue
            ref = float(klines[0][1])
            for m, k in zip(OFFSETS, klines):
                costs[m].append(float(k[4]) / ref - 1)

        print(f'{pair}: {len(starts)} long starts, {len(costs[1])} with 1m data')
        for m in OFFSETS:
            c = np.array(costs[m]) * 100
            print(f'  live fill at +{m} min: mean {c.mean():+.3f}%  median {np.median(c):+.3f}%  '
                  f'p10 {np.percentile(c, 10):+.3f}%  p90 {np.percentile(c, 90):+.3f}%  '
                  f'mean |cost| {np.abs(c).mean():.3f}%')
        if trades is not None:
            t = trades[trades['pair'] == pair]
            move = (t['close_rate'] / t['open_rate'] - 1) * 100
            print(f'  backtest trades {len(t)}: mean price move {move.mean():+.3f}%, '
                  f'median {move.median():+.3f}% (unleveraged, before fees)')


if __name__ == '__main__':
    main()
