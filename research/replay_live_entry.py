"""Replay real long-signal starts through the live producer -> trader path.

Research only: never imported by the strategy or data manager. Reads the
producer's custom feathers read-only. For the last N long-run starts per pair:
  1. the detector on history ending one candle BEFORE the start gives 0 there;
  2. history ending AT the start, re-dated so that candle closed minutes ago, is
     published by the real producer function into a temp dir, then read by the
     real dry-run strategy path: the newest candle must give enter_long=1.
Parameters come from HARMONIC.json, as in production. Exit code 1 on any failure.

Run (research/ is not in the compose volume, so mount it):
    docker compose run --rm --no-deps -v "$PWD/research:/freqtrade/research" \
        --entrypoint python freqtrade -B research/replay_live_entry.py --last 3
About 1 minute per start at the 0.1 CPU limit (two full-history detector runs).
"""
import argparse
import logging
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from freqtrade.enums import RunMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))
import DATA_MANAGER as dm  # noqa: E402
import HARMONIC as strategy_module  # noqa: E402

COLS = ['date', 'open', 'high', 'low', 'close', 'volume']


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--last', type=int, default=3, help='long-run starts per pair')
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    manager = dm.DataManager()
    params = dm._load_strategy_hyperparams(manager.user_data_dir)
    strategy = strategy_module.HARMONIC({})
    strategy.dp = SimpleNamespace(runmode=RunMode.DRY_RUN)
    strategy.sigma.value, strategy.err_thresh.value, strategy.kept_patterns.value = params
    sources = {pair: manager._get_feather_path(pair, '5m') for pair in manager.pairs}
    failures = 0
    print(f'params sigma={params[0]} err_thresh={params[1]} kept_patterns={params[2]}')

    for pair, source in sources.items():
        history = pd.read_feather(source, columns=COLS + ['signal'])
        sig = history['signal'].to_numpy()
        starts = np.flatnonzero((sig[1:] == 1) & (sig[:-1] != 1)) + 1
        for i in starts[-args.last:]:
            prefix = history.iloc[:i + 1][COLS].copy()
            before = manager.compute_signals(prefix.iloc[:-1], *params)[-1]
            # Re-date just before publishing so the start candle closed 5-10 min ago
            # (the trader holds on signals older than 15 min).
            newest_open = pd.Timestamp.now(tz='UTC').floor('5min') - pd.Timedelta(minutes=10)
            prefix['date'] += newest_open - prefix['date'].iloc[-1]
            with tempfile.TemporaryDirectory() as tmp, patch.object(strategy_module, 'data_manager', manager):
                manager.data_dir = tmp
                prefix.to_feather(manager._get_feather_path(pair, '5m'))
                manager.ensure_data_and_signals_up_to_date(pair, *params, update_ohlcv=False)
                strategy._signal_cache = {}
                live_df = prefix.tail(1500).reset_index(drop=True)
                out = strategy.populate_indicators(live_df, {'pair': pair})
                out = strategy.populate_exit_trend(strategy.populate_entry_trend(out, {}), {})
            last = out.iloc[-1]
            ok = before == 0 and bool(last.data_ok) and last.enter_long == 1 and last.exit_long == 0
            failures += not ok
            print(f"{'PASS' if ok else 'FAIL'} {pair} long start {history['date'].iloc[i]}: "
                  f"signal one candle earlier={before}, newest candle signal={last.signal} "
                  f"data_ok={last.data_ok} enter_long={last.enter_long} exit_long={last.exit_long}")
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
