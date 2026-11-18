"""Compare what the live trader saw (user_data/logs/live_decisions.csv) with the
producer's final signals, which a backtest reproduces (check_backtest_parity.py).

Research only, read-only. The detector is prefix-invariant (unit tests and
real-data prefix checks), so the producer's current file already holds, for every
candle, the signal computed from origin through that candle on the final candles.
A ledger row whose data_ok decision differs from it means either a later input
repair (REPAIRED lines in data-manager.log*, within the previous 12 candles) or a
gating/merge bug. --recompute also re-derives the file from its own candles.
Reports: mismatches, missed decisions (candles never seen with data_ok before
the next candle) and signal latency (first data_ok sighting - candle close).

Run:
    docker compose run --rm --no-deps -v "$PWD/research:/freqtrade/research" \
        --entrypoint python freqtrade -B research/compare_live_ledger.py [--recompute]
"""
import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))
import DATA_MANAGER as dm  # noqa: E402

STEP = pd.Timedelta(minutes=5)
REPAIR = re.compile(r'REPAIRED (\S+) (\S+) old=')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--recompute', action='store_true', help='also re-derive the file signals')
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    manager = dm.DataManager()
    logs = Path(manager.user_data_dir) / 'logs'
    ledger = pd.read_csv(logs / 'live_decisions.csv')
    ledger['candle_date'] = pd.to_datetime(ledger['candle_date'], utc=True)
    ledger['seen_at_utc'] = pd.to_datetime(ledger['seen_at_utc'], utc=True)
    repairs = set()
    for f in logs.glob('data-manager.log*'):
        for m in REPAIR.finditer(f.read_text(encoding='utf-8', errors='replace')):
            repairs.add((m[1], pd.Timestamp(m[2], tz='UTC')))
    params = dm._load_strategy_hyperparams(manager.user_data_dir)
    expected_fp = manager._fingerprint(*params)
    failures = 0

    for pair, rows in ledger.groupby('pair'):
        file = pd.read_feather(manager._get_feather_path(pair, '5m'))
        file['date'] = pd.to_datetime(file['date'], utc=True)
        if args.recompute:
            fresh = manager.compute_signals(file, *params)
            n = int((fresh != file['signal'].to_numpy()).sum())
            print(f'{pair}: file vs fresh recompute differs on {n} rows')
            failures += n
        final = file.set_index('date')['signal']
        ok = rows[rows['data_ok'].astype(bool)].copy()
        # The decision that counts is the last data_ok row before the next candle.
        acted = ok.sort_values('seen_at_utc').groupby('candle_date').last()
        acted['final'] = final.reindex(acted.index)
        diff = acted[acted['signal'] != acted['final']]
        print(f'{pair}: {len(rows)} ledger rows, {len(acted)} candles decided with data_ok, '
              f'{rows["candle_date"].min()} .. {rows["candle_date"].max()}')
        wrong_fp = rows[rows['fingerprint'] != expected_fp]
        if len(wrong_fp):
            print(f'  {len(wrong_fp)} rows carry fingerprint != {expected_fp}')
        for date, r in diff.iterrows():
            repaired = [d for p, d in repairs if p == pair and date - 12 * STEP <= d <= date]
            print(f'  MISMATCH {date}: live {r.signal} final {r.final}; '
                  f'{"repaired inputs " + str(sorted(map(str, repaired))) if repaired else "NO repair logged"}')
        failures += len(diff)
        # The newest candle may still be awaiting its signal: undecided, not missed.
        span = pd.date_range(rows['candle_date'].min(), rows['candle_date'].max() - STEP, freq=STEP)
        missed = span.difference(acted.index)
        print(f'  missed decisions (no data_ok sighting): {len(missed)} of {len(span)} candles'
              + (f', e.g. {[str(d) for d in missed[:5]]}' if len(missed) else ''))
        first_seen = ok.groupby('candle_date')['seen_at_utc'].min()
        latency = (first_seen - (first_seen.index + STEP).to_series(index=first_seen.index)).dt.total_seconds()
        if len(latency):
            print(f'  latency after candle close: median {latency.median():.0f} s, '
                  f'p90 {np.percentile(latency, 90):.0f} s, max {latency.max():.0f} s')
        long_starts = acted[(acted['signal'] == 1)].index
        print(f'  live +1 decisions: {len(long_starts)}')

    print('PASS' if failures == 0 else f'FAIL ({failures})')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
