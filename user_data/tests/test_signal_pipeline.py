"""Small science/production checks: run with python -B -m unittest discover -s user_data/tests."""
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from freqtrade.enums import RunMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'strategies'))
import DATA_MANAGER as dm
import HARMONIC as strategy_module

PARAMS = (.035, .16, '1011000')
PAIR = 'BTC/USDT:USDT'


def candles():
    # Exact Gartley ratios; the last downward leg is observable before the next
    # turning point is confirmed. Prices are positive and units are arbitrary.
    close = np.array([110, 100, 120, 107.64, 115.27848, 104.28, 104.30, 110.])
    return pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=len(close), freq='5min', tz='UTC'),
        'open': close, 'high': close, 'low': close, 'close': close, 'volume': 1.,
    })


class SignalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.manager = dm.DataManager()
        self.strategy = strategy_module.HARMONIC({})
        self.strategy.sigma.value, self.strategy.err_thresh.value, self.strategy.kept_patterns.value = PARAMS
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manager.data_dir = self.tmp.name
        # HARMONIC can import the module under its package name; replace only
        # this instance to keep tests independent of production files.
        self.manager_patch = patch.object(strategy_module, 'data_manager', self.manager)
        self.manager_patch.start()
        self.addCleanup(self.manager_patch.stop)

    def test_causality_and_price_unit_invariance(self):
        data = candles()
        full = self.manager.compute_signals(data, *PARAMS)
        self.assertEqual(full[5], 1)
        for n in range(1, len(data) + 1):
            np.testing.assert_array_equal(
                self.manager.compute_signals(data.iloc[:n], *PARAMS), full[:n])
        scaled = data.copy()
        scaled[['open', 'high', 'low', 'close']] *= 1000
        np.testing.assert_array_equal(self.manager.compute_signals(scaled, *PARAMS), full)

    def test_empty_flat_and_monotonic_markets(self):
        data = candles()
        for prices in [np.ones(len(data)), np.arange(100., 100. + len(data))]:
            for col in ['open', 'high', 'low', 'close']:
                data[col] = prices
            self.assertFalse(self.manager.compute_signals(data, *PARAMS).any())
        self.assertEqual(len(self.manager.compute_signals(data.iloc[:0], *PARAMS)), 0)

    def test_missing_signal_holds_then_retries_same_candle(self):
        data = candles().iloc[:6].copy()
        prepared = data.assign(signal=1)
        with patch.object(self.strategy, '_load_signals', return_value=(prepared.iloc[:-1], True)):
            out = self.strategy.populate_indicators(data, {'pair': PAIR})
        # Existing flags must also be cleared when a previously good row goes bad.
        out['enter_long'] = out['exit_long'] = 1
        out = self.strategy.populate_entry_trend(out, {})
        out = self.strategy.populate_exit_trend(out, {})
        self.assertFalse(out.data_ok.iloc[-1])
        self.assertEqual(out.enter_long.iloc[-1], 0)
        self.assertEqual(out.exit_long.iloc[-1], 0)
        with patch.object(self.strategy, '_load_signals', return_value=(prepared, True)):
            ready = self.strategy.populate_indicators(data, {'pair': PAIR})
        self.assertFalse(self.strategy.process_only_new_candles)
        self.assertEqual(self.strategy.populate_entry_trend(ready, {}).enter_long.iloc[-1], 1)

    def test_hyperopt_uses_requested_params_and_ignores_future_custom_rows(self):
        data = candles()
        # Intentionally wrong cached signals/metadata cannot affect offline runs.
        path = self.manager._get_feather_path(PAIR, '5m')
        data.assign(signal=-1).to_feather(path)
        self.strategy.dp = SimpleNamespace(runmode=RunMode.HYPEROPT)
        with self.assertRaisesRegex(ValueError, 'analyze-per-epoch'):
            self.strategy.populate_indicators(data.iloc[:6], {'pair': PAIR})
        self.strategy.config['analyze_per_epoch'] = True
        self.strategy.sigma.value = .036
        out = self.strategy.populate_indicators(data.iloc[:6], {'pair': PAIR})
        expected = self.manager.compute_signals(data.iloc[:6], .036, .16, '1011000')
        np.testing.assert_array_equal(out.signal, expected)
        self.assertEqual(out.signal.iloc[-1], 1)
        self.assertTrue(out.data_ok.all())
        np.testing.assert_array_equal(pd.read_feather(path).signal, -np.ones(len(data)))
        # A later backtest start must use ONLY the custom prehistory for warmup.
        later = self.strategy.populate_indicators(data.iloc[4:6], {'pair': PAIR})
        np.testing.assert_array_equal(later.signal, expected[4:6])
        # This parameter change must alter the result, not merely be accepted.
        self.strategy.kept_patterns.value = '0111000'  # exclude Gartley
        different = self.strategy.populate_indicators(data.iloc[:6], {'pair': PAIR})
        self.assertFalse(different.signal.any())

    def test_invalid_ohlcv_rejected_in_shared_paths(self):
        for col, value in [('close', -1.), ('high', np.inf), ('high', 1.), ('volume', -1.)]:
            data = candles().assign(signal=0)
            data.loc[2, col] = value
            with self.subTest(col=col, value=value):
                with self.assertRaises(ValueError):
                    self.manager.compute_signals(data, *PARAMS)
                with self.assertRaises(ValueError):
                    self.manager._normalize_ohlcv(data)
                meta = self.manager._build_signal_metadata(
                    PAIR, data, self.manager._fingerprint(*PARAMS), *PARAMS, 'full')
                fatal, _, _, _ = self.manager._validate_signal_dataframe(PAIR, data, meta, *PARAMS, None)
                self.assertTrue(fatal)

    def test_old_algorithm_cache_and_large_nan_tail_rejected(self):
        data = pd.concat([candles()] * 10, ignore_index=True).assign(signal=0)
        data['date'] = pd.date_range('2026-01-01', periods=len(data), freq='5min', tz='UTC')
        meta = self.manager._build_signal_metadata(
            PAIR, data, self.manager._fingerprint(*PARAMS), *PARAMS, 'full')
        old = dict(meta, fingerprint='1461968a1fe98e38')
        self.assertTrue(self.manager._validate_signal_dataframe(PAIR, data, old, *PARAMS, None)[0])
        data.loc[20:, 'signal'] = np.nan
        self.assertTrue(self.manager._validate_signal_dataframe(PAIR, data, meta, *PARAMS, None)[0])

    def test_downloading_does_not_publish_unsigned_tail_or_lose_old_data(self):
        data = candles().assign(signal=1)
        path = self.manager._get_feather_path(PAIR, '5m')
        data.to_feather(path)
        t = int((data.date.iloc[-1] + pd.Timedelta(minutes=5)).timestamp() * 1000)
        kline = [t, '111', '111', '111', '111', '1', t + 299999]
        with patch.object(self.manager, '_fetch_klines_chunk', side_effect=[[kline], []]), \
                patch.object(dm.time, 'sleep'):
            downloaded = self.manager.download_data(PAIR, '5m', persist=False)
        self.assertEqual(len(downloaded), len(data) + 1)
        pd.testing.assert_frame_equal(pd.read_feather(path), data)
        with patch.object(self.manager, '_fetch_klines_chunk', return_value=None):
            failed = self.manager.download_data(PAIR, '5m', persist=False)
        pd.testing.assert_frame_equal(failed, data)

    def test_tail_repair_replaces_tail_without_copying_history(self):
        # Large enough that an unstable sort can reorder duplicate dates.
        n = 3000
        close = 100 + np.sin(np.arange(n) / 50.)
        data = pd.DataFrame({
            'date': pd.date_range('2026-01-01', periods=n, freq='5min', tz='UTC'),
            'open': close, 'high': close, 'low': close, 'close': close, 'volume': 1., 'signal': 0,
        })
        path = self.manager._get_feather_path(PAIR, '5m')
        data.to_feather(path)
        ms = lambda ts: int(ts.timestamp() * 1000)
        revised = [[ms(d), *[str(c * 1.01)] * 4, '1', ms(d) + 299999]
                   for d, c in zip(data.date.iloc[-12:], data.close.iloc[-12:])]
        t = ms(data.date.iloc[-1] + pd.Timedelta(minutes=5))
        revised.append([t, '100', '100', '100', '100', '1', t + 299999])
        with patch.object(self.manager, '_fetch_klines_chunk', side_effect=[revised, []]), \
                patch.object(dm.time, 'sleep'), \
                patch.object(self.manager, '_normalize_ohlcv', wraps=self.manager._normalize_ohlcv) as norm:
            out = self.manager.download_data(PAIR, '5m', persist=False)
        self.assertEqual(len(out), n + 1)
        np.testing.assert_allclose(out.close.iloc[-13:-1], data.close.iloc[-12:] * 1.01)
        self.assertLessEqual(max(len(c.args[0]) for c in norm.call_args_list), n + 13)

    def test_cache_rechecks_freshness_and_file_revision(self):
        data = candles().assign(signal=1)
        data['date'] = pd.date_range(
            end=pd.Timestamp.now(tz='UTC').floor('5min') - pd.Timedelta(minutes=5),
            periods=len(data), freq='5min')
        path = Path(self.manager._get_feather_path(PAIR, '5m'))
        path.touch()
        meta_path = Path(self.manager._meta_path(str(path)))
        meta_path.touch()
        with patch.object(self.manager, 'load_precomputed_signals', return_value=(data, True)) as load:
            self.assertTrue(self.strategy._load_signals(PAIR)[1])
            data['date'] -= pd.Timedelta(hours=1)
            self.assertFalse(self.strategy._load_signals(PAIR)[1])
            self.assertEqual(load.call_count, 1)
            meta_path.write_text('changed')
            self.strategy._load_signals(PAIR)
            self.assertEqual(load.call_count, 2)

    def test_persistence_failure_is_not_reported_as_success(self):
        data = candles()
        path = self.manager._get_feather_path(PAIR, '5m')
        data.to_feather(path)
        with patch.object(self.manager, '_atomic_write_feather', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.manager.ensure_data_and_signals_up_to_date(
                    PAIR, *PARAMS, update_ohlcv=False, force_full_recompute=True)
        pd.testing.assert_frame_equal(pd.read_feather(path), data)


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    unittest.main()
