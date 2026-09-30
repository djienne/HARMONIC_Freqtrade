# HARMONIC Freqtrade

A long-only harmonic-pattern strategy for BTC/USDT and ETH/USDT perpetual
futures on Binance, using Freqtrade 2025.10 and five-minute candles. The supplied
configuration is **dry-run**. Bearish patterns close longs; they do not open shorts.

## Model and data

`user_data/strategies/DATA_MANAGER.py` contains directional-change detection and
the XABCD ratio matcher. X, A, B and C must already be confirmed; D is evaluated
from the current closed candle. An entry persists until the next directional
change is confirmed. Fibonacci ratios are model assumptions, not evidence of a
profitable market mechanism.

Two services share `user_data`:

- `data-manager` downloads closed candles, repairs the last 12 candles, computes
  signals over the full history, and publishes the feather plus parameter/version
  metadata. It never publishes an unfinished download on the signal path.
- `freqtrade` validates the prepared files before starting and reads them during
  trading. Missing, invalid or stale signals block entries **and signal exits**.
  A missing candle is not a flat signal. Stop-loss handling remains active.

The trader polls every five seconds and can retry the same candle after the
producer finishes. Validated files are cached by file revision; age is checked
on every use. The API is bound to `127.0.0.1:3009`.

Full signal computation replaces the old approximate tail computation: resetting
the detector on a slice changed historical signals at the slice boundary. If
full computation no longer fits a five-minute cycle, preserve exact detector
state before attempting incremental optimization.

## Run

```sh
docker compose up -d --build
docker compose logs --tail 50 freqtrade data-manager
```

Initial history acquisition takes time. The trader restarts until prepared
signals pass validation. Existing algorithm caches are invalidated by the
versioned fingerprint and recomputed by the producer. Never delete the trade
database to update code or regenerate signals.

To apply a strategy change to an already running deployment, restart both
services: imported Python modules are not hot-reloaded. Do not start a second
writer against the same custom data directory.

## Checks and research

```sh
docker compose run --rm --no-deps --entrypoint python freqtrade -B -m unittest discover -s user_data/tests -v
docker compose run --rm --no-deps --entrypoint python freqtrade -B user_data/tests/test_validator_write_window.py
docker compose run --rm --no-deps --entrypoint freqtrade freqtrade backtesting --strategy HARMONIC --config user_data/config.json --timeframe 5m --timerange 20250101-20251127 --cache none
```

Backtesting requires Freqtrade's native OHLCV, mark and funding data. Its strategy
path recomputes signals in memory for the selected parameters. Custom candles
strictly before the native input provide warmup if available; the native input
takes precedence, and future custom rows and precomputed signals are excluded.
With no custom history, only the native input/startup candles initialize the
detector. Record the input coverage when comparing runs.

Hyperopt must use `--analyze-per-epoch` when optimizing `sigma`, `err_thresh` or
`kept_patterns`. The documented command in `commands.txt` includes it. It must
not modify the producer's signal files while evaluating parameters.

The tests check prefix causality, invariance to price units, flat/empty limits,
missing-candle recovery, parameter changes, cache expiry, download failures,
and invalid OHLCV rejection. A passing test is not evidence of profitability.
Native backtests with only five-minute candles do not resolve intrabar order
execution; use finer detail candles and out-of-sample periods for performance
research, including fees, funding, slippage and parameter sensitivity.

The September 30, 2026 review passed 9 pipeline tests and 8 legacy validator
checks. Full computation on 709,857 real candles per pair agreed with six
historical prefixes ending on signals, and adding 13 candles preserved both
histories. At the configured 0.1 CPU limit, full computation took about 20–22
seconds per pair. The native 2025-01-01 through 2025-11-26 backtest completed
with 13 trades, using the supplied parameters, fees, funding data and five-minute
candles (no finer detail data). This is an execution check, not a performance
qualification.

**Historical backtests made before the September 2026 causality fix are not
reliable performance evidence.** The old matcher required a future confirmed
extreme before evaluating the final leg, so historical entries could be absent
in real time. Retained Git history/results are archival, not a benchmark to match.

## Intentional risk settings

The existing defaults remain: leverage up to 5, stop-loss at -0.85 leveraged
trade return, two open trades, almost all available balance allocated, and no
practical ROI profit target. These are aggressive model settings. Dry-run
execution and a successful backtest do not justify switching to real funds.

Only project source/configuration should be added to new commits. Runtime
databases, market data, logs, reports, bytecode and `.env` files are local state.
The exchange credential fields in the supplied configuration are empty.
