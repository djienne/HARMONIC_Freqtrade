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
  signals over the full history, and publishes ONE feather whose Arrow schema
  carries the parameter/version metadata, so data and metadata change in a single
  atomic replace. It never publishes an unfinished download on the signal path.
- `freqtrade` logs a validation report of the prepared files, then starts and
  reads them during trading. Missing, invalid or stale signals block entries
  **and signal exits**. A missing candle is not a flat signal. Stop-loss handling
  remains active because the trader starts even when validation fails.

Connection loss and restarts: the producer retries downloads with backoff, never
touches the signal file while offline, and after any gap re-downloads from the
last 12 stored candles to now, which reproduces identical signals (tested with a
350-candle gap). The trader holds (stop-loss active) while signals are stale and
exits at startup if Binance is unreachable, so the compose restart policy retries
it about once a minute. Docker Desktop must itself be running: it starts at
Windows login, and a PC reboot recovered by itself in under 10 minutes.

The trader polls every five seconds and can retry the same candle after the
producer finishes. Validated files are cached by file revision; age is checked
on every use. The API is bound to `127.0.0.1:3009`.

Memory: the trader validates the full file but caches only the newest 5,000
`date`/`signal` rows, and both services use Arrow's system allocator. Before
this (November 2026) the trader grew to 1.2 GB RAM + 1 GB swap: a pandas `.dt`
reference cycle pinned two full-history columns per reload until a rare full
GC. Check with `docker stats --no-stream HARMONIC` (swap shows in
`docker exec HARMONIC cat /sys/fs/cgroup/memory.swap.current`).

Full signal computation replaces the old approximate tail computation: resetting
the detector on a slice changed historical signals at the slice boundary. If
full computation no longer fits a five-minute cycle, preserve exact detector
state before attempting incremental optimization.

## Run

```sh
docker compose up -d --build
docker compose logs --tail 50 freqtrade data-manager
```

Initial history acquisition takes time. Until prepared signals pass validation
the trader runs but holds: no entries and no signal exits, stop-loss still
active. Existing algorithm caches are invalidated by the
versioned fingerprint and recomputed by the producer. Never delete the trade
database to update code or regenerate signals.

To apply a strategy change to an already running deployment, restart both
services: imported Python modules are not hot-reloaded. Do not start a second
writer against the same custom data directory.

## Checks and research

```sh
docker compose run --rm --no-deps --entrypoint python freqtrade -B -m unittest discover -s user_data/tests -v
docker compose run --rm --no-deps --entrypoint freqtrade freqtrade backtesting --strategy HARMONIC --config user_data/config.json --timeframe 5m --timerange 20250101- --cache none --export trades
docker compose run --rm --no-deps --entrypoint freqtrade freqtrade lookahead-analysis --strategy HARMONIC --config user_data/config.json --timeframe 5m --timerange 20240101-20251127
docker compose run --rm --no-deps --entrypoint freqtrade freqtrade recursive-analysis --strategy HARMONIC --config user_data/config.json --timeframe 5m --timerange 20250101-20250401
```

`research/replay_live_entry.py` replays the latest real long starts on the
producer's candles through the real producer → file → dry-run strategy path and
fails unless each becomes `enter_long` on the newest candle (mount `research/`
as in `param_sensitivity.py`). On 2026-11-09 all six recent starts passed. A
throwaway dry-run copy (fresh database, separate port) force-entered BTC: a 5x
long with a stake of about 500 filled at once, and the live exit signal closed
it 41 s later (`exit_signal`). The production database had never held a trade
because the pre-fix pipeline published each long one candle after its start.

### Backtest ≡ live (2026-11-18)

Research scripts run in Docker with `research/` mounted. From Git Bash, prefix
`MSYS_NO_PATHCONV=1` and mount `"$(pwd -W)/research:/freqtrade/research"`.

- `research/check_backtest_parity.py --start 2025-01-01` (after the backtest above,
  with native data downloaded to now) compares three things. It checks native against
  producer OHLC, then the strategy's backtest signals against the live file, then the
  exported trades against the live signals. Over 2025-01-01..2026-11-18 it found
  **0 differing signal rows** across 9 BTC and 15 ETH long starts. All 24 trades opened
  on the candle after a live +1, and all signal exits followed a live ≤ 0. No live start
  was missed.
- That input check found 7 candles per pair, 2026-10-27 00:00–00:30, where the producer
  still held Binance's flat zero-volume placeholders. Binance revised them more than an
  hour later, past the 12-candle tail repair. Signals were unaffected. The producer now
  re-downloads from one candle before any zero-volume candle within 30 days.
  Repaired candles are logged as `REPAIRED <pair> <date> old=… new=…`.
- **Warmup is part of the model.** The directional-change state starts at bar 0, so the
  start bar matters. A 500-candle warmup missed 4 of 5 recent long starts. Of 60 random
  truncated starts, 12 differed for up to 28 days. Offline runs therefore start the
  detector at the producer's first candle (2020-01-01) and log that origin. Native
  candles before it get no signal. Without custom history, a WARNING is logged.
- **Latency.** Live signals are published a median of 41 s (BTC) / 75 s (ETH) after
  the candle closes, while backtests fill at the open. `research/entry_latency_cost.py`
  brackets the live fill with 1m closes at +1, +2 and +3 min across all historical long
  starts:

  | | +1 min mean (median) | +2 min mean | mean \|cost\| | backtest mean move |
  |---|---|---|---|---|
  | BTC (38 starts) | +0.13% (+0.12%) | +0.14% | 0.40% | +2.56% (9 trades) |
  | ETH (70 starts) | −0.04% (−0.03%) | −0.24% | 0.36–0.64% | +1.09% (15 trades) |

  Positive means a worse fill for a long. There is no significant systematic bias
  (BTC is about 1.6 SE from zero). Per-trade noise is small next to BTC's edge but
  roughly half of ETH's.
- **Live decision ledger.** The trader appends its newest-candle decision to
  `user_data/logs/live_decisions.csv`. It writes a row on first sight and again on any
  same-candle change, but only in a real trade run. Each row has the candle, the
  sighting time, the file revision, the fingerprint, signal, data_ok, the flags and OHLC.
  `research/compare_live_ledger.py [--recompute]` diffs those decisions against the
  producer's final signals and links any mismatch to logged repairs. It also reports
  missed decisions and latency. The first natural live long is the end-to-end proof:
  its trade, ledger row and backtest trade must share the entry candle.

Only live and dry-run read the producer's signal files. Every other run mode,
including Freqtrade's `lookahead-analysis` and `recursive-analysis`, runs the
detector in memory, so those tools test the detector itself. Lookahead analysis
needs at least 10 trades (`--minimum-trade-amount`), hence its longer range.

Backtesting requires Freqtrade's native OHLCV, mark and funding data. Its strategy
path recomputes signals in memory for the selected parameters. Custom candles
strictly before the native input provide warmup if available; the native input
takes precedence, and future custom rows and precomputed signals are excluded.
With no custom history, only the native input/startup candles initialize the
detector. Record the input coverage when comparing runs.

Hyperopt must use `--analyze-per-epoch` when optimizing `sigma`, `err_thresh` or
`kept_patterns`. The documented command in `commands.txt` includes it. It must
not modify the producer's signal files while evaluating parameters.

Both `freqtrade hyperopt` and `freqtrade hyperopt-show` overwrite
`user_data/strategies/HARMONIC.json` unless given `--disable-param-export`; the
commands in `commands.txt` include it. Both containers read that file at start,
so an unintended export changes live parameters (including `LEV`) at the next
restart, and a restart of only one service leaves the trader holding on a
fingerprint mismatch. Adopting parameters is a deliberate step: run
`hyperopt-show -n <epoch>` without the flag, review `HARMONIC.json`, then restart
both services together.

The tests check prefix causality, invariance to price units, flat/empty limits,
missing-candle recovery, parameter changes, cache expiry and cached failures,
download failures, tail repair and its REPAIRED log, zero-volume placeholder
re-download, the offline detector origin, the live decision ledger, run-mode
routing, date resolution and invalid OHLCV rejection. A passing test is not evidence of profitability.
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

**Parameter provenance.** `HARMONIC.json` (sigma 0.035, err_thresh 0.16,
kept_patterns 1011000, LEV 5) was exported by hyperopt on 2025-03-10, so it was
selected with the pre-fix, look-ahead matcher. `research/param_sensitivity.py`
re-runs the causal detector over a sigma × err_thresh grid on the producer's
candles. On Binance 5m archives from 2020-01 to 2026-08 (both pairs; crude
next-open trade walk, price-level returns, 0.1% fee per side) it gave:

| sigma, err_thresh | before 2025-03-10 | after 2025-03-10 |
|---|---|---|
| 0.030, 0.16 | 102 trades, +0.77%, 76% wins | 19 trades, −0.06%, 42% wins |
| 0.035, 0.16 (live) | 89 trades, +1.14%, 82% wins | 19 trades, +1.04%, 74% wins |
| 0.040, 0.16 | 65 trades, +0.46%, 65% wins | 10 trades, +1.25%, 80% wins |

The live point is an in-sample peak. Nineteen later trades cannot confirm it,
and lowering sigma by 0.005 removes the later edge. No trade reached the
stop-loss in these 6.7 years, so the data say nothing about the leverage tail.
Treat the parameters as unvalidated; changing them is an explicit decision.

## Intentional risk settings

The existing defaults remain: leverage up to 5, stop-loss at -0.85 leveraged
trade return, two open trades, almost all available balance allocated, and no
practical ROI profit target. These are aggressive model settings. Dry-run
execution and a successful backtest do not justify switching to real funds.

Only project source/configuration should be added to new commits. Runtime
databases, market data, logs, reports, bytecode and `.env` files are local state.
The exchange credential fields in the supplied configuration are empty.
