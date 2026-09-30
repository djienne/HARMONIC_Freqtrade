# Project guidance

Read README.md for the current model, data flow, validation commands and limits.
Workspace rules in ../AGENTS.md also apply.

## Identity and runtime

- Folder/strategy: HARMONIC; trader: HARMONIC; producer: HARMONIC_DATA_MANAGER.
- Docker image: ft-harmonic:2025.10, from Dockerfile.technical.
- Five-minute Binance futures candles; BTC/USDT:USDT and ETH/USDT:USDT.
- Live configuration is dry-run and long-only. Do not change risk settings or
  switch to real trading without explicit authorization.
- API: http://127.0.0.1:3009. The committed local API username/password and JWT
  secret are intentional and accepted. The port must remain loopback-only.
- Preserve user_data/tradesv3.sqlite and all other operational state.

## Signal ownership

DATA_MANAGER.py owns the detector and live custom signal files in
user_data/data/binance/futures_custom_data. The producer runs --loop; the trader
runs --validate-only before starting as a diagnostic only: it must never block the
trader from starting, or open trades lose stop-loss handling. Do not run another
writing data-manager against the active directory. Standalone
user_data/validate_data.py is optional, not the compose preflight entrypoint.

Downloaded candles remain in memory until signals are computed. Publishing the
feather and metadata uses separate atomic replacements; readers may briefly hold
until both match. The validator retains bounded tolerance for legacy checkpoint
windows, but missing per-candle signals must never become actionable exits.

The trader retries the same candle because external signals arrive later than
Freqtrade's candles. Cache unchanged files, but check age on every call. A valid
file alone does not establish coverage of the trader's latest candle.

Offline backtest/hyperopt computes signals with the same detector in memory.
Only custom candles preceding the supplied native input may be used as warmup.
Never use future custom rows, ignore epoch parameters, or persist offline results
into the live signal files. Hyperopt requires --analyze-per-epoch and runs in the
native freqtrade conda environment as required by ../AGENTS.md.

## Scientific checks

Preserve causal prefix invariance, not old output bytes. The pre-fix algorithm
had future dependence; old performance results are not truth. Test representative
real candles as well as the small runnable tests. Full recomputation is deliberate:
resetting a detector on a fixed warmup slice erased historical boundary signals.
Version the signal fingerprint when calculation semantics change.

Tests and native backtests use Docker. The test commands are in README.md.
research/ holds historical prototypes and analysis scripts; production never
imports them.

CPU ceilings and exchange rate limits belong to the workspace's central tools.
Do not hand-edit either policy here. No new dependency is needed for the signal
pipeline: it uses the packages already supplied with the Freqtrade image.
