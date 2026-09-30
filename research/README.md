# research/

Exploratory / reference scripts. **Not used by the running bot.**

The production harmonic-pattern code lives in
`user_data/strategies/DATA_MANAGER.py` (functions `directional_change`,
`get_extremes`, `find_xabcd`, `get_error`) and `user_data/strategies/HARMONIC.py`.

These files are kept for reference only and intentionally diverge from production:

- `harmonic_patterns.py` — early prototype; uses only 3 patterns
  (`ALL_PATTERNS = [GARTLEY, BUTTERFLY, CRAB]`) and a different point-D lookback
  slice (`ohlc.iloc[last_conf_i:i]` vs the production `[last_conf_i:i + 1]`).
- `directional_change.py` — standalone NumPy version of the zig-zag detector with
  a matplotlib `__main__` demo.

Do not import from here in the strategy or data manager.
