"""Self-test for the signal validator's write-window tolerance.

Reproduces the exact state the trader sees while DATA_MANAGER is mid-update
(tail-repair blanks the last TAIL_REPAIR_CANDLES signals, meta still describes
the previous larger state) and asserts it is tolerated, while genuine
corruption is still rejected.

Run inside the container:
    python -B user_data/tests/test_validator_write_window.py
"""
import numpy as np
import pandas as pd

import os, sys
# DATA_MANAGER lives in the strategies dir; this test lives outside it so
# freqtrade never imports it while scanning for strategies.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "strategies"))
import DATA_MANAGER as dm

SIGMA, ERR, KEPT = 0.035, 0.16, "1011000"


def make_manager():
    for attr in dir(dm):
        obj = getattr(dm, attr)
        if isinstance(obj, type) and hasattr(obj, "_validate_signal_dataframe"):
            try:
                return obj(timeframe="5m")
            except TypeError:
                try:
                    return obj()
                except TypeError:
                    continue
    raise SystemExit("could not instantiate the data manager class")


def frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.date_range("2026-08-01", periods=n, freq="5min", tz="UTC"),
        "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0,
        "signal": rng.choice([-1, 0, 1], size=n),
    })


def meta_for(mgr, df, mode="incremental"):
    # Use the manager's real fingerprint; a fake one fails an unrelated check.
    fp = mgr._fingerprint(SIGMA, ERR, KEPT)
    return mgr._build_signal_metadata(
        "BTC/USDT:USDT", df, fp, SIGMA, ERR, KEPT, mode, previous_meta=None)


def run(mgr, label, df, meta, expect_fatal):
    fatal, soft, fresh, out = mgr._validate_signal_dataframe(
        "BTC/USDT:USDT", df, meta, SIGMA, ERR, KEPT, None)
    ok = bool(fatal) == expect_fatal
    print(f"[{'PASS' if ok else 'FAIL'}] {label:44} fatal={len(fatal)} soft={len(soft)} rows_out={len(out)}")
    if fatal and not expect_fatal:
        print(f"        unexpected fatal: {fatal}")
    if not fatal and expect_fatal:
        print("        expected fatal but got none")
    return ok


def main():
    mgr = make_manager()
    results = []

    # 1. clean steady state -> must pass untouched
    df = frame()
    results.append(run(mgr, "clean, meta matches", df, meta_for(mgr, df), False))

    # 2. THE REAL WINDOW: meta from the pre-checkpoint state, feather has 12
    #    tail-repaired rows blanked plus 1 new candle appended.
    full = frame()
    meta_prev = meta_for(mgr, full)
    window = full.copy()
    new_row = window.iloc[[-1]].copy()
    new_row["date"] = window["date"].iloc[-1] + pd.Timedelta(minutes=5)
    window = pd.concat([window, new_row], ignore_index=True)
    window.loc[window.index[-13:], "signal"] = np.nan
    results.append(run(mgr, "write window (12 blanked + 1 new)", window, meta_prev, False))


    # 2b. THE OTHER WINDOW: signals just persisted to the feather, meta write
    #     has not landed yet -> data fully signalled, meta 1 row BEHIND.
    persisted = frame(n=201)
    meta_behind = meta_for(mgr, persisted.iloc[:200])
    results.append(run(mgr, "write window (meta 1 row behind)", persisted, meta_behind, False))

    # 2c. meta far behind is NOT a write gap -> must stay fatal
    far = frame(n=260)
    results.append(run(mgr, "meta far behind (corruption)", far, meta_for(mgr, frame(200)), True))

    # 3. corruption: NaN in the middle must stay fatal
    mid = frame()
    mid.loc[mid.index[50], "signal"] = np.nan
    results.append(run(mgr, "NaN mid-series (corruption)", mid, meta_for(mgr, frame()), True))

    # 4. meta BEHIND the data by less than the tolerance: the persist gap, tolerated
    behind = frame(n=210)  # lag 10 <= tolerance -> treated as the persist gap
    results.append(run(mgr, "meta behind by 10 (within tolerance)", behind, meta_for(mgr, frame(200)), False))

    # 5. corruption: wrong signal params must stay fatal
    df5 = frame()
    m5 = meta_for(mgr, df5)
    m5["sigma"] = 0.99
    fatal, _, _, _ = mgr._validate_signal_dataframe(
        "BTC/USDT:USDT", df5, m5, SIGMA, ERR, KEPT, None)
    ok = bool(fatal)
    print(f"[{'PASS' if ok else 'FAIL'}] {'wrong sigma (corruption)':44} fatal={len(fatal)}")
    results.append(ok)

    # 6. all-NaN signals must stay fatal
    allnan = frame()
    allnan["signal"] = np.nan
    results.append(run(mgr, "all signals NaN (corruption)", allnan, meta_for(mgr, frame()), True))

    print(f"\n{sum(results)}/{len(results)} passed")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
