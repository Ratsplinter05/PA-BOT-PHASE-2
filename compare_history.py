"""Compare a freshly collected data folder with the existing one.

Read-only. Shows, for every instrument and timeframe, how the new data
overlaps the old, whether overlapping candles agree, what would be lost or
gained, and whether all timeframes of an instrument end at the same time.

Example:
    python -m research.compare_history --new data/historical_refresh_test
"""

import argparse
from pathlib import Path

import pandas as pd

TIMEFRAME_MINUTES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240}
PRICE_COLUMNS = ["open", "high", "low", "close"]
TOLERANCE = 1e-9


def load(path):
    data = pd.read_parquet(path)
    data["time"] = pd.to_datetime(data["time"], utc=True)
    return data.sort_values("time").reset_index(drop=True)


def compare_frames(old, new):
    """Return a dict describing how new relates to old."""
    old_times = set(old["time"])
    new_times = set(new["time"])
    overlap = sorted(old_times & new_times)

    mismatches = 0
    if overlap:
        a = old[old["time"].isin(overlap)].set_index("time")[PRICE_COLUMNS]
        b = new[new["time"].isin(overlap)].set_index("time")[PRICE_COLUMNS]
        diff = (a.sort_index() - b.sort_index()).abs().max(axis=1)
        mismatches = int((diff > TOLERANCE).sum())

    return {
        "old_rows": len(old), "new_rows": len(new),
        "overlap": len(overlap), "mismatched": mismatches,
        "lost_old_rows": len(old_times - new_times),
        "gained_new_rows": len(new_times - old_times),
        "old_start": old["time"].min(), "old_end": old["time"].max(),
        "new_start": new["time"].min(), "new_end": new["time"].max(),
    }


def end_alignment(last_closes, tolerance_minutes=240):
    """True if every timeframe's last candle closes within one H4 of the rest."""
    values = list(last_closes.values())
    if not values:
        return True
    spread = (max(values) - min(values)).total_seconds() / 60
    return spread <= tolerance_minutes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--new", required=True, help="new data folder")
    parser.add_argument("--old", default="data/historical",
                        help="existing data folder")
    args = parser.parse_args(argv)

    new_raw = Path(args.new) / "raw"
    old_raw = Path(args.old) / "raw"
    if not new_raw.exists():
        raise SystemExit(f"No raw data folder found in {args.new}")

    print("=== NEW DATA VS EXISTING DATA ===")
    for symbol_dir in sorted(p for p in new_raw.iterdir() if p.is_dir()):
        symbol = symbol_dir.name
        closes = {}
        print(f"\n{symbol}")
        for timeframe, minutes in TIMEFRAME_MINUTES.items():
            new_path = symbol_dir / f"{symbol}_{timeframe}.parquet"
            if not new_path.exists():
                print(f"  {timeframe}: not collected")
                continue
            new = load(new_path)
            closes[timeframe] = (
                new["time"].max() + pd.Timedelta(minutes=minutes)
            )
            line = (f"  {timeframe}: {len(new)} candles "
                    f"{new['time'].min():%Y-%m-%d} -> {new['time'].max():%Y-%m-%d %H:%M}")
            old_path = old_raw / symbol / f"{symbol}_{timeframe}.parquet"
            if not old_path.exists():
                print(line + "  (no existing data to compare)")
                continue
            r = compare_frames(load(old_path), new)
            print(line)
            print(f"      overlap {r['overlap']} | mismatched {r['mismatched']} | "
                  f"old rows missing from new {r['lost_old_rows']} | "
                  f"new rows added {r['gained_new_rows']}")

        if closes:
            ok = end_alignment(closes)
            ends = ", ".join(f"{k} {v:%m-%d %H:%M}" for k, v in closes.items())
            print(f"  Timeframes end together: {'YES' if ok else 'NO'} ({ends})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())