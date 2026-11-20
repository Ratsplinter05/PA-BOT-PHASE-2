"""Run the historical strategy replay on stored data and print a summary.

Reads stored candles only (MT5 does not need to be open). It evaluates
strategy signals; it does not simulate orders or profit and loss.

Examples (from the project folder):
    python -m research.run_replay --interval 144
    python -m research.run_replay --interval 1 --start 2026-10-01
"""

import argparse
import time
from datetime import datetime, timezone

import pandas as pd

from research.historical_strategy_replay import HistoricalStrategyReplay
from strategy.strategy_orchestrator import StrategyOrchestrator


class SetupRecorder:
    """Wraps the orchestrator and remembers every distinct setup it reports.

    The pipeline re-reports past setups on every assessment, so counting
    per assessment repeats the same setup many times. Tracking by setup_id
    gives the number of distinct setups and each one's latest state.
    """

    def __init__(self):
        self.inner = StrategyOrchestrator()
        self.setups = {}

    def analyze(self, **kwargs):
        result = self.inner.analyze(**kwargs)
        for item in result.m5_confirmations.confirmations:
            previous = self.setups.get(item.setup_id)
            was_confirmed = bool(previous and previous["ever_confirmed"])
            confirmed_now = item.state == "CONFIRMED"

            # A confirmed setup later ages out to EXPIRED, so remember that
            # it was confirmed once and keep its first confirmation details.
            if was_confirmed and not confirmed_now:
                state = "CONFIRMED"
                confirmation_time = previous["confirmation_time"]
                reason = previous["reason"]
            else:
                state = item.state
                confirmation_time = item.confirmation_time
                reason = item.reason

            self.setups[item.setup_id] = {
                "setup_id": item.setup_id,
                "direction": item.direction,
                "state": state,
                "ever_confirmed": was_confirmed or confirmed_now,
                "qualification_time": item.qualification_time,
                "confirmation_time": confirmation_time,
                "candles_elapsed": item.candles_elapsed,
                "reason": reason,
            }
        return result

    def table(self):
        if not self.setups:
            return pd.DataFrame()
        table = pd.DataFrame(self.setups.values())
        return table.sort_values("setup_id").reset_index(drop=True)


def build_rows(result):
    rows = []
    for item in result.assessments:
        alignment = item.alignment
        rows.append({
            "assessment_time": item.assessment_time,
            "alignment_state": alignment.alignment_state,
            "directional_permission": alignment.directional_permission,
            "permitted_direction": alignment.permitted_direction,
            "qualified": item.qualified_setup_count,
            "confirmed": item.confirmed_setup_count,
            "waiting": item.waiting_setup_count,
            "expired": item.expired_setup_count,
            "rejected": item.rejected_setup_count,
        })
    return pd.DataFrame(rows)


def print_summary(result, table, seconds):
    total = result.assessment_count
    print("\n=== REPLAY SUMMARY ===")
    print(f"Symbol:               {result.symbol}")
    print(f"Period assessed:      {result.start_time} -> {result.end_time}")
    print(f"Assessments run:      {total}")
    print(f"Skipped (too little history): {result.skipped_insufficient_history}")
    print(f"Skipped (stale data):         {result.skipped_stale_data}")
    print(f"Run time:             {seconds:.1f} seconds")

    if total == 0:
        print("\nNo assessments were run.")
        return

    print("\n--- H4/H1 alignment (share of assessments) ---")
    states = table["alignment_state"].value_counts()
    for state, count in states.items():
        print(f"  {state:<28} {count:>6}  {count / total:6.1%}")

    permitted = int(table["directional_permission"].sum())
    print(f"\nDirection permitted:  {permitted} ({permitted / total:.1%})")
    for direction, count in (
        table["permitted_direction"].dropna().value_counts().items()
    ):
        print(f"  {direction:<12} {count}")


def print_setup_summary(setups):
    print("\n--- Distinct setups (each counted once) ---")
    if setups.empty:
        print("  none recorded")
        return

    print(f"Total distinct M15 setups seen: {len(setups)}")
    order = ["CONFIRMED", "WAITING", "EXPIRED", "REJECTED"]
    for state in order:
        subset = setups[setups["state"] == state]
        if subset.empty:
            continue
        by_dir = subset["direction"].value_counts().to_dict()
        print(f"  {state:<10} {len(subset):>5}   {by_dir}")

    confirmed = setups[setups["state"] == "CONFIRMED"].copy()
    if not confirmed.empty:
        gap = (
            pd.to_datetime(confirmed["confirmation_time"])
            - pd.to_datetime(confirmed["qualification_time"])
        ).dt.total_seconds() / 60
        print(
            f"\nConfirmed setups: {len(confirmed)}. Minutes from qualified "
            f"to confirmed: median {gap.median():.0f}, "
            f"min {gap.min():.0f}, max {gap.max():.0f}"
        )

    rejected = setups[setups["state"] == "REJECTED"]
    if not rejected.empty:
        print("\nTop reasons for rejection:")
        for reason, count in rejected["reason"].value_counts().head(5).items():
            print(f"  {count:>5}  {reason}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--interval", type=int, default=144,
                        help="assess every Nth M5 candle (144 = every 12 hours)")
    parser.add_argument("--start", default=None, help="e.g. 2026-10-01")
    parser.add_argument("--end", default=None, help="e.g. 2026-11-05")
    parser.add_argument("--max", type=int, default=None,
                        help="stop after this many assessments")
    parser.add_argument("--save", action="store_true",
                        help="save the per-assessment table as a CSV report")
    args = parser.parse_args(argv)

    recorder = SetupRecorder()
    replay = HistoricalStrategyReplay(strategy_orchestrator=recorder)

    started = time.time()
    result = replay.run(
        symbol=args.symbol,
        start_time=args.start,
        end_time=args.end,
        assessment_interval=args.interval,
        max_assessments=args.max,
    )
    seconds = time.time() - started

    table = build_rows(result)
    print_summary(result, table, seconds)
    setups = recorder.table()
    print_setup_summary(setups)

    if args.save and not table.empty:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = replay.storage.save_report(
            table, f"{args.symbol}_replay_{stamp}"
        )
        print(f"\nSaved: {path}")
        if not setups.empty:
            path = replay.storage.save_report(
                setups, f"{args.symbol}_replay_setups_{stamp}"
            )
            print(f"Saved: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())