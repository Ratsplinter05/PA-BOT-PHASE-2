"""Phase 1 connectivity and market-data check (read-only).

Connects to the MT5 terminal, discovers and resolves the configured
instruments, then validates the latest closed candles on every timeframe.
No orders are placed and trading execution is not implemented.
"""

from datetime import datetime, timedelta, timezone

from src.config import load_settings
from src.mt5_connection import MT5Connection
from src.market_data import MarketData
from src.data_validation import CandleValidator
from src.logger import get_logger


log = get_logger("PHASE1")

SYMBOL_CANDIDATES = {
    "XAUUSD": ["XAUUSD.0", "XAUUSD"],
    "GBPUSD": ["GBPUSD.0", "GBPUSD"],
    "BTCUSD": ["BTCUSD.0", "BTCUSD"],
    "EURUSD": ["EURUSD.0", "EURUSD"],
    "US30": ["Wall Street 30.0", "US30"],
    "NAS100": ["US Tech 100.0", "NAS100"],
}


TIMEFRAME_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440,
}

# Candle data older than this (in market-open time) is flagged as stale.
STALE_AFTER_CANDLES = 2
MIN_STALE_MINUTES = 120

# Weekly market closure window (UTC): Friday 22:00 to Sunday 22:00.
WEEKEND_START_WEEKDAY = 4
WEEKEND_HOUR = 22


def weekend_minutes(start, end):
    """Minutes between start and end that fall inside the weekly closure."""
    if end <= start:
        return 0.0

    total = 0.0
    day = (start - timedelta(days=7)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    while day <= end:
        if day.weekday() == WEEKEND_START_WEEKDAY:
            window_start = day.replace(hour=WEEKEND_HOUR)
            window_end = window_start + timedelta(days=2)
            overlap = min(end, window_end) - max(start, window_start)
            total += max(overlap.total_seconds(), 0.0) / 60
        day += timedelta(days=1)

    return total


def stale_candle_message(latest_open_time, timeframe, now):
    """Return a warning if the latest closed candle is too old, else None."""
    minutes = TIMEFRAME_MINUTES.get(timeframe)
    if minutes is None:
        return None

    close_time = latest_open_time + timedelta(minutes=minutes)
    gap = (now - close_time).total_seconds() / 60
    gap -= weekend_minutes(close_time, now)

    limit = max(STALE_AFTER_CANDLES * minutes, MIN_STALE_MINUTES)

    if gap <= limit:
        return None

    return (
        f"Latest closed candle is {gap / 60:.1f} hours behind "
        f"(limit {limit / 60:.1f}). MT5 history may not be synced; "
        "open a chart for this symbol and run again."
    )


def resolve_broker_symbol(market, requested, available):
    """Return the broker symbol for a requested instrument, or None."""
    for candidate in SYMBOL_CANDIDATES.get(requested, [requested]):
        actual = market.resolve_symbol(candidate, available)
        if actual:
            return actual
    return None


def check_instrument(market, validator, requested, actual,
                     timeframes, candle_count, now=None):
    """Validate closed candles for one instrument; return result rows."""
    rows = []
    now = now or datetime.now(timezone.utc)

    tick = market.get_tick(actual)
    print(f"\n{requested} ({actual})")
    print(f"  Tick: {tick}")

    for timeframe in timeframes:
        try:
            candles = market.get_closed_candles(
                actual, timeframe, count=candle_count
            )

            if candles.empty:
                rows.append({
                    "instrument": requested,
                    "symbol": actual,
                    "timeframe": timeframe,
                    "status": "FAIL",
                    "candles": 0,
                    "details": "No candle data returned.",
                })
                print(f"  {timeframe}: NO DATA")
                continue

            validation = validator.validate(candles, timeframe=timeframe)
            status = validation["status"]
            problems = list(validation["errors"])
            warnings = list(validation["warnings"])

            if validation["valid"] and len(candles) < candle_count:
                status = "WARNING"
                warnings.append(
                    f"Only {len(candles)} of {candle_count} "
                    "requested candles returned."
                )

            latest = candles.iloc[-1]

            if validation["valid"]:
                stale = stale_candle_message(
                    latest["time"].to_pydatetime(), timeframe, now
                )
                if stale:
                    status = "WARNING"
                    warnings.append(stale)

            details = "; ".join(problems + warnings) or "All checks passed."

            rows.append({
                "instrument": requested,
                "symbol": actual,
                "timeframe": timeframe,
                "status": status,
                "candles": len(candles),
                "details": details,
            })
            print(
                f"  {timeframe}: {status} | {len(candles)} candles | "
                f"latest closed={latest['time']} | "
                f"close={latest['close']} | {details}"
            )

        except Exception as error:
            log.exception("Check failed for %s %s", actual, timeframe)
            rows.append({
                "instrument": requested,
                "symbol": actual,
                "timeframe": timeframe,
                "status": "FAIL",
                "candles": 0,
                "details": str(error),
            })
            print(f"  {timeframe}: FAIL | {error}")

    return rows


def main():
    """Run the Phase 1 check. Returns 0 if nothing failed, otherwise 1."""

    settings = load_settings()

    requested_symbols = settings["market"]["instruments"]
    timeframes = list(settings["market"]["timeframes"].keys())
    candle_count = int(settings["market"]["candles_per_timeframe"])
    execution_enabled = settings["trading"]["execution_enabled"]

    market = MarketData()
    validator = CandleValidator()
    connection = MT5Connection()

    results = []

    try:
        terminal, account = connection.connect()

        print("\n=== DERIV / MT5 ACCOUNT CHECK ===")
        print(f"Terminal: {terminal.name}")
        print(f"Company:  {account.company}")
        print(f"Server:   {account.server}")
        print(f"Login:    {account.login}")
        print(f"Currency: {account.currency}")
        print(f"Balance:  {account.balance}")
        print(f"Equity:   {account.equity}")
        print(f"Trading execution enabled: {execution_enabled}")

        available = market.discover_symbols()
        print(f"\nBroker symbols discovered: {len(available)}")

        print("\n=== INSTRUMENT DISCOVERY ===")
        resolved = {}

        for requested in requested_symbols:
            actual = resolve_broker_symbol(market, requested, available)

            if actual is None:
                print(
                    f"[NOT FOUND] {requested} - "
                    "check Deriv Market Watch / symbol suffix."
                )
                possible = [
                    s for s in available if requested.upper() in s.upper()
                ]
                if possible:
                    print(f"  Possible matches: {', '.join(possible[:10])}")

                for timeframe in timeframes:
                    results.append({
                        "instrument": requested,
                        "symbol": "NOT FOUND",
                        "timeframe": timeframe,
                        "status": "FAIL",
                        "candles": 0,
                        "details": "Broker symbol not found.",
                    })
                continue

            try:
                info = market.enable_symbol(actual)
            except Exception as error:
                print(f"[ERROR] {requested} -> {actual}: {error}")
                for timeframe in timeframes:
                    results.append({
                        "instrument": requested,
                        "symbol": actual,
                        "timeframe": timeframe,
                        "status": "FAIL",
                        "candles": 0,
                        "details": str(error),
                    })
                continue

            resolved[requested] = actual
            print(
                f"[FOUND] {requested} -> {actual} | "
                f"digits={info.digits} | "
                f"point={info.point} | "
                f"trade_mode={info.trade_mode} | "
                f"volume_min={info.volume_min} | "
                f"volume_step={info.volume_step}"
            )

        print("\n=== MARKET DATA TEST ===")
        for requested, actual in resolved.items():
            results.extend(
                check_instrument(
                    market, validator, requested, actual,
                    timeframes, candle_count,
                )
            )

        passed = sum(r["status"] == "PASS" for r in results)
        warned = sum(r["status"] == "WARNING" for r in results)
        failed = sum(r["status"] == "FAIL" for r in results)

        print("\n=== SUMMARY ===")
        print(f"Total checks: {len(results)}")
        print(f"PASS: {passed}  WARNING: {warned}  FAIL: {failed}")

        print("\nPhase 1 connectivity and market-data check completed.")
        print("No orders were sent. Trading execution is not implemented in Phase 1.")

        return 1 if failed else 0

    except Exception as exc:
        log.exception("Phase 1 check failed: %s", exc)
        raise
    finally:
        connection.disconnect()
