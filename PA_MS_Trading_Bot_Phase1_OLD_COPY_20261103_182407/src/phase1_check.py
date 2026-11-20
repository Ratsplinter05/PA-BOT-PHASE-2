from src.config import load_settings
from src.mt5_connection import MT5Connection
from src.market_data import MarketData
from src.logger import get_logger

log = get_logger("PHASE1")

def main():
    settings = load_settings()
    requested_symbols = settings["market"]["instruments"]
    timeframes = list(settings["market"]["timeframes"].keys())
    candle_count = int(settings["market"]["candles_per_timeframe"])

    market = MarketData()
    connection = MT5Connection()

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

        available = market.discover_symbols()
        print(f"\nBroker symbols discovered: {len(available)}")

        print("\n=== INSTRUMENT DISCOVERY ===")
        resolved = {}
        for requested in requested_symbols:
            actual = market.resolve_symbol(requested, available)
            if actual is None:
                print(f"[NOT FOUND] {requested} - check Deriv Market Watch / symbol suffix.")
                # Show likely candidates to help map broker-specific names.
                candidates = [s for s in available if requested.upper() in s.upper()]
                if candidates:
                    print(f"  Possible matches: {', '.join(candidates[:10])}")
                continue

            info = market.enable_symbol(actual)
            resolved[requested] = actual
            print(
                f"[FOUND] {requested} -> {actual} | digits={info.digits} "
                f"| point={info.point} | trade_mode={info.trade_mode} "
                f"| volume_min={info.volume_min} | volume_step={info.volume_step}"
            )

        print("\n=== MARKET DATA TEST ===")
        for requested, actual in resolved.items():
            tick = market.get_tick(actual)
            print(f"\n{requested} ({actual})")
            print(f"  Tick: {tick}")

            for timeframe in timeframes:
                candles = market.get_candles(actual, timeframe, candle_count)
                if candles.empty:
                    print(f"  {timeframe}: NO DATA")
                    continue
                last_closed = candles.iloc[-2] if len(candles) > 1 else candles.iloc[-1]
                print(
                    f"  {timeframe}: {len(candles)} candles | "
                    f"latest closed={last_closed['time']} | "
                    f"close={last_closed['close']}"
                )

        print("\nPhase 1 connectivity and market-data check completed.")
        print("No orders were sent. Trading execution is not implemented in Phase 1.")

    except Exception as exc:
        log.exception("Phase 1 check failed: %s", exc)
        raise
    finally:
        connection.disconnect()

if __name__ == "__main__":
    main()
