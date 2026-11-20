"""Entry point for the live scanner.

Run from the project root:
    python -m src.run_scanner
"""

from src.live_scanner import LiveScanner
from src.config import load_settings


def main():
    settings = load_settings()
    scan_interval = int(
        settings.get("market", {}).get("scan_interval_seconds", 15)
    )
    scanner = LiveScanner(
        scan_interval_seconds=scan_interval,
        rr_ratio=2.0,
        risk_percent=1.0,
    )
    scanner.run()


if __name__ == "__main__":
    main()