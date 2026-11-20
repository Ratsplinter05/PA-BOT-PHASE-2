from pathlib import Path

p = Path("src/live_scanner.py")
src = p.read_text(encoding="utf-8")

# 1. Add import
old_import = "from src.trade_parameters import TradeParameterCalculator"
new_import = (
    "from src.risk_manager import RiskManager\n"
    "from src.trade_parameters import TradeParameterCalculator"
)
assert old_import in src, "import not found"
src = src.replace(old_import, new_import, 1)

# 2. Add risk_manager to __init__
old_init = (
    "        self.calculator = TradeParameterCalculator(\n"
    "            rr_ratio=rr_ratio,\n"
    "            risk_percent=risk_percent,\n"
    "        )"
)
new_init = (
    "        self.calculator = TradeParameterCalculator(\n"
    "            rr_ratio=rr_ratio,\n"
    "            risk_percent=risk_percent,\n"
    "        )\n"
    "        self.risk_manager = RiskManager(\n"
    "            max_risk_percent=risk_percent,\n"
    "        )"
)
assert old_init in src, "init block not found"
src = src.replace(old_init, new_init, 1)

# 3. Add risk check inside _process_confirmations after param_result is valid
old_log = (
    "            if param_result.valid:\n"
    "                p = param_result.parameters\n"
    "                log.info(\n"
    "                    \"NEW CONFIRMED SETUP | Instrument=%s | Direction=%s | \"\n"
    "                    \"Entry=%.5f | SL=%.5f | TP=%.5f | Lots=%.2f | \"\n"
    "                    \"Risk=%.2f | RR=1:%.1f | SetupID=%s\","
)
new_log = (
    "            if param_result.valid:\n"
    "                p = param_result.parameters\n"
    "                risk_decision = self.risk_manager.check(\n"
    "                    symbol=instrument,\n"
    "                    account_balance=account_balance,\n"
    "                    proposed_lot_size=p.lot_size,\n"
    "                    sl_distance=p.sl_distance,\n"
    "                    pip_value_per_lot=pip_value,\n"
    "                )\n"
    "                if not risk_decision.approved:\n"
    "                    log.warning(\n"
    "                        \"Trade blocked by risk manager | \"\n"
    "                        \"Instrument=%s | Reason=%s\",\n"
    "                        instrument,\n"
    "                        risk_decision.reason,\n"
    "                    )\n"
    "                    continue\n"
    "                p = param_result.parameters\n"
    "                log.info(\n"
    "                    \"NEW CONFIRMED SETUP | Instrument=%s | Direction=%s | \"\n"
    "                    \"Entry=%.5f | SL=%.5f | TP=%.5f | Lots=%.2f | \"\n"
    "                    \"Risk=%.2f | RR=1:%.1f | SetupID=%s\","
)
assert old_log in src, "log block not found"
src = src.replace(old_log, new_log, 1)

p.write_text(src, encoding="utf-8")
print("Done.", p.stat().st_size, "bytes")
