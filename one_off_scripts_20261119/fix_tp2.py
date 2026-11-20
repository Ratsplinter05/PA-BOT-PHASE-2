from pathlib import Path

p = Path(r"src\trade_parameters.py")
src = p.read_text(encoding="utf-8")

old = "from strategy.m15_setup_detector import M15Setup\nfrom strategy.m5_confirmation_detector import M5Confirmation"
new = "# M15Setup and M5Confirmation are accepted by duck-typing (hasattr checks)."

assert old in src, "OLD TEXT NOT FOUND"
out = src.replace(old, new, 1)
p.write_text(out, encoding="utf-8")
print("Done.")