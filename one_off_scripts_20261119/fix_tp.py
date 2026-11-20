from pathlib import Path

p = Path(r"src\trade_parameters.py")
src = p.read_text(encoding="utf-8")

old = """        if not isinstance(setup, M15Setup):
            return self._reject("setup must be an M15Setup instance.")

        if not isinstance(confirmation, M5Confirmation):
            return self._reject(
                "confirmation must be an M5Confirmation instance."
            )"""

new = """        for attr in ("direction", "state", "atr_value", "broken_level"):
            if not hasattr(setup, attr):
                return self._reject(
                    f"setup is missing required attribute: {attr!r}."
                )

        for attr in ("direction", "state", "setup_id", "confirmation_time",
                     "confirmation_event"):
            if not hasattr(confirmation, attr):
                return self._reject(
                    f"confirmation is missing required attribute: {attr!r}."
                )"""

assert old in src, "OLD TEXT NOT FOUND"
out = src.replace(old, new, 1)
p.write_text(out, encoding="utf-8")
print("Done.")