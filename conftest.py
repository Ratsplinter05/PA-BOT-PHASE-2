"""Shared test setup.

MetaTrader5 only installs on Windows. If it cannot be imported, register a
harmless stand-in so offline tests still run on Linux/macOS/CI. On a normal
Windows setup the real package is used and this does nothing.
"""

import sys
import types

try:
    import MetaTrader5  # noqa: F401
except ImportError:
    stub = types.ModuleType("MetaTrader5")
    stub.__getattr__ = lambda name: (lambda *args, **kwargs: None)
    sys.modules["MetaTrader5"] = stub
