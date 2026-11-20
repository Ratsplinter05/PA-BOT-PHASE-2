from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

def load_settings():
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    settings["mt5"]["terminal_path"] = (
        os.getenv("MT5_PATH") or settings["mt5"].get("terminal_path")
    )
    settings["mt5"]["login"] = os.getenv("MT5_LOGIN") or None
    settings["mt5"]["password"] = os.getenv("MT5_PASSWORD") or None
    settings["mt5"]["server"] = os.getenv("MT5_SERVER") or None
    return settings
