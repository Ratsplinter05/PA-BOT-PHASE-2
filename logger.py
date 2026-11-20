import logging
from pathlib import Path
from src.config import PROJECT_ROOT, load_settings

def get_logger(name="PA-MS"):
    settings = load_settings()
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "pa_ms_phase1.log"

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, settings["app"].get("log_level", "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
