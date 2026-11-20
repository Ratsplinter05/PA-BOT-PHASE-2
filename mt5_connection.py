import MetaTrader5 as mt5
from src.config import load_settings
from src.logger import get_logger

log = get_logger("MT5")

class MT5Connection:
    def __init__(self):
        self.settings = load_settings()
        self.connected = False

    def connect(self):
        cfg = self.settings["mt5"]
        kwargs = {"timeout": int(cfg.get("timeout_ms", 60000))}
        if cfg.get("terminal_path"):
            kwargs["path"] = cfg["terminal_path"]

        login = cfg.get("login")
        password = cfg.get("password")
        server = cfg.get("server")

        # If all credentials are supplied, initialize and log in explicitly.
        # Otherwise use the account already authenticated in the MT5 terminal.
        if login and password and server:
            initialized = mt5.initialize(**kwargs, login=int(login),
                                         password=password, server=server)
        else:
            initialized = mt5.initialize(**kwargs)

        if not initialized:
            error = mt5.last_error()
            log.error("MT5 initialization failed: %s", error)
            raise RuntimeError(f"Could not initialize MT5: {error}")

        terminal = mt5.terminal_info()
        account = mt5.account_info()
        if terminal is None or account is None:
            error = mt5.last_error()
            mt5.shutdown()
            raise RuntimeError(f"MT5 terminal/account information unavailable: {error}")

        self.connected = True
        log.info("Connected to MT5. Account login=%s server=%s",
                 account.login, account.server)
        return terminal, account

    def ensure_connected(self):
        if not self.connected:
            return self.connect()
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        if terminal is None or account is None:
            log.warning("MT5 connection appears unavailable; reconnecting.")
            self.disconnect()
            return self.connect()
        return terminal, account

    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
            log.info("MT5 connection closed.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
