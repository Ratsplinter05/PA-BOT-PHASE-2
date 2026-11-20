# PA-MS Trading Bot - Phase 1

Phase 1 is a read-only Deriv MT5 connectivity and market-data diagnostic.
It does not place, modify, or close trades.

## Requirements

- Windows 10 or Windows 11
- Deriv MT5 terminal installed
- A Deriv MT5 demo account is recommended for initial testing
- Python 3.11 (64-bit) recommended
- Internet connection

## Setup

1. Install Python 3.11 (64-bit) from python.org. During installation, tick
   "Add Python to PATH".
2. Install and open Deriv MT5. Log in to your Deriv MT5 demo account.
3. In MT5, open Market Watch and make sure the instruments you want are
   available. The broker may use symbol suffixes or different names.
4. Extract this project folder to a simple path, for example:
   `C:\Trading\PA_MS_Trading_Bot_Phase1`
5. Open the folder in VS Code, or open PowerShell in that folder.
6. Create a virtual environment:

   `py -3.11 -m venv .venv`

7. Activate it in PowerShell:

   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

   ` .\.venv\Scripts\Activate.ps1`

   (The leading space before the activation command is optional.)

8. Upgrade pip and install packages:

   `python -m pip install --upgrade pip`

   `pip install -r requirements.txt`

9. Copy `.env.example` to `.env`. Leave credentials blank initially if you
   have already logged into the MT5 terminal. Do not share the `.env` file.
10. Run the diagnostic:

    `python main.py`

## If MT5 cannot connect

- Confirm Deriv MT5 is open and logged in.
- Confirm you are using an MT5 account, not a Deriv cTrader or Deriv X account.
- In MT5, check that Algo Trading is not relevant to this read-only diagnostic;
  this phase does not submit orders.
- If multiple MT5 terminals are installed, set `MT5_PATH` in `.env` to the
  full path of the Deriv terminal executable, usually terminal64.exe.
- Check that your account/server details are correct if you choose explicit
  credentials in `.env`.
- Run `python main.py` again and inspect `logs/pa_ms_phase1.log`.

## Symbol mapping

If a requested symbol is not found, the diagnostic prints possible matches.
Update `config/settings.yaml` with the exact Deriv symbol names once confirmed.
Do not assume that a symbol name from another broker will work on Deriv.

## Data notes

MT5 timestamps are converted to UTC in the diagnostic output. MT5's newest
returned candle is normally still forming. The script labels it as such.
Later strategy modules must use completed candles for confirmed signals.

## Security

- Never commit `.env` or share account passwords.
- Start with a demo account.
- Do not enable live execution until later phases have been independently
  tested, including risk checks, order validation and manual approval.
