
from pathlib import Path
import re

import pandas as pd

from src.config import load_settings
from src.logger import get_logger


log = get_logger("HISTORICAL_STORAGE")


class HistoricalDataStorage:
    """
    Stores and retrieves historical market data.

    Features:
    - Configurable output directory.
    - Parquet storage for historical candle datasets.
    - CSV storage for collection and validation reports.
    - Separate raw data and report directories.
    - Prevents accidental overwriting by default.
    - Supports reading saved Parquet datasets.
    - Does not modify the supplied DataFrame.
    """

    def __init__(self):
        self.settings = load_settings()
        self.config = self.settings["research"]["historical_data"]

        self.storage_format = str(
            self.config.get("storage_format", "parquet")
        ).lower()

        if self.storage_format != "parquet":
            raise ValueError(
                f"Unsupported storage format: {self.storage_format}"
            )

        self.output_directory = Path(
            self.config.get(
                "output_directory",
                "data/historical"
            )
        )

        self.raw_directory = self.output_directory / "raw"
        self.reports_directory = self.output_directory / "reports"

        self.raw_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.reports_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    @staticmethod
    def _validate_name(value, name):
        """Validate a symbol or timeframe used in a filename."""

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{name} must be a non-empty string."
            )

        value = value.strip()

        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError(
                f"Invalid {name}: {value}"
            )

        return value

    def _dataset_path(self, symbol, timeframe):
        """Build the output path for one dataset."""

        symbol = self._validate_name(symbol, "symbol")
        timeframe = self._validate_name(timeframe, "timeframe")

        symbol_directory = self.raw_directory / symbol

        return (
            symbol_directory
            / f"{symbol}_{timeframe}.parquet"
        )

    def save_dataset(
        self,
        df,
        symbol,
        timeframe,
        overwrite=False
    ):
        """
        Save one historical candle dataset.

        Args:
            df: Historical candle DataFrame.
            symbol: Resolved broker symbol.
            timeframe: M5, M15, H1 or H4.
            overwrite: Whether an existing dataset may be replaced.

        Returns:
            Path to the saved Parquet file.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Dataset must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Cannot save an empty historical dataset."
            )

        output_path = self._dataset_path(
            symbol,
            timeframe
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Dataset already exists: {output_path}. "
                "Use overwrite=True to replace it."
            )

        try:
            df.to_parquet(
                output_path,
                engine="pyarrow",
                index=False
            )

        except Exception:
            log.exception(
                "Failed to save historical dataset | "
                "Symbol=%s | Timeframe=%s",
                symbol,
                timeframe
            )
            raise

        log.info(
            "Historical dataset saved | "
            "Symbol=%s | Timeframe=%s | "
            "Candles=%s | Path=%s",
            symbol,
            timeframe,
            len(df),
            output_path
        )

        return output_path

    def load_dataset(self, symbol, timeframe):
        """
        Load one previously saved historical dataset.

        Returns:
            pandas DataFrame.
        """

        input_path = self._dataset_path(
            symbol,
            timeframe
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Historical dataset not found: {input_path}"
            )

        return pd.read_parquet(
            input_path,
            engine="pyarrow"
        )

    def save_report(
        self,
        report,
        report_name,
        overwrite=False
    ):
        """
        Save a collection or validation report as CSV.

        Args:
            report: pandas DataFrame or list of dictionaries.
            report_name: Report filename without extension.
            overwrite: Whether an existing report may be replaced.

        Returns:
            Path to the saved CSV report.
        """

        report_name = self._validate_name(
            report_name,
            "report_name"
        )

        if isinstance(report, pd.DataFrame):
            report_df = report.copy()

        elif isinstance(report, list):
            report_df = pd.DataFrame(report)

        else:
            raise TypeError(
                "Report must be a DataFrame or a list of dictionaries."
            )

        if report_df.empty:
            raise ValueError(
                "Cannot save an empty report."
            )

        output_path = (
            self.reports_directory
            / f"{report_name}.csv"
        )

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Report already exists: {output_path}. "
                "Use overwrite=True to replace it."
            )

        report_df.to_csv(
            output_path,
            index=False
        )

        log.info(
            "Research report saved | "
            "Name=%s | Rows=%s | Path=%s",
            report_name,
            len(report_df),
            output_path
        )

        return output_path
