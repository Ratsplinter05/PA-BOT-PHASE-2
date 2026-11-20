
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from research.historical_storage import HistoricalDataStorage


class TestHistoricalDataStorage(unittest.TestCase):

    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.output_directory = Path(self.temp_directory.name)

        self.config_patch = patch(
            "research.historical_storage.load_settings",
            return_value={
                "research": {
                    "historical_data": {
                        "storage_format": "parquet",
                        "output_directory": str(self.output_directory),
                    }
                }
            }
        )

        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

        self.addCleanup(self.temp_directory.cleanup)

        self.storage = HistoricalDataStorage()

        self.sample_data = pd.DataFrame({
            "time": pd.date_range(
                "2025-01-01",
                periods=3,
                freq="5min",
                tz="UTC"
            ),
            "open": [1.1000, 1.1005, 1.1010],
            "high": [1.1010, 1.1015, 1.1020],
            "low": [1.0995, 1.1000, 1.1005],
            "close": [1.1005, 1.1010, 1.1015],
            "tick_volume": [100, 120, 110],
            "spread": [10, 11, 10],
            "real_volume": [0, 0, 0],
        })

    def test_creates_storage_directories(self):
        self.assertTrue(
            (self.output_directory / "raw").is_dir()
        )

        self.assertTrue(
            (self.output_directory / "reports").is_dir()
        )

    def test_saves_parquet_dataset(self):
        output_path = self.storage.save_dataset(
            self.sample_data,
            "EURUSD",
            "M5"
        )

        self.assertTrue(output_path.exists())
        self.assertEqual(
            output_path.name,
            "EURUSD_M5.parquet"
        )

    def test_loads_saved_dataset(self):
        self.storage.save_dataset(
            self.sample_data,
            "EURUSD",
            "M5"
        )

        loaded_data = self.storage.load_dataset(
            "EURUSD",
            "M5"
        )

        pd.testing.assert_frame_equal(
            loaded_data,
            self.sample_data,
            check_dtype=False
        )

    def test_prevents_overwriting_existing_dataset(self):
        self.storage.save_dataset(
            self.sample_data,
            "EURUSD",
            "M5"
        )

        with self.assertRaises(FileExistsError):
            self.storage.save_dataset(
                self.sample_data,
                "EURUSD",
                "M5"
            )

    def test_allows_explicit_dataset_overwrite(self):
        self.storage.save_dataset(
            self.sample_data,
            "EURUSD",
            "M5"
        )

        updated_data = self.sample_data.copy()
        updated_data.loc[0, "close"] = 1.2000

        self.storage.save_dataset(
            updated_data,
            "EURUSD",
            "M5",
            overwrite=True
        )

        loaded_data = self.storage.load_dataset(
            "EURUSD",
            "M5"
        )

        self.assertEqual(
            loaded_data.loc[0, "close"],
            1.2000
        )

    def test_rejects_empty_dataset(self):
        with self.assertRaises(ValueError):
            self.storage.save_dataset(
                pd.DataFrame(),
                "EURUSD",
                "M5"
            )

    def test_rejects_invalid_dataset_type(self):
        with self.assertRaises(TypeError):
            self.storage.save_dataset(
                [],
                "EURUSD",
                "M5"
            )

    def test_rejects_invalid_symbol_path(self):
        with self.assertRaises(ValueError):
            self.storage.save_dataset(
                self.sample_data,
                "../EURUSD",
                "M5"
            )

    def test_does_not_modify_original_dataset(self):
        original_data = self.sample_data.copy(deep=True)

        self.storage.save_dataset(
            self.sample_data,
            "EURUSD",
            "M5"
        )

        pd.testing.assert_frame_equal(
            self.sample_data,
            original_data
        )

    def test_saves_dataframe_report(self):
        report = pd.DataFrame({
            "symbol": ["EURUSD"],
            "timeframe": ["M5"],
            "status": ["PASS"],
        })

        output_path = self.storage.save_report(
            report,
            "validation_report"
        )

        self.assertTrue(output_path.exists())

        loaded_report = pd.read_csv(output_path)

        self.assertEqual(
            loaded_report.loc[0, "status"],
            "PASS"
        )

    def test_saves_list_report(self):
        report = [
            {
                "symbol": "EURUSD",
                "status": "PASS"
            },
            {
                "symbol": "GBPUSD",
                "status": "WARNING"
            }
        ]

        output_path = self.storage.save_report(
            report,
            "collection_report"
        )

        self.assertTrue(output_path.exists())

        loaded_report = pd.read_csv(output_path)

        self.assertEqual(len(loaded_report), 2)

    def test_prevents_overwriting_existing_report(self):
        report = [
            {"symbol": "EURUSD", "status": "PASS"}
        ]

        self.storage.save_report(
            report,
            "validation_report"
        )

        with self.assertRaises(FileExistsError):
            self.storage.save_report(
                report,
                "validation_report"
            )

    def test_rejects_empty_report(self):
        with self.assertRaises(ValueError):
            self.storage.save_report(
                [],
                "empty_report"
            )

    def test_rejects_invalid_report_type(self):
        with self.assertRaises(TypeError):
            self.storage.save_report(
                "invalid report",
                "validation_report"
            )

    def test_rejects_missing_dataset(self):
        with self.assertRaises(FileNotFoundError):
            self.storage.load_dataset(
                "EURUSD",
                "M5"
            )

    def test_rejects_invalid_storage_format(self):
        with patch(
            "research.historical_storage.load_settings",
            return_value={
                "research": {
                    "historical_data": {
                        "storage_format": "csv",
                        "output_directory": str(self.output_directory),
                    }
                }
            }
        ):
            with self.assertRaises(ValueError):
                HistoricalDataStorage()


    def test_rejects_dot_symbol(self):
        with self.assertRaises(ValueError):
            self.storage.save_dataset(
                self.sample_data,
                ".",
                "M5"
            )

    def test_rejects_dotdot_symbol(self):
        with self.assertRaises(ValueError):
            self.storage.save_dataset(
                self.sample_data,
                "..",
                "M5"
            )

    def test_rejects_dotdot_timeframe(self):
        with self.assertRaises(ValueError):
            self.storage.save_dataset(
                self.sample_data,
                "EURUSD",
                ".."
            )

    def test_rejects_dotdot_report_name(self):
        with self.assertRaises(ValueError):
            self.storage.save_report(
                [{"status": "PASS"}],
                ".."
            )

    def test_dataset_write_failure_removes_temporary_file(self):
        output_path = self.storage._dataset_path("EURUSD", "M5")

        with patch.object(
            pd.DataFrame,
            "to_parquet",
            side_effect=OSError("Simulated Parquet write failure")
        ):
            with self.assertRaises(OSError):
                self.storage.save_dataset(
                    self.sample_data,
                    "EURUSD",
                    "M5"
                )

        self.assertFalse(output_path.exists())
        self.assertEqual(
            list(output_path.parent.glob(".*.parquet")),
            []
        )

    def test_report_write_failure_removes_temporary_file(self):
        output_path = (
            self.storage.reports_directory
            / "failure_report.csv"
        )

        with patch.object(
            pd.DataFrame,
            "to_csv",
            side_effect=OSError("Simulated CSV write failure")
        ):
            with self.assertRaises(OSError):
                self.storage.save_report(
                    [{"status": "PASS"}],
                    "failure_report"
                )

        self.assertFalse(output_path.exists())
        self.assertEqual(
            list(output_path.parent.glob(".*.csv")),
            []
        )

    def test_no_overwrite_race_preserves_existing_dataset(self):
        output_path = self.storage._dataset_path("EURUSD", "M5")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        competing_content = "created by competing process"

        def simulate_competing_creation(source, destination):
            Path(destination).write_text(
                competing_content,
                encoding="utf-8"
            )
            raise FileExistsError("Simulated concurrent file creation")

        with patch(
            "research.historical_storage.os.link",
            side_effect=simulate_competing_creation
        ):
            with self.assertRaises(FileExistsError):
                self.storage.save_dataset(
                    self.sample_data,
                    "EURUSD",
                    "M5"
                )

        self.assertEqual(
            output_path.read_text(encoding="utf-8"),
            competing_content
        )

        self.assertEqual(
            list(output_path.parent.glob(".*.parquet")),
            []
        )

    def test_report_write_failure_is_logged(self):
        with patch.object(
            pd.DataFrame,
            "to_csv",
            side_effect=OSError("Simulated CSV write failure")
        ), patch(
            "research.historical_storage.log.exception"
        ) as log_exception:
            with self.assertRaises(OSError):
                self.storage.save_report(
                    [{"status": "PASS"}],
                    "logged_failure_report"
                )

        log_exception.assert_called_once()


if __name__ == "__main__":
    unittest.main()
