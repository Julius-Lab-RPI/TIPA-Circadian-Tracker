import tempfile
import unittest
from pathlib import Path

import pandas as pd

from convert_garmin_data import convert_subject
from data_io import load_minute_data


def _write_garmin_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = [
        "Header Length,5",
        "Powered by Labfront",
        "Documentation,https://example.test",
        "projectId,projectTitle,participantId",
        "project-1,Example,subject-1",
        "",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        output.write("\n".join(metadata) + "\n")
        pd.DataFrame(rows, columns=headers).to_csv(output, index=False)


class GarminDataConversionTests(unittest.TestCase):
    def test_converts_per_minute_samples_and_leaves_missing_values_blank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory) / "subject-1"
            heart_rate_folder = subject / "garmin-connect-daily-heart-rate"
            activity_folder = subject / "garmin-device-actigraphy"
            _write_garmin_csv(
                heart_rate_folder / "261017_heart.csv",
                ["isoDate", "beatsPerMinute"],
                [
                    ["2026-10-17T13:08:15.000+09:00", 71],
                    ["2026-10-17T13:08:30.000+09:00", 72],
                    ["2026-10-17T13:08:45.000+09:00", 73],
                    ["2026-10-17T13:09:00.000+09:00", 74],
                    ["2026-10-17T13:10:00.000+09:00", 75],
                ],
            )
            _write_garmin_csv(
                activity_folder / "261017_activity.csv",
                ["isoDate", "zeroCrossingCount"],
                [
                    ["2026-10-17T13:08:59.000+09:00", 8],
                    ["2026-10-17T13:11:00.000+09:00", 10],
                ],
            )

            report = convert_subject(subject)
            result = pd.read_csv(report.output_path)

            self.assertEqual(
                list(result.columns),
                ["time", "ZeroCrossingCount", "HeartRate"],
            )
            self.assertEqual(report.row_count, 3)
            self.assertEqual(report.missing_heart_rate, 1)
            self.assertEqual(report.missing_zero_crossing_count, 1)
            self.assertEqual(result.loc[0, "ZeroCrossingCount"], 8)
            self.assertEqual(result.loc[0, "HeartRate"], 74)
            self.assertTrue(pd.isna(result.loc[1, "ZeroCrossingCount"]))
            self.assertEqual(result.loc[1, "HeartRate"], 75)
            self.assertEqual(result.loc[2, "ZeroCrossingCount"], 10)
            self.assertTrue(pd.isna(result.loc[2, "HeartRate"]))

            loaded = load_minute_data(report.output_path)

        self.assertEqual(len(loaded.timestamps), 3)
        self.assertEqual(
            list(loaded.columns.columns),
            ["ZeroCrossingCount", "HeartRate"],
        )
        self.assertEqual(loaded.columns.iloc[0].tolist(), [8.0, 74.0])
        self.assertTrue(pd.isna(loaded.columns.iloc[1, 0]))

    def test_reports_when_both_source_folders_have_no_csv_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = convert_subject(Path(directory) / "subject-without-data")

        self.assertIsNone(report.output_path)
        self.assertEqual(report.row_count, 0)


if __name__ == "__main__":
    unittest.main()
