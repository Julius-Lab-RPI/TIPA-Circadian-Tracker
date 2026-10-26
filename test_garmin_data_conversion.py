import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from convert_garmin_data import convert_subject
from data_io import load_minute_data
from run_tpbm_gui import _run_dataset, _subject_directories


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
    def test_folder_gui_workflow_converts_and_estimates_each_subject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            subjects = [dataset / "subject-1", dataset / "subject-2"]
            for subject in subjects:
                subject.mkdir()
            generated = [
                subject / "Minute-by-minute.csv" for subject in subjects
            ]
            reports = [
                SimpleNamespace(subject=subject.name, output_path=output)
                for subject, output in zip(subjects, generated, strict=True)
            ]
            run_results = [
                {"summary": {"subject": subject.name, "output_dir": str(dataset / f"run-{subject.name}")}}
                for subject in subjects
            ]
            progress_messages: list[str] = []

            with (
                patch("run_tpbm_gui.convert_subject", side_effect=reports) as convert,
                patch("run_tpbm_gui.run_pipeline", side_effect=run_results) as run,
            ):
                result = _run_dataset(
                    dataset,
                    dataset / "results",
                    "ZeroCrossingCount",
                    "KF",
                    progress_messages.append,
                )

        self.assertEqual(convert.call_count, 2)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            [entry["subject"] for entry in result["completed"]],
            ["subject-1", "subject-2"],
        )
        self.assertEqual(result["failed"], [])
        self.assertEqual(result["skipped"], [])
        self.assertEqual(len(progress_messages), 2)
        self.assertEqual(run.call_args_list[0].args[0].option, 2)
        self.assertEqual(run.call_args_list[0].args[0].signal_label, "ZeroCrossingCount")
        self.assertEqual(run.call_args_list[0].args[0].phase_estimator, "KF")

    def test_folder_discovery_accepts_a_single_subject_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory) / "subject-1"
            (subject / "garmin-connect-daily-heart-rate").mkdir(parents=True)
            (subject / "garmin-device-actigraphy").mkdir()

            self.assertEqual(_subject_directories(subject), [subject])

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
