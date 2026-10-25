"""Convert Garmin heart-rate and actigraphy exports to minute-level CSV files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

HEART_RATE_FOLDER = "garmin-connect-daily-heart-rate"
ACTIGRAPHY_FOLDER = "garmin-device-actigraphy"
GARMIN_METADATA_ROWS = 6


@dataclass(frozen=True, slots=True)
class SubjectConversionReport:
    subject: str
    output_path: Path | None
    row_count: int
    missing_heart_rate: int
    missing_zero_crossing_count: int
    heart_rate_only_file_dates: tuple[str, ...]
    actigraphy_only_file_dates: tuple[str, ...]


def _source_file_dates(paths: list[Path]) -> set[str]:
    dates: set[str] = set()
    for path in paths:
        try:
            dates.add(datetime.strptime(path.name[:6], "%y%m%d").date().isoformat())
        except ValueError:
            continue
    return dates


def _read_minute_values(
    folder: Path,
    column: str,
    *,
    select_minute_boundary: bool = False,
    round_to_minute: bool = False,
) -> tuple[pd.Series, list[Path]]:
    files = sorted(folder.glob("*.csv")) if folder.is_dir() else []
    parts: list[pd.DataFrame] = []
    for path in files:
        try:
            frame = pd.read_csv(
                path,
                skiprows=GARMIN_METADATA_ROWS,
                usecols=["isoDate", column],
            )
        except (OSError, pd.errors.ParserError, ValueError) as error:
            raise ValueError(f"could not read required data from {path}: {error}") from error

        try:
            timestamps = pd.DatetimeIndex(
                [
                    datetime.fromisoformat(str(value)).replace(tzinfo=None)
                    for value in frame["isoDate"]
                ]
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path} contains an invalid isoDate timestamp") from error

        values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
        if select_minute_boundary:
            selected = (timestamps.second == 0) & (timestamps.microsecond == 0)
            timestamps = timestamps[selected]
            values = values[selected]
        if round_to_minute:
            timestamps = timestamps.round("min")
        parts.append(pd.DataFrame({column: values}, index=timestamps))

    if not parts:
        return pd.Series(
            index=pd.DatetimeIndex([]),
            dtype="float64",
            name=column,
        ), files

    values = pd.concat(parts).sort_index()
    if values.index.has_duplicates:
        duplicate = values.index[values.index.duplicated()][0]
        raise ValueError(
            f"duplicate {column} readings after minute alignment in {folder}: "
            f"{duplicate}"
        )
    return values[column], files


def convert_subject(subject_directory: str | Path) -> SubjectConversionReport:
    """Write one subject's aligned minute-by-minute CSV beside its source folders."""
    subject_path = Path(subject_directory)
    heart_rate, heart_rate_files = _read_minute_values(
        subject_path / HEART_RATE_FOLDER,
        "beatsPerMinute",
        select_minute_boundary=True,
    )
    activity, activity_files = _read_minute_values(
        subject_path / ACTIGRAPHY_FOLDER,
        "zeroCrossingCount",
        round_to_minute=True,
    )
    heart_rate_dates = _source_file_dates(heart_rate_files)
    activity_dates = _source_file_dates(activity_files)

    if heart_rate.empty and activity.empty:
        return SubjectConversionReport(
            subject=subject_path.name,
            output_path=None,
            row_count=0,
            missing_heart_rate=0,
            missing_zero_crossing_count=0,
            heart_rate_only_file_dates=tuple(sorted(heart_rate_dates - activity_dates)),
            actigraphy_only_file_dates=tuple(sorted(activity_dates - heart_rate_dates)),
        )

    timestamps = heart_rate.index.union(activity.index).sort_values()
    minute_grid = pd.date_range(timestamps.min(), timestamps.max(), freq="min")
    aligned = pd.DataFrame(
        {
            "time": minute_grid,
            "ZeroCrossingCount": activity.reindex(minute_grid).to_numpy(),
            "HeartRate": heart_rate.reindex(minute_grid).to_numpy(),
        }
    )
    output_path = subject_path / "Minute-by-minute.csv"
    aligned.to_csv(output_path, index=False, na_rep="")

    return SubjectConversionReport(
        subject=subject_path.name,
        output_path=output_path,
        row_count=len(aligned),
        missing_heart_rate=int(aligned["HeartRate"].isna().sum()),
        missing_zero_crossing_count=int(aligned["ZeroCrossingCount"].isna().sum()),
        heart_rate_only_file_dates=tuple(sorted(heart_rate_dates - activity_dates)),
        actigraphy_only_file_dates=tuple(sorted(activity_dates - heart_rate_dates)),
    )


def convert_dataset(dataset_root: str | Path) -> list[SubjectConversionReport]:
    """Convert every immediate subject subfolder under a dataset directory."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise NotADirectoryError(f"dataset directory does not exist: {root}")
    subject_directories = sorted(path for path in root.iterdir() if path.is_dir())
    return [convert_subject(path) for path in subject_directories]


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="directory containing one subject folder per participant",
    )
    arguments = parser.parse_args()

    reports = convert_dataset(arguments.dataset_root)
    for report in reports:
        print(f"{report.subject}: {report.row_count:,} minute rows")
        if report.output_path is None:
            print("  No usable timestamps; no output file generated.")
            continue
        print(f"  Output: {report.output_path}")
        if report.missing_heart_rate or report.missing_zero_crossing_count:
            print(
                "  Missing minute values: "
                f"HeartRate={report.missing_heart_rate:,}; "
                f"ZeroCrossingCount={report.missing_zero_crossing_count:,}"
            )
        if report.heart_rate_only_file_dates:
            print(
                "  Heart-rate source files without actigraphy files: "
                + ", ".join(report.heart_rate_only_file_dates)
            )
        if report.actigraphy_only_file_dates:
            print(
                "  Actigraphy source files without heart-rate files: "
                + ", ".join(report.actigraphy_only_file_dates)
            )


if __name__ == "__main__":
    _main()
