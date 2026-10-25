"""Excel/CSV/MAT data loading translated from load_xlsx_data.m and Utils.m."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.io import loadmat
from scipy.io import savemat

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class MinuteData:
    """One-minute-grid data and audit details from a source spreadsheet."""

    timestamps: pd.DatetimeIndex
    time_hours: FloatArray
    minute_data_days: FloatArray
    columns: pd.DataFrame
    source_rows_kept: NDArray[np.int64]
    inserted_minutes: NDArray[np.bool_]
    duplicate_rows: NDArray[np.int64]

    @property
    def start_day(self) -> pd.Timestamp:
        return self.timestamps[0]

    @property
    def start_midnight(self) -> pd.Timestamp:
        return self.timestamps[0].normalize()


def _regularize_minute_frame(
    timestamps: pd.Series | pd.DatetimeIndex,
    data: pd.DataFrame,
) -> tuple[pd.DatetimeIndex, pd.DataFrame, NDArray[np.int64], NDArray[np.bool_], NDArray[np.int64]]:
    time = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="coerce"))
    if time.isna().any():
        raise ValueError("source data contains invalid timestamps")
    aligned = time.round("min")
    # Stable ordering and first-row duplicate retention match the main MATLAB trial.
    source_order = np.arange(len(aligned), dtype=np.int64)
    ordered = pd.DataFrame({"timestamp": aligned, "source_row": source_order})
    ordered = ordered.sort_values(["timestamp", "source_row"], kind="stable")
    keep = ~ordered["timestamp"].duplicated(keep="first")
    kept_rows = ordered.loc[keep, "source_row"].to_numpy(dtype=np.int64)
    duplicate_rows = np.setdiff1d(source_order, kept_rows, assume_unique=False)
    sorted_times = pd.DatetimeIndex(ordered.loc[keep, "timestamp"])
    kept_data = data.iloc[kept_rows].copy()
    kept_data.index = sorted_times
    full_index = pd.date_range(sorted_times[0], sorted_times[-1], freq="min")
    regular = kept_data.reindex(full_index)
    inserted = ~full_index.isin(sorted_times)
    return full_index, regular, kept_rows, inserted, duplicate_rows


def load_minute_xlsx(path: str | Path) -> MinuteData:
    """Load the workflow spreadsheet and enforce its strict one-minute grid."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"spreadsheet not found: {source}")
    frame = pd.read_excel(source)
    if frame.shape[1] < 3:
        raise ValueError("spreadsheet needs timestamp, activity, and signal columns")
    data = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    index, regular, kept, inserted, duplicates = _regularize_minute_frame(
        frame.iloc[:, 0], data
    )
    t = np.asarray((index - index[0].normalize()) / pd.Timedelta(hours=1), dtype=np.float64)
    minute_data = np.column_stack((t / 24.0, regular.to_numpy(dtype=np.float64)))
    return MinuteData(index, t, minute_data, regular, kept, inserted, duplicates)


def load_xlsx_data(
    folder: str | Path = "Data",
    subjects: tuple[str, ...] = ("3_3", "3_4", "4_3", "4_5"),
) -> dict[str, MinuteData]:
    """Python equivalent of the subject spreadsheet import loop."""
    root = Path(folder)
    return {
        subject: load_minute_xlsx(root / f"{subject}_minute-by-minute-data.xlsx")
        for subject in subjects
    }


def convert_minute_workbooks(
    folder: str | Path = "Data",
    output_dir: str | Path = "MatData",
    subjects: tuple[str, ...] = ("3_3", "3_4", "4_3", "4_5"),
) -> dict[str, MinuteData]:
    """Convert the subject workbooks to the MAT layout used by the MATLAB scripts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaded = load_xlsx_data(folder, subjects)
    for subject, data in loaded.items():
        matlab_datenum = data.start_day.to_pydatetime().toordinal() + 366
        matlab_datenum += (
            data.start_day.hour * 3600
            + data.start_day.minute * 60
            + data.start_day.second
            + data.start_day.microsecond / 1_000_000
        ) / 86400
        savemat(
            output / f"{subject}_Minute_Data.mat",
            {
                "MinuteData_array": data.minute_data_days,
                "start_day": matlab_datenum,
            },
            do_compression=True,
        )
    return loaded


def load_csv_series(
    folder: str | Path,
    time_column: int = 3,
    target_column: int = 5,
) -> tuple[pd.DatetimeIndex, FloatArray]:
    """Combine day-level CSVs, keeping the MATLAB one-based column defaults."""
    root = Path(folder)
    files = sorted(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files found in {root}")
    time_parts: list[pd.Series] = []
    value_parts: list[pd.Series] = []
    for path in files:
        frame = pd.read_csv(path)
        if max(time_column, target_column) > frame.shape[1] or min(time_column, target_column) < 1:
            raise ValueError(f"{path} does not contain the configured time/target columns")
        time_parts.append(pd.to_datetime(frame.iloc[:, time_column - 1], errors="coerce"))
        value_parts.append(pd.to_numeric(frame.iloc[:, target_column - 1], errors="coerce"))
    timestamps = pd.concat(time_parts, ignore_index=True)
    values = pd.concat(value_parts, ignore_index=True)
    if timestamps.isna().any():
        raise ValueError("CSV input contains invalid timestamps")
    data = pd.DataFrame({"target": values.to_numpy()})
    index, regular, _, _, _ = _regularize_minute_frame(timestamps, data)
    return index, regular["target"].to_numpy(dtype=np.float64)


def combine_csv_minute_data(
    folder: str | Path,
    output_path: str | Path = "Combined_Daily_Data.mat",
    time_column: int = 3,
    target_column: int = 5,
) -> tuple[pd.DatetimeIndex, FloatArray]:
    """Combine timestamped CSV files and write MATLAB-compatible numeric data."""
    timestamps, values = load_csv_series(folder, time_column, target_column)
    midnight = timestamps[0].normalize()
    time_days = np.asarray((timestamps - midnight) / pd.Timedelta(days=1), dtype=np.float64)
    matlab_datenum = timestamps[0].to_pydatetime().toordinal() + 366
    matlab_datenum += (
        timestamps[0].hour * 3600
        + timestamps[0].minute * 60
        + timestamps[0].second
        + timestamps[0].microsecond / 1_000_000
    ) / 86400
    savemat(
        output_path,
        {
            "FinalData_array": np.column_stack((time_days, values)),
            "start_day": matlab_datenum,
        },
        do_compression=True,
    )
    return timestamps, values


def load_real_data(subject: int | str, folder: str | Path = "Data") -> tuple[FloatArray, ...]:
    """Read the legacy Utils.loadRealData MAT layout."""
    path = Path(folder) / f"A{subject}.mat"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.asarray(loadmat(path)["Data"], dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 11:
        raise ValueError(f"{path} Data matrix must have at least 11 columns")
    return data[:, 0], data[:, 10], data[:, 5], data[:, 4]


def load_synth_data(subject: int | str, folder: str | Path = "Data") -> dict[str, Any]:
    path = Path(folder) / f"SynthData{subject}.mat"
    if not path.is_file():
        raise FileNotFoundError(path)
    contents = loadmat(path, squeeze_me=True)
    missing = {"t", "y", "phaseShift"} - contents.keys()
    if missing:
        raise ValueError(f"{path} is missing variables: {', '.join(sorted(missing))}")
    return {name: contents[name] for name in ("t", "y", "phaseShift")}


def load_optimal_gains(filename: str | Path) -> NDArray[Any]:
    path = Path(filename)
    contents = loadmat(path, squeeze_me=True)
    if "optimGain" not in contents:
        raise ValueError(f"{path} does not contain 'optimGain'")
    return np.asarray(contents["optimGain"])


def save_timestamp_audit(
    path: str | Path,
    original: pd.DatetimeIndex,
    aligned: pd.DatetimeIndex,
    kept_rows: NDArray[np.int64],
) -> None:
    kept = np.zeros(len(original), dtype=bool)
    kept[kept_rows] = True
    pd.DataFrame(
        {
            "SourceRow": np.arange(1, len(original) + 1),
            "OriginalTimestamp": original,
            "AlignedTimestamp": aligned,
            "AdjustmentSeconds": (aligned - original).total_seconds(),
            "KeptAfterAlignment": kept,
        }
    ).to_csv(path, index=False)
