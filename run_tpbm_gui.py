"""Small Windows-friendly file picker and launcher for the tPBM pipeline."""

from __future__ import annotations

import os
import queue
import re
import sys
import threading
import time
from collections.abc import Callable
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from convert_garmin_data import convert_subject
from run_tpbm import PipelineConfig, run_pipeline

_SIGNAL_CHOICES = ("Heart beat per minute", "Actigraphy")


def _pipeline_signal(display_signal: str, source_suffix: str) -> str:
    if display_signal not in _SIGNAL_CHOICES:
        raise ValueError(f"unsupported signal selection: {display_signal}")
    is_csv = source_suffix.lower() == ".csv"
    if display_signal == "Heart beat per minute":
        return "HeartRate" if is_csv else "BPM"
    return "ZeroCrossingCount" if is_csv else "Act"


def _phase_estimator_code(display_choice: str) -> str:
    if display_choice == "KF":
        return "KF"
    if display_choice == "KS (too slow)":
        return "KS"
    raise ValueError(f"unsupported phase estimator selection: {display_choice}")


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"Elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}"


def _default_output_root() -> Path:
    if getattr(sys, "frozen", False):
        documents = Path.home() / "Documents"
        return (documents if documents.is_dir() else Path.home()) / "tPBM Results"
    return Path(__file__).resolve().parent / "tPBM Results"


def _subject_directories(dataset_root: Path) -> list[Path]:
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"data folder does not exist: {dataset_root}")
    if any(
        (dataset_root / name).is_dir()
        for name in (
            "garmin-connect-daily-heart-rate",
            "garmin-device-actigraphy",
        )
    ):
        return [dataset_root]
    return sorted(path for path in dataset_root.iterdir() if path.is_dir())


def _run_dataset(
    dataset_root: Path,
    output_root: Path,
    subject_label: str,
    signal: str,
    phase_estimator: str,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    subjects = _subject_directories(dataset_root)
    if not subjects:
        raise ValueError(
            "The selected folder contains no subject folders. Select a dataset "
            "folder containing participant folders or one participant folder."
        )

    completed: list[dict[str, Any]] = []
    skipped: list[str] = []
    failed: list[str] = []
    for index, subject_directory in enumerate(subjects, start=1):
        subject = (
            subject_label
            if len(subjects) == 1
            else f"{subject_label}_{subject_directory.name}"
        )
        progress(f"Converting and analyzing {subject} ({index}/{len(subjects)})...")
        try:
            report = convert_subject(subject_directory)
            if report.output_path is None:
                skipped.append(subject)
                continue
            signal_label = _pipeline_signal(signal, ".csv")
            option = 2 if signal == "Actigraphy" else 3
            result = run_pipeline(
                PipelineConfig(
                    subject=subject,
                    input_xlsx=report.output_path,
                    output_root=output_root,
                    option=option,
                    signal_label=signal_label,
                    phase_estimator=phase_estimator,
                )
            )
            completed.append(result["summary"])
        except Exception as error:
            failed.append(f"{subject}: {type(error).__name__}: {error}")

    if not completed and failed:
        raise RuntimeError("No subject completed analysis.\n" + "\n".join(failed))
    if not completed:
        raise ValueError("No usable Garmin data was found in the selected folder.")
    return {
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
    }


class TpbmLauncher:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("tPBM Analysis")
        self.root.geometry("900x470")
        self.root.minsize(800, 440)
        self.messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.started_at: float | None = None
        self.result_directory: Path | None = None

        self.input_path = tk.StringVar()
        self.output_root = tk.StringVar(value=str(_default_output_root()))
        self.subject = tk.StringVar(value="4_5")
        self.signal = tk.StringVar(value="Heart beat per minute")
        self.phase_estimator = tk.StringVar(value="KF")
        self.status = tk.StringVar(
            value="Choose a workbook, Garmin CSV, or Garmin data folder to begin."
        )
        self.elapsed = tk.StringVar(value="")

        self._build_form()
        self.root.after(150, self._poll_worker)

    def _build_form(self) -> None:
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(
            frame,
            text="Run the tPBM circadian analysis",
            font=("Segoe UI", 15, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))
        ttk.Label(
            frame,
            text=(
                "Choose an XLSX/CSV file, a participant folder, or a dataset "
                "folder containing participant folders with Garmin exports."
            ),
            wraplength=700,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 16))

        ttk.Label(frame, text="Input data").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.input_path).grid(
            row=2, column=1, sticky="ew", padx=8, pady=5
        )
        self.browse_file_button = ttk.Button(
            frame, text="Browse...", command=self._browse_input
        )
        self.browse_file_button.grid(row=2, column=2, sticky="ew", pady=5)
        self.browse_folder_button = ttk.Button(
            frame, text="Data folder...", command=self._browse_dataset
        )
        self.browse_folder_button.grid(row=2, column=3, sticky="ew", padx=(8, 0), pady=5)

        ttk.Label(frame, text="Subject").grid(row=3, column=0, sticky="w", pady=5)
        self.subject_box = ttk.Entry(
            frame,
            textvariable=self.subject,
            width=14,
        )
        self.subject_box.grid(row=3, column=1, sticky="w", padx=8, pady=5)

        ttk.Label(frame, text="Signal to analyze").grid(
            row=4, column=0, sticky="w", pady=5
        )
        self.signal_box = ttk.Combobox(
            frame,
            textvariable=self.signal,
            values=_SIGNAL_CHOICES,
            state="readonly",
            width=24,
        )
        self.signal_box.grid(row=4, column=1, sticky="w", padx=8, pady=5)

        ttk.Label(frame, text="Phase estimator").grid(
            row=5, column=0, sticky="w", pady=5
        )
        self.estimator_box = ttk.Combobox(
            frame,
            textvariable=self.phase_estimator,
            values=("KF", "KS (too slow)"),
            state="readonly",
            width=14,
        )
        self.estimator_box.grid(row=5, column=1, sticky="w", padx=8, pady=5)
        ttk.Label(
            frame,
            text="KF = forward filter; KS = RTS smoother (too slow).",
        ).grid(row=5, column=2, columnspan=2, sticky="w", padx=8)

        ttk.Label(frame, text="Results folder").grid(
            row=6, column=0, sticky="w", pady=5
        )
        ttk.Entry(frame, textvariable=self.output_root).grid(
            row=6, column=1, sticky="ew", padx=8, pady=5
        )
        self.browse_output_button = ttk.Button(
            frame, text="Browse...", command=self._browse_output
        )
        self.browse_output_button.grid(
            row=6, column=2, columnspan=2, sticky="ew", padx=(8, 0), pady=5
        )

        buttons = ttk.Frame(frame)
        buttons.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(18, 8))
        buttons.columnconfigure(0, weight=1)
        self.run_button = ttk.Button(
            buttons, text="Run analysis", command=self._start_run
        )
        self.run_button.grid(row=0, column=0, sticky="w")
        self.open_results_button = ttk.Button(
            buttons,
            text="Open latest results folder",
            command=self._open_results,
            state="disabled",
        )
        self.open_results_button.grid(row=0, column=1, sticky="e")

        ttk.Label(frame, textvariable=self.status).grid(
            row=8, column=0, columnspan=4, sticky="w", pady=(8, 2)
        )
        ttk.Label(frame, textvariable=self.elapsed).grid(
            row=9, column=0, columnspan=4, sticky="w"
        )

    def _browse_input(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select the tPBM spreadsheet",
            filetypes=(
                ("Supported data", "*.xlsx *.csv"),
                ("Excel workbooks", "*.xlsx"),
                ("Converted Garmin data", "*.csv"),
                ("All files", "*.*"),
            ),
        )
        if selected:
            self._configure_selected_input(Path(selected))

    def _browse_dataset(self) -> None:
        current_path = self.input_path.get().strip()
        current = Path(current_path).expanduser() if current_path else Path.home()
        initial_directory = current if current.is_dir() else current.parent
        selected = filedialog.askdirectory(
            title="Select a Garmin dataset or participant folder",
            initialdir=str(initial_directory),
        )
        if selected:
            selected_path = Path(selected)
            self.input_path.set(selected)
            self.signal_box.configure(values=_SIGNAL_CHOICES)
            self.signal.set("Heart beat per minute")
            self.subject.set(selected_path.name)
            self.status.set(
                "Folder selected. The app will create Minute-by-minute.csv "
                "and analyze each participant with usable data. "
                "For a multi-participant folder, the Subject value will prefix "
                "each participant folder name. Choose Heart beat per minute "
                "or Actigraphy."
            )

    def _configure_selected_input(self, path: Path) -> None:
        self.input_path.set(str(path))
        self.signal_box.configure(values=_SIGNAL_CHOICES)
        if path.suffix.lower() == ".csv":
            self.subject.set(path.parent.name)
            self.signal.set("Heart beat per minute")
            self.status.set(
                "Garmin CSV selected. Heart beat per minute will be analyzed by default."
            )
            return

        self.signal.set("Heart beat per minute")
        subject_match = re.match(r"(.+)_minute-by-minute-data$", path.stem)
        self.subject.set(
            subject_match.group(1) if subject_match else path.parent.name
        )
        self.status.set(
            "Workbook selected. Heart beat per minute will be analyzed by default."
        )

    def _browse_output(self) -> None:
        selected = filedialog.askdirectory(
            title="Select a folder for analysis results",
            initialdir=self.output_root.get() or str(Path.cwd()),
            mustexist=False,
        )
        if selected:
            self.output_root.set(selected)

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.run_button.configure(state=state)
        self.browse_file_button.configure(state=state)
        self.browse_folder_button.configure(state=state)
        self.browse_output_button.configure(state=state)
        self.subject_box.configure(state="disabled" if running else "normal")
        self.signal_box.configure(state="disabled" if running else "readonly")
        self.estimator_box.configure(state="disabled" if running else "readonly")

    def _start_run(self) -> None:
        source = Path(self.input_path.get().strip()).expanduser()
        is_dataset = source.is_dir()
        if not is_dataset and (
            not source.is_file() or source.suffix.lower() not in {".xlsx", ".csv"}
        ):
            messagebox.showerror(
                "Select supported input data",
                "Choose an existing .xlsx/CSV file or a Garmin data folder.",
                parent=self.root,
            )
            return
        subject = self.subject.get().strip()
        if not subject or re.search(r'[<>:"/\\|?*\x00-\x1f]', subject):
            messagebox.showerror(
                "Check subject name",
                "Enter a subject name without filename-invalid characters.",
                parent=self.root,
            )
            return
        output = Path(self.output_root.get().strip()).expanduser()
        try:
            output.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            messagebox.showerror(
                "Cannot use results folder",
                str(error),
                parent=self.root,
            )
            return

        config = PipelineConfig(
            subject=subject,
            input_xlsx=source,
            output_root=output,
            option=2 if self.signal.get() == "Actigraphy" else 3,
            signal_label=_pipeline_signal(self.signal.get(), source.suffix),
            phase_estimator=_phase_estimator_code(self.phase_estimator.get()),
        )
        self._set_running(True)
        self.started_at = time.perf_counter()
        self.status.set("Running analysis. This window will remain responsive.")
        self.elapsed.set(_format_elapsed(0))
        if is_dataset:
            worker = threading.Thread(
                target=self._run_dataset_worker,
                args=(
                    source,
                    output,
                    subject,
                    self.signal.get(),
                    _phase_estimator_code(self.phase_estimator.get()),
                ),
                daemon=True,
            )
        else:
            worker = threading.Thread(
                target=self._run_pipeline_worker,
                args=(config,),
                daemon=True,
            )
        worker.start()

    def _run_pipeline_worker(self, config: PipelineConfig) -> None:
        try:
            result = run_pipeline(config)
            self.messages.put(("success", result["summary"]))
        except Exception as error:
            self.messages.put(("error", error))

    def _run_dataset_worker(
        self,
        dataset_root: Path,
        output_root: Path,
        subject_label: str,
        signal: str,
        phase_estimator: str,
    ) -> None:
        try:
            result = _run_dataset(
                dataset_root,
                output_root,
                subject_label,
                signal,
                phase_estimator,
                lambda message: self.messages.put(("progress", message)),
            )
            self.messages.put(("dataset_success", result))
        except Exception as error:
            self.messages.put(("error", error))

    def _poll_worker(self) -> None:
        if self.started_at is not None:
            self.elapsed.set(_format_elapsed(time.perf_counter() - self.started_at))

        try:
            kind, payload = self.messages.get_nowait()
        except queue.Empty:
            self.root.after(150, self._poll_worker)
            return

        if kind == "progress":
            self.status.set(str(payload))
            self.root.after(150, self._poll_worker)
            return

        self._set_running(False)
        if self.started_at is not None:
            self.elapsed.set(_format_elapsed(time.perf_counter() - self.started_at))
        self.started_at = None
        if kind == "success":
            summary = payload
            self.result_directory = Path(summary["output_dir"])
            self.status.set(f"Finished. Results: {self.result_directory}")
            self.open_results_button.configure(state="normal")
            messagebox.showinfo(
                "Analysis complete",
                f"Results were saved to:\n{self.result_directory}",
                parent=self.root,
            )
        elif kind == "dataset_success":
            completed = payload["completed"]
            self.result_directory = Path(completed[-1]["output_dir"])
            self.open_results_button.configure(state="normal")
            details = [
                f"Completed analyses: {len(completed)}",
                f"Skipped empty subjects: {len(payload['skipped'])}",
                f"Failed subjects: {len(payload['failed'])}",
                f"Results folder: {self.result_directory.parent}",
            ]
            if payload["failed"]:
                details.extend(["", *payload["failed"]])
            self.status.set(
                f"Folder processing finished: {len(completed)} analyzed, "
                f"{len(payload['skipped'])} skipped, {len(payload['failed'])} failed."
            )
            if payload["failed"]:
                messagebox.showwarning(
                    "Folder processing finished with errors",
                    "\n".join(details),
                    parent=self.root,
                )
            else:
                messagebox.showinfo(
                    "Folder processing complete",
                    "\n".join(details),
                    parent=self.root,
                )
        else:
            self.status.set("Analysis failed. See the error message for details.")
            messagebox.showerror(
                "Analysis failed",
                f"{type(payload).__name__}: {payload}",
                parent=self.root,
            )
        self.root.after(150, self._poll_worker)

    def _open_results(self) -> None:
        if self.result_directory is not None and self.result_directory.is_dir():
            os.startfile(str(self.result_directory))
        else:
            messagebox.showerror(
                "Results folder unavailable",
                "The most recent results folder no longer exists.",
                parent=self.root,
            )


def main() -> None:
    root = tk.Tk()
    TpbmLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
