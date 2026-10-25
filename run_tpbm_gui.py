"""Small Windows-friendly file picker and launcher for the tPBM pipeline."""

from __future__ import annotations

import os
import queue
import re
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from run_tpbm import PipelineConfig, run_pipeline


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"Elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}"


class TpbmLauncher:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("tPBM Analysis")
        self.root.geometry("760x390")
        self.root.minsize(680, 360)
        self.messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.started_at: float | None = None
        self.result_directory: Path | None = None

        self.input_path = tk.StringVar()
        self.output_root = tk.StringVar(
            value=str(Path(__file__).resolve().parent / "tPBM Results")
        )
        self.subject = tk.StringVar(value="4_5")
        self.signal = tk.StringVar(value="BPM")
        self.phase_estimator = tk.StringVar(value="KF")
        self.status = tk.StringVar(value="Choose an XLSX workbook or Garmin CSV to begin.")
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
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
        ttk.Label(
            frame,
            text=(
                "Choose a legacy XLSX workbook or a converted "
                "Minute-by-minute.csv. Subject and signal are detected when possible."
            ),
            wraplength=700,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 16))

        ttk.Label(frame, text="Input data").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.input_path).grid(
            row=2, column=1, sticky="ew", padx=8, pady=5
        )
        self.browse_file_button = ttk.Button(
            frame, text="Browse...", command=self._browse_input
        )
        self.browse_file_button.grid(row=2, column=2, sticky="ew", pady=5)

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
            values=("BPM", "Act"),
            state="readonly",
            width=14,
        )
        self.signal_box.grid(row=4, column=1, sticky="w", padx=8, pady=5)

        ttk.Label(frame, text="Phase estimator").grid(
            row=5, column=0, sticky="w", pady=5
        )
        self.estimator_box = ttk.Combobox(
            frame,
            textvariable=self.phase_estimator,
            values=("KF", "KS"),
            state="readonly",
            width=14,
        )
        self.estimator_box.grid(row=5, column=1, sticky="w", padx=8, pady=5)
        ttk.Label(
            frame,
            text="KF = forward filter; KS = RTS smoother.",
        ).grid(row=5, column=2, sticky="w", padx=8)

        ttk.Label(frame, text="Results folder").grid(
            row=6, column=0, sticky="w", pady=5
        )
        ttk.Entry(frame, textvariable=self.output_root).grid(
            row=6, column=1, sticky="ew", padx=8, pady=5
        )
        self.browse_output_button = ttk.Button(
            frame, text="Browse...", command=self._browse_output
        )
        self.browse_output_button.grid(row=6, column=2, sticky="ew", pady=5)

        buttons = ttk.Frame(frame)
        buttons.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(18, 8))
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
            row=8, column=0, columnspan=3, sticky="w", pady=(8, 2)
        )
        ttk.Label(frame, textvariable=self.elapsed).grid(
            row=9, column=0, columnspan=3, sticky="w"
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

    def _configure_selected_input(self, path: Path) -> None:
        self.input_path.set(str(path))
        if path.suffix.lower() == ".csv":
            self.subject.set(path.parent.name)
            self.signal_box.configure(values=("HeartRate", "ZeroCrossingCount"))
            self.signal.set("HeartRate")
            self.status.set("Garmin CSV selected. HeartRate will be analyzed by default.")
            return

        self.signal_box.configure(values=("BPM", "Act"))
        self.signal.set("BPM")
        subject_match = re.match(r"(.+)_minute-by-minute-data$", path.stem)
        self.subject.set(
            subject_match.group(1) if subject_match else path.parent.name
        )
        self.status.set("Workbook selected. BPM will be analyzed by default.")

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
        self.browse_output_button.configure(state=state)
        self.subject_box.configure(state="disabled" if running else "normal")
        self.signal_box.configure(state="disabled" if running else "readonly")
        self.estimator_box.configure(state="disabled" if running else "readonly")

    def _start_run(self) -> None:
        source = Path(self.input_path.get().strip()).expanduser()
        if not source.is_file() or source.suffix.lower() not in {".xlsx", ".csv"}:
            messagebox.showerror(
                "Select supported input data",
                "Browse to an existing .xlsx workbook or converted Garmin CSV.",
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
            option=2 if self.signal.get() in ("Act", "ZeroCrossingCount") else 3,
            signal_label=self.signal.get(),
            phase_estimator=self.phase_estimator.get(),
        )
        self._set_running(True)
        self.started_at = time.perf_counter()
        self.status.set("Running analysis. This window will remain responsive.")
        self.elapsed.set(_format_elapsed(0))
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

    def _poll_worker(self) -> None:
        if self.started_at is not None:
            self.elapsed.set(_format_elapsed(time.perf_counter() - self.started_at))

        try:
            kind, payload = self.messages.get_nowait()
        except queue.Empty:
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
