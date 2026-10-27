"""Regenerate the three-panel PNG visualization from a saved tPBM run folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from run_tpbm import _save_results_visualization, _visualization_path


def plot_saved_run(run_directory: Path, output_path: Path | None = None) -> Path:
    """Load saved PF and DLMO artifacts and write the visualization PNG."""
    run_directory = run_directory.resolve()
    summary_path = run_directory / "Summary.json"
    diagnostics_path = run_directory / "PF" / "Diagnostics.csv"
    summary_path_npz = run_directory / "PF" / "PF_Summary.npz"
    candidates_path = run_directory / "CBT" / "Candidates.csv"
    events_path = run_directory / "CBT" / "Events.csv"
    required = (
        summary_path,
        diagnostics_path,
        summary_path_npz,
        candidates_path,
        events_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "run folder is missing required results files: " + ", ".join(missing)
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    diagnostics = pd.read_csv(diagnostics_path)
    with np.load(summary_path_npz) as particle_filter:
        time_hours = np.asarray(particle_filter["time_hours"], dtype=np.float64)
        state_mean = np.asarray(particle_filter["state_mean"], dtype=np.float64)
        if "mean_state_theta_rad" in particle_filter:
            phase = np.asarray(
                particle_filter["mean_state_theta_rad"], dtype=np.float64
            )
        else:
            phase_column = (
                "MeanStateThetaRad"
                if "MeanStateThetaRad" in diagnostics
                else "PFOutputPhaseRad"
            )
            phase = diagnostics[phase_column].to_numpy(dtype=np.float64)

    if len(diagnostics) != len(time_hours) or len(phase) != len(time_hours):
        raise ValueError("saved diagnostics and particle-filter arrays have different lengths")
    if state_mean.ndim != 2 or state_mean.shape[0] != len(time_hours):
        raise ValueError("saved PF state_mean has an unexpected shape")
    if not np.all(np.isfinite(phase)):
        raise ValueError("saved particle-filter output phase contains non-finite values")

    pf_result = SimpleNamespace(
        time_hours=time_hours,
        mean_state_theta_rad=np.mod(phase, 2 * np.pi),
        state_mean=state_mean,
    )
    detector_options = summary.get("config", {}).get("detector_options", {})
    dlmo = {
        "candidates": pd.read_csv(candidates_path),
        "events": pd.read_csv(events_path, parse_dates=["DLMODateTime"]),
        "options": {"offset_hours": float(detector_options.get("offset_hours", 7.0))},
    }
    start_midnight = pd.Timestamp(summary["start_midnight"])
    destination = output_path or _visualization_path(run_directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save_results_visualization(destination, pf_result, dlmo, start_midnight)
    return destination.resolve()


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_directory",
        type=Path,
        help="pipeline run folder containing Summary.json, PF, and CBT results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional PNG output path (defaults to Results_Visualization.png in the run folder)",
    )
    arguments = parser.parse_args()
    output = plot_saved_run(arguments.run_directory, arguments.output)
    print(f"Visualization saved to: {output}")


if __name__ == "__main__":
    _main()
