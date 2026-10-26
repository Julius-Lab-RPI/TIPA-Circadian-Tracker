# tPBM Circadian Analysis (Python)

This package ports the MATLAB `Matlab Compact` workflows to Python. It
supports spreadsheet-based KS/RTS smoother or forward-KF phase estimation,
phase particle filtering, core-body-temperature (CBT) state estimation, DLMO
candidate detection and tracking, and plotting/exporting results.

## Package contents

- `run_tpbm.py`: complete XLSX/CSV-to-DLMO workflow; command-line entry point.
- `run_tpbm_gui.py`: Windows file-picker interface for selecting input,
  subject, signal, phase estimator, and output folder.
- `convert_garmin_data.py`: convert Labfront Garmin heart-rate and actigraphy
  exports into one-minute subject CSV files.
- `plot_tpbm_results.py`: regenerate the three-panel SVG visualization from a
  saved run folder.
- `particle_filter.py`: particle filter; can also be run independently from a
  CSV of phase observations.
- `kalman_smoothing.py`: Kalman state-space, RTS smoothing, KS parameter
  optimization, and related utilities.
- `kalman_filter.py`: forward KF and steady-state KF implementations and
  optimizers.
- `circadian_models.py`: oscillator models, activity-to-light mapping, phase
  calculations, and CBT helpers.
- `dlmo.py`: CBT trough candidates, event sequence tracking, daily DLMO report,
  and optional target-day assignment.
- `data_io.py`: XLSX, CSV, and MAT-file loading/conversion helpers.
- `matlab_utils.py`: Python counterparts for MATLAB `Utils` calculations and
  plotting. The pipeline writes its three-panel SVG directly; the legacy
  MATLAB-style plotting helpers use Matplotlib.

## Requirements and setup

Use Python 3.10 or newer. Dependencies are listed in `requirements.txt`.
From a terminal in the project directory:

```cmd
"E:\miniforge3\envs\tpbm\python.exe" -m pip install -r requirements.txt
```

Use your own Python executable path if it differs. Conda activation is
optional: calling the environment's `python.exe` directly avoids shell
activation-policy issues.

## Build and distribute the Windows app

Build on a Windows computer with Python installed. From PowerShell in the
project folder, install the build dependencies and create the app package:

```powershell
& "E:\miniforge3\Scripts\conda.exe" create -y -n tpbm-build python=3.12 pip
& "E:\miniforge3\envs\tpbm-build\python.exe" -m pip install -r requirements-build.txt
& ".\build_windows.cmd" "E:\miniforge3\envs\tpbm-build\python.exe"
```

The executable and shareable ZIP are created in `dist`. Send
`TpbmAnalysis-Windows.zip` to Windows users; extract the ZIP and run
`TpbmAnalysis.exe` from the extracted `TpbmAnalysis` folder. Keep the other
bundled files beside the executable. Analysis results default to a
`tPBM Results` folder under the user's Documents folder (or their home folder
if Documents is unavailable). Build from a clean Python/Conda environment to
avoid bundling unrelated installed packages. The build is an unsigned Windows application,
not an installer; Windows SmartScreen may show a warning when it is first run.

## Use case 1: Run the complete analysis with the file-picker UI

From VS Code's `cmd.exe` terminal:

```cmd
"E:\miniforge3\envs\tpbm\python.exe" "E:\tPBM Python\run_tpbm_gui.py"
```

In the window:

1. Click **Browse...** to choose an `.xlsx` workbook or converted Garmin CSV,
   or click **Data folder...** to choose a Garmin participant or dataset folder.
2. Confirm or edit the subject name (it is inferred from the selected file).
3. Confirm the signal; the GUI defaults to `BPM` for workbooks and
   `HeartRate` for Garmin CSV files. Garmin `ZeroCrossingCount` can also be
   selected as the phase-estimation signal.
4. Select the phase estimator (`KF` or `KS`).
5. Optionally choose a results folder, then click **Run analysis**.
6. When it finishes, choose **Open latest results folder** to inspect outputs.

The GUI defaults to the forward KF estimator and saves into `tPBM Results`
beside the Python scripts. It remains responsive while analysis runs. Errors
are shown in a dialog.

When a data folder is selected, the GUI converts and analyzes each participant
folder in sequence. You can select either one participant folder or the parent
dataset folder containing participant subfolders. Each participant should
contain `garmin-connect-daily-heart-rate` and
`garmin-device-actigraphy` folders. The converter creates
`Minute-by-minute.csv` inside each participant folder; the GUI then runs the
selected phase estimator and particle-filter/DLMO analysis for that participant.
Progress is shown in the status line, and completion reports any skipped or
failed participant folders.

## Convert Garmin experiment exports

Each subject folder should contain `garmin-connect-daily-heart-rate` and
`garmin-device-actigraphy` subfolders with the Labfront CSV exports. Run:

```cmd
"E:\miniforge3\envs\tpbm\python.exe" "E:\tPBM Python\convert_garmin_data.py" "C:\path\to\experiment-data"
```

The converter writes `Minute-by-minute.csv` into each subject folder with
`time`, `ZeroCrossingCount`, and `HeartRate` columns. It selects heart-rate
readings recorded exactly at the minute boundary, rounds actigraphy timestamps
to the nearest minute, and leaves unmatched readings blank. The command reports
missing minute values and dates where only one data type has a source file.
After conversion, the analysis GUI can open that subject's
`Minute-by-minute.csv` directly; no temporary Excel conversion is needed.

## Use case 2: Run the complete analysis from the command line

Example for subject 4_5, BPM signal, and the forward KF:

```cmd
"E:\miniforge3\envs\tpbm\python.exe" "E:\tPBM Python\run_tpbm.py" "C:\path\to\4_5_minute-by-minute-data.xlsx" --subject 4_5 --signal BPM --phase-estimator KF
```

The same command using the KS/RTS smoother:

```cmd
"E:\miniforge3\envs\tpbm\python.exe" "E:\tPBM Python\run_tpbm.py" "C:\path\to\4_5_minute-by-minute-data.xlsx" --subject 4_5 --signal BPM --phase-estimator KS
```

Replace the input path with a workbook or converted Garmin CSV on your
computer. Paths containing spaces must be enclosed in quotes. The required
`--subject` value is used for run identification. If omitted, `--signal`
defaults to BPM for XLSX and HeartRate for Garmin CSV. Use `--signal Act` or
`--signal BPM` for legacy workbooks, and `--signal HeartRate` or
`--signal ZeroCrossingCount` for Garmin CSV files. Activity counts are mapped
to light input for the particle filter in either format.

See all command-line options:

```cmd
"E:\miniforge3\envs\tpbm\python.exe" "E:\tPBM Python\run_tpbm.py" --help
```

Useful options:

| Option | Purpose | Default |
|---|---|---|
| `--phase-estimator KS` or `KF` | Select KS/RTS or forward-KF phase estimation | `KS` |
| `--signal Act`, `BPM`, `ZeroCrossingCount`, or `HeartRate` | Select which input signal is analyzed | Auto: `BPM` for XLSX; `HeartRate` for CSV |
| `--output-root "folder"` | Choose where timestamped run folders are created | `tPBM Results` |
| `--ks-iterations`, `--ks-population`, `--ks-offspring` | Set KS optimizer size | `150`, `100`, `50` |
| `--kf-iterations`, `--kf-population`, `--kf-offspring` | Set KF optimizer size | `150`, `100`, `50` |
| `--ks-seed`, `--kf-seed` | Set the phase-estimator random seed | `12345` |
| `--particles` | Set particle count | `500` |
| `--pf-seed` | Set particle-filter random seed | `1` |
| `--start-after-hours` | Start PF processing this many hours after the first midnight | `72` |
| `--store-history` | Save particle-level history (can require substantial disk space) | Off |

For a quick smoke test, reduce the optimizer and particle counts:

```cmd
"E:\miniforge3\envs\tpbm\python.exe" "E:\tPBM Python\run_tpbm.py" "C:\path\to\4_5_minute-by-minute-data.xlsx" --subject 4_5 --signal BPM --phase-estimator KF --kf-iterations 1 --kf-population 10 --kf-offspring 5 --particles 20
```

Reduced settings verify the workflow but are not appropriate for comparing
production results or estimating the default optimizer's runtime. The hidden
legacy `--prefilter KF` alias is still accepted, but new commands should use
`--phase-estimator KF`.

### Input data formats

Legacy XLSX workbooks need at least three columns, in this order:

1. Timestamp
2. Activity (also used to estimate light input)
3. BPM

Column headers are not used to choose these fields; their order matters.
Converted Garmin CSV files need the columns `time`, `ZeroCrossingCount`, and
`HeartRate`. The loader rounds timestamps to the nearest minute, retains the
first row for duplicate minute labels, and inserts missing minutes on a
regular one-minute grid. Non-numeric signal cells become missing values.
Input files are read-only from the pipeline's perspective; they are not
modified.

### Full-pipeline outputs

Each run creates a new timestamped directory under the output root:

- `Input/`: aligned input data and timestamp audit.
- `KS/` or `KF/`: selected estimator result MAT file and optimization costs.
- `PF/`: diagnostics CSV, particle-filter summary NPZ, and optional full
  particle history.
- `CBT/`: `Candidates.csv`, tracked `Events.csv`, `Daily.csv`, and
  `AllDays.csv`.
- `Results_Visualization.svg`: phase, CBT state, and DLMO plot.
- `Summary.json`: selected settings, elapsed stage timings, DLMO counts, and
  output location.

`AllDays.csv` contains the report for every DLMO day in the available PF
recording span; it is not limited to days 7, 14, and 21. Status values identify
warmup, incomplete, ambiguous, missing, or tracked-event days. The detector
also writes `Candidates.csv` and `Events.csv`, retaining candidate times,
confidence and tracking status.

The SVG has three panels:

1. PF output phase `theta_hat + delta_hat`, wrapped to `[0, 2*pi)` without
   subtracting the circadian trend.
2. The first CBT oscillator state.
3. Candidate and tracked DLMO clock hours. Candidate troughs, tracked events,
   and ambiguous events use different markers.

## Use case 3: Regenerate a plot from an existing run

No re-analysis is needed if the run folder has its saved PF and CBT files:

```cmd
"E:\miniforge3\envs\tpbm\python.exe" "E:\tPBM Python\plot_tpbm_results.py" "E:\tPBM Python\tPBM Results\<run-folder>"
```

To choose another output filename or location:

```cmd
"E:\miniforge3\envs\tpbm\python.exe" "E:\tPBM Python\plot_tpbm_results.py" "E:\tPBM Python\tPBM Results\<run-folder>" --output "C:\path\to\my_plot.svg"
```

## Use case 4: Run the particle filter independently

Use this when phase estimates are already available and you do not want to run
the workbook or Kalman stages. Create a CSV with these required columns:

```csv
time_hours,light,theta_rad
72.0,200.0,1.25
72.0166666667,200.0,1.26
72.0333333333,0.0,
```

`time_hours` must be finite and strictly increasing; `light` is nonnegative;
`theta_rad` is in radians and may be blank for missing phase observations.
Run:

```cmd
"E:\miniforge3\envs\tpbm\python.exe" "E:\tPBM Python\particle_filter.py" "C:\path\to\phase_input.csv" "C:\path\to\pf_output" --particles 500 --seed 1
```

Particle-filter options:

- `--particles`: number of particles (default 500).
- `--seed`: random seed (default 1).
- `--start-after-hours`: PF start time relative to the first midnight
  (default 72).
- `--smoothing-samples`: phase observation smoothing window in samples
  (default 1440).
- `--r`: phase measurement standard deviation in radians (default 0.15).
- `--q X XC N DELTA TAU`: five process-noise standard deviations (defaults
  `0.025 0.025 0.005 0.05 0.05`).
- `--store-history`: save full particle histories; output can be large.

This writes `estimates.csv`; with `--store-history`, it also writes
`particle_filter_history.npz`.

## Use case 5: Use the Python functions in another script

The modules can be imported directly. For example, run the full workflow and
access the summary and DLMO event table:

```python
from pathlib import Path

from run_tpbm import PipelineConfig, run_pipeline

result = run_pipeline(
    PipelineConfig(
        subject="4_5",
        input_xlsx=Path(r"C:\path\to\4_5_minute-by-minute-data.xlsx"),
        phase_estimator="KF",
    )
)

print(result["summary"]["output_dir"])
print(result["dlmo"]["events"])
print(result["targets"])  # all available days in this pipeline
```

For individual components:

- `data_io.load_minute_xlsx(path)`: load and regularize one XLSX.
- `data_io.load_xlsx_data(folder, subjects=...)`: load a set of standard-named
  workbooks.
- `data_io.convert_minute_workbooks(folder, output_dir, subjects=...)`: create
  MATLAB-compatible MAT files from those workbooks.
- `data_io.load_csv_series(folder, time_column=3, target_column=5)`: combine
  timestamped CSV series on a one-minute grid. Column numbers are one-based,
  matching the MATLAB helper.
- `data_io.combine_csv_minute_data(...)`: combine CSV series and write a MAT
  file.
- `kalman_smoothing.run_kalman_smoothing(time, observations, ...)`: run the
  KS/RTS stage directly on numeric arrays.
- `kalman_filter.simulate_kalman_filter(...)` and
  `kalman_filter.optimize_kf_parameters(...)`: run or optimize forward KF
  filtering outside the complete pipeline. `KFSS` and
  `optimize_steady_state_filter(...)` provide steady-state filtering.
- `particle_filter.run_particle_filter(time_hours, light, theta_rad, config)`: run
  the PF directly on arrays.
- `dlmo.compare_dlmo_extract(time_hours, cbt_state, start_midnight, options)`: find
  candidates and track events.
- `dlmo.compare_dlmo_targets(..., study_days, ...)`: assign events to an
  explicit day list; pass `None` for every available day.
- `circadian_models`: calculate wrapped/continuous phase, estimate phase
  shifts, map activity to light, run the Jewett-99 oscillator, or generate
  Huang initial conditions.
- `data_io.load_real_data(subject)`, `load_synth_data(subject)`, and
  `load_optimal_gains(filename)`: read legacy MATLAB `.mat` inputs and
  optimization results.

For details on function arguments and return values, inspect the corresponding
Python docstrings or use `help(function)` in Python.

## Runtime and troubleshooting

- `Summary.json` records `phase_estimation_seconds` and `pf_seconds`.
- To measure total time from `cmd.exe`, invoke PowerShell explicitly:

  ```cmd
  powershell -NoProfile -Command "Measure-Command { & 'E:\miniforge3\envs\tpbm\python.exe' 'E:\tPBM Python\run_tpbm.py' 'C:\path\to\input.xlsx' --subject 4_5 --signal BPM --phase-estimator KF }"
  ```

  `Measure-Command` by itself is a PowerShell command and will not work at a
  `cmd.exe` prompt.
- If the terminal says `& was unexpected`, you are in `cmd.exe`; omit the
  PowerShell call operator and run the quoted executable and script path as
  shown in the examples.
- If a file path contains spaces, quote the entire path.
- For malformed workbooks, verify the first three columns and timestamp
  formatting. Check `Input/TimestampAudit.csv` and `Input/Input.csv` in a run
  folder to inspect alignment and inserted minutes.
- The MATLAB and Python random number generators differ, so runs with the same
  seed are reproducible in Python but not bit-for-bit identical to MATLAB.
- The particle-filter oscillator uses the MATLAB RK4 transition. The separate
  MATLAB `ode23` helper is represented with RK4 and interpolated light, so its
  results may differ slightly.

## Tests

Run all unit tests from the project folder:

```cmd
"E:\miniforge3\envs\tpbm\python.exe" -m unittest discover -v
```
