"""End-to-end Python equivalent of the MATLAB KS and KF spreadsheet trials."""

from __future__ import annotations

import argparse
from html import escape
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import savemat

from circadian_models import activity_to_light, estimate_available_phase_shift
from data_io import load_minute_xlsx, save_timestamp_audit
from dlmo import compare_dlmo_extract, compare_dlmo_targets
from kalman_smoothing import (
    compute_cost,
    compute_spectrum,
    create_kalman_state_space,
    optimize_parameters,
    simulate_kalman_smoother,
)
from kalman_filter import (
    optimize_kf_parameters,
    simulate_kalman_filter,
)
from particle_filter import (
    ParticleFilterConfig,
    ParticleFilterResult,
    run_particle_filter,
)

_SUBJECTS = ("3_3", "3_4", "4_3", "4_5")


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    subject: str
    input_xlsx: Path
    output_root: Path = Path("tPBM Results")
    option: int = 3
    ks_seed: int = 12345
    ks_iterations: int = 150
    ks_population: int = 100
    ks_offspring: int = 50
    ks_parents: int = 2
    ks_deviation_factor: float = float("inf")
    ks_zeros_are_missing: bool = True
    phase_estimator: str = "KS"
    kf_seed: int = 12345
    kf_iterations: int = 150
    kf_population: int = 100
    kf_offspring: int = 50
    kf_parents: int = 2
    kf_deviation_factor: float = float("inf")
    kf_zeros_are_missing: bool = True
    pf: ParticleFilterConfig = field(default_factory=ParticleFilterConfig)
    day_boundary_hour: float = 12.0
    detector_options: dict[str, float] = field(
        default_factory=lambda: {"relax_final_trough": 1.0}
    )

    @property
    def signal_name(self) -> str:
        return "Act" if self.option == 2 else "BPM"

def _fill_light(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_light = activity_to_light(activity)
    missing = ~np.isfinite(activity) | ~np.isfinite(raw_light)
    if np.all(missing):
        raise ValueError("no finite activity values available to estimate light")
    indices = np.arange(len(raw_light))
    light = raw_light.copy()
    valid = ~missing
    light[missing] = np.interp(indices[missing], indices[valid], raw_light[valid])
    if np.any(~np.isfinite(light)) or np.any(light < 0):
        raise ValueError("light imputation produced invalid values")
    return raw_light, light, missing


def _run_ks(
    time_hours: np.ndarray,
    signal: np.ndarray,
    config: PipelineConfig,
) -> dict[str, object]:
    rng = np.random.default_rng(config.ks_seed)
    raw = np.asarray(signal, dtype=np.float64).copy()
    zero_mask = (raw == 0) & config.ks_zeros_are_missing
    raw[zero_mask] = np.nan
    mean = float(np.nanmean(raw))
    std = float(np.nanstd(raw, ddof=1))
    signal_adjustment_mask = np.abs(raw - mean) > config.ks_deviation_factor * std
    raw[signal_adjustment_mask] = np.nan
    normalized_mean = float(np.nanmean(raw))
    normalized_std = float(np.nanstd(raw, ddof=1))
    if not np.isfinite(normalized_std) or normalized_std <= 0:
        raise ValueError("selected signal has no usable variation after missing-value handling")
    y = (raw - normalized_mean) / normalized_std
    if not np.allclose(np.diff(time_hours), 1.0 / 60.0, atol=1e-8, rtol=0.0):
        raise ValueError("Kalman smoother expects a one-minute regular time grid")

    order = 1
    a, b, c, d = create_kalman_state_space(order, 3, time_hours[1] - time_hours[0])
    costs, average_cost, q_pop, r_pop = optimize_parameters(
        config.ks_iterations,
        config.ks_population,
        config.ks_offspring,
        config.ks_parents,
        order,
        0,
        18,
        -5,
        1e2,
        1e8,
        a,
        c,
        time_hours,
        y,
        seed=int(rng.integers(0, np.iinfo(np.uint32).max)),
    )
    best = int(np.nanargmin(costs))
    q = q_pop[best].reshape((3, 3), order="F")
    q = np.einsum("ij,kj->ik", q, q)
    x_smooth, y_smooth, x_post = simulate_kalman_smoother(a, c, q, r_pop[best], y)
    y_reconstructed = y.copy()
    y_reconstructed[~np.isfinite(y_reconstructed)] = y_smooth[~np.isfinite(y_reconstructed)]
    original_spectrum, frequencies, _ = compute_spectrum(time_hours, y_reconstructed)
    filtered_spectrum, _, _ = compute_spectrum(time_hours, y_smooth)
    estimated_cost = compute_cost(original_spectrum, filtered_spectrum, frequencies, order)
    theta, phase_shift = estimate_available_phase_shift(
        2 * np.pi / 24, time_hours, x_smooth[0], x_smooth[1]
    )
    return {
        "t": time_hours,
        "y_raw": signal,
        "y": raw,
        "y_norm": y,
        "y_rec": y_reconstructed,
        "x_smooth": x_smooth,
        "y_smooth": y_smooth,
        "x_post": x_post,
        "theta": theta,
        "Q": q,
        "R": float(r_pop[best]),
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "Cost": costs,
        "avgCost": average_cost,
        "Q_pop": q_pop,
        "R_pop": r_pop,
        "minCost": float(np.nanmin(costs)),
        "best_index": best,
        "estimCost": estimated_cost,
        "rectifiedSpectrum": original_spectrum,
        "filteredSpectrum": filtered_spectrum,
        "frequencies": frequencies,
        "mu_y": normalized_mean,
        "sigma_y": normalized_std,
        "missing_mask": ~np.isfinite(raw),
        "zero_mask": zero_mask,
        "signal_adjustment_mask": signal_adjustment_mask,
        "estimPhaseShift": phase_shift,
    }


def _run_kf(
    time_hours: np.ndarray,
    signal: np.ndarray,
    config: PipelineConfig,
) -> dict[str, object]:
    raw = np.asarray(signal, dtype=np.float64).copy()
    zero_mask = (raw == 0) & config.kf_zeros_are_missing
    raw[zero_mask] = np.nan
    raw_mean = float(np.nanmean(raw))
    raw_std = float(np.nanstd(raw, ddof=1))
    signal_adjustment_mask = (
        np.abs(raw - raw_mean) > config.kf_deviation_factor * raw_std
    )
    raw[signal_adjustment_mask] = np.nan
    normalized_mean = float(np.nanmean(raw))
    normalized_std = float(np.nanstd(raw, ddof=1))
    if not np.isfinite(normalized_std) or normalized_std <= 0:
        raise ValueError("selected signal has no usable variation after missing-value handling")
    y = (raw - normalized_mean) / normalized_std
    if len(time_hours) < 2 or not np.allclose(
        np.diff(time_hours), 1.0 / 60.0, atol=1e-8, rtol=0.0
    ):
        raise ValueError("Kalman filter expects a one-minute regular time grid")

    order = 1
    a, b, c, d = create_kalman_state_space(
        order, 3, time_hours[1] - time_hours[0]
    )
    costs, average_cost, q_pop, r_pop = optimize_kf_parameters(
        config.kf_iterations,
        config.kf_population,
        config.kf_offspring,
        config.kf_parents,
        order,
        0,
        18,
        -5,
        1e2,
        1e8,
        a,
        c,
        time_hours,
        y,
        seed=config.kf_seed,
    )
    best = int(np.nanargmin(costs))
    q_factor = q_pop[best].reshape((3, 3), order="F")
    q = np.einsum("ij,kj->ik", q_factor, q_factor)
    states, y_filtered, kf_history = simulate_kalman_filter(
        a, c, q, float(r_pop[best]), y, full_log=True
    )
    if kf_history is None:
        raise RuntimeError("full Kalman filter history was requested but not returned")
    y_reconstructed = y.copy()
    y_reconstructed[~np.isfinite(y_reconstructed)] = y_filtered[
        ~np.isfinite(y_reconstructed)
    ]
    original_spectrum, frequencies, _ = compute_spectrum(
        time_hours, y_reconstructed
    )
    filtered_spectrum, _, _ = compute_spectrum(time_hours, y_filtered)
    estimated_cost = compute_cost(
        original_spectrum, filtered_spectrum, frequencies, order
    )
    theta, phase_shift = estimate_available_phase_shift(
        2 * np.pi / 24, time_hours, states[0], states[1]
    )
    phi = np.mod(np.unwrap(theta) - (2 * np.pi / 24) * time_hours, 2 * np.pi)
    return {
        "t": time_hours,
        "y_raw": signal,
        "y": raw,
        "y_norm": y,
        "y_rec": y_reconstructed,
        "x_filtered": states,
        "y_filtered": y_filtered,
        "kf_history": kf_history,
        "theta": theta,
        "phi": phi,
        "Q": q,
        "R": float(r_pop[best]),
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "Cost": costs,
        "avgCost": average_cost,
        "Q_pop": q_pop,
        "R_pop": r_pop,
        "minCost": float(np.nanmin(costs)),
        "best_index": best,
        "estimCost": estimated_cost,
        "rectifiedSpectrum": original_spectrum,
        "filteredSpectrum": filtered_spectrum,
        "frequencies": frequencies,
        "mu_y": normalized_mean,
        "sigma_y": normalized_std,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "missing_mask": ~np.isfinite(raw),
        "zero_mask": zero_mask,
        "signal_adjustment_mask": signal_adjustment_mask,
        "estimPhaseShift": phase_shift,
        "seed": config.kf_seed,
    }


def _save_results_visualization(
    output_path: Path,
    pf_result: ParticleFilterResult,
    dlmo: dict[str, Any],
    start_midnight: pd.Timestamp,
) -> None:
    """Save un-detrended PF output phase, CBT state, and DLMO events as SVG."""
    time_hours = pf_result.time_hours
    phase = np.mod(pf_result.mean_state_theta_rad, 2 * np.pi)
    cbt_state = pf_result.state_mean[:, 0]
    candidates = dlmo["candidates"]
    events = dlmo["events"]
    offset_hours = float(dlmo["options"]["offset_hours"])
    width, height = 1600, 1230
    left, right = 115, 1540
    panel_tops = [105, 465, 825]
    panel_bottoms = [365, 725, 1085]
    x_min, x_max = float(time_hours[0]), float(time_hours[-1])
    if x_max <= x_min:
        raise ValueError("visualization requires an increasing time range")
    fragments = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<rect width='100%' height='100%' fill='white'/>",
        "<style>text{font-family:Arial,sans-serif;fill:#222} .grid{stroke:#ddd;stroke-width:1} .axis{stroke:#444;stroke-width:1.4} .line{fill:none;stroke-width:1.5}</style>",
        "<text x='800' y='42' text-anchor='middle' font-size='25' font-weight='bold'>tPBM Particle-Filter Results</text>",
    ]

    def add_text(x: float, y: float, text: str, *, size: int = 15, anchor: str = "middle", weight: str = "normal") -> None:
        fragments.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}">{escape(text)}</text>'
        )

    def x_pixel(hour: float) -> float:
        return left + (hour - x_min) / (x_max - x_min) * (right - left)

    date_tick_step = 24 * max(1, int(np.ceil((x_max - x_min) / 24 / 8)))
    first_tick = int(np.ceil(x_min / date_tick_step) * date_tick_step)
    x_ticks = list(range(first_tick, int(x_max) + 1, date_tick_step))
    if not x_ticks or x_ticks[0] > x_min + 1:
        x_ticks.insert(0, int(np.ceil(x_min)))
    if x_ticks[-1] < x_max - 1:
        x_ticks.append(int(x_max))

    def draw_panel(
        index: int,
        title: str,
        y_label: str,
        y_min: float,
        y_max: float,
        y_ticks: list[tuple[float, str]],
    ) -> Any:
        top, bottom = panel_tops[index], panel_bottoms[index]
        add_text(left, top - 24, title, size=19, anchor="start", weight="bold")
        for value, label in y_ticks:
            y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
            fragments.append(
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>'
            )
            add_text(left - 12, y + 5, label, size=13, anchor="end")
        for hour in x_ticks:
            x = x_pixel(hour)
            fragments.append(
                f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>'
            )
            tick_date = (start_midnight + pd.to_timedelta(hour, unit="h")).strftime("%b %d")
            add_text(x, bottom + 23, tick_date, size=12)
        fragments.extend(
            [
                f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
                f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
            ]
        )
        add_text(28, (top + bottom) / 2, y_label, size=14)
        return lambda value: bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    def draw_line(
        x_values: np.ndarray,
        y_values: np.ndarray,
        y_transform: Any,
        color: str,
        *,
        opacity: float = 1.0,
        series_name: str = "series",
    ) -> None:
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        starts = np.flatnonzero(np.diff(np.r_[False, finite, False].astype(np.int8)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, finite, False].astype(np.int8)) == -1)
        for start, end in zip(starts, ends, strict=True):
            points = " ".join(
                f"{x_pixel(float(x_values[i])):.2f},{y_transform(float(y_values[i])):.2f}"
                for i in range(start, end)
            )
            if end - start > 1:
                fragments.append(
                    f'<polyline class="line" data-series="{escape(series_name)}" points="{points}" stroke="{color}" opacity="{opacity:.2f}"/>'
                )

    y_phase = draw_panel(
        0,
        "1. Wrapped particle-filter output phase: y_hat = theta_hat + delta_hat",
        "Phase (rad)",
        0,
        2 * np.pi,
        [
            (0, "0"),
            (np.pi / 2, "pi/2"),
            (np.pi, "pi"),
            (3 * np.pi / 2, "3pi/2"),
            (2 * np.pi, "2pi"),
        ],
    )
    draw_line(
        time_hours,
        phase,
        y_phase,
        "#1769aa",
        series_name="wrapped_theta_hat_plus_delta_hat",
    )

    cbt_min = float(np.nanmin(cbt_state))
    cbt_max = float(np.nanmax(cbt_state))
    cbt_pad = max((cbt_max - cbt_min) * 0.06, 1e-6)
    y_cbt = draw_panel(
        1,
        "2. Core-body-temperature trajectory: first state x_hat[0]",
        "CBT state",
        cbt_min - cbt_pad,
        cbt_max + cbt_pad,
        [
            (value, f"{value:.3g}")
            for value in np.linspace(cbt_min, cbt_max, 5)
        ],
    )
    draw_line(
        time_hours, cbt_state, y_cbt, "#d97706", series_name="first_cbt_state"
    )

    y_dlmo = draw_panel(
        2,
        "3. DLMO clock hours: candidate troughs and tracked events",
        "Clock hour",
        0,
        24,
        [(value, f"{value:02d}:00") for value in range(0, 25, 4)],
    )
    if not candidates.empty:
        candidate_hours_since_midnight = (
            candidates["CBTHour"].to_numpy(dtype=np.float64) - offset_hours
        )
        for hour, dlmo_hour in zip(
            candidate_hours_since_midnight,
            np.mod(candidate_hours_since_midnight, 24.0),
            strict=True,
        ):
            x, y = x_pixel(float(hour)), y_dlmo(float(dlmo_hour))
            fragments.append(
                f'<path d="M{x-5:.1f},{y-5:.1f} L{x+5:.1f},{y+5:.1f} M{x-5:.1f},{y+5:.1f} L{x+5:.1f},{y-5:.1f}" stroke="#777" stroke-width="1.7"/>'
            )

    if not events.empty:
        event_hours_since_midnight = (
            pd.to_datetime(events["DLMODateTime"]) - start_midnight
        ).dt.total_seconds().to_numpy(dtype=np.float64) / 3600
        event_clock_hours = events["DLMOHour"].to_numpy(dtype=np.float64)
        ambiguous = events["Ambiguous"].to_numpy(dtype=bool)
        for hour, dlmo_hour, is_ambiguous, event_id, status in zip(
            event_hours_since_midnight,
            event_clock_hours,
            ambiguous,
            events["EventID"],
            events["Status"],
            strict=True,
        ):
            x, y = x_pixel(float(hour)), y_dlmo(float(dlmo_hour))
            if is_ambiguous:
                fragments.append(
                    f'<path d="M{x:.1f},{y-7:.1f} L{x+7:.1f},{y:.1f} L{x:.1f},{y+7:.1f} L{x-7:.1f},{y:.1f} Z" fill="#e08b00" stroke="#222"/>'
                )
            else:
                fragments.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#25834a" stroke="#222"/>'
                )
            add_text(
                x + 8,
                y - 8,
                f"E{int(event_id)} {str(status).replace('_', ' ')}",
                size=10,
                anchor="start",
            )
    add_text(left, 1150, "x = candidate DLMO; green circle = tracked event; orange diamond = ambiguous tracked event; labels show event status", size=13, anchor="start")
    add_text(800, 1200, "Date", size=15)
    fragments.append("</svg>")
    output_path.write_text("\n".join(fragments), encoding="utf-8")


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    """Run XLSX ingestion, Kalman phase estimation, particle filtering, and DLMO tracking."""
    if config.option not in (2, 3):
        raise ValueError("option must be 2 (activity) or 3 (BPM)")
    if config.phase_estimator not in ("KS", "KF"):
        raise ValueError("phase_estimator must be 'KS' or 'KF'")
    if config.pf.n_particles < 2:
        raise ValueError("particle count must be at least two")
    source = config.input_xlsx.resolve()
    data = load_minute_xlsx(source)
    frame = data.columns
    activity = frame.iloc[:, 0].to_numpy(dtype=np.float64)
    signal_index = config.option - 2
    if signal_index >= frame.shape[1]:
        raise ValueError("spreadsheet is missing the selected Act/BPM signal column")
    raw_signal = frame.iloc[:, signal_index].to_numpy(dtype=np.float64)
    raw_light, light, light_gap_mask = _fill_light(activity)

    output_dir = config.output_root / (
        f"A{config.subject}_{config.signal_name}_{config.phase_estimator}_phi_Q5_R015_seed{config.pf.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    estimator_name = config.phase_estimator
    for child in ("Input", estimator_name, "PF", "CBT"):
        (output_dir / child).mkdir()

    original_times = pd.DatetimeIndex(pd.to_datetime(pd.read_excel(source).iloc[:, 0]))
    aligned_times = original_times.round("min")
    save_timestamp_audit(
        output_dir / "Input" / "TimestampAudit.csv",
        original_times,
        aligned_times,
        data.source_rows_kept,
    )
    input_frame = pd.DataFrame(
        {
            "DateTime": data.timestamps,
            "TimeHours": data.time_hours,
            "RawSignal": raw_signal,
            "Activity": activity,
            "LightRaw": raw_light,
            "LightFilled": light,
            "InsertedMinute": data.inserted_minutes,
            "LightGapFilled": light_gap_mask,
        }
    )
    input_frame.to_csv(output_dir / "Input" / "Input.csv", index=False)

    started = time.perf_counter()
    phase_result = (
        _run_kf(data.time_hours, raw_signal, config)
        if config.phase_estimator == "KF"
        else _run_ks(data.time_hours, raw_signal, config)
    )
    phase_estimation_seconds = time.perf_counter() - started
    result_file_stem = f"{estimator_name}_Result"
    savemat(
        output_dir / estimator_name / f"{result_file_stem}.mat",
        phase_result,
        do_compression=True,
    )
    pd.DataFrame(
        {
            "iteration": np.arange(1, len(phase_result["avgCost"]) + 1),
            "average_cost": phase_result["avgCost"],
        }
    ).to_csv(output_dir / estimator_name / "Optimization.csv", index=False)

    started = time.perf_counter()
    pf_result = run_particle_filter(
        data.time_hours, light, phase_result["theta"], config.pf
    )
    pf_seconds = time.perf_counter() - started
    pf_state_phase = pf_result.mean_state_phase_rad
    detector_options = {
        "warmup_hours": 48.0,
        **config.detector_options,
    }
    dlmo = compare_dlmo_extract(
        pf_result.time_hours,
        pf_result.state_mean[:, 0],
        data.start_midnight,
        detector_options,
    )
    targets = compare_dlmo_targets(
        dlmo,
        pf_result.time_hours,
        pf_result.state_mean[:, 0],
        data.start_midnight,
        None,
        config.day_boundary_hour,
    )
    dlmo["events"].to_csv(output_dir / "CBT" / "Events.csv", index=False)
    dlmo["candidates"].to_csv(output_dir / "CBT" / "Candidates.csv", index=False)
    dlmo["daily"].to_csv(output_dir / "CBT" / "Daily.csv", index=False)
    targets.to_csv(output_dir / "CBT" / "AllDays.csv", index=False)
    visualization_path = output_dir / "Results_Visualization.svg"
    _save_results_visualization(
        visualization_path, pf_result, dlmo, data.start_midnight
    )

    diagnostics = pd.DataFrame(
        {
            "TimeHours": pf_result.time_hours,
            "Light": pf_result.light,
            "ObservedPhiRad": pf_result.observed_phase_rad,
            "SmoothedPhiRad": pf_result.smoothed_phase_rad,
            "MeanStatePhiRad": pf_state_phase,
            "MeanStateThetaRad": pf_result.mean_state_theta_rad,
            "PFOutputPhaseRad": pf_result.mean_state_theta_rad,
            "CBTStateX": pf_result.state_mean[:, 0],
            "Neff": pf_result.effective_sample_size,
            "MaxWeight": pf_result.max_weight,
            "Resampled": pf_result.resampled,
            "PriorResidualVarianceRad2": pf_result.prior_residual_variance_rad2,
            "PosteriorResidualVarianceRad2": pf_result.posterior_residual_variance_rad2,
            "CumulativeRSDRad": pf_result.cumulative_r_sd_rad,
        }
    )
    diagnostics.to_csv(output_dir / "PF" / "Diagnostics.csv", index=False)
    np.savez_compressed(
        output_dir / "PF" / "PF_Summary.npz",
        time_hours=pf_result.time_hours,
        state_mean=pf_result.state_mean,
        state_sd=pf_result.state_sd,
        mean_state_phase_rad=pf_result.mean_state_phase_rad,
        mean_state_theta_rad=pf_result.mean_state_theta_rad,
        circular_phase_mean_rad=pf_result.circular_phase_mean_rad,
        effective_sample_size=pf_result.effective_sample_size,
        resampled=pf_result.resampled,
    )
    if pf_result.history is not None:
        np.savez_compressed(output_dir / "PF" / "PF_History.npz", **pf_result.history)

    config_summary = asdict(config)
    config_summary.pop("ks_deviation_factor", None)
    config_summary.pop("kf_deviation_factor", None)
    summary = {
        "subject": config.subject,
        "signal": config.signal_name,
        "input_xlsx": str(source),
        "start_day": data.start_day.isoformat(),
        "start_midnight": data.start_midnight.isoformat(),
        "phase_estimator": config.phase_estimator,
        "phase_estimation_seconds": phase_estimation_seconds,
        "pf_seconds": pf_seconds,
        "ks_seed": config.ks_seed,
        "kf_seed": config.kf_seed,
        "pf_seed": config.pf.seed,
        "phase_estimation_cost": float(phase_result["estimCost"]),
        "phase_shift_hours": (
            float(phase_result["estimPhaseShift"])
            if np.isfinite(phase_result["estimPhaseShift"])
            else None
        ),
        "fixed_pf_r_rad": config.pf.r,
        "pf_residual_diagnostic_rad": pf_result.r_estimate_rad,
        "resampling_events": pf_result.resampling_events,
        "mean_neff_fraction": pf_result.mean_neff_fraction,
        "dlmo_days_reported": len(targets),
        "dlmo_events_tracked": len(dlmo["events"]),
        "output_dir": str(output_dir.resolve()),
        "visualization": str(visualization_path.resolve()),
        "config": {
            **config_summary,
            "input_xlsx": str(source),
            "output_root": str(config.output_root),
            "pf": asdict(config.pf),
        },
    }
    # JSON cannot encode Infinity; it is reported explicitly as a string.
    (output_dir / "Summary.json").write_text(
        json.dumps(summary, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "summary": summary,
        "phase_estimation": phase_result,
        "ks": phase_result if config.phase_estimator == "KS" else None,
        "particle_filter": pf_result,
        "dlmo": dlmo,
        "targets": targets,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_xlsx", type=Path)
    parser.add_argument("--subject", choices=_SUBJECTS, required=True)
    parser.add_argument("--signal", choices=("Act", "BPM"), default="BPM")
    parser.add_argument("--output-root", type=Path, default=Path("tPBM Results"))
    parser.add_argument("--ks-seed", type=int, default=12345)
    parser.add_argument("--ks-iterations", type=int, default=150)
    parser.add_argument("--ks-population", type=int, default=100)
    parser.add_argument("--ks-offspring", type=int, default=50)
    parser.add_argument(
        "--phase-estimator",
        dest="phase_estimator",
        choices=("KS", "KF"),
        default="KS",
        help="Kalman-based phase estimation method (default: KS)",
    )
    parser.add_argument(
        "--prefilter",
        dest="phase_estimator",
        choices=("KS", "KF"),
        help=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--kf-seed", type=int, default=12345)
    parser.add_argument("--kf-iterations", type=int, default=150)
    parser.add_argument("--kf-population", type=int, default=100)
    parser.add_argument("--kf-offspring", type=int, default=50)
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--pf-seed", type=int, default=1)
    parser.add_argument("--start-after-hours", type=float, default=72.0)
    parser.add_argument("--store-history", action="store_true")
    args = parser.parse_args()
    config = PipelineConfig(
        subject=args.subject,
        input_xlsx=args.input_xlsx,
        output_root=args.output_root,
        option=2 if args.signal == "Act" else 3,
        ks_seed=args.ks_seed,
        ks_iterations=args.ks_iterations,
        ks_population=args.ks_population,
        ks_offspring=args.ks_offspring,
        phase_estimator=args.phase_estimator,
        kf_seed=args.kf_seed,
        kf_iterations=args.kf_iterations,
        kf_population=args.kf_population,
        kf_offspring=args.kf_offspring,
        pf=ParticleFilterConfig(
            n_particles=args.particles,
            seed=args.pf_seed,
            start_after_hours=args.start_after_hours,
            store_history=args.store_history,
        ),
    )
    outcome = run_pipeline(config)
    print(json.dumps(outcome["summary"], indent=2, default=str, allow_nan=True))


if __name__ == "__main__":
    _main()
