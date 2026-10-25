"""Circadian oscillator and phase helpers translated from the MATLAB project."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat, savemat
from scipy.signal import find_peaks

FloatArray = NDArray[np.float64]
_TWO_PI = 2.0 * np.pi


def wrap_to_2pi(angle: FloatArray | float) -> FloatArray:
    return np.mod(angle, _TWO_PI)


def wrap_to_pi(angle: FloatArray | float) -> FloatArray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def compute_phi(time_hours: FloatArray, theta: FloatArray, omega: float) -> FloatArray:
    time = np.asarray(time_hours, dtype=np.float64)
    angle = np.asarray(theta, dtype=np.float64)
    if time.shape != angle.shape:
        raise ValueError("time_hours and theta must have matching shapes")
    result = np.full(angle.shape, np.nan)
    valid = np.isfinite(angle)
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        result[start:end] = wrap_to_2pi(np.unwrap(angle[start:end]) - omega * time[start:end])
    return result


def estimate_phase_shift(
    day1: int,
    day2: int,
    omega: float,
    time_hours: FloatArray,
    x1: FloatArray,
    x2: FloatArray,
) -> tuple[FloatArray, float]:
    """Translate ``estimate_phase_shift.m``; day numbers are one-based."""
    t = np.asarray(time_hours, dtype=np.float64)
    first = np.asarray(x1, dtype=np.float64)
    second = np.asarray(x2, dtype=np.float64)
    if t.shape != first.shape or t.shape != second.shape or omega == 0:
        raise ValueError("time and state vectors must match and omega must be nonzero")
    theta = wrap_to_2pi(-np.arctan2(second, omega * first))
    relative_t = t - t[0]
    a = np.unwrap(theta[(relative_t >= (day1 - 1) * 24) & (relative_t <= day1 * 24)])
    b = np.unwrap(theta[(relative_t >= (day2 - 1) * 24) & (relative_t <= day2 * 24)])
    if not a.size or not b.size:
        raise ValueError("recording does not contain samples for both requested days")
    count = min(a.size, b.size)
    shift = float(np.mod(12.0 + np.mean(a[:count] - b[:count]) / omega, 24.0) - 12.0)
    return theta, shift


def estimate_available_phase_shift(
    omega: float,
    time_hours: FloatArray,
    x1: FloatArray,
    x2: FloatArray,
) -> tuple[FloatArray, float]:
    """Estimate phase shift over available complete days, if at least two exist."""
    t = np.asarray(time_hours, dtype=np.float64)
    first = np.asarray(x1, dtype=np.float64)
    second = np.asarray(x2, dtype=np.float64)
    if t.ndim != 1 or t.shape != first.shape or t.shape != second.shape:
        raise ValueError("time and state vectors must be matching one-dimensional arrays")
    if t.size == 0 or not np.all(np.isfinite(t)) or omega == 0:
        raise ValueError("time must be finite and nonempty and omega must be nonzero")
    relative_time = t - t[0]
    last_complete_day = int(np.floor((relative_time[-1] + 1e-9) / 24.0))
    if last_complete_day < 2:
        theta = wrap_to_2pi(-np.arctan2(second, omega * first))
        return theta, float("nan")
    return estimate_phase_shift(
        1, last_complete_day, omega, t, first, second
    )


def activity_to_light(activity: FloatArray) -> FloatArray:
    """Map activity samples to 0/100/200/500/2000 lux as in MATLAB."""
    values = np.asarray(activity, dtype=np.float64)
    result = np.zeros(values.shape, dtype=np.float64)
    finite_positive = np.isfinite(values) & (values > 0)
    half_max = np.nanmax(values) / 2 if np.any(np.isfinite(values)) else 0.0
    result[finite_positive & (values < 0.1 * half_max)] = 100.0
    result[finite_positive & (values >= 0.1 * half_max) & (values < 0.25 * half_max)] = 200.0
    result[finite_positive & (values >= 0.25 * half_max) & (values < 0.4 * half_max)] = 500.0
    result[finite_positive & (values >= 0.4 * half_max)] = 2000.0
    return result


def _jewett_derivative(states: FloatArray, light: FloatArray, tau: FloatArray) -> FloatArray:
    x, xc, n = states.T
    alpha = 0.16 * np.power(light / 9500.0, 0.6)
    u = 19.875 * alpha * (1.0 - n)
    b = (1.0 - 0.4 * x) * (1.0 - 0.4 * xc) * u
    return np.column_stack(
        (
            np.pi / 12.0 * (xc + 0.13 * (x / 3 + 4 * x**3 / 3 - 256 * x**7 / 105) + b),
            np.pi / 12.0 * (xc * b / 3 - x * ((24 / 0.99729 / tau) ** 2 + 0.55 * b)),
            60.0 * (alpha * (1.0 - n) - 0.013 * n),
        )
    )


def jewett99_rk4_particles(
    states: FloatArray,
    light: FloatArray,
    dt_hours: float,
    tau: FloatArray,
) -> FloatArray:
    """Vectorized RK4 state transition for particles with individual tau values."""
    particle_states = np.asarray(states, dtype=np.float64)
    illumination = np.asarray(light, dtype=np.float64)
    tau_values = np.asarray(tau, dtype=np.float64)
    if particle_states.ndim != 2 or particle_states.shape[1] != 3:
        raise ValueError("states must have shape (n_particles, 3)")
    if illumination.shape != (len(particle_states),) or tau_values.shape != illumination.shape:
        raise ValueError("light and tau must each have one value per particle")
    if np.any(tau_values == 0) or np.any(illumination < 0) or dt_hours <= 0:
        raise ValueError("tau must be nonzero, light nonnegative, and dt positive")
    k1 = _jewett_derivative(particle_states, illumination, tau_values)
    k2 = _jewett_derivative(particle_states + dt_hours * k1 / 2, illumination, tau_values)
    k3 = _jewett_derivative(particle_states + dt_hours * k2 / 2, illumination, tau_values)
    k4 = _jewett_derivative(particle_states + dt_hours * k3, illumination, tau_values)
    return particle_states + dt_hours * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def jewett99_rk4_tau(
    time_hours: FloatArray,
    light: FloatArray,
    initial_conditions: Sequence[float],
    tau: float | FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Fixed-step RK4 version used in the particle filter and initial-state setup."""
    t = np.asarray(time_hours, dtype=np.float64).reshape(-1)
    illumination = np.asarray(light, dtype=np.float64).reshape(-1)
    initial = np.asarray(initial_conditions, dtype=np.float64).reshape(-1)
    tau_values = np.asarray(tau, dtype=np.float64)
    if len(t) < 2 or len(t) != len(illumination) or initial.size != 3:
        raise ValueError("time/light lengths must match (at least two); initial state has 3 values")
    if not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0) or not np.all(np.isfinite(illumination)):
        raise ValueError("time must increase and time/light must be finite")
    if np.any(illumination < 0) or np.any(tau_values == 0):
        raise ValueError("light must be nonnegative and tau must be nonzero")
    tau_values = np.broadcast_to(tau_values, t.shape)
    states = np.empty((t.size, 3), dtype=np.float64)
    states[0] = initial
    for index in range(1, t.size):
        h = t[index] - t[index - 1]
        states[index] = jewett99_rk4_particles(
            states[index - 1 : index],
            illumination[index : index + 1],
            h,
            tau_values[index : index + 1],
        )[0]
    alpha = 0.16 * np.power(illumination / 9500.0, 0.6)
    process_u = 19.875 * alpha * (1.0 - states[:, 2])
    return t, process_u, states


def jewett99(
    time_hours: FloatArray,
    light: FloatArray,
    initial_conditions: Sequence[float],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Integrate the Jewett-99 oscillator with RK4 and linearly interpolated light."""
    t = np.asarray(time_hours, dtype=np.float64).reshape(-1)
    illumination = np.asarray(light, dtype=np.float64).reshape(-1)
    initial = np.asarray(initial_conditions, dtype=np.float64).reshape(-1)
    if len(t) < 2 or len(t) != len(illumination) or initial.size != 3:
        raise ValueError("time/light lengths must match (at least two); initial state has 3 values")
    if (
        not np.all(np.isfinite(t))
        or np.any(np.diff(t) <= 0)
        or not np.all(np.isfinite(illumination))
        or np.any(illumination < 0)
    ):
        raise ValueError("time must increase and light must be finite and nonnegative")

    states = np.empty((t.size, 3), dtype=np.float64)
    states[0] = initial
    for index in range(1, t.size):
        h = t[index] - t[index - 1]
        start_state = states[index - 1 : index]
        light_start = illumination[index - 1]
        light_end = illumination[index]

        def derivative(fraction: float, values: FloatArray) -> FloatArray:
            current_light = np.array([light_start + (light_end - light_start) * fraction])
            return _jewett_derivative(values, current_light, np.array([24.2]))

        k1 = derivative(0.0, start_state)
        k2 = derivative(0.5, start_state + h * k1 / 2)
        k3 = derivative(0.5, start_state + h * k2 / 2)
        k4 = derivative(1.0, start_state + h * k3)
        states[index] = (start_state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)[0]
    alpha = 0.16 * np.power(illumination / 9500.0, 0.6)
    process_u = 19.875 * alpha * (1.0 - states[:, 2])
    return t, process_u, states


def get_cbt_mins(
    time_hours: FloatArray,
    x: FloatArray,
    xc: FloatArray,
    model: str = "jewett99",
) -> FloatArray:
    """Translate ``getCBTMins.m`` for Jewett-99 and St. Hilaire-07 models."""
    t = np.asarray(time_hours, dtype=np.float64).reshape(-1)
    x_values = np.asarray(x, dtype=np.float64).reshape(-1)
    xc_values = np.asarray(xc, dtype=np.float64).reshape(-1)
    if not t.size or not (t.size == x_values.size == xc_values.size):
        raise ValueError("time_hours, x, and xc must have matching lengths")
    if model == "jewett99":
        new_start = 0.0
        if t[0] != 0:
            keep = t >= 24
            t, x_values = t[keep], x_values[keep]
            if not t.size:
                return np.empty(0)
            new_start = float(t[0])
        relative = t - t[0]
        day_count = int(np.ceil(relative[-1] / 24))
        result = np.full(day_count, np.nan)
        for day in range(day_count):
            indices = np.flatnonzero((relative >= day * 24) & (relative < (day + 1) * 24))
            if indices.size < 3:
                continue
            peaks, _ = find_peaks(-x_values[indices], distance=min(120, indices.size - 2))
            if peaks.size:
                selected = peaks[np.argmin(x_values[indices[peaks]])]
                result[day] = t[indices[selected]] + 0.8
        if new_start != 0:
            result += new_start
            result = np.r_[np.nan, result]
        return result
    if model == "hilaire07":
        angles = np.arctan2(xc_values, x_values)
        target = np.deg2rad(-170.7)
        tolerance = np.deg2rad(1.5)
        mins = t[(angles <= target + tolerance) & (angles > target - tolerance)]
        if not mins.size:
            return np.full(int(np.ceil(t[-1] / 24)), np.nan)
        groups = np.split(mins, np.flatnonzero(np.diff(mins) > 1.0) + 1)
        cbt = np.array([group.mean() for group in groups]) + 0.97
        result = np.full(int(np.ceil(t[-1] / 24)), np.nan)
        for value in cbt:
            day = int(np.floor(value / 24))
            if 0 <= day < result.size:
                result[day] = value
        return result
    raise ValueError("model must be 'jewett99' or 'hilaire07'")


def huang_initial_conditions_for_hour(start_hour: int) -> FloatArray:
    """Generate Huang-style stable initial conditions from a 30-day light cycle."""
    if not 0 <= start_hour <= 23:
        raise ValueError("start_hour must be in 0..23")
    dt = 1.0 / 60.0
    t = np.arange(0.0, 30 * 24 + dt / 2, dt)
    hour_of_day = np.mod(t, 24.0)
    light = np.where(hour_of_day >= 8.0, 800.0, 0.0)
    _, _, states = jewett99(t, light, (0.1, 0.1, 0.5))
    last_day = t >= t[-1] - 24.0
    mask = last_day & (np.floor(hour_of_day) == start_hour)
    if not np.any(mask):
        raise RuntimeError("no matching samples found for the requested initial hour")
    return np.mean(states[mask], axis=0)


def generate_initial_conditions_huang(
    subject: str,
    mat_data_dir: str | Path = "MatData",
) -> FloatArray:
    """Load a subject's minute MAT file, compute Huang initials, and append ``init``."""
    path = Path(mat_data_dir) / f"{subject}_Minute_Data.mat"
    if not path.is_file():
        raise FileNotFoundError(path)
    contents = loadmat(path)
    if "MinuteData_array" not in contents:
        raise ValueError(f"{path} does not contain 'MinuteData_array'")
    minute_data = np.asarray(contents["MinuteData_array"], dtype=np.float64)
    if minute_data.ndim != 2 or minute_data.shape[0] == 0:
        raise ValueError("'MinuteData_array' must be a nonempty matrix")
    start_hour = int(np.floor(minute_data[0, 0] * 24.0))
    initial = huang_initial_conditions_for_hour(start_hour)
    variables = {
        name: value
        for name, value in contents.items()
        if not name.startswith("__")
    }
    variables["init"] = initial.reshape(1, 3)
    savemat(path, variables, do_compression=True)
    return initial
