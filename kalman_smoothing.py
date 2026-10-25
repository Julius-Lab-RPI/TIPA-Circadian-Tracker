"""Kalman state-space, RTS smoothing, and parameter optimization from KS.m."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from scipy.io import savemat

from circadian_models import estimate_available_phase_shift

FloatArray = NDArray[np.float64]


def _solve_small_system(matrix: FloatArray, right_hand_side: FloatArray) -> FloatArray:
    """Pivoted Gaussian elimination for the small covariance systems in RTS."""
    a = np.asarray(matrix, dtype=np.float64).copy()
    b = np.asarray(right_hand_side, dtype=np.float64).copy()
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape[0] != a.shape[0]:
        raise ValueError("incompatible linear system dimensions")
    size = a.shape[0]
    for column in range(size):
        pivot = column + int(np.argmax(np.abs(a[column:, column])))
        if abs(a[pivot, column]) <= np.finfo(np.float64).tiny:
            raise np.linalg.LinAlgError("singular covariance matrix in RTS smoother")
        if pivot != column:
            a[[column, pivot]] = a[[pivot, column]]
            b[[column, pivot]] = b[[pivot, column]]
        for row in range(column + 1, size):
            factor = a[row, column] / a[column, column]
            a[row, column:] -= factor * a[column, column:]
            b[row] -= factor * b[column]
    result = np.empty_like(b)
    for row in range(size - 1, -1, -1):
        remainder = (
            np.sum(a[row, row + 1 :, None] * result[row + 1 :], axis=0)
            if row + 1 < size
            else 0.0
        )
        result[row] = (b[row] - remainder) / a[row, row]
    return result


def create_kalman_state_space(
    order: int,
    state_length: int,
    dt: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    if order < 1 or state_length < 2 * order + 1 or dt <= 0:
        raise ValueError("invalid order, state_length, or sample interval")
    omega = 2.0 * np.pi / 24.0
    a = np.zeros((state_length, state_length))
    c = np.zeros((1, state_length))
    for harmonic in range(1, order + 1):
        index = harmonic * 2 - 1
        a[index - 1 : index + 1, index - 1 : index + 1] = (
            (0.0, 1.0),
            (-(harmonic * omega) ** 2, 0.0),
        )
        c[0, index] = 2.0 / (harmonic * omega)
    c[0, -1] = 1.0
    # MATLAB's impulse-invariant c2d result for these unforced oscillator
    # blocks is expm(A*dt), with B=D=0 and unchanged C. The block exponential
    # has a closed form, avoiding a general-purpose matrix exponential.
    ad = np.eye(state_length)
    for harmonic in range(1, order + 1):
        index = harmonic * 2 - 2
        frequency = harmonic * omega
        angle = frequency * dt
        ad[index : index + 2, index : index + 2] = (
            (np.cos(angle), np.sin(angle) / frequency),
            (-frequency * np.sin(angle), np.cos(angle)),
        )
    b = np.zeros((state_length, 1))
    d = np.zeros((1, 1))
    return ad, b, c, d


def simulate_kalman_smoother(
    a: FloatArray,
    c: FloatArray,
    q: FloatArray,
    r: float,
    observations: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Run the MATLAB forward Kalman filter and backward RTS smoother."""
    a = np.asarray(a, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64).reshape(1, -1)
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64).reshape(-1)
    dimension = a.shape[0]
    if a.shape != (dimension, dimension) or c.shape[1] != dimension:
        raise ValueError("A and C dimensions are incompatible")
    if q.shape != (dimension, dimension) or r <= 0 or not np.isfinite(r):
        raise ValueError("Q must be square and R must be a positive finite variance")
    if not y.size:
        raise ValueError("observations cannot be empty")

    count = y.size
    x_prior = np.zeros((count, dimension))
    p_prior = np.zeros((count, dimension, dimension))
    x_post = np.zeros_like(x_prior)
    p_post = np.zeros_like(p_prior)
    identity = np.eye(dimension)
    for k in range(count):
        if k == 0:
            p_prior[k] = 1e3 * identity
        else:
            x_prior[k] = np.einsum("ij,j->i", a, x_post[k - 1])
            propagated = np.einsum("ij,jk->ik", a, p_post[k - 1])
            p_prior[k] = np.einsum("ij,kj->ik", propagated, a) + q
        if not np.isfinite(y[k]):
            x_post[k] = x_prior[k]
            p_post[k] = p_prior[k]
        else:
            observation = c[0]
            innovation_variance = float(
                np.einsum("i,ij,j->", observation, p_prior[k], observation) + r
            )
            gain = np.einsum("ij,j->i", p_prior[k], observation) / innovation_variance
            innovation = y[k] - np.sum(observation * x_prior[k])
            x_post[k] = x_prior[k] + gain * innovation
            correction = identity - np.einsum("i,j->ij", gain, observation)
            p_post[k] = np.einsum("ij,jk->ik", correction, p_prior[k])

    x_smooth = np.zeros_like(x_post)
    p_smooth = np.zeros_like(p_post)
    x_smooth[-1] = x_post[-1]
    p_smooth[-1] = p_post[-1]
    for k in range(count - 2, -1, -1):
        propagated_covariance = np.einsum("ij,jk->ik", a, p_post[k])
        gain = _solve_small_system(p_prior[k + 1], propagated_covariance).T
        x_smooth[k] = x_post[k] + np.einsum(
            "ij,j->i", gain, x_smooth[k + 1] - x_prior[k + 1]
        )
        covariance_delta = p_smooth[k + 1] - p_prior[k + 1]
        smoothed_delta = np.einsum("ij,jk->ik", gain, covariance_delta)
        smoothed_delta = np.einsum("ij,kj->ik", smoothed_delta, gain)
        p_smooth[k] = p_post[k] + smoothed_delta
    y_smooth = np.einsum("i,ji->j", c[0], x_smooth)
    return x_smooth.T, y_smooth, x_post.T


def initialize_population(
    state_length: int,
    population_size: int,
    l_end: float,
    r_end: float,
    lower_bound: float,
    r_lower: float,
    r_upper: float,
    rng: np.random.Generator | None = None,
) -> tuple[FloatArray, FloatArray]:
    rng = rng or np.random.default_rng()
    q_size = state_length**2
    midpoint = (r_end + l_end) / 2.0
    draws = (r_end - l_end) * rng.random((population_size, q_size)) + l_end
    q_population = np.zeros_like(draws)
    upper = draws > midpoint + 0.5
    lower = draws < midpoint - 0.5
    q_population[upper] = 10.0 ** (draws[upper] - midpoint + lower_bound)
    q_population[lower] = -(10.0 ** (midpoint - draws[lower] + lower_bound))
    r_population = r_lower + (r_upper - r_lower) * rng.random(population_size)
    return q_population, r_population


def compute_spectrum(time: FloatArray, values: FloatArray) -> tuple[FloatArray, FloatArray, int]:
    t = np.asarray(time, dtype=np.float64).reshape(-1)
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    if t.size != signal.size or t.size < 2:
        raise ValueError("time and values must have equal lengths >= 2")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(signal)):
        raise ValueError("spectrum inputs must be finite")
    dt = t[1] - t[0]
    if dt <= 0:
        raise ValueError("time must be strictly increasing")
    length = signal.size
    spectrum = np.abs(np.fft.fft(signal) / length)[: length // 2 + 1]
    if spectrum.size > 2:
        spectrum[1:-1] *= 2.0
    frequencies = np.fft.rfftfreq(length, d=dt)
    return spectrum, frequencies, length


def compute_cost(
    original_spectrum: FloatArray,
    filtered_spectrum: FloatArray,
    frequencies: FloatArray,
    order: int,
) -> float:
    """Translate the harmonic-preservation/noise cost in Utils.computeCost."""
    original = np.asarray(original_spectrum, dtype=np.float64)
    filtered = np.asarray(filtered_spectrum, dtype=np.float64)
    f = np.asarray(frequencies, dtype=np.float64)
    if original.shape != filtered.shape or original.size != f.size:
        raise ValueError("spectra and frequency arrays must have matching shapes")
    harmonic_indices = [int(np.argmin(np.abs(f - harmonic / 24.0))) for harmonic in range(1, 6)]
    noise_index = int(np.argmin(np.abs(f - 0.0309)))
    width = harmonic_indices[0] - noise_index
    if width <= 0 or order < 1 or order > len(harmonic_indices):
        raise ValueError("signal is too short for the requested harmonic cost")
    harmonic_cost = trapezoid((filtered[:width] - original[:width]) ** 2)
    noise_cost = trapezoid(
        filtered[width : harmonic_indices[0] - width + 1] ** 2
    )
    for harmonic in range(order):
        index = harmonic_indices[harmonic]
        left = max(0, index - width)
        right = min(filtered.size, index + width + 1)
        harmonic_cost += trapezoid((filtered[left:right] - original[left:right]) ** 2)
        if harmonic != order - 1:
            next_index = harmonic_indices[harmonic + 1]
            noise_cost += trapezoid(filtered[index + width + 1 : next_index - width] ** 2)
    noise_cost += trapezoid(filtered[harmonic_indices[order - 1] + width + 1 :] ** 2)
    return float(harmonic_cost + noise_cost)


def _evaluate_population(
    q_vectors: FloatArray,
    r_values: FloatArray,
    a: FloatArray,
    c: FloatArray,
    observations: FloatArray,
    time: FloatArray,
    order: int,
    batch_size: int = 64,
) -> FloatArray:
    """Evaluate many Q/R candidates together while preserving the KS/RTS equations."""
    vectors = np.asarray(q_vectors, dtype=np.float64)
    variances = np.asarray(r_values, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) != len(variances):
        raise ValueError("Q and R candidate arrays must have matching candidate counts")
    count, state_length = len(observations), a.shape[0]
    if vectors.shape[1] != state_length**2 or batch_size < 1:
        raise ValueError("candidate dimension or batch_size is invalid")
    if count != len(time):
        raise ValueError("time and observation vectors must have the same length")
    frequencies = np.fft.rfftfreq(count, d=float(time[1] - time[0]))
    harmonic_indices = np.array(
        [np.argmin(np.abs(frequencies - harmonic / 24.0)) for harmonic in range(1, 6)]
    )
    noise_index = int(np.argmin(np.abs(frequencies - 0.0309)))
    band_width = int(harmonic_indices[0] - noise_index)
    if band_width <= 0:
        raise ValueError("signal is too short for the requested harmonic cost")
    missing = ~np.isfinite(observations)
    filled_observations = np.where(missing, 0.0, observations)
    observation = np.asarray(c, dtype=np.float64).reshape(-1)
    identity = np.eye(state_length)
    costs = np.empty(len(vectors), dtype=np.float64)

    for batch_start in range(0, len(vectors), batch_size):
        batch_stop = min(batch_start + batch_size, len(vectors))
        # MATLAB reshapes each Q candidate column-major before Q*Q'.
        batch_q = vectors[batch_start:batch_stop].reshape(
            (-1, state_length, state_length)
        ).transpose(0, 2, 1)
        batch_q = np.einsum("bij,bkj->bik", batch_q, batch_q)
        batch_r = variances[batch_start:batch_stop]
        batch_count = len(batch_r)
        x_post = np.zeros((count, batch_count, state_length), dtype=np.float64)
        p_post = np.zeros(
            (count, batch_count, state_length, state_length), dtype=np.float64
        )
        prior_covariance = np.broadcast_to(
            1e3 * identity, (batch_count, state_length, state_length)
        ).copy()
        for step in range(count):
            if step:
                prior_mean = np.einsum("ij,bj->bi", a, x_post[step - 1])
                prior_covariance = (
                    np.einsum(
                        "ij,bjk,lk->bil",
                        a,
                        p_post[step - 1],
                        a,
                    )
                    + batch_q
                )
            else:
                prior_mean = np.zeros((batch_count, state_length))
            if np.isfinite(observations[step]):
                innovation_variance = (
                    np.einsum("i,bij,j->b", observation, prior_covariance, observation)
                    + batch_r
                )
                gain = np.einsum(
                    "bij,j->bi", prior_covariance, observation
                ) / innovation_variance[:, None]
                innovation = filled_observations[step] - np.einsum(
                    "i,bi->b", observation, prior_mean
                )
                x_post[step] = prior_mean + gain * innovation[:, None]
                correction = identity[None, :, :] - np.einsum(
                    "bi,j->bij", gain, observation
                )
                p_post[step] = np.einsum(
                    "bij,bjk->bik", correction, prior_covariance
                )
            else:
                x_post[step] = prior_mean
                p_post[step] = prior_covariance

        x_smooth = np.empty_like(x_post)
        x_smooth[-1] = x_post[-1]
        for step in range(count - 2, -1, -1):
            prior_covariance = (
                np.einsum("ij,bjk,lk->bil", a, p_post[step], a) + batch_q
            )
            lower = np.zeros_like(prior_covariance)
            lower[:, 0, 0] = np.sqrt(prior_covariance[:, 0, 0])
            lower[:, 1, 0] = prior_covariance[:, 1, 0] / lower[:, 0, 0]
            lower[:, 2, 0] = prior_covariance[:, 2, 0] / lower[:, 0, 0]
            lower[:, 1, 1] = np.sqrt(
                prior_covariance[:, 1, 1] - lower[:, 1, 0] ** 2
            )
            lower[:, 2, 1] = (
                prior_covariance[:, 2, 1]
                - lower[:, 2, 0] * lower[:, 1, 0]
            ) / lower[:, 1, 1]
            diagonal_22 = prior_covariance[:, 2, 2] - (
                lower[:, 2, 0] ** 2 + lower[:, 2, 1] ** 2
            )
            if np.any(diagonal_22 <= 0.0) or not np.all(np.isfinite(diagonal_22)):
                raise FloatingPointError("candidate Kalman covariance is not positive definite")
            lower[:, 2, 2] = np.sqrt(diagonal_22)

            rhs = np.einsum("ij,bjk->bik", a, p_post[step])
            z = np.empty_like(rhs)
            z[:, 0] = rhs[:, 0] / lower[:, 0, 0, None]
            z[:, 1] = (
                rhs[:, 1] - lower[:, 1, 0, None] * z[:, 0]
            ) / lower[:, 1, 1, None]
            z[:, 2] = (
                rhs[:, 2]
                - lower[:, 2, 0, None] * z[:, 0]
                - lower[:, 2, 1, None] * z[:, 1]
            ) / lower[:, 2, 2, None]
            solved = np.empty_like(rhs)
            solved[:, 2] = z[:, 2] / lower[:, 2, 2, None]
            solved[:, 1] = (
                z[:, 1] - lower[:, 2, 1, None] * solved[:, 2]
            ) / lower[:, 1, 1, None]
            solved[:, 0] = (
                z[:, 0]
                - lower[:, 1, 0, None] * solved[:, 1]
                - lower[:, 2, 0, None] * solved[:, 2]
            ) / lower[:, 0, 0, None]
            smoother_gain = solved.transpose(0, 2, 1)
            prior_mean = np.einsum("ij,bj->bi", a, x_post[step])
            difference = x_smooth[step + 1] - prior_mean
            x_smooth[step] = x_post[step] + np.einsum(
                "bij,bj->bi", smoother_gain, difference
            )

        smooth_output = np.einsum("i,nbi->bn", observation, x_smooth)
        filled = np.where(missing[None, :], smooth_output, observations[None, :])
        original_spectrum = np.abs(np.fft.rfft(filled, axis=1) / count)
        filtered_spectrum = np.abs(np.fft.rfft(smooth_output, axis=1) / count)
        if count % 2 == 0:
            original_spectrum[:, 1:-1] *= 2.0
            filtered_spectrum[:, 1:-1] *= 2.0
        else:
            original_spectrum[:, 1:-1] *= 2.0
            filtered_spectrum[:, 1:-1] *= 2.0
        harmonic_cost = np.zeros(batch_count)
        noise_cost = np.zeros(batch_count)
        harmonic_cost += _trapezoid_unit(
            (filtered_spectrum[:, :band_width] - original_spectrum[:, :band_width]) ** 2
        )
        noise_start = band_width
        noise_stop = int(harmonic_indices[0] - band_width + 1)
        noise_cost += _trapezoid_unit(filtered_spectrum[:, noise_start:noise_stop] ** 2)
        for harmonic in range(order):
            index = int(harmonic_indices[harmonic])
            left = max(0, index - band_width)
            right = min(filtered_spectrum.shape[1], index + band_width + 1)
            harmonic_cost += _trapezoid_unit(
                (
                    filtered_spectrum[:, left:right]
                    - original_spectrum[:, left:right]
                )
                ** 2,
            )
            if harmonic != order - 1:
                next_index = int(harmonic_indices[harmonic + 1])
                noise_cost += _trapezoid_unit(
                    filtered_spectrum[:, index + band_width + 1 : next_index - band_width] ** 2
                )
        noise_cost += _trapezoid_unit(
            filtered_spectrum[:, int(harmonic_indices[order - 1]) + band_width + 1 :] ** 2
        )
        costs[batch_start:batch_stop] = harmonic_cost + noise_cost
    return costs


def _trapezoid_unit(values: FloatArray) -> FloatArray:
    if values.shape[-1] < 2:
        return np.zeros(values.shape[:-1], dtype=np.float64)
    return np.sum((values[..., :-1] + values[..., 1:]) * 0.5, axis=-1)


def optimize_parameters(
    iterations: int,
    mu: int,
    lam: int,
    rho: int,
    order: int,
    l_end: float,
    r_end: float,
    lower_bound: float,
    r_lower: float,
    r_upper: float,
    a: FloatArray,
    c: FloatArray,
    time: FloatArray,
    observations: FloatArray,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Genetic Q/R search from KS.m; R here is a measurement variance."""
    if iterations < 0 or mu < 2 or lam < 1 or rho < 1 or rho > min(50, mu):
        raise ValueError("invalid optimization population or iteration settings")
    rng = np.random.default_rng(seed)
    state_length = np.asarray(a).shape[0]
    q_size = state_length**2
    q_pop, r_pop = initialize_population(
        state_length, mu, l_end, r_end, lower_bound, r_lower, r_upper, rng
    )
    costs = np.empty(mu, dtype=np.float64)
    average_cost = np.empty(iterations, dtype=np.float64)
    time_values = np.asarray(time, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64).reshape(-1)
    combinations_pool = list(combinations(range(min(50, mu)), rho))
    if lam > len(combinations_pool):
        raise ValueError("lambda exceeds the number of available parent combinations")

    costs[:] = _evaluate_population(q_pop, r_pop, a, c, y, time_values, order)
    for iteration in range(iterations):
        selected = rng.permutation(len(combinations_pool))[:lam]
        offspring_q = np.empty((lam, q_size))
        offspring_r = np.empty(lam)
        for child, selection in enumerate(selected):
            parents = combinations_pool[selection]
            offspring_q[child] = np.mean(q_pop[list(parents)], axis=0)
            offspring_r[child] = np.mean(r_pop[list(parents)])
        child_cost = _evaluate_population(
            offspring_q, offspring_r, a, c, y, time_values, order
        )
        q_pop = np.vstack((q_pop, offspring_q))
        r_pop = np.r_[r_pop, offspring_r]
        costs = np.r_[costs, child_cost]
        nan_indices = np.flatnonzero(np.isnan(costs))
        ranked_non_nan = np.flatnonzero(~np.isnan(costs))
        worst_cost = ranked_non_nan[
            np.argsort(costs[ranked_non_nan], kind="stable")[::-1]
        ]
        remove = np.concatenate((nan_indices, worst_cost))[:lam]
        keep_mask = np.ones(costs.size, dtype=np.bool_)
        keep_mask[remove] = False
        keep = np.flatnonzero(keep_mask)
        q_pop, r_pop, costs = q_pop[keep], r_pop[keep], costs[keep]
        average_cost[iteration] = np.nanmean(costs) if np.any(np.isfinite(costs)) else np.nan
    return costs, average_cost, q_pop, r_pop


def calculate_theta(x_hat: FloatArray, omega: float) -> FloatArray:
    states = np.asarray(x_hat, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("x_hat must contain at least two state rows")
    return np.mod(-np.arctan2(states[1], omega * states[0]) + np.pi / 2, 2 * np.pi) - np.pi


def calculate_continuous_phase(time: FloatArray, theta: FloatArray, omega: float) -> FloatArray:
    t = np.asarray(time, dtype=np.float64)
    angle = np.asarray(theta, dtype=np.float64)
    if t.shape != angle.shape or omega == 0:
        raise ValueError("time and theta must match and omega must be nonzero")
    return (np.unwrap(angle) - omega * t) / omega


def estimate_cbt_phase_shift(
    time: FloatArray,
    x: FloatArray,
    day1: int,
    day2: int,
) -> tuple[FloatArray, float]:
    t = np.asarray(time, dtype=np.float64)
    values = np.asarray(x, dtype=np.float64)
    interval_count = int(np.floor((t[-1] - t[0]) / 24.0))
    intervals = t[0] + 24.0 * np.arange(interval_count + 1)
    markers: list[float] = []
    for start, end in zip(intervals[:-1], intervals[1:], strict=True):
        indices = np.flatnonzero((t >= start) & (t < end))
        if not indices.size:
            markers.append(float("nan"))
        else:
            markers.append(float(t[indices[np.argmin(values[indices])]] % 24.0))
    result = np.asarray(markers)
    if day1 < 1 or day2 < 1 or day1 > len(result) or day2 > len(result):
        raise ValueError("requested day is outside available CBT markers")
    return result, float(result[day2 - 1] - result[day1 - 1])


def find_daily_min(values: FloatArray, window_size: int) -> tuple[FloatArray, NDArray[np.int64]]:
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    if window_size < 1:
        raise ValueError("window_size must be positive")
    count = signal.size // window_size
    minima = np.empty(count)
    indices = np.empty(count, dtype=np.int64)
    for group in range(count):
        start = group * window_size
        local = int(np.argmin(signal[start : start + window_size]))
        minima[group] = signal[start + local]
        indices[group] = start + local
    return minima, indices


def run_kalman_smoothing(
    time: FloatArray,
    observations: FloatArray,
    *,
    iterations: int = 150,
    population_size: int = 100,
    offspring_count: int = 50,
    parent_count: int = 2,
    seed: int | None = None,
    output_path: str | None = None,
) -> dict[str, object]:
    """High-level equivalent of run_kalman_smoothing.m."""
    t = np.asarray(time, dtype=np.float64).reshape(-1)
    raw = np.asarray(observations, dtype=np.float64).reshape(-1).copy()
    if len(t) != len(raw) or len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError("time and observations must match with increasing times")
    raw[raw == 0] = np.nan
    mean = float(np.nanmean(raw))
    standard_deviation = float(np.nanstd(raw, ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        raise ValueError("selected signal has no usable variation")
    normalized = (raw - mean) / standard_deviation
    order = 1
    a, _, c, _ = create_kalman_state_space(order, 3, t[1] - t[0])
    costs, averages, q_pop, r_pop = optimize_parameters(
        iterations,
        population_size,
        offspring_count,
        parent_count,
        order,
        0,
        18,
        -5,
        1e2,
        1e8,
        a,
        c,
        t,
        normalized,
        seed,
    )
    best = int(np.nanargmin(costs))
    q = q_pop[best].reshape((3, 3), order="F")
    q = np.einsum("ij,kj->ik", q, q)
    x_smooth, y_smooth, x_post = simulate_kalman_smoother(a, c, q, r_pop[best], normalized)
    theta, phase_shift = _estimate_phase_shift_for_smoother(t, x_smooth)
    reconstructed = normalized.copy()
    missing = ~np.isfinite(reconstructed)
    reconstructed[missing] = y_smooth[missing]
    original_spectrum, frequencies, _ = compute_spectrum(t, reconstructed)
    filtered_spectrum, _, _ = compute_spectrum(t, y_smooth)
    estimated_cost = compute_cost(original_spectrum, filtered_spectrum, frequencies, order)
    result: dict[str, object] = {
        "t": t,
        "y_raw": raw,
        "y": normalized,
        "y_rec": reconstructed,
        "y_norm": normalized,
        "x_smooth": x_smooth,
        "y_smooth": y_smooth,
        "x_post": x_post,
        "theta": theta,
        "phi": np.mod(np.unwrap(theta) - (2 * np.pi / 24) * t, 2 * np.pi),
        "Q": q,
        "R": float(r_pop[best]),
        "Cost": costs,
        "avgCost": averages,
        "Q_pop": q_pop,
        "R_pop": r_pop,
        "minCost": float(np.nanmin(costs)),
        "estimCost": estimated_cost,
        "rectifiedSpectrum": original_spectrum,
        "filteredSpectrum": filtered_spectrum,
        "frequencies": frequencies,
        "estimPhaseShift": phase_shift,
        "mu_y": mean,
        "sigma_y": standard_deviation,
        "x_post": x_post,
    }
    if output_path is not None:
        savemat(output_path, result, do_compression=True)
    return result


def _estimate_phase_shift_for_smoother(
    time: FloatArray, x_smooth: FloatArray
) -> tuple[FloatArray, float]:
    return estimate_available_phase_shift(
        2 * np.pi / 24, time, x_smooth[0], x_smooth[1]
    )


def save_kalman_result(path: str, result: dict[str, object]) -> None:
    """Save compatible numeric arrays to a MATLAB .mat file."""
    savemat(path, result, do_compression=True)


class KS:
    """MATLAB ``KS`` static-method names retained for easy migration."""

    createKalmanStateSpace = staticmethod(create_kalman_state_space)
    initializePopulation = staticmethod(initialize_population)
    simulateKalmanSmoother = staticmethod(simulate_kalman_smoother)
    optimizeParameters = staticmethod(optimize_parameters)
