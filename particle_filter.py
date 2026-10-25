"""Python port of the MATLAB tPBM phase particle filter.

The state order and update order follow ``run_pf_tpbm_phi_Q5_R015_full_log.m``:
``[x, xc, n, delta, tau]``, Jewett-99 RK4 propagation, wrapped Gaussian phase
likelihood, ESS-triggered systematic resampling, then process-noise roughening.
"""

from __future__ import annotations

import argparse
import csv
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from filterpy.monte_carlo import systematic_resample
from numpy.typing import NDArray

from circadian_models import jewett99_rk4_particles

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_TWO_PI = 2.0 * np.pi
_OMEGA = _TWO_PI / 24.0
_RESAMPLING_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class ParticleFilterConfig:
    """Settings matching the MATLAB full-log trial defaults."""

    n_particles: int = 500
    q: tuple[float, float, float, float, float] = (
        0.025,
        0.025,
        0.005,
        0.05,
        0.05,
    )
    r: float = 0.15
    seed: int = 1
    start_after_hours: float = 72.0
    observation_smoothing_samples: int = 1440
    resample_threshold_fraction: float = 0.25
    store_history: bool = False


@dataclass(slots=True)
class ParticleFilterResult:
    """Posterior summaries and optional particle-level histories."""

    time_hours: FloatArray
    light: FloatArray
    observed_phase_rad: FloatArray
    smoothed_phase_rad: FloatArray
    state_mean: FloatArray
    state_sd: FloatArray
    mean_state_phase_rad: FloatArray
    mean_state_theta_rad: FloatArray
    circular_phase_mean_rad: FloatArray
    phase_resultant: FloatArray
    effective_sample_size: FloatArray
    max_weight: FloatArray
    zero_weight_count: NDArray[np.int64]
    weight_entropy: FloatArray
    log_weight_normalizer: FloatArray
    resampled: BoolArray
    unique_resampled_states: FloatArray
    prior_residual_variance_rad2: FloatArray
    posterior_residual_variance_rad2: FloatArray
    cumulative_r_sd_rad: FloatArray
    r_estimate_rad: float
    r_estimated_variance_rad2: float
    r_observation_count: int
    resampling_events: int
    events_per_hour: float
    mean_neff_fraction: float
    final_particles: FloatArray
    final_weights: FloatArray
    history: dict[str, FloatArray] | None


def wrap_to_2pi(values: FloatArray) -> FloatArray:
    """Wrap angles to [0, 2*pi)."""
    return np.mod(values, _TWO_PI)


def wrap_to_pi(values: FloatArray) -> FloatArray:
    """Wrap angular differences to [-pi, pi], preserving positive pi."""
    return np.arctan2(np.sin(values), np.cos(values))


def _centered_moving_mean(values: FloatArray, window: int) -> FloatArray:
    """Match MATLAB movmean's centered, shortened-window edge behavior."""
    count = values.size
    left = window // 2
    right = window - left
    starts = np.maximum(0, np.arange(count) - left)
    stops = np.minimum(count, np.arange(count) + right)
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return (cumulative[stops] - cumulative[starts]) / (stops - starts)


def phase_observations(
    time_hours: FloatArray,
    theta_rad: FloatArray,
    smoothing_samples: int,
) -> tuple[FloatArray, FloatArray]:
    """Convert theta to wrapped phase and smooth each contiguous finite segment."""
    if smoothing_samples < 1:
        raise ValueError("smoothing_samples must be a positive integer")
    if int(smoothing_samples) != smoothing_samples:
        raise ValueError("smoothing_samples must be a positive integer")

    observed = np.full(theta_rad.shape, np.nan, dtype=np.float64)
    smoothed = np.full(theta_rad.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(theta_rad)
    edges = np.diff(np.concatenate(([False], finite, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    for start, end in zip(starts, ends, strict=True):
        unwrapped_phi = np.unwrap(theta_rad[start:end]) - _OMEGA * time_hours[start:end]
        observed[start:end] = wrap_to_2pi(unwrapped_phi)
        smoothed[start:end] = wrap_to_2pi(
            _centered_moving_mean(
                np.unwrap(observed[start:end]),
                int(smoothing_samples),
            )
        )
    return observed, smoothed


def jewett99_rk4_step(
    states: FloatArray,
    light: FloatArray,
    dt_hours: float,
    tau: FloatArray,
) -> FloatArray:
    """Advance the Jewett-99 oscillator one fixed RK4 step per particle."""
    states = np.asarray(states, dtype=np.float64)
    light = np.asarray(light, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    return jewett99_rk4_particles(states, light, dt_hours, tau)


def _filterpy_resample(weights: FloatArray, rng: np.random.Generator) -> NDArray[np.int64]:
    """Use FilterPy's systematic resampler without changing NumPy's global RNG."""
    local_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    with _RESAMPLING_LOCK:
        global_state = np.random.get_state()
        try:
            np.random.seed(local_seed)
            return np.asarray(systematic_resample(weights), dtype=np.int64)
        finally:
            np.random.set_state(global_state)


def _weighted_mean_and_sd(
    states: FloatArray,
    weights: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    mean = np.sum(weights[:, None] * states, axis=0)
    delta_complex = np.sum(weights * np.exp(1j * states[:, 3]))
    mean[3] = wrap_to_2pi(np.asarray(np.arctan2(
        np.sum(weights * np.sin(states[:, 3])),
        np.sum(weights * np.cos(states[:, 3])),
    )))

    linear_columns = np.array([0, 1, 2, 4])
    centered = states[:, linear_columns] - mean[linear_columns]
    sd = np.zeros(5, dtype=np.float64)
    sd[linear_columns] = np.sqrt(np.sum(weights[:, None] * centered**2, axis=0))
    sd[3] = np.sqrt(max(0.0, -2.0 * np.log(min(1.0, abs(delta_complex)))))
    return mean, sd


def run_particle_filter(
    time_hours: FloatArray,
    light: FloatArray,
    theta_rad: FloatArray,
    config: ParticleFilterConfig | None = None,
) -> ParticleFilterResult:
    """Run the phase particle filter on minute/hourly aligned vectors.

    ``theta_rad`` is the preprocessed phase angle from the MATLAB KS stage;
    this function performs the MATLAB theta-to-phi conversion and smoothing.
    """
    config = config or ParticleFilterConfig()
    time_hours = np.asarray(time_hours, dtype=np.float64).reshape(-1)
    light = np.asarray(light, dtype=np.float64).reshape(-1)
    theta_rad = np.asarray(theta_rad, dtype=np.float64).reshape(-1)

    if len(time_hours) < 2 or len(time_hours) != len(light) or len(time_hours) != len(theta_rad):
        raise ValueError("time_hours, light, and theta_rad must have equal length >= 2")
    if not np.all(np.isfinite(time_hours)) or not np.all(np.diff(time_hours) > 0.0):
        raise ValueError("time_hours must be finite and strictly increasing")
    if not np.all(np.isfinite(light)):
        raise ValueError("light must contain only finite values")
    if np.any(light < 0.0):
        raise ValueError("light must be nonnegative")
    if config.n_particles < 2 or int(config.n_particles) != config.n_particles:
        raise ValueError("n_particles must be an integer >= 2")
    if len(config.q) != 5 or not np.all(np.isfinite(config.q)) or np.any(np.asarray(config.q) < 0):
        raise ValueError("q must contain five finite, nonnegative standard deviations")
    if not np.isfinite(config.r) or config.r <= 0:
        raise ValueError("r must be a positive finite standard deviation")
    if not 0.0 < config.resample_threshold_fraction <= 1.0:
        raise ValueError("resample_threshold_fraction must be in (0, 1]")

    observed_full, smoothed_full = phase_observations(
        time_hours,
        theta_rad,
        config.observation_smoothing_samples,
    )
    start_reference_hour = 24.0 * np.floor(time_hours[0] / 24.0)
    requested_start_hour = start_reference_hour + config.start_after_hours
    start_index = int(np.searchsorted(time_hours, requested_start_hour, side="left"))
    if start_index >= len(time_hours):
        raise ValueError("recording ends before the requested particle-filter start")

    t = time_hours[start_index:]
    inputs = light[start_index:]
    observed = observed_full[start_index:]
    y = smoothed_full[start_index:]
    n_steps = len(t)
    n_particles = int(config.n_particles)
    threshold = config.resample_threshold_fraction * n_particles
    measurement_update = np.isfinite(y)
    if not np.any(measurement_update):
        raise ValueError("no finite phase observations after the particle-filter start")

    rng = np.random.default_rng(config.seed)
    x_min = np.array([-1.5, -1.5, 0.0, 0.0, 23.8])
    x_max = np.array([1.5, 1.5, 1.0, _TWO_PI, 24.6])
    uniform_draws = rng.random((5, n_particles)).T
    particles = x_min + (x_max - x_min) * uniform_draws
    particles[:, 4] = rng.normal(24.15, 0.2, size=n_particles)
    weights = np.full(n_particles, 1.0 / n_particles)
    q = np.asarray(config.q, dtype=np.float64)

    state_mean = np.zeros((n_steps, 5), dtype=np.float64)
    state_sd = np.zeros((n_steps, 5), dtype=np.float64)
    circular_phase_mean = np.full(n_steps, np.nan)
    phase_resultant = np.full(n_steps, np.nan)
    effective_sample_size = np.zeros(n_steps)
    max_weight = np.zeros(n_steps)
    zero_weight_count = np.zeros(n_steps, dtype=np.int64)
    weight_entropy = np.zeros(n_steps)
    log_weight_normalizer = np.full(n_steps, np.nan)
    resampled = np.zeros(n_steps, dtype=np.bool_)
    unique_resampled_states = np.full(n_steps, np.nan)
    prior_residual_variance = np.full(n_steps, np.nan)
    posterior_residual_variance = np.full(n_steps, np.nan)

    history: dict[str, FloatArray] | None = None
    if config.store_history:
        history = {
            "X_prior_history": np.empty((n_steps, n_particles, 5)),
            "X_after_history": np.empty((n_steps, n_particles, 5)),
            "weights_prior_history": np.empty((n_steps, n_particles)),
            "weights_posterior_history": np.empty((n_steps, n_particles)),
            "weights_after_history": np.empty((n_steps, n_particles)),
            "predicted_phase_history": np.empty((n_steps, n_particles)),
            "residual_history": np.full((n_steps, n_particles), np.nan),
            "log_likelihood_history": np.zeros((n_steps, n_particles)),
            "log_weight_unnormalized_history": np.empty((n_steps, n_particles)),
        }

    for step in range(n_steps):
        if step > 0:
            dt = t[step] - t[step - 1]
            particles[:, :3] = jewett99_rk4_step(
                particles[:, :3],
                np.full(n_particles, inputs[step]),
                dt,
                particles[:, 4],
            )
            particles[:, 3] = wrap_to_2pi(particles[:, 3])
        if not np.all(np.isfinite(particles)):
            raise FloatingPointError(f"nonfinite particle state at filter step {step + 1}")

        prior_weights = weights.copy()
        predicted = wrap_to_2pi(
            np.arctan2(particles[:, 0], particles[:, 1])
            - _OMEGA * t[step]
            + particles[:, 3]
        )
        residual = np.full(n_particles, np.nan)
        log_likelihood = np.zeros(n_particles)
        log_weights = np.log(weights)

        if measurement_update[step]:
            residual = wrap_to_pi(predicted - y[step])
            prior_residual_variance[step] = np.sum(weights * residual**2)
            log_likelihood = -0.5 * (residual / config.r) ** 2
            log_weights = np.log(weights) + log_likelihood
            max_log_weight = np.max(log_weights)
            unnormalized = np.exp(log_weights - max_log_weight)
            weight_sum = np.sum(unnormalized)
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                raise FloatingPointError(f"particle weights collapsed at filter step {step + 1}")
            weights = unnormalized / weight_sum
            log_weight_normalizer[step] = max_log_weight + np.log(weight_sum)
            posterior_residual_variance[step] = np.sum(weights * residual**2)

        state_mean[step], state_sd[step] = _weighted_mean_and_sd(particles, weights)
        phase_complex = np.sum(weights * np.exp(1j * predicted))
        circular_phase_mean[step] = wrap_to_2pi(np.asarray(np.angle(phase_complex)))
        phase_resultant[step] = abs(phase_complex)
        effective_sample_size[step] = 1.0 / np.sum(weights**2)
        max_weight[step] = np.max(weights)
        zero_weight_count[step] = np.count_nonzero(weights == 0.0)
        positive_weights = weights[weights > 0.0]
        weight_entropy[step] = -np.sum(positive_weights * np.log(positive_weights))

        if history is not None:
            history["X_prior_history"][step] = particles
            history["weights_prior_history"][step] = prior_weights
            history["weights_posterior_history"][step] = weights
            history["predicted_phase_history"][step] = predicted
            history["residual_history"][step] = residual
            history["log_likelihood_history"][step] = log_likelihood
            history["log_weight_unnormalized_history"][step] = log_weights

        if effective_sample_size[step] < threshold:
            resampled[step] = True
            ancestors = _filterpy_resample(weights, rng)
            particles = particles[ancestors].copy()
            unique_resampled_states[step] = np.unique(particles, axis=0).shape[0]
            weights = np.full(n_particles, 1.0 / n_particles)
            particles += rng.normal(0.0, q, size=(n_particles, 5))
            particles[:, 4] = np.clip(particles[:, 4], x_min[4], x_max[4])

        if history is not None:
            history["X_after_history"][step] = particles
            history["weights_after_history"][step] = weights

    mean_state_phase = wrap_to_2pi(
        np.arctan2(state_mean[:, 0], state_mean[:, 1])
        - _OMEGA * t
        + state_mean[:, 3]
    )
    mean_state_theta = wrap_to_2pi(mean_state_phase + _OMEGA * t)

    valid = np.isfinite(posterior_residual_variance)
    r_estimate = (
        float(np.sqrt(np.mean(posterior_residual_variance[valid])))
        if np.any(valid)
        else float("nan")
    )
    valid_values = np.where(valid, posterior_residual_variance, 0.0)
    counts = np.cumsum(valid)
    cumulative_variance = np.divide(
        np.cumsum(valid_values),
        counts,
        out=np.full(n_steps, np.nan),
        where=counts > 0,
    )
    cumulative_r_sd = np.sqrt(cumulative_variance)
    elapsed_hours = max(float(t[-1] - t[0]), np.finfo(np.float64).eps)

    return ParticleFilterResult(
        time_hours=t,
        light=inputs,
        observed_phase_rad=observed,
        smoothed_phase_rad=y,
        state_mean=state_mean,
        state_sd=state_sd,
        mean_state_phase_rad=mean_state_phase,
        mean_state_theta_rad=mean_state_theta,
        circular_phase_mean_rad=circular_phase_mean,
        phase_resultant=phase_resultant,
        effective_sample_size=effective_sample_size,
        max_weight=max_weight,
        zero_weight_count=zero_weight_count,
        weight_entropy=weight_entropy,
        log_weight_normalizer=log_weight_normalizer,
        resampled=resampled,
        unique_resampled_states=unique_resampled_states,
        prior_residual_variance_rad2=prior_residual_variance,
        posterior_residual_variance_rad2=posterior_residual_variance,
        cumulative_r_sd_rad=cumulative_r_sd,
        r_estimate_rad=r_estimate,
        r_estimated_variance_rad2=(
            float(np.mean(posterior_residual_variance[valid]))
            if np.any(valid)
            else float("nan")
        ),
        r_observation_count=int(np.sum(valid)),
        resampling_events=int(np.sum(resampled)),
        events_per_hour=float(np.sum(resampled) / elapsed_hours),
        mean_neff_fraction=float(np.mean(effective_sample_size) / n_particles),
        final_particles=particles.copy(),
        final_weights=weights.copy(),
        history=history,
    )


def _read_input_csv(path: Path) -> tuple[FloatArray, FloatArray, FloatArray]:
    required = ("time_hours", "light", "theta_rad")
    with path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or not all(name in reader.fieldnames for name in required):
            raise ValueError(
                "input CSV must contain columns: time_hours, light, theta_rad"
            )
        rows = list(reader)
    try:
        time = np.asarray([float(row["time_hours"]) for row in rows], dtype=np.float64)
        light = np.asarray([float(row["light"]) for row in rows], dtype=np.float64)
        theta = np.asarray(
            [
                float(value) if value is not None and value.strip() else np.nan
                for value in (row["theta_rad"] for row in rows)
            ],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("input CSV columns must contain numeric values") from error
    return time, light, theta


def _write_estimates(path: Path, result: ParticleFilterResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            (
                "time_hours",
                "light",
                "observed_phase_rad",
                "smoothed_phase_rad",
                "mean_state_phase_rad",
                "mean_state_theta_rad",
                "x_mean",
                "xc_mean",
                "n_mean",
                "delta_mean_rad",
                "tau_mean_hours",
                "ess",
                "resampled",
                "posterior_residual_variance_rad2",
                "cumulative_r_sd_rad",
            )
        )
        for index, time in enumerate(result.time_hours):
            writer.writerow(
                (
                    time,
                    result.light[index],
                    result.observed_phase_rad[index],
                    result.smoothed_phase_rad[index],
                    result.mean_state_phase_rad[index],
                    result.mean_state_theta_rad[index],
                    *result.state_mean[index],
                    result.effective_sample_size[index],
                    bool(result.resampled[index]),
                    result.posterior_residual_variance_rad2[index],
                    result.cumulative_r_sd_rad[index],
                )
            )


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="CSV with time_hours, light, theta_rad")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start-after-hours", type=float, default=72.0)
    parser.add_argument("--smoothing-samples", type=int, default=1440)
    parser.add_argument("--r", type=float, default=0.15)
    parser.add_argument(
        "--q",
        type=float,
        nargs=5,
        default=(0.025, 0.025, 0.005, 0.05, 0.05),
        metavar=("X", "XC", "N", "DELTA", "TAU"),
    )
    parser.add_argument(
        "--store-history",
        action="store_true",
        help="also write the full particle-level .npz history (large files)",
    )
    args = parser.parse_args()

    time, light, theta = _read_input_csv(args.input_csv)
    config = ParticleFilterConfig(
        n_particles=args.particles,
        q=tuple(args.q),
        r=args.r,
        seed=args.seed,
        start_after_hours=args.start_after_hours,
        observation_smoothing_samples=args.smoothing_samples,
        store_history=args.store_history,
    )
    result = run_particle_filter(time, light, theta, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_estimates(args.output_dir / "estimates.csv", result)
    if result.history is not None:
        np.savez_compressed(
            args.output_dir / "particle_filter_history.npz",
            **result.history,
            time_hours=result.time_hours,
        )
    print(f"Processed {len(result.time_hours)} samples.")
    print(f"Resampling events: {result.resampling_events}")
    print(f"R residual diagnostic (rad): {result.r_estimate_rad:.6g}")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    _cli()
