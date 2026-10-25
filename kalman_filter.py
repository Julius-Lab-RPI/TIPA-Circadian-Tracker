"""Forward and steady-state Kalman filter ports for the tPBM MATLAB project."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from kalman_smoothing import (
    _solve_small_system,
    _trapezoid_unit,
    create_kalman_state_space,
    initialize_population,
)

FloatArray = NDArray[np.float64]


def simulate_kalman_filter(
    a: FloatArray,
    c: FloatArray,
    q: FloatArray,
    r: float,
    observations: FloatArray,
    *,
    initial_covariance: float = 1e3,
    initial_state: FloatArray | None = None,
    full_log: bool = False,
) -> tuple[FloatArray, FloatArray, dict[str, FloatArray] | None]:
    """Run the same forward KF initialization and measurement update as KF pipeline."""
    transition = np.asarray(a, dtype=np.float64)
    observation = np.asarray(c, dtype=np.float64).reshape(-1)
    process_covariance = np.asarray(q, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64).reshape(-1)
    dimension = transition.shape[0]
    if transition.shape != (dimension, dimension) or observation.size != dimension:
        raise ValueError("A and C dimensions are incompatible")
    if process_covariance.shape != (dimension, dimension):
        raise ValueError("Q must match the state dimension")
    if not np.isfinite(r) or r <= 0 or initial_covariance <= 0:
        raise ValueError("R and initial_covariance must be positive and finite")
    if not y.size:
        raise ValueError("observations cannot be empty")
    state = (
        np.zeros(dimension)
        if initial_state is None
        else np.asarray(initial_state, dtype=np.float64).reshape(-1).copy()
    )
    if state.size != dimension:
        raise ValueError("initial_state must match the state dimension")
    covariance = initial_covariance * np.eye(dimension)
    states = np.empty((dimension, len(y)), dtype=np.float64)
    logs: dict[str, FloatArray] | None = None
    if full_log:
        logs = {
            "x_prior": np.zeros((dimension, len(y))),
            "P_prior": np.zeros((dimension, dimension, len(y))),
            "P_post": np.zeros((dimension, dimension, len(y))),
            "gain": np.zeros((dimension, len(y))),
            "innovation": np.full(len(y), np.nan),
            "innovation_variance": np.full(len(y), np.nan),
            "measurement_used": np.isfinite(y),
        }

    for index, sample in enumerate(y):
        if index:
            state = np.einsum("ij,j->i", transition, state)
            propagated = np.einsum("ij,jk->ik", transition, covariance)
            covariance = np.einsum("ij,kj->ik", propagated, transition) + process_covariance
        if logs is not None:
            logs["x_prior"][:, index] = state
            logs["P_prior"][:, :, index] = covariance
        if np.isfinite(sample):
            innovation = sample - np.sum(observation * state)
            innovation_variance = (
                np.einsum("i,ij,j->", observation, covariance, observation) + r
            )
            gain = np.einsum("ij,j->i", covariance, observation) / innovation_variance
            state = state + gain * innovation
            correction = np.eye(dimension) - np.einsum("i,j->ij", gain, observation)
            covariance = np.einsum("ij,jk->ik", correction, covariance)
            if logs is not None:
                logs["gain"][:, index] = gain
                logs["innovation"][index] = innovation
                logs["innovation_variance"][index] = innovation_variance
        states[:, index] = state
        if logs is not None:
            logs["P_post"][:, :, index] = covariance

    filtered = np.einsum("i,ij->j", observation, states)
    return states, filtered, logs


def simulate_matlab_kf(
    a: FloatArray,
    c: FloatArray,
    q: FloatArray,
    r: float,
    observations: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Match the separate KF.simulateKalmanFilter class initialization."""
    transition = np.asarray(a, dtype=np.float64)
    observation = np.asarray(c, dtype=np.float64).reshape(-1)
    process_covariance = np.asarray(q, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64).reshape(-1)
    dimension = transition.shape[0]
    if (
        transition.shape != (dimension, dimension)
        or observation.size != dimension
        or process_covariance.shape != (dimension, dimension)
    ):
        raise ValueError("A, C, and Q dimensions are incompatible")
    if not np.isfinite(r) or r <= 0:
        raise ValueError("R must be a positive finite variance")
    if not y.size:
        raise ValueError("observations cannot be empty")
    state = np.zeros(dimension)
    state[-1] = 1500.0
    covariance = 0.01 * np.eye(dimension)
    states = np.zeros((dimension, len(y)))
    states[-1, 0] = 1500.0
    identity = np.eye(dimension)
    for index in range(1, len(y)):
        prior_state = np.einsum("ij,j->i", transition, state)
        prior_covariance = (
            np.einsum(
                "ij,jk,lk->il",
                transition,
                covariance,
                transition,
            )
            + process_covariance
        )
        if np.isfinite(y[index]):
            innovation = y[index] - np.sum(observation * prior_state)
            innovation_variance = (
                np.einsum("i,ij,j->", observation, prior_covariance, observation) + r
            )
            gain = np.einsum("ij,j->i", prior_covariance, observation) / innovation_variance
            state = prior_state + gain * innovation
            covariance = np.einsum(
                "ij,jk->ik",
                identity - np.outer(gain, observation),
                prior_covariance,
            )
        else:
            state, covariance = prior_state, prior_covariance
        states[:, index] = state
    return states, np.einsum("i,ij->j", observation, states)


def _evaluate_kf_population(
    q_vectors: FloatArray,
    r_values: FloatArray,
    a: FloatArray,
    c: FloatArray,
    observations: FloatArray,
    time: FloatArray,
    order: int,
    batch_size: int = 64,
) -> FloatArray:
    vectors = np.asarray(q_vectors, dtype=np.float64)
    variances = np.asarray(r_values, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64)
    count = len(y)
    dimension = a.shape[0]
    if vectors.ndim != 2 or len(vectors) != len(variances):
        raise ValueError("Q and R candidates must have matching lengths")
    if count != len(time) or vectors.shape[1] != dimension**2 or batch_size < 1:
        raise ValueError("invalid input lengths, Q dimension, or batch size")
    observation = np.asarray(c, dtype=np.float64).reshape(-1)
    identity = np.eye(dimension)
    missing = ~np.isfinite(y)
    filled_y = np.where(missing, 0.0, y)
    frequencies = np.fft.rfftfreq(count, d=float(time[1] - time[0]))
    harmonic_indices = np.array(
        [np.argmin(np.abs(frequencies - harmonic / 24.0)) for harmonic in range(1, 6)]
    )
    band_width = int(
        harmonic_indices[0] - np.argmin(np.abs(frequencies - 0.0309))
    )
    if band_width <= 0:
        raise ValueError("signal is too short for the requested harmonic cost")
    costs = np.empty(len(vectors), dtype=np.float64)

    for batch_start in range(0, len(vectors), batch_size):
        batch_stop = min(batch_start + batch_size, len(vectors))
        batch_count = batch_stop - batch_start
        raw_q = vectors[batch_start:batch_stop].reshape(
            (-1, dimension, dimension)
        ).transpose(0, 2, 1)
        q = np.einsum("bij,bkj->bik", raw_q, raw_q)
        r = variances[batch_start:batch_stop]
        state = np.zeros((batch_count, dimension))
        covariance = np.broadcast_to(
            1e3 * identity, (batch_count, dimension, dimension)
        ).copy()
        filtered = np.empty((batch_count, count))
        for step in range(count):
            if step:
                state = np.einsum("ij,bj->bi", a, state)
                propagated = np.einsum("ij,bjk,lk->bil", a, covariance, a)
                covariance = propagated + q
            if np.isfinite(y[step]):
                innovation_variance = (
                    np.einsum("i,bij,j->b", observation, covariance, observation) + r
                )
                gain = np.einsum("bij,j->bi", covariance, observation) / innovation_variance[:, None]
                innovation = filled_y[step] - np.einsum("i,bi->b", observation, state)
                state += gain * innovation[:, None]
                correction = identity[None] - np.einsum("bi,j->bij", gain, observation)
                covariance = np.einsum("bij,bjk->bik", correction, covariance)
            filtered[:, step] = np.einsum("i,bi->b", observation, state)

        reconstructed = np.where(missing[None, :], filtered, y[None, :])
        original_spectrum = np.abs(np.fft.rfft(reconstructed, axis=1) / count)
        filtered_spectrum = np.abs(np.fft.rfft(filtered, axis=1) / count)
        if count % 2 == 0:
            original_spectrum[:, 1:-1] *= 2
            filtered_spectrum[:, 1:-1] *= 2
        else:
            original_spectrum[:, 1:-1] *= 2
            filtered_spectrum[:, 1:-1] *= 2
        harmonic_cost = _trapezoid_unit(
            (filtered_spectrum[:, :band_width] - original_spectrum[:, :band_width]) ** 2
        )
        noise_cost = _trapezoid_unit(
            filtered_spectrum[
                :, band_width : int(harmonic_indices[0] - band_width + 1)
            ]
            ** 2
        )
        for harmonic in range(order):
            index = int(harmonic_indices[harmonic])
            low = max(0, index - band_width)
            high = min(filtered_spectrum.shape[1], index + band_width + 1)
            harmonic_cost += _trapezoid_unit(
                (filtered_spectrum[:, low:high] - original_spectrum[:, low:high]) ** 2
            )
            if harmonic + 1 < order:
                next_index = int(harmonic_indices[harmonic + 1])
                noise_cost += _trapezoid_unit(
                    filtered_spectrum[:, index + band_width + 1 : next_index - band_width] ** 2
                )
        noise_cost += _trapezoid_unit(
            filtered_spectrum[
                :, int(harmonic_indices[order - 1]) + band_width + 1 :
            ]
            ** 2
        )
        costs[batch_start:batch_stop] = harmonic_cost + noise_cost
    return costs


def optimize_kf_parameters(
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
    """Genetic Q/R optimization for the forward KF, matching the MATLAB KF script."""
    if iterations < 0 or mu < 2 or lam < 1 or rho < 1 or rho > min(50, mu):
        raise ValueError("invalid optimization population or iteration settings")
    rng = np.random.default_rng(seed)
    dimension = np.asarray(a).shape[0]
    q_pop, r_pop = initialize_population(
        dimension, mu, l_end, r_end, lower_bound, r_lower, r_upper, rng
    )
    y = np.asarray(observations, dtype=np.float64).reshape(-1)
    t = np.asarray(time, dtype=np.float64).reshape(-1)
    combinations_pool = list(combinations(range(min(50, mu)), rho))
    if lam > len(combinations_pool):
        raise ValueError("lambda exceeds the number of parent combinations")
    costs = _evaluate_kf_population(q_pop, r_pop, a, c, y, t, order)
    average_cost = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        labels = rng.permutation(len(combinations_pool))[:lam]
        children_q = np.empty((lam, dimension**2))
        children_r = np.empty(lam)
        for child, label in enumerate(labels):
            parents = combinations_pool[label]
            children_q[child] = np.mean(q_pop[list(parents)], axis=0)
            children_r[child] = np.mean(r_pop[list(parents)])
        children_cost = _evaluate_kf_population(
            children_q, children_r, a, c, y, t, order
        )
        q_pop = np.vstack((q_pop, children_q))
        r_pop = np.r_[r_pop, children_r]
        costs = np.r_[costs, children_cost]
        nan_indices = np.flatnonzero(np.isnan(costs))
        non_nan = np.flatnonzero(~np.isnan(costs))
        worst = non_nan[np.argsort(costs[non_nan], kind="stable")[::-1]]
        remove = np.r_[nan_indices, worst][:lam]
        keep_mask = np.ones(len(costs), dtype=np.bool_)
        keep_mask[remove] = False
        keep = np.flatnonzero(keep_mask)
        q_pop, r_pop, costs = q_pop[keep], r_pop[keep], costs[keep]
        average_cost[iteration] = np.nanmean(costs)
    return costs, average_cost, q_pop, r_pop


def optimize_class_kf(
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
    frequencies: FloatArray,
    original_spectrum: FloatArray,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """MATLAB KF.optimizeFilter API, including its 1500-state initialization."""
    from kalman_smoothing import compute_cost, compute_spectrum

    if iterations < 0 or mu < 2 or lam < 1 or rho < 1 or rho > min(50, mu):
        raise ValueError("invalid optimization population or iteration settings")
    rng = np.random.default_rng(seed)
    dimension = np.asarray(a).shape[0]
    q_pop, r_pop = initialize_population(
        dimension, mu, l_end, r_end, lower_bound, r_lower, r_upper, rng
    )
    y = np.asarray(observations, dtype=np.float64).reshape(-1)
    t = np.asarray(time, dtype=np.float64).reshape(-1)
    f = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    spectrum = np.asarray(original_spectrum, dtype=np.float64).reshape(-1)

    def evaluate(q_candidates: FloatArray, r_candidates: FloatArray) -> FloatArray:
        result = np.empty(len(q_candidates))
        for index, (vector, variance) in enumerate(zip(q_candidates, r_candidates)):
            factor = vector.reshape((dimension, dimension), order="F")
            q = np.einsum("ij,kj->ik", factor, factor)
            _, filtered = simulate_matlab_kf(a, c, q, float(variance), y)
            filtered_spectrum, _, _ = compute_spectrum(t, filtered)
            result[index] = compute_cost(
                spectrum, filtered_spectrum, f, order
            )
        return result

    combinations_pool = list(combinations(range(min(50, mu)), rho))
    if lam > len(combinations_pool):
        raise ValueError("lambda exceeds the number of parent combinations")
    costs = evaluate(q_pop, r_pop)
    average_cost = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        labels = rng.permutation(len(combinations_pool))[:lam]
        children_q = np.empty((lam, dimension**2))
        children_r = np.empty(lam)
        for child, label in enumerate(labels):
            parents = combinations_pool[label]
            children_q[child] = np.mean(q_pop[list(parents)], axis=0)
            children_r[child] = np.mean(r_pop[list(parents)])
        child_costs = evaluate(children_q, children_r)
        q_pop = np.vstack((q_pop, children_q))
        r_pop = np.r_[r_pop, children_r]
        costs = np.r_[costs, child_costs]
        nan_indices = np.flatnonzero(np.isnan(costs))
        non_nan = np.flatnonzero(~np.isnan(costs))
        worst = non_nan[np.argsort(costs[non_nan], kind="stable")[::-1]]
        remove = np.r_[nan_indices, worst][:lam]
        keep_mask = np.ones(len(costs), dtype=np.bool_)
        keep_mask[remove] = False
        keep = np.flatnonzero(keep_mask)
        q_pop, r_pop, costs = q_pop[keep], r_pop[keep], costs[keep]
        average_cost[iteration] = np.nanmean(costs)
    return costs, average_cost, q_pop, r_pop


def _steady_state_objective(
    q_vectors: FloatArray,
    r_values: FloatArray,
    a: FloatArray,
    c: FloatArray,
    time: FloatArray,
    observations: FloatArray,
    frequencies: FloatArray,
    original_spectrum: FloatArray,
    order: int,
    *,
    normalized: bool,
) -> FloatArray:
    from kalman_smoothing import compute_cost, compute_spectrum

    result = np.empty(len(q_vectors), dtype=np.float64)
    for index, (vector, variance) in enumerate(zip(q_vectors, r_values)):
        factor = vector.reshape((a.shape[0], a.shape[0]), order="F")
        q = np.einsum("ij,kj->ik", factor, factor)
        gain = KFSS.steady_state_gain(a, c, q, float(variance))
        states, filtered = simulate_steady_state_filter(
            a, c, gain, observations, zero_is_missing=not normalized
        )
        del states
        filtered_spectrum, _, _ = compute_spectrum(time, filtered)
        result[index] = compute_cost(
            original_spectrum, filtered_spectrum, frequencies, order
        )
    return result


def _optimize_steady_state(
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
    frequencies: FloatArray,
    original_spectrum: FloatArray,
    *,
    normalized: bool,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    if iterations < 0 or mu < 2 or lam < 1 or rho < 1 or rho > min(50, mu):
        raise ValueError("invalid optimization population or iteration settings")
    rng = np.random.default_rng(seed)
    dimension = np.asarray(a).shape[0]
    q_pop, r_pop = initialize_population(
        dimension, mu, l_end, r_end, lower_bound, r_lower, r_upper, rng
    )
    y = np.asarray(observations, dtype=np.float64).reshape(-1)
    t = np.asarray(time, dtype=np.float64).reshape(-1)
    f = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    spectrum = np.asarray(original_spectrum, dtype=np.float64).reshape(-1)
    combinations_pool = list(combinations(range(min(50, mu)), rho))
    if lam > len(combinations_pool):
        raise ValueError("lambda exceeds the number of parent combinations")
    costs = _steady_state_objective(
        q_pop, r_pop, a, c, t, y, f, spectrum, order, normalized=normalized
    )
    average_cost = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        labels = rng.permutation(len(combinations_pool))[:lam]
        children_q = np.empty((lam, dimension**2))
        children_r = np.empty(lam)
        for child, label in enumerate(labels):
            parents = combinations_pool[label]
            children_q[child] = np.mean(q_pop[list(parents)], axis=0)
            children_r[child] = np.mean(r_pop[list(parents)])
        child_costs = _steady_state_objective(
            children_q,
            children_r,
            a,
            c,
            t,
            y,
            f,
            spectrum,
            order,
            normalized=normalized,
        )
        q_pop = np.vstack((q_pop, children_q))
        r_pop = np.r_[r_pop, children_r]
        costs = np.r_[costs, child_costs]
        nan_indices = np.flatnonzero(np.isnan(costs))
        non_nan = np.flatnonzero(~np.isnan(costs))
        worst = non_nan[np.argsort(costs[non_nan], kind="stable")[::-1]]
        remove = np.r_[nan_indices, worst][:lam]
        keep_mask = np.ones(len(costs), dtype=np.bool_)
        keep_mask[remove] = False
        keep = np.flatnonzero(keep_mask)
        q_pop, r_pop, costs = q_pop[keep], r_pop[keep], costs[keep]
        average_cost[iteration] = np.nanmean(costs)
    return costs, average_cost, q_pop, r_pop


def simulate_steady_state_filter(
    a: FloatArray,
    c: FloatArray,
    gain: FloatArray,
    observations: FloatArray,
    *,
    zero_is_missing: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """MATLAB KFSS.simulateFilter / simulateFilterNorm recursion."""
    transition = np.asarray(a, dtype=np.float64)
    observation = np.asarray(c, dtype=np.float64).reshape(-1)
    kalman_gain = np.asarray(gain, dtype=np.float64).reshape(-1)
    y = np.asarray(observations, dtype=np.float64).reshape(-1)
    dimension = transition.shape[0]
    if transition.shape != (dimension, dimension):
        raise ValueError("A must be square")
    if observation.size != dimension or kalman_gain.size != dimension:
        raise ValueError("A, C, and L dimensions are incompatible")
    states = np.zeros((dimension, len(y)))
    for step in range(1, len(y)):
        prior = np.einsum("ij,j->i", transition, states[:, step - 1])
        if np.isfinite(y[step - 1]) and not (zero_is_missing and y[step - 1] == 0):
            states[:, step] = (
                np.einsum("ij,j->i", transition - np.outer(kalman_gain, observation), states[:, step - 1])
                + kalman_gain * y[step - 1]
            )
        else:
            states[:, step] = prior
    return states, np.einsum("i,ij->j", observation, states)


def optimize_steady_state_filter(
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
    frequencies: FloatArray,
    original_spectrum: FloatArray,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    return _optimize_steady_state(
        iterations,
        mu,
        lam,
        rho,
        order,
        l_end,
        r_end,
        lower_bound,
        r_lower,
        r_upper,
        a,
        c,
        time,
        observations,
        frequencies,
        original_spectrum,
        normalized=False,
        seed=seed,
    )


def optimize_steady_state_filter_norm(
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
    frequencies: FloatArray,
    original_spectrum: FloatArray,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    return _optimize_steady_state(
        iterations,
        mu,
        lam,
        rho,
        order,
        l_end,
        r_end,
        lower_bound,
        r_lower,
        r_upper,
        a,
        c,
        time,
        observations,
        frequencies,
        original_spectrum,
        normalized=True,
        seed=seed,
    )


def simulate_steady_state_filter_raw(
    a: FloatArray,
    c: FloatArray,
    gain: FloatArray,
    observations: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    return simulate_steady_state_filter(
        a, c, gain, observations, zero_is_missing=True
    )


class KF:
    """Names matching the MATLAB ``KF`` class."""

    createKalmanStateSpace = staticmethod(create_kalman_state_space)
    initializePopulation = staticmethod(initialize_population)
    simulateKalmanFilter = staticmethod(simulate_matlab_kf)
    optimizeFilter = staticmethod(optimize_class_kf)


class KFSS:
    """Steady-state filter helpers translated from the MATLAB ``KFSS`` class."""

    simulateFilter = staticmethod(simulate_steady_state_filter_raw)
    optimizeFilter = staticmethod(optimize_steady_state_filter)
    optimizeFilterNorm = staticmethod(optimize_steady_state_filter_norm)

    @staticmethod
    def simulateFilterNorm(
        a: FloatArray,
        c: FloatArray,
        gain: FloatArray,
        observations: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        return simulate_steady_state_filter(
            a, c, gain, observations, zero_is_missing=False
        )

    @staticmethod
    def steady_state_gain(
        a: FloatArray,
        c: FloatArray,
        q: FloatArray,
        r: float,
        *,
        tolerance: float = 1e-12,
        max_iterations: int = 100_000,
    ) -> FloatArray:
        """Iterate the discrete Riccati equation to obtain the steady-state gain."""
        transition = np.asarray(a, dtype=np.float64)
        observation = np.asarray(c, dtype=np.float64).reshape(-1)
        covariance = np.asarray(q, dtype=np.float64).copy()
        if (
            transition.ndim != 2
            or transition.shape[0] != transition.shape[1]
            or observation.size != transition.shape[0]
            or covariance.shape != transition.shape
            or not np.isfinite(r)
            or r <= 0
        ):
            raise ValueError("A, C, Q, and R dimensions or values are invalid")
        identity = np.eye(transition.shape[0])
        doubling_a = transition.T.copy()
        doubling_g = np.einsum("i,j->ij", observation, observation) / r
        doubling_h = covariance
        for _ in range(max_iterations):
            left = identity + np.einsum("ij,jk->ik", doubling_g, doubling_h)
            right = identity + np.einsum("ij,jk->ik", doubling_h, doubling_g)
            a_right = _solve_small_system(left.T, doubling_a.T).T
            ag = np.einsum("ij,jk->ik", doubling_a, doubling_g)
            ag_right = _solve_small_system(right.T, ag.T).T
            ha = np.einsum("ij,jk->ik", doubling_h, doubling_a)
            h_correction = np.einsum(
                "ij,jk->ik",
                doubling_a.T,
                _solve_small_system(right, ha),
            )
            next_a = np.einsum("ij,jk->ik", a_right, doubling_a)
            next_g = doubling_g + np.einsum(
                "ij,jk->ik", ag_right, doubling_a.T
            )
            next_h = doubling_h + h_correction
            next_h = (next_h + next_h.T) / 2.0
            if np.max(np.abs(next_h - doubling_h)) <= tolerance * max(
                1.0, float(np.max(np.abs(next_h)))
            ):
                innovation_variance = (
                    np.einsum("i,ij,j->", observation, next_h, observation) + r
                )
                apc = np.einsum(
                    "ij,j->i",
                    np.einsum("ij,jk->ik", transition, next_h),
                    observation,
                )
                return apc / innovation_variance
            doubling_a, doubling_g, doubling_h = next_a, next_g, next_h
        raise RuntimeError("steady-state Riccati iteration did not converge")
