"""Python counterparts for the static methods in the MATLAB Utils class."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from circadian_models import calculate_theta, calculate_continuous_phase
from data_io import load_optimal_gains, load_real_data, load_synth_data
from kalman_smoothing import compute_cost, compute_spectrum, find_daily_min


class Utils:
    loadRealData = staticmethod(load_real_data)
    loadSynthData = staticmethod(load_synth_data)
    loadOptimalGains = staticmethod(load_optimal_gains)
    computeSpectrum = staticmethod(compute_spectrum)
    computeCost = staticmethod(compute_cost)
    calculateTheta = staticmethod(calculate_theta)
    calculateContinuousPhase = staticmethod(calculate_continuous_phase)
    find_daily_min = staticmethod(find_daily_min)

    @staticmethod
    def estimatePhaseShift(
        x_hat: NDArray[np.float64],
        omega: float,
        day1: int,
        day2: int,
    ) -> tuple[NDArray[np.float64], float]:
        return _estimate_phase_shift(x_hat, omega, day1, day2)

    @staticmethod
    def estimateCBTPhaseShift(time: NDArray[np.float64], x: NDArray[np.float64], day1: int, day2: int) -> tuple[NDArray[np.float64], float]:
        from kalman_smoothing import estimate_cbt_phase_shift

        return estimate_cbt_phase_shift(time, x, day1, day2)

    @staticmethod
    def plotAvgCostEvolution(avg_cost: NDArray[np.float64]) -> None:
        plt.figure(1)
        plt.clf()
        plt.semilogy(avg_cost)
        plt.title("Evolution of Average Population Cost")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.grid(True)

    @staticmethod
    def plotFilterOutputs(
        time: NDArray[np.float64],
        y: NDArray[np.float64],
        y_hat: NDArray[np.float64],
        x_hat: NDArray[np.float64],
        theta: NDArray[np.float64],
        day1: int,
        day2: int,
        label: str,
    ) -> None:
        t = np.asarray(time)
        states = np.asarray(x_hat)
        figure, axes = plt.subplots(5, 1, num=2, clear=True, figsize=(11, 12))
        axes[0].plot(t, y, "b-.", label="y")
        axes[0].plot(t, y_hat, "r-.", linewidth=2, label="yHat")
        axes[0].set_title(f"Filter Output vs {label}")
        axes[0].set_ylabel(f"{label} Magnitude")
        axes[0].legend()
        phase = np.unwrap(np.asarray(theta)) - (2 * np.pi / 24) * t
        baseline = (t >= (day1 - 1) * 24) & (t <= day1 * 24)
        axes[1].plot(t, phase * 24 / (2 * np.pi) - np.mean(phase[baseline]))
        for day in (day1 - 1, day1, day2 - 1, day2):
            axes[1].axvline(day * 24, linewidth=1)
        axes[1].set_title(f"Phase Offset Relative to Day {day1}")
        for index, axis in enumerate(axes[2:]):
            axis.plot(t, states[index] if index < states.shape[0] else states[-1])
            axis.set_ylabel("State")
        figure.tight_layout()

    @staticmethod
    def plotSpectra(frequency: NDArray[np.float64], original: NDArray[np.float64], filtered: NDArray[np.float64], label: str) -> None:
        plt.plot(frequency, original, label="Original Spectrum")
        plt.plot(frequency, filtered, label="Filtered Spectrum")
        plt.legend()
        plt.title(f"Original and Filtered Spectra of {label}")
        plt.xlim(0, 1)

    @staticmethod
    def plotPhaseShiftBoxPlot(results: NDArray[np.float64], time_shift: NDArray[np.float64], subject: int, dlmo_subject: int, filter_name: str) -> None:
        selected = np.asarray(results)[np.asarray(results)[:, 0] == subject, 3]
        plt.boxplot(selected)
        _draw_reference_lines(time_shift, dlmo_subject)
        plt.ylim(-5, 5)
        plt.title(f"{filter_name} Phase Shift Estimate for Subject {subject}")
        plt.ylabel("Phase Shift (h)")

    @staticmethod
    def plotPhaseShiftHistogram(results: NDArray[np.float64], time_shift: NDArray[np.float64], subject: int, dlmo_subject: int, filter_name: str) -> None:
        selected = np.asarray(results)[np.asarray(results)[:, 0] == subject, 3]
        plt.hist(selected, bins=24, label="OBF Estimates")
        _draw_reference_lines(time_shift, dlmo_subject, vertical=True)
        plt.title(f"{filter_name} Phase Shift Estimates for Subject {subject}")
        plt.xlabel("Phase Shift (h)")
        plt.ylabel("Count")
        plt.xlim(-3, 3)
        plt.legend()

    @staticmethod
    def plotCosts(results: NDArray[np.float64], subject: int, filter_name: str) -> None:
        values = np.asarray(results)
        plt.hist(values[values[:, 0] == subject, 5], bins=24)
        plt.title(f"{filter_name} Optimal Cost for Subject {subject}")
        plt.ylabel("Count")
        plt.xlabel("Optimal Costs")
        plt.xlim(9e4, 2e6)

    @staticmethod
    def plotRuntimes(results: NDArray[np.float64], subject: int, filter_name: str) -> None:
        values = np.asarray(results)
        plt.boxplot(values[values[:, 0] == subject, 4])
        plt.title(f"{filter_name} Runtimes for Subject {subject}")
        plt.ylabel("Runtime (s)")
        plt.ylim(0, 100)


def _draw_reference_lines(time_shift: NDArray[np.float64], subject: int, vertical: bool = False) -> None:
    colors = ("r", "k", "c")
    labels = ("Lewy", "Eastman", "Voultsios")
    for index, (color, label) in enumerate(zip(colors, labels, strict=True)):
        value = time_shift[subject, index]
        if np.isfinite(value):
            if vertical:
                plt.axvline(value, linestyle="-.", color=color, label=label)
            else:
                plt.axhline(value, linestyle="-.", color=color, label=label)


def _estimate_phase_shift(
    x_hat: NDArray[np.float64],
    omega: float,
    day1: int,
    day2: int,
) -> tuple[NDArray[np.float64], float]:
    states = np.asarray(x_hat, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] < 2 or omega == 0:
        raise ValueError("x_hat must contain two state rows and omega must be nonzero")
    theta = np.mod(-np.arctan2(states[1], omega * states[0]) + np.pi / 2, 2 * np.pi) - np.pi
    first = slice((day1 - 1) * 1440, day1 * 1440)
    second = slice((day2 - 1) * 1440, day2 * 1440)
    theta_first = np.unwrap(theta[first])
    theta_second = np.unwrap(theta[second])
    if not theta_first.size or not theta_second.size:
        raise ValueError("x_hat does not span the requested days")
    count = min(theta_first.size, theta_second.size)
    shift = float(np.mean(theta_first[:count] - theta_second[:count]) / omega)
    shift = float(np.mod(12 + shift, 24) - 12)
    return theta, shift
