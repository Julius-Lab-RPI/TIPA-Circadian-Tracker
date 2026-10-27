import queue
import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from circadian_models import (
    activity_to_light,
    estimate_available_phase_shift,
    get_cbt_mins,
    jewett99,
    jewett99_rk4_tau,
)
from data_io import load_minute_xlsx
from dlmo import compare_dlmo_extract, compare_dlmo_sequence, compare_dlmo_targets
from kalman_smoothing import (
    compute_cost,
    compute_spectrum,
    create_kalman_state_space,
    simulate_kalman_smoother,
)
from kalman_filter import (
    KF,
    KFSS,
    _evaluate_kf_population,
    simulate_kalman_filter,
)
from run_tpbm import _save_dlmo_summary, _save_results_visualization
from run_tpbm_gui import TpbmLauncher


class TranslatedModuleTests(unittest.TestCase):
    def test_gui_elapsed_time_updates_while_running_and_on_completion(self) -> None:
        class FakeVariable:
            value = ""

            def set(self, value: str) -> None:
                self.value = value

        launcher = TpbmLauncher.__new__(TpbmLauncher)
        launcher.started_at = time.perf_counter() - 2.2
        launcher.elapsed = FakeVariable()
        launcher.messages = queue.Queue()
        launcher.root = SimpleNamespace(after=lambda *_args: None)

        launcher._poll_worker()
        self.assertEqual(launcher.elapsed.value, "Elapsed: 00:00:02")

        launcher.status = FakeVariable()
        launcher.result_directory = None
        launcher.open_results_button = SimpleNamespace(
            configure=lambda **_kwargs: None
        )
        launcher.messages.put(
            (
                "success",
                {"output_dir": tempfile.gettempdir()},
            )
        )
        launcher._set_running = lambda _running: None
        with patch("run_tpbm_gui.messagebox.showinfo"):
            launcher._poll_worker()

        self.assertEqual(launcher.elapsed.value, "Elapsed: 00:00:02")
        self.assertIsNone(launcher.started_at)

    def test_gui_detects_garmin_csv_and_selects_heart_rate(self) -> None:
        class FakeVariable:
            def __init__(self) -> None:
                self.value = ""

            def set(self, value: str) -> None:
                self.value = value

        launcher = TpbmLauncher.__new__(TpbmLauncher)
        launcher.input_path = FakeVariable()
        launcher.subject = FakeVariable()
        launcher.signal = FakeVariable()
        launcher.status = FakeVariable()
        launcher.signal_box = SimpleNamespace(
            configure=lambda **kwargs: setattr(launcher, "signal_values", kwargs["values"])
        )
        path = Path("C:/experiment/subject-13/Minute-by-minute.csv")

        launcher._configure_selected_input(path)

        self.assertEqual(launcher.input_path.value, str(path))
        self.assertEqual(launcher.subject.value, "subject-13")
        self.assertEqual(launcher.signal.value, "Heart beat per minute")
        self.assertEqual(
            launcher.signal_values, ("Heart beat per minute", "Actigraphy")
        )

    def test_gui_signal_labels_map_to_input_columns(self) -> None:
        from run_tpbm_gui import _phase_estimator_code, _pipeline_signal

        self.assertEqual(_pipeline_signal("Heart beat per minute", ".xlsx"), "BPM")
        self.assertEqual(_pipeline_signal("Heart beat per minute", ".csv"), "HeartRate")
        self.assertEqual(_pipeline_signal("Actigraphy", ".xlsx"), "Act")
        self.assertEqual(
            _pipeline_signal("Actigraphy", ".csv"), "ZeroCrossingCount"
        )
        self.assertEqual(_phase_estimator_code("KS (too slow)"), "KS")

    def test_results_visualization_saves_png_only(self) -> None:
        time_hours = np.arange(0, 72, 1 / 60)
        pf_result = SimpleNamespace(
            time_hours=time_hours,
            mean_state_theta_rad=np.mod(time_hours * np.pi / 12, 2 * np.pi),
            state_mean=np.column_stack(
                (
                    np.sin(time_hours * np.pi / 12),
                    np.cos(time_hours * np.pi / 12),
                )
            ),
        )
        start = pd.Timestamp("2025-01-01")
        dlmo = {
            "options": {"offset_hours": 7.0},
            "candidates": pd.DataFrame({"CBTHour": [30.0, 54.0]}),
            "events": pd.DataFrame(
                {
                    "DLMODateTime": [
                        start + pd.Timedelta(hours=21.75),
                        start + pd.Timedelta(hours=45.75),
                    ],
                    "DLMOHour": [21.75, 21.75],
                    "Ambiguous": [False, True],
                    "EventID": [1, 2],
                    "Status": ["tracked", "ambiguous"],
                }
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "chart.png"
            _save_results_visualization(output, pf_result, dlmo, start)
            with Image.open(output) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (1600, 1230))
                colors = {color for _, color in image.getcolors(1600 * 1230)}
                self.assertIn((37, 131, 74, 255), colors)
                self.assertIn((224, 139, 0, 255), colors)

            with self.assertRaisesRegex(ValueError, r"\.png extension"):
                _save_results_visualization(
                    Path(directory) / "chart.svg", pf_result, dlmo, start
                )

    def test_state_space_and_rts_smoother_shapes(self) -> None:
        a, b, c, d = create_kalman_state_space(1, 3, 1 / 60)
        self.assertEqual(a.shape, (3, 3))
        self.assertEqual(b.shape, (3, 1))
        self.assertEqual(c.shape, (1, 3))
        self.assertEqual(d.shape, (1, 1))
        time = np.arange(120, dtype=np.float64) / 60
        observations = np.sin(2 * np.pi * time / 24)
        observations[20:25] = np.nan
        states, smooth, posterior = simulate_kalman_smoother(
            a, c, np.eye(3) * 1e-4, 0.1, observations
        )
        self.assertEqual(states.shape, (3, len(time)))
        self.assertEqual(smooth.shape, time.shape)
        self.assertEqual(posterior.shape, (3, len(time)))
        self.assertTrue(np.all(np.isfinite(smooth)))

    def test_batched_kf_objective_matches_single_candidate_evaluation(self) -> None:
        count = 7 * 24 * 60 + 1
        time = np.arange(count, dtype=np.float64) / 60
        a, _, c, _ = create_kalman_state_space(1, 3, 1 / 60)
        observations = np.sin(2 * np.pi * time / 24)
        observations[100:103] = np.nan
        candidates = np.array(
            [
                np.linspace(-0.04, 0.05, 9),
                np.linspace(0.03, -0.02, 9),
                np.linspace(-0.01, 0.06, 9),
            ]
        )
        variances = np.array([0.2, 0.5, 1.2])
        actual = _evaluate_kf_population(
            candidates, variances, a, c, observations, time, 1, batch_size=2
        )
        expected = []
        for candidate, variance in zip(candidates, variances):
            factor = candidate.reshape((3, 3), order="F")
            q = np.einsum("ij,kj->ik", factor, factor)
            _, filtered, _ = simulate_kalman_filter(
                a, c, q, variance, observations
            )
            reconstructed = observations.copy()
            missing = ~np.isfinite(reconstructed)
            reconstructed[missing] = filtered[missing]
            original_spectrum, frequencies, _ = compute_spectrum(time, reconstructed)
            filtered_spectrum, _, _ = compute_spectrum(time, filtered)
            expected.append(
                compute_cost(original_spectrum, filtered_spectrum, frequencies, 1)
            )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_forward_kf_updates_finite_samples_and_skips_missing_samples(self) -> None:
        a, _, c, _ = create_kalman_state_space(1, 3, 1 / 60)
        observations = np.array([0.2, np.nan, -0.1])
        states, filtered, history = simulate_kalman_filter(
            a, c, np.eye(3) * 1e-4, 0.1, observations, full_log=True
        )
        self.assertIsNotNone(history)
        self.assertEqual(states.shape, (3, len(observations)))
        self.assertEqual(filtered.shape, observations.shape)
        np.testing.assert_array_equal(history["measurement_used"], [True, False, True])
        self.assertTrue(np.all(np.isfinite(states)))

    def test_matlab_kf_api_keeps_1500_initial_state(self) -> None:
        a, _, c, _ = create_kalman_state_space(1, 3, 1 / 60)
        states, filtered = KF.simulateKalmanFilter(
            a, c, np.eye(3) * 1e-4, 0.1, np.array([0.0, 0.1, np.nan])
        )
        self.assertEqual(states[-1, 0], 1500.0)
        self.assertAlmostEqual(filtered[0], 1500.0)
        self.assertTrue(np.all(np.isfinite(states)))

    def test_steady_state_gain_and_zero_missing_convention(self) -> None:
        a, _, c, _ = create_kalman_state_space(1, 3, 1 / 60)
        q = np.diag([1e-4, 1e-5, 1e-3])
        gain = KFSS.steady_state_gain(a, c, q, 100.0)
        self.assertEqual(gain.shape, (3,))
        self.assertTrue(np.all(np.isfinite(gain)))
        time = np.arange(7 * 24 * 60, dtype=np.float64) / 60
        observations = np.sin(2 * np.pi * time / 24)
        _, filtered = KFSS.simulateFilterNorm(a, c, gain, observations)
        self.assertTrue(np.all(np.isfinite(filtered)))
        states, _ = KFSS.simulateFilter(
            a, c, gain, np.array([1.0, 0.0, np.nan, 2.0])
        )
        np.testing.assert_allclose(
            states[:, 2], np.einsum("ij,j->i", a, states[:, 1])
        )

    def test_results_visualization_contains_all_three_panels_and_dlmo_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.png"
            start = pd.Timestamp("2025-04-21")
            pf_result = SimpleNamespace(
                time_hours=np.arange(0, 96, 1 / 60),
                mean_state_theta_rad=np.full(len(np.arange(0, 96, 1 / 60)), np.pi),
                smoothed_phase_rad=np.mod(
                    2 * np.pi * np.arange(0, 96, 1 / 60) / 24, 2 * np.pi
                ),
                state_mean=np.column_stack(
                    (
                        np.cos(2 * np.pi * np.arange(0, 96, 1 / 60) / 24),
                        np.zeros(len(np.arange(0, 96, 1 / 60))),
                    )
                ),
            )
            dlmo = {
                "options": {"offset_hours": 8.25},
                "candidates": pd.DataFrame({"CBTHour": [30.0, 54.0]}),
                "events": pd.DataFrame(
                    {
                        "DLMODateTime": [
                            start + pd.Timedelta(hours=21.75),
                            start + pd.Timedelta(hours=45.75),
                        ],
                        "DLMOHour": [21.75, 21.75],
                        "Ambiguous": [False, True],
                        "EventID": [1, 2],
                        "Status": ["anchor", "confirmed_ambiguous"],
                    }
                ),
            }
            _save_results_visualization(output, pf_result, dlmo, start)
            with Image.open(output) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (1600, 1230))
                colors = {color for _, color in image.getcolors(1600 * 1230)}
        self.assertIn((37, 131, 74, 255), colors)
        self.assertIn((224, 139, 0, 255), colors)
        self.assertIn((119, 119, 119, 255), colors)

    def test_root_dlmo_summary_contains_only_requested_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            targets = pd.DataFrame(
                {
                    "RecordingDay": [1, 2],
                    "DLMODateTime": [
                        pd.Timestamp("2025-01-01 21:00"),
                        pd.NaT,
                    ],
                    "Status": ["anchor", "incomplete_end"],
                    "EveningDate": pd.to_datetime(
                        ["2025-01-01", "2025-01-02"]
                    ),
                    "DLMOHour": [21.0, np.nan],
                }
            )

            path = _save_dlmo_summary(output_dir, targets)
            saved = pd.read_excel(path)

        self.assertEqual(path, output_dir / "DLMO_Summary.xlsx")
        self.assertEqual(
            list(saved.columns), ["RecordingDay", "DLMODateTime", "Status"]
        )
        self.assertEqual(saved["RecordingDay"].tolist(), [1, 2])
        self.assertEqual(saved["Status"].tolist(), ["anchor", "incomplete_end"])
        self.assertEqual(
            pd.Timestamp(saved.loc[0, "DLMODateTime"]),
            pd.Timestamp("2025-01-01 21:00"),
        )
        self.assertTrue(pd.isna(saved.loc[1, "DLMODateTime"]))

    def test_available_phase_shift_supports_short_and_variable_recordings(self) -> None:
        omega = 2 * np.pi / 24
        short_time = np.arange(0, 36, 1 / 60)
        short_states = np.cos(omega * short_time)
        theta, shift = estimate_available_phase_shift(
            omega, short_time, short_states, np.sin(omega * short_time)
        )
        self.assertEqual(theta.shape, short_time.shape)
        self.assertTrue(np.isnan(shift))

        long_time = np.arange(0, 9 * 24, 1 / 60)
        long_states = np.cos(omega * long_time)
        _, long_shift = estimate_available_phase_shift(
            omega, long_time, long_states, np.sin(omega * long_time)
        )
        self.assertTrue(np.isfinite(long_shift))

    def test_spectrum_matches_single_sided_sine_amplitude(self) -> None:
        time = np.arange(1440, dtype=np.float64) / 60
        signal = np.sin(2 * np.pi * time / 24)
        spectrum, frequency, length = compute_spectrum(time, signal)
        peak = np.argmax(spectrum[1:]) + 1
        self.assertEqual(length, len(time))
        self.assertAlmostEqual(frequency[peak], 1 / 24, delta=1 / (time[-1] * 60))
        self.assertAlmostEqual(spectrum[peak], 1.0, places=3)

    def test_oscillator_implementations_agree_on_constant_light(self) -> None:
        time = np.arange(61, dtype=np.float64) / 60
        light = np.full(time.shape, 500.0)
        initial = (0.1, 0.1, 0.5)
        _, _, fixed = jewett99_rk4_tau(time, light, initial, 24.2)
        _, _, variable = jewett99(time, light, initial)
        np.testing.assert_allclose(fixed, variable, rtol=1e-11, atol=1e-12)

    def test_activity_mapping_and_daily_cbt_output(self) -> None:
        np.testing.assert_array_equal(
            activity_to_light(np.array([0.0, 0.2, 1.0, 1.5, 10.0])),
            [0, 100, 200, 500, 2000],
        )
        time = np.arange(0, 72, 1 / 60)
        x = np.sin(2 * np.pi * time / 24)
        cbt = get_cbt_mins(time, x, np.zeros_like(x))
        self.assertEqual(cbt.shape, (3,))
        self.assertTrue(np.all(np.isfinite(cbt)))

    def test_excel_loader_regularizes_and_audits_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.xlsx"
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        [
                            "2025-01-01 00:00:20",
                            "2025-01-01 00:02:00",
                            "2025-01-01 00:02:10",
                        ]
                    ),
                    "activity": [1, 2, 3],
                    "bpm": [60, 61, 62],
                }
            )
            frame.to_excel(path, index=False)
            loaded = load_minute_xlsx(path)
        self.assertEqual(len(loaded.timestamps), 3)
        self.assertEqual(loaded.duplicate_rows.tolist(), [2])
        self.assertEqual(loaded.inserted_minutes.tolist(), [False, True, False])
        self.assertEqual(loaded.minute_data_days.shape, (3, 3))

    def test_sequence_path_and_dlmo_targets(self) -> None:
        options = {
            "event_reward": 3.0,
            "restart_penalty": 2.0,
            "min_separation_hours": 12.0,
            "restart_hours": 36.0,
            "expected_period_hours": 24.0,
            "timing_scale_hours": 6.0,
            "ambiguity_separation_hours": 2.0,
            "ambiguity_margin": 0.75,
        }
        chosen, margin, score = compare_dlmo_sequence(
            np.array([24.0, 48.0, 72.0]),
            np.array([0.5, 0.5, 0.5]),
            np.ones(3, dtype=np.int64),
            options,
        )
        np.testing.assert_array_equal(chosen, [0, 1, 2])
        self.assertEqual(margin.shape, (3,))
        self.assertGreater(score, 0)

        time = np.arange(0, 8 * 24, 1 / 60)
        cbt_state = np.cos(2 * np.pi * (time - 4.0) / 24)
        result = compare_dlmo_extract(
            time,
            cbt_state,
            pd.Timestamp("2025-01-01"),
            {"warmup_hours": 24.0},
        )
        self.assertGreaterEqual(len(result["events"]), 5)
        targets = compare_dlmo_targets(
            result,
            time,
            cbt_state,
            pd.Timestamp("2025-01-01"),
            [7],
            12,
        )
        self.assertEqual(len(targets), 1)
        self.assertIn("Status", targets.columns)

    def test_dlmo_targets_cover_all_days_in_recording_span(self) -> None:
        time = np.arange(0, 8 * 24, 1 / 60)
        cbt_state = np.cos(2 * np.pi * (time - 4.0) / 24)
        start = pd.Timestamp("2025-01-01")
        result = compare_dlmo_extract(
            time,
            cbt_state,
            start,
            {"warmup_hours": 24.0},
        )
        targets = compare_dlmo_targets(
            result,
            time,
            cbt_state,
            start,
            None,
            12,
        )
        expected_count = (
            int(np.floor((time[-1] - 12 - result["options"]["offset_hours"]) / 24))
            - int(np.floor((time[0] - 12 - result["options"]["offset_hours"]) / 24))
            + 1
        )
        self.assertEqual(len(targets), expected_count)
        self.assertIn("RecordingDay", targets.columns)
        self.assertNotIn("StudyDay", targets.columns)
        self.assertTrue(targets["EveningDate"].is_monotonic_increasing)
        self.assertGreater(len(targets), 3)

    def test_dlmo_handles_a_recording_with_no_candidate_troughs(self) -> None:
        time = np.arange(0, 96, 1 / 60)
        result = compare_dlmo_extract(
            time,
            np.zeros_like(time),
            pd.Timestamp("2025-01-01"),
            {"warmup_hours": 24.0},
        )
        self.assertTrue(result["events"].empty)
        self.assertTrue(result["candidates"].empty)
        self.assertGreater(len(result["daily"]), 0)


if __name__ == "__main__":
    unittest.main()
