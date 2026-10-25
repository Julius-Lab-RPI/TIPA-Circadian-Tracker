import tempfile
import unittest
from pathlib import Path

import numpy as np

from particle_filter import (
    ParticleFilterConfig,
    _read_input_csv,
    jewett99_rk4_step,
    phase_observations,
    run_particle_filter,
    wrap_to_pi,
    wrap_to_2pi,
)


class ParticleFilterTests(unittest.TestCase):
    def test_csv_reader_accepts_blank_phase_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text(
                "time_hours,light,theta_rad\n0,0,0.1\n0.0167,100,\n",
                encoding="utf-8",
            )
            time, light, theta = _read_input_csv(path)

        np.testing.assert_allclose(time, [0.0, 0.0167])
        np.testing.assert_allclose(light, [0.0, 100.0])
        self.assertTrue(np.isnan(theta[1]))

    def test_angle_wrapping(self) -> None:
        wrapped = wrap_to_2pi(np.array([-0.1, 0.0, 2.0 * np.pi + 0.1]))
        np.testing.assert_allclose(wrapped, [2.0 * np.pi - 0.1, 0.0, 0.1])
        self.assertAlmostEqual(float(wrap_to_pi(np.array([np.pi]))[0]), np.pi)

    def test_phase_conversion_smooths_each_finite_segment(self) -> None:
        time = np.arange(6, dtype=np.float64) / 60.0
        theta = np.array([0.1, 0.2, 0.3, np.nan, 0.5, 0.6])
        observed, smoothed = phase_observations(time, theta, 3)
        self.assertTrue(np.all(np.isfinite(observed[[0, 1, 2, 4, 5]])))
        self.assertTrue(np.isnan(observed[3]))
        self.assertTrue(np.isnan(smoothed[3]))
        self.assertTrue(np.all((smoothed[[0, 1, 2, 4, 5]] >= 0.0)))
        self.assertTrue(np.all((smoothed[[0, 1, 2, 4, 5]] < 2.0 * np.pi)))

    def test_jewett_step_at_dark_input(self) -> None:
        states = np.array([[0.1, 0.2, 0.5]], dtype=np.float64)
        next_state = jewett99_rk4_step(
            states,
            np.array([0.0]),
            1.0 / 60.0,
            np.array([24.2]),
        )
        self.assertEqual(next_state.shape, (1, 3))
        self.assertTrue(np.all(np.isfinite(next_state)))
        self.assertLess(next_state[0, 2], states[0, 2])

    def test_filter_is_reproducible_and_returns_normalized_histories(self) -> None:
        time = np.arange(12, dtype=np.float64) / 60.0
        light = np.full(time.shape, 200.0)
        theta = np.mod(0.2 + 2.0 * np.pi * time / 24.0, 2.0 * np.pi)
        config = ParticleFilterConfig(
            n_particles=40,
            seed=73,
            start_after_hours=0.0,
            observation_smoothing_samples=3,
            store_history=True,
        )

        first = run_particle_filter(time, light, theta, config)
        second = run_particle_filter(time, light, theta, config)

        np.testing.assert_array_equal(first.state_mean, second.state_mean)
        np.testing.assert_array_equal(first.resampled, second.resampled)
        self.assertEqual(first.state_mean.shape, (len(time), 5))
        self.assertIsNotNone(first.history)
        assert first.history is not None
        self.assertEqual(first.history["X_prior_history"].shape, (len(time), 40, 5))
        np.testing.assert_allclose(
            first.history["weights_posterior_history"].sum(axis=1),
            np.ones(len(time)),
        )
        self.assertTrue(np.all(np.isfinite(first.state_mean)))
        self.assertTrue(np.isfinite(first.r_estimate_rad))

    def test_resampling_preserves_numpy_global_random_state(self) -> None:
        time = np.arange(12, dtype=np.float64) / 60.0
        light = np.full(time.shape, 200.0)
        theta = np.mod(0.2 + 2.0 * np.pi * time / 24.0, 2.0 * np.pi)
        original_state = np.random.get_state()
        try:
            np.random.seed(42)
            expected_state = np.random.get_state()
            result = run_particle_filter(
                time,
                light,
                theta,
                ParticleFilterConfig(
                    n_particles=40,
                    seed=73,
                    start_after_hours=0.0,
                    observation_smoothing_samples=3,
                ),
            )
            actual_state = np.random.get_state()
            self.assertGreater(result.resampling_events, 0)
            self.assertEqual(expected_state[0], actual_state[0])
            np.testing.assert_array_equal(expected_state[1], actual_state[1])
            self.assertEqual(expected_state[2:], actual_state[2:])
        finally:
            np.random.set_state(original_state)


if __name__ == "__main__":
    unittest.main()
