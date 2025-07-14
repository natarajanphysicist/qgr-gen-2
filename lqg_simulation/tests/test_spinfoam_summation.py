import unittest
from lqg_simulation.dynamics.amplitudes import monte_carlo_spinfoam_sum, parallel_spinfoam_sum

def dummy_amplitude(spins):
    # Simple amplitude: product of spins (avoid zero)
    prod = 1.0
    for s in spins:
        prod *= (s if s != 0 else 1)
    return prod

class TestSpinfoamSummation(unittest.TestCase):
    def test_monte_carlo_sum(self):
        spin_ranges = [(0.5, 2.5), (0.5, 2.5)]
        result = monte_carlo_spinfoam_sum(dummy_amplitude, spin_ranges, n_samples=1000, seed=42)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_parallel_sum(self):
        spin_ranges = [(0.5, 1.0), (0.5, 1.0)]  # Only (0.5, 1.0) for each spin
        result = parallel_spinfoam_sum(dummy_amplitude, spin_ranges, n_workers=2)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

if __name__ == '__main__':
    unittest.main()
