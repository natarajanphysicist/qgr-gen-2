# Efficient Spinfoam Amplitude Summation

This toolkit supports several strategies for summing over internal spins in spinfoam amplitudes:

## Methods
- **Monte Carlo Summation:**
  - Use `monte_carlo_spinfoam_sum` for large or high-dimensional sums.
- **Parallel Summation:**
  - Use `parallel_spinfoam_sum` for small sums over all configurations, utilizing multiple CPU cores.
- **GPU/Tensor Network (Planned):**
  - Placeholder for future GPU or tensor network acceleration.

## Example Usage
```
from lqg_simulation.dynamics.amplitudes import monte_carlo_spinfoam_sum

def my_amplitude(spins):
    # Compute amplitude for a given spin configuration
    return 1.0  # Replace with real calculation

spin_ranges = [(0.5, 2.5), (0.5, 2.5)]  # Example: two internal spins
result = monte_carlo_spinfoam_sum(my_amplitude, spin_ranges, n_samples=10000)
print(result)
```

## Extending
- Implement your own efficient summation strategies or plug in GPU/tensor network backends.
- Use the plugin system to register new summation methods.
