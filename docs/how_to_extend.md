# How to Extend the LQG Simulation Toolkit

## Adding a New Amplitude or Observable
- Create a new function in the appropriate module (e.g., `dynamics/amplitudes.py` or `observables/geometry.py`).
- Add a test in `tests/`.
- Document the function with a clear docstring and usage example.

## Adding a New Visualization
- Add a function to `plotting/visualize.py` or `plotting/plot3d.py`.
- Ensure it works with the core `SpinNetwork` data structure.

## Adding Quantum/Parallel Features
- Place quantum code in `quantum/` and parallel code in `utils/` or relevant modules.
- Use stubs as templates for new features.
