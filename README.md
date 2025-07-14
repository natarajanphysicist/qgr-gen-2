# LQG Simulation Tool

A Python-based simulation tool for exploring concepts in Loop Quantum Gravity (LQG).

## Project Structure

- `lqg_simulation/`: Main package directory.
  - `core/`: Core data structures like `SpinNetwork`, `Node`, `Link`.
  - `mathematics/`: Mathematical functions, including Wigner symbols, etc.
  - `observables/`: Functions for calculating geometric observables (volume, area).
  - `dynamics/`: Code related to the evolution of spin networks, transition amplitudes. (Planned)
  - `plotting/`: Visualization utilities.
  - `utils/`: Helper functions and utilities.
  - `examples/`: Example scripts demonstrating how to use the tool.
  - `tests/`: Unit tests for the package.
- `README.md`: This file.
- `requirements.txt`: Python package dependencies. (To be added)
- `setup.py`: Script for packaging and distribution. (To be added)

## Current Features (In Development)

- Basic spin network representation (nodes, links with spin-j values).
- Wigner 3j, 6j, 9j, and 10j symbol calculations.
- Placeholder vertex amplitude calculation.
- Spinfoam Amplitudes (Ponzano-Regge Model for 3D Gravity):
  - Generic amplitude calculation for an arbitrary 2-complex (triangulation) by automatically identifying all tetrahedra (vertices) and links (faces).
  - Summation of the Pachner complex amplitude over a range of internal spins.
- Basic Spin Network Dynamics:
  - Asymptotic EPRL-FK vertex amplitude for 4D spinfoams (using the Regge action with the Immirzi parameter).
  - Simple move: changing the spin of a link.
  - Complex move: 1-to-4 Pachner move (subdividing a tetrahedron).
- Geometric Observables:
  - Calculation of the volume of a single quantum tetrahedron.
  - Generic volume calculation for an arbitrary 2-complex by summing the volumes of all constituent tetrahedra.
  - Calculation of 4D dihedral angles for a 4-simplex (using Cayley-Menger determinant).
  - Calculation of the area of a surface (sum over pierced links).
- Simple 2D spin network visualization (using NetworkX & Matplotlib).

## Advanced Features

- **Plugin System:** Easily extend the toolkit with new observables, amplitudes, moves, or visualization methods. See `lqg_simulation/plugins/` and `docs/plugins.md`.
- **Quantum Simulation:** Simulate spin networks as quantum circuits using Qiskit. See `lqg_simulation/quantum/quantum_sim.py` and `docs/quantum_sim.md`.
- **Efficient Spinfoam Summation:** Monte Carlo and parallel summation for large-scale spinfoam calculations. See `lqg_simulation/dynamics/amplitudes.py` and `docs/spinfoam_summation.md`.
- **3D/Animated Visualization:** Visualize spin networks in 3D. See `lqg_simulation/plotting/plot3d.py` and `docs/plot3d.md`.
- **Jupyter/Interactive Workflows:** Example notebook in `lqg_simulation/examples/lqg_interactive_demo.ipynb` demonstrates interactive research workflows.

## Getting Started (Example)

```python
# (This is a placeholder - will be updated as features are implemented)
# from lqg_simulation.core import SpinNetwork

# Create a spin network
# sn = SpinNetwork()
# n1 = sn.add_node(node_name="N1")
# n2 = sn.add_node(node_name="N2")
# sn.add_link(n1, n2, spin_j=1.0, link_name="L12")

# sn.display()

# Visualize (planned)
# from lqg_simulation.plotting import plot_spin_network
# plot_spin_network(sn)
```

## Example: Interactive Research Workflow (Jupyter)

See `lqg_simulation/examples/lqg_interactive_demo.ipynb` for a full demo.

```python
from lqg_simulation.core.spin_network import SpinNetwork
from lqg_simulation.plotting.plot3d import plot_spin_network_3d

sn = SpinNetwork()
n1 = sn.add_node(node_name="A", pos=[0,0,0])
n2 = sn.add_node(node_name="B", pos=[1,0,0])
sn.add_link(n1, n2, spin=1)
fig, ax = plot_spin_network_3d(sn)
fig.show()
```

## Development

(Instructions for setting up a development environment, running tests, etc., will be added here.)

```bash
# Example:
# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# pytest
```

## Contributing

(Guidelines for contributing will be added here.)

## License

(License information will be added here, e.g., MIT, GPL.)
