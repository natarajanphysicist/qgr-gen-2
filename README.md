# Quantum Gravity Simulation Platform

## Overview
This platform is an advanced research and educational tool for Loop Quantum Gravity (LQG) and spinfoam models. It provides interactive modules for constructing, visualizing, and simulating spin networks, calculating Wigner symbols, exploring spinfoam amplitudes, and running quantum gravity-inspired quantum computing simulations.


## Features
- **Spin Networks**: Build, visualize, and analyze spin networks. Add nodes/links manually or generate random networks. View network statistics and node/link details.
- **Entanglement Entropy for Spin Networks**: Select a region (set of nodes) and compute the entanglement entropy for that region directly in the Spin Networks page (Quantum Information panel).
- **Wigner Symbols**: Compute 3j, 6j, 9j, 12j, and 15j Wigner symbols. Visualize properties and calculation history.
- **Spinfoam Models**: Generate and visualize spinfoam complexes (EPRL-FK, Ooguri, Barrett-Crane). Calculate vertex and transition amplitudes, analyze quantum fluctuations, and compute partition functions. Includes a research/citation panel and recent calculations log.
- **Boundary Entanglement Entropy for Spinfoam Models**: Select a set of boundary vertices and compute the entanglement entropy for the boundary region in the Spinfoam Models page (Quantum Information panel).
- **Geometric Observables**: Calculate area, volume, curvature, torsion, holonomy, Wilson loops, Ricci/Weyl tensors, field strength, quantum spectra, and semiclassical limits. Visualize area, volume, and length spectra.
- **Quantum Computing**: Simulate quantum evolution, quantum bounce, entanglement, error correction, state tomography, and holographic duality using Qiskit or classical backends. Visualize results and quantum circuit statistics.
- **Advanced Simulations**: Explore spacetime emergence, black hole physics, and cosmological models with interactive parameters and visualizations.
- **Research Examples**: Run case studies on simple/constrained/thermal/hamiltonian evolution, singularity resolution, primordial perturbations, black hole information, and quantum cosmology.
- **Research Trends & Citations**: Each major module includes a collapsible panel with recent research trends, long-standing problems, and key literature references.
## How to Use Entanglement Entropy Features

- **Spin Networks Page**: In the right panel, scroll to "Entanglement Entropy (Quantum Information)". Select one or more nodes to define a region and click "Compute Entanglement Entropy". The entropy for the selected region will be displayed.
- **Spinfoam Models Page**: In the left panel, scroll to "Boundary Entanglement Entropy (Quantum Information)". Select one or more boundary vertices and click "Compute Boundary Entanglement Entropy". The entropy for the selected boundary region will be displayed.

These features use a toy model for entropy (log of number of boundary links/faces). For research, you can extend these with more advanced quantum information calculations.

## Getting Started
1. **Install Requirements**
   - Python 3.8+
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
2. **Run the App**
   ```bash
   streamlit run app.py
   ```
3. **Navigate**
   - Use the sidebar to select modules: Spin Networks, Wigner Symbols, Spinfoam Models, Geometric Observables, Quantum Computing, Advanced Simulations, Research Examples.

## Project Structure
- `app.py` — Main Streamlit app and UI logic
- `core/` — Core simulation modules (spin networks, spinfoam, geometry, quantum computing, wigner symbols)
- `utils/` — Visualization, plotting, and helper utilities
- `examples/` — Research example modules
- `attached_assets/` — Images and documentation assets

## Research & Citations
- Each module features a panel with:
  - Recent research trends
  - Long-standing open problems
  - Key review articles and citations

## Roadmap & Advanced Research Features

### Quantum Information & Holography (Planned)
- Entanglement entropy calculations for spin networks and spinfoam boundaries
- Holographic correspondence tools: compare bulk and boundary observables, visualize entanglement wedges
- Quantum information panel: display entropy, mutual information, and holographic diagnostics

### Advanced Visualization (Planned)
- Interactive 3D/4D visualization of spin networks and spinfoam complexes (WebGL/advanced Plotly)
- Real-time manipulation: rotate, zoom, select/highlight nodes, links, faces
- Animated evolution: play/pause controls and time sliders for network evolution, amplitude fluctuations, and cosmological bounces
- Export animations as GIF/MP4 for research and presentations

### Future Directions
- Parameter scans and batch simulations for automated research workflows
- Machine learning integration for pattern discovery and symbolic regression
- HPC/parallel computation support for large-scale simulations
- Community features: export/import workflows, notebook integration, open problem playground

## Notes
- Spinfoam amplitudes are generated with valid spin configurations to avoid zero results due to triangle inequality violations.
- The platform is designed for both research and educational use in quantum gravity, LQG, and related fields.

## License
For research and educational use only.

---
**Quantum Gravity Simulation Platform** — Developed for theoretical physics research and education.
