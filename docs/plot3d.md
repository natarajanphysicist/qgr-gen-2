# 3D and Animated Visualization

This toolkit supports 3D visualization of spin networks for research and teaching.

## Usage
```
from lqg_simulation.plotting.plot3d import plot_spin_network_3d
fig, ax = plot_spin_network_3d(spin_network)
fig.show()
```
- Nodes should have `.pos` or `.xyz` attributes (3D coordinates). If missing, random positions are used.
- Edges are drawn between nodes using the `links` attribute.

## Animation (Planned)
- The `animate_spin_network_evolution` function is a placeholder for future animated visualization of network evolution.

## Extending
- You can add support for Plotly, PyVista, or other 3D/VR libraries.
- Use the plugin system to register new visualization methods.
