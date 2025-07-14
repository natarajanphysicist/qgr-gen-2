# Quantum Simulation of Spin Networks

This toolkit supports quantum circuit simulation of spin network states using Qiskit (or other backends in the future).

## How It Works
- The `simulate_spin_network_on_qubits` function in `lqg_simulation.quantum.quantum_sim` maps a spin network to a quantum circuit.
- The Qiskit backend is used if available; otherwise, a dummy backend is used.
- Plugins can register new quantum simulation methods.

## Example Usage
```
from lqg_simulation.quantum.quantum_sim import simulate_spin_network_on_qubits
result = simulate_spin_network_on_qubits(spin_network)
print(result)
```

## Plugin Example
See `lqg_simulation/plugins/quantum_spin_network_plugin.py` for a sample plugin that registers quantum simulation functionality.

## Requirements
- Qiskit must be installed for quantum simulation: `pip install qiskit`

## Extending
- You can add new backends (e.g., Cirq) or more advanced mappings from spin networks to circuits.
- Use the plugin system to register new quantum algorithms or encodings.
