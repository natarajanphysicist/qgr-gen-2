from lqg_simulation.utils.plugin_loader import PluginBase
from lqg_simulation.quantum.quantum_sim import simulate_spin_network_on_qubits

class QuantumSpinNetworkPlugin(PluginBase):
    def register(self):
        print("[Plugin] QuantumSpinNetworkPlugin registered! You can now run quantum simulations of spin networks.")
        # Example: register a quantum simulation method in a global registry (not implemented here)
        # quantum_registry['simulate'] = self.simulate

    def simulate(self, spin_network):
        return simulate_spin_network_on_qubits(spin_network)
