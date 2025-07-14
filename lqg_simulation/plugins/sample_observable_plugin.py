from lqg_simulation.utils.plugin_loader import PluginBase

class SampleObservablePlugin(PluginBase):
    def register(self):
        print("[Plugin] SampleObservablePlugin registered! You can now add custom observables here.")
        # Example: register a new observable in a global registry (not implemented here)
        # observables_registry['sample'] = self.compute_sample_observable

    def compute_sample_observable(self, spin_network):
        # Dummy observable: count nodes
        return len(spin_network.nodes)
