"""
Basic plugin loader for the LQG simulation toolkit.
Allows dynamic discovery and loading of plugins from the plugins/ directory.
Plugins can register new observables, amplitudes, moves, or visualization methods.
"""
import importlib
import os
import sys

PLUGIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'plugins')

class PluginBase:
    """Base class for all plugins. Plugins should inherit from this."""
    def register(self):
        raise NotImplementedError("Plugin must implement register() method.")

def discover_plugins():
    """Discover and import all plugins in the plugins/ directory."""
    plugins = []
    sys.path.insert(0, PLUGIN_DIR)
    for fname in os.listdir(PLUGIN_DIR):
        if fname.endswith('.py') and not fname.startswith('_'):
            mod_name = fname[:-3]
            try:
                mod = importlib.import_module(mod_name)
                for attr in dir(mod):
                    obj = getattr(mod, attr)
                    if isinstance(obj, type) and issubclass(obj, PluginBase) and obj is not PluginBase:
                        plugins.append(obj())
            except Exception as e:
                print(f"[PluginLoader] Failed to load plugin {mod_name}: {e}")
    sys.path.pop(0)
    return plugins

def register_all_plugins():
    """Call register() on all discovered plugins."""
    for plugin in discover_plugins():
        plugin.register()
