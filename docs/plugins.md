# Plugin System for LQG Simulation Toolkit

This toolkit supports a simple plugin system for adding new observables, amplitudes, moves, or visualization methods without modifying the core codebase.

## How Plugins Work
- Place your plugin Python files in the `lqg_simulation/plugins/` directory.
- Each plugin should define a class inheriting from `PluginBase` and implement a `register()` method.
- Use the `register()` method to add your functionality (e.g., register new observables).

## Example Plugin
```
from lqg_simulation.utils.plugin_loader import PluginBase

class MyCustomObservable(PluginBase):
    def register(self):
        print("[Plugin] MyCustomObservable registered!")
        # Register your observable here
```

## Loading Plugins
To load and register all plugins, call:
```
from lqg_simulation.utils.plugin_loader import register_all_plugins
register_all_plugins()
```

## Notes
- Plugins can be used to extend the toolkit for research, teaching, or experimentation.
- You can add new plugin types (amplitudes, moves, etc.) by following the same pattern.
