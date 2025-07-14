"""
Command-line entry point for the LQG simulation toolkit.
"""
import argparse
from lqg_simulation.core.spin_network import SpinNetwork
from lqg_simulation.plotting.visualize import plot_spin_network_2d

def main():
    parser = argparse.ArgumentParser(description="LQG Simulation Toolkit CLI")
    parser.add_argument('--demo', action='store_true', help='Run a simple spin network demo')
    args = parser.parse_args()
    if args.demo:
        sn = SpinNetwork()
        n1 = sn.add_node(node_name="A")
        n2 = sn.add_node(node_name="B")
        sn.add_link(n1, n2, spin=1)
        plot_spin_network_2d(sn, title="CLI Demo Spin Network")

if __name__ == "__main__":
    main()
