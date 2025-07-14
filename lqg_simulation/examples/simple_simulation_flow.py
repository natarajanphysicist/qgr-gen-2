# lqg_simulation/examples/simple_simulation_flow.py

"""
Example script demonstrating a simple flow:
1. Create a SpinNetwork.
2. Visualize it.
3. Calculate a placeholder amplitude for a node.
4. (Optional) Modify the network.
5. Visualize it again.
"""
import matplotlib.pyplot as plt
from sympy import S

from lqg_simulation.core import SpinNetwork
from lqg_simulation.plotting.visualize import plot_spin_network
from lqg_simulation.dynamics.amplitudes import calculate_placeholder_vertex_amplitude

def run_simple_simulation_example():
    print("Starting simple simulation flow example...")

    # 1. Create a SpinNetwork
    sn = SpinNetwork()

    n1 = sn.add_node(node_name="N1")
    n2 = sn.add_node(node_name="N2")
    n3 = sn.add_node(node_name="N3")
    n_center = sn.add_node(node_name="NC") # A central node

    # Connect outer nodes to the center node, forming a 3-valent central node initially
    sn.add_link(n_center, n1, spin_j=S(1)/2, link_name="L_C1")
    sn.add_link(n_center, n2, spin_j=S(1),   link_name="L_C2")
    sn.add_link(n_center, n3, spin_j=S(1),   link_name="L_C3")

    print("\nInitial Spin Network:")
    sn.display()

    # 2. Visualize the initial spin network
    print("\nGenerating plot for the initial spin network (initial_network.png)...")
    fig_initial, ax_initial = plot_spin_network(sn, title="Initial Spin Network")
    plt.savefig("initial_network.png")
    plt.close(fig_initial) # Close the figure
    print("Plot saved as initial_network.png")

    # 3. Calculate placeholder vertex amplitude for the central node
    # NC is 3-valent, so placeholder amplitude will be 1.0
    print(f"\nCalculating placeholder amplitude for node {n_center.name} (3-valent)...")
    amplitude_nc_3valent = calculate_placeholder_vertex_amplitude(n_center, sn, intertwiner_spin=S(1))
    print(f"Placeholder amplitude for {n_center.name}: {amplitude_nc_3valent}")

    # 4. Modify the network: Make NC 4-valent by adding another node and link
    print("\nModifying the spin network: adding N4 and link L_C4 to make NC 4-valent...")
    n4 = sn.add_node(node_name="N4")
    sn.add_link(n_center, n4, spin_j=S(3)/2, link_name="L_C4")

    print("\nModified Spin Network:")
    sn.display()

    # Calculate new amplitude for NC (now 4-valent)
    # Spins incident on NC: 1/2, 1, 1, 3/2. Sorted: 0.5, 1.0, 1.0, 1.5
    # If intertwiner_spin=1: 6j is {0.5, 1.0, 1.0 / 1.5, 1.0, 1.0} (sum of 1st triad is 2.5, not int)
    # So, 6j will be 0. Amplitude = (2*1+1)*0 = 0.
    print(f"\nCalculating placeholder amplitude for node {n_center.name} (now 4-valent)...")
    amplitude_nc_4valent = calculate_placeholder_vertex_amplitude(n_center, sn, intertwiner_spin=S(1))
    print(f"Placeholder amplitude for {n_center.name} (J_int=1): {amplitude_nc_4valent}")

    # Try with default intertwiner spin for NC (4-valent)
    # Incident spins [1/2, 1, 1, 3/2]. Smallest non-zero is 1/2. J_int = 1/2.
    # 6j is {0.5, 1.0, 1.0 / 1.5, 0.5, 0.5} (sum of 1st triad is 2.5, not int)
    # So, 6j will be 0. Amplitude = (2*0.5+1)*0 = 0.
    amplitude_nc_4valent_default_J = calculate_placeholder_vertex_amplitude(n_center, sn)
    print(f"Placeholder amplitude for {n_center.name} (default J_int): {amplitude_nc_4valent_default_J}")

    # 5. Visualize the modified network
    print("\nGenerating plot for the modified spin network (modified_network.png)...")
    fig_modified, ax_modified = plot_spin_network(sn, title="Modified Spin Network (NC is 4-valent)")
    plt.savefig("modified_network.png")
    plt.close(fig_modified) # Close the figure
    print("Plot saved as modified_network.png")

    print("\nSimple simulation flow example finished.")
    print("Check for 'initial_network.png' and 'modified_network.png' in the current directory.")

if __name__ == '__main__':
    run_simple_simulation_example()
