# Example: Compute area, volume, deficit angle, and spinfoam amplitude for a simple spin network
from lqg_simulation.core.spin_network import SpinNetwork
from lqg_simulation.observables.geometry import calculate_total_area, calculate_total_volume, calculate_deficit_angle
from lqg_simulation.dynamics.amplitudes import calculate_total_spinfoam_amplitude, sum_amplitudes_over_internal_spin
from lqg_simulation.plotting.visualize import plot_spin_network, plot_observable_evolution
import matplotlib.pyplot as plt

# Build a simple tetrahedral spin network
sn = SpinNetwork()
n1 = sn.add_node(node_name="N1")
n2 = sn.add_node(node_name="N2")
n3 = sn.add_node(node_name="N3")
n4 = sn.add_node(node_name="N4")
l1 = sn.add_link(n1, n2, spin_j=1)
l2 = sn.add_link(n1, n3, spin_j=1)
l3 = sn.add_link(n1, n4, spin_j=1)
l4 = sn.add_link(n2, n3, spin_j=1)
l5 = sn.add_link(n2, n4, spin_j=1)
l6 = sn.add_link(n3, n4, spin_j=1)

# Compute observables
area = calculate_total_area(sn)
volume = calculate_total_volume(sn)
deficit_n1 = calculate_deficit_angle(n1, sn)

total_amp, vertex_amp, face_prod = calculate_total_spinfoam_amplitude(sn)

print(f"Total area: {area:.4f}")
print(f"Total volume: {volume:.4f}")
print(f"Deficit angle at N1: {deficit_n1:.4f}")
print(f"Spinfoam total amplitude: {total_amp:.4f} (vertex: {vertex_amp:.4f}, face prod: {face_prod:.4f})")

# Spinfoam sum over one internal spin (l1)
results, summed = sum_amplitudes_over_internal_spin(sn, l1, j_min=0, j_max=2)
print("\nSum over l1 spin:")
for j, amp in results:
    print(f"  j={j}: amplitude={amp:.4f}")
print(f"Total summed amplitude: {summed:.4f}")

# Plot the network and observable evolution (dummy evolution)
fig1, ax1 = plot_spin_network(sn, title="Tetrahedral Spin Network")
plt.show(block=False)

# Dummy evolution: change l1 spin from 0 to 2 and record area
evo_areas = []
from sympy import S
for i in range(0, 5):
    l1.spin_j = S(i)/2
    evo_areas.append(calculate_total_area(sn))
l1.spin_j = 1  # restore
fig2, ax2 = plot_observable_evolution(evo_areas, observable_name="Total Area")
plt.show()
