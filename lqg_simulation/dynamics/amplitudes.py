# lqg_simulation/dynamics/amplitudes.py
"""
Defines functions for calculating transition amplitudes, starting with placeholders.
"""
import math
import itertools
import concurrent.futures
import random

from lqg_simulation.core.spin_network import Node, SpinNetwork, Link
from lqg_simulation.observables.geometry import _find_tetrahedra, calculate_dihedral_angles_placeholder
from lqg_simulation.mathematics.wigner_symbols import (
    calculate_wigner_6j, calculate_10j_symbol
)
from sympy import S


def calculate_ponzano_regge_vertex_amplitude(network: SpinNetwork, tetra_nodes: list[Node]):
    """
    Calculates the Ponzano-Regge/Ooguri model vertex amplitude for a spinfoam.

    This model applies to 3D quantum gravity. The amplitude for a vertex of the
    spinfoam (dual to a tetrahedron in the triangulation) is given by the
    Wigner 6j symbol evaluated on the spins of the six faces (links) that
    form the tetrahedron.

    Args:
        network: The SpinNetwork object containing the tetrahedron.
        tetra_nodes: A list of the four Node objects that form the tetrahedron.

    Returns:
        The numerical value of the Ponzano-Regge amplitude (float).

    Raises:
        ValueError: If the list does not contain 4 nodes or if the nodes do
                    not form a complete tetrahedron in the network.
    """
    if not isinstance(network, SpinNetwork):
        raise TypeError("Input must be a SpinNetwork object.")

    if len(tetra_nodes) != 4:
        raise ValueError("A tetrahedron must be defined by exactly 4 nodes.")

    n1, n2, n3, n4 = tetra_nodes
    try: # pragma: no cover
        spins = [
            network.get_link_between(n1, n2).spin_j, network.get_link_between(n1, n3).spin_j,
            network.get_link_between(n1, n4).spin_j, network.get_link_between(n2, n3).spin_j,
            network.get_link_between(n2, n4).spin_j, network.get_link_between(n3, n4).spin_j
        ]
    except AttributeError as e:
        raise ValueError("The provided nodes do not form a complete tetrahedron in the network.") from e

    # The vertex amplitude is the Wigner 6j symbol of the six spins.
    return calculate_wigner_6j(*spins)


def calculate_face_amplitude(spin_j):
    """
    Calculates the amplitude for a single face of a spinfoam.

    In many SU(2)-based spinfoam models, the face amplitude is simply the
    dimension of the irreducible representation associated with the face's spin `j`.

    Args:
        spin_j: The spin `j` of the face.

    Returns:
        The face amplitude, (2j + 1).
    """
    return 2 * spin_j + 1


def calculate_total_spinfoam_amplitude(network: SpinNetwork):
    """
    Calculates the total spinfoam amplitude for a simple 2-complex.

    This is a simplified calculation for a 2-complex represented by a single
    tetrahedron. The total amplitude is the product of the vertex amplitude
    and all the face amplitudes.

    Z_spinfoam = A_vertex * product(A_face)

    This implementation assumes the Ponzano-Regge model for the vertex and
    a (2j+1) weighting for the faces.

    Args:
        network: The SpinNetwork representing the 2-complex (must be a tetrahedron).

    Returns:
        A tuple containing (total_amplitude, vertex_amplitude, face_amplitudes_product).
    """
    # 1. Calculate the vertex amplitude
    # This function assumes the network itself is a single tetrahedron
    if not (len(network.nodes) == 4 and len(network.links) == 6):
        raise ValueError("calculate_total_spinfoam_amplitude is for a single tetrahedron network.")

    vertex_amplitude = calculate_ponzano_regge_vertex_amplitude(network, network.nodes)

    # 2. Calculate the product of all face amplitudes
    face_amplitudes_product = 1.0
    for link in network.links:  # Each link in the triangulation is a face in the spinfoam
        face_amp = calculate_face_amplitude(link.spin_j)
        face_amplitudes_product *= face_amp

    # 3. Calculate the total amplitude
    total_amplitude = vertex_amplitude * face_amplitudes_product

    return float(total_amplitude), float(vertex_amplitude), float(face_amplitudes_product)


def calculate_pachner_complex_amplitude(network: SpinNetwork, original_nodes: list[Node], internal_node: Node):
    """
    Calculates the total spinfoam amplitude for a 4-tetrahedron complex
    that results from a 1-to-4 Pachner move.

    The amplitude is calculated as the product of the vertex amplitudes (6j-symbol)
    for each of the four tetrahedra, multiplied by the face amplitudes (2j+1)
    for each of the four internal faces.

    Z = (Π_{v=1 to 4} A_v) * (Π_{f_int=1 to 4} A_f)
      = (Π_{k=1 to 4} {6j}_k) * (Π_{i=1 to 4} (2*j_int_i + 1))

    This function assumes the internal spins are fixed (i.e., it does not perform
    a sum over them).

    Args:
        network: The SpinNetwork object containing the complex.
        original_nodes: A list of the four original boundary nodes.
        internal_node: The new internal node created by the Pachner move.

    Returns:
        A tuple containing (total_amplitude, product_of_vertex_amps, product_of_internal_face_amps).
    """
    if len(original_nodes) != 4:
        raise ValueError("The complex must be defined by 4 original boundary nodes.")

    # 1. Calculate the product of internal face amplitudes
    internal_face_product = 1.0
    internal_links = network.get_links_for_node(internal_node)
    if len(internal_links) != 4:
        raise ValueError(f"Internal node '{internal_node.name}' should have 4 links, but has {len(internal_links)}.")

    for link in internal_links:
        internal_face_product *= calculate_face_amplitude(link.spin_j)

    # 2. Calculate the product of the four vertex amplitudes (6j symbols)
    vertex_product = 1.0
    tetra_node_combinations = itertools.combinations(original_nodes, 3)

    for combo in tetra_node_combinations:
        n1, n2, n3 = combo[0], combo[1], combo[2]

        # Get the 6 spins for the tetrahedron (internal_node, n1, n2, n3)
        try:
            j_int1 = network.get_link_between(internal_node, n1).spin_j
            j_int2 = network.get_link_between(internal_node, n2).spin_j
            j_int3 = network.get_link_between(internal_node, n3).spin_j
            j_ext12 = network.get_link_between(n1, n2).spin_j
            j_ext13 = network.get_link_between(n1, n3).spin_j
            j_ext23 = network.get_link_between(n2, n3).spin_j
        except AttributeError:  # If a link is missing
            raise ValueError(f"The nodes {internal_node.name}, {n1.name}, {n2.name}, {n3.name} do not form a tetrahedron in the network.")

        spins = [j_int1, j_int2, j_int3, j_ext12, j_ext13, j_ext23]
        vertex_amp = calculate_wigner_6j(*spins)
        vertex_product *= vertex_amp

    # 3. Calculate total amplitude
    total_amplitude = vertex_product * internal_face_product

    return float(total_amplitude), float(vertex_product), float(internal_face_product)


def calculate_summed_pachner_amplitude(network: SpinNetwork, original_nodes: list[Node], internal_node: Node, j_max):
    """
    Calculates the spinfoam amplitude for a 4-tetrahedron complex by summing
    over a range of internal spin configurations.

    This function approximates the full quantum amplitude by summing the contributions
    from different internal geometries up to a maximum spin `j_max`.

    Z_summed = Σ_{j_int} (Π_{v} A_v) * (Π_{f_int} A_f)
             = Σ_{j_int_1...j_int_4 <= j_max} [ (Π_{k=1 to 4} {6j}_k) * (Π_{i=1 to 4} (2*j_int_i + 1)) ]

    Args:
        network: The SpinNetwork object containing the complex.
        original_nodes: A list of the four original boundary nodes.
        internal_node: The new internal node created by the Pachner move.
        j_max: The maximum spin value (integer or half-integer) to include in the sum.

    Returns:
        The total summed amplitude as a float.
    """
    internal_links = network.get_links_for_node(internal_node)
    if len(internal_links) != 4:
        raise ValueError(f"Internal node '{internal_node.name}' should have 4 links, but has {len(internal_links)}.")

    # Generate the list of spin values to iterate over
    spin_range = [S(i) / 2 for i in range(int(2 * j_max + 1))]

    # Generate all combinations of 4 spins from the spin range
    spin_combinations = list(itertools.product(spin_range, repeat=4))

    total_amplitude = 0.0
    num_combinations = len(spin_combinations)
    print(f"Summing over {num_combinations} internal spin configurations (j_max={j_max})...")

    # Store original spins to restore them later
    original_internal_spins = [link.spin_j for link in internal_links]

    for i, combo in enumerate(spin_combinations):
        # Assign the new combination of internal spins
        for link_idx, spin_val in enumerate(combo):
            internal_links[link_idx].spin_j = spin_val

        # Calculate the amplitude for this specific configuration
        amp_for_config, _, _ = calculate_pachner_complex_amplitude(network, original_nodes, internal_node)
        total_amplitude += amp_for_config

        if (i + 1) % 20 == 0 or (i + 1) == num_combinations:
            print(f"  ...processed {i+1}/{num_combinations} combinations.", end='\r')

    # Restore original spins to leave the network in its initial state
    for link_idx, spin_val in enumerate(original_internal_spins):
        internal_links[link_idx].spin_j = spin_val

    print("\nSummation complete.")
    return total_amplitude


def calculate_generic_spinfoam_amplitude(network: SpinNetwork):
    """
    Calculates the total spinfoam amplitude for a generic 2-complex.

    This function identifies all tetrahedra (vertices) and links (faces) in the
    given spin network and computes the total Ponzano-Regge amplitude:

    Z = (Π_v A_v) * (Π_f A_f)
      = (Π_{tetrahedra} {6j}) * (Π_{links} (2j+1))

    This provides a general way to calculate the amplitude for any triangulation
    represented by the spin network.

    Args:
        network: The SpinNetwork representing the 2-complex.

    Returns:
        A tuple containing (total_amplitude, product_of_vertex_amps, product_of_face_amps).
    """
    # 1. Calculate the product of all face amplitudes
    product_of_face_amps = 1.0
    for link in network.links:
        product_of_face_amps *= calculate_face_amplitude(link.spin_j)

    # 2. Find all tetrahedra in the network and calculate vertex amplitude product
    tetrahedra = _find_tetrahedra(network)
    print(f"Found {len(tetrahedra)} tetrahedra in the network.")
    product_of_vertex_amps = 1.0
    for i, tetra_nodes in enumerate(tetrahedra):
        vertex_amp = calculate_ponzano_regge_vertex_amplitude(network, tetra_nodes)
        print(f"  - Amplitude for tetrahedron {i+1} ({[n.name for n in tetra_nodes]}): {vertex_amp:.6f}")
        product_of_vertex_amps *= vertex_amp

    # 3. Calculate total amplitude
    total_amplitude = product_of_vertex_amps * product_of_face_amps
    return float(total_amplitude), float(product_of_vertex_amps), float(product_of_face_amps)


def calculate_4simplex_vertex_amplitude(face_spins, intertwiner_spins):
    """
    Calculates the vertex amplitude for a 4-simplex (dual to a vertex in a 4D spinfoam),
    using the Wigner 15j symbol (first kind, pentagon diagram).

    Args:
        face_spins: List or tuple of 10 spins (j1..j10) for the 10 triangle faces of the 4-simplex.
        intertwiner_spins: List or tuple of 5 spins (k1..k5) for the 5 tetrahedral intertwiners.
    Returns:
        The value of the vertex amplitude as a float.
    """
    if len(face_spins) != 10 or len(intertwiner_spins) != 5:
        raise ValueError("Need 10 face spins and 5 intertwiner spins for a 4-simplex vertex amplitude.")
    from lqg_simulation.mathematics.wigner_symbols import calculate_15j_symbol
    # The 15j symbol takes the 10 face spins and 5 intertwiner spins in a standard order.
    j = list(face_spins) + list(intertwiner_spins)
    return calculate_15j_symbol(j)


def calculate_placeholder_vertex_amplitude(node, network, intertwiner_spin=None):
    """
    Placeholder for a vertex amplitude for a node in a spin network.
    - For 3-valent nodes: returns 1.0
    - For 4-valent nodes: returns a 6j-based amplitude
    - For other valence: returns 1.0
    """
    # Get incident links
    connected_links = [l for l in network.links if l.node1 == node or l.node2 == node]
    valence = len(connected_links)
    if valence == 3:
        return 1.0
    elif valence == 4:
        spins = sorted([l.spin_j for l in connected_links])
        j1, j2, j3, j4 = spins[0], spins[1], spins[2], spins[3]
        from sympy import S
        if intertwiner_spin is None:
            non_zero_spins = [s for s in spins if s != 0]
            chosen_J_int = min(non_zero_spins) if non_zero_spins else S(1)
        else:
            chosen_J_int = S(intertwiner_spin)
        val_6j = calculate_wigner_6j(j1, j2, j3, j4, chosen_J_int, chosen_J_int)
        amplitude = (2 * chosen_J_int + 1) * val_6j
        return amplitude
    else:
        return 1.0


def calculate_ooguri_vertex_amplitude(j1, j2, j3, j4, j5, j6):
    """
    Calculates the Ooguri model vertex amplitude for a tetrahedron.
    The amplitude is the Wigner 6j symbol of the six edge spins.
    """
    return float(calculate_wigner_6j(j1, j2, j3, j4, j5, j6))


def sum_amplitudes_over_internal_spin(network, link, j_min=0, j_max=2):
    """
    Sum the total spinfoam amplitude over all possible values of a single internal spin (link).
    Args:
        network: SpinNetwork object.
        link: Link object whose spin_j will be varied.
        j_min: Minimum spin value (default 0).
        j_max: Maximum spin value (default 2).
    Returns:
        List of (spin, amplitude) tuples and the total sum.
    """
    from sympy import S
    results = []
    total = 0.0
    original_spin = link.spin_j
    for i in range(int(2*j_min), int(2*j_max)+1):
        j = S(i)/2
        link.spin_j = j
        amp, _, _ = calculate_total_spinfoam_amplitude(network)
        results.append((j, amp))
        total += amp
    link.spin_j = original_spin  # Restore
    return results, total


def monte_carlo_spinfoam_sum(amplitude_func, spin_ranges, n_samples=1000, seed=None):
    """
    Monte Carlo summation over internal spins for spinfoam amplitudes.
    Args:
        amplitude_func: Function taking a list of spins and returning amplitude.
        spin_ranges: List of (min, max) tuples for each internal spin.
        n_samples: Number of Monte Carlo samples.
        seed: Random seed (optional).
    Returns:
        Estimated sum (float).
    """
    if seed is not None:
        random.seed(seed)
    total = 0.0
    for _ in range(n_samples):
        spins = [random.randint(int(a*2), int(b*2))/2 for a, b in spin_ranges]
        total += amplitude_func(spins)
    return total / n_samples


def parallel_spinfoam_sum(amplitude_func, spin_ranges, n_workers=4):
    """
    Parallel summation over all possible internal spin configurations (small cases only).
    Args:
        amplitude_func: Function taking a list of spins and returning amplitude.
        spin_ranges: List of (min, max) tuples for each internal spin.
        n_workers: Number of parallel workers.
    Returns:
        Total sum (float).
    """
    from itertools import product
    spin_values = [ [a + 0.5*i for i in range(int((b-a)*2)+1)] for a, b in spin_ranges ]
    configs = list(product(*spin_values))
    total = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(amplitude_func, configs))
        total = sum(results)
    return total

# Placeholder for future GPU/tensor network support
def gpu_tensor_spinfoam_sum(*args, **kwargs):
    print("[Spinfoam] GPU/tensor network summation not yet implemented.")
    return None


if __name__ == '__main__':
    # Example usage:
    from lqg_simulation.core import SpinNetwork

    sn = SpinNetwork()
    n1 = sn.add_node(node_name="N1")
    n2 = sn.add_node(node_name="N2")
    n3 = sn.add_node(node_name="N3")
    n4 = sn.add_node(node_name="N4")
    n_center = sn.add_node(node_name="NC")

    # Connect outer nodes to the center node
    l1 = sn.add_link(n_center, n1, spin_j=S(1)/2, link_name="L_C1")
    l2 = sn.add_link(n_center, n2, spin_j=S(1),   link_name="L_C2")
    l3 = sn.add_link(n_center, n3, spin_j=S(3)/2, link_name="L_C3")
    l4 = sn.add_link(n_center, n4, spin_j=S(1),   link_name="L_C4")

    # Calculate placeholder amplitude for the center node
    # Spins are 1/2, 1, 1, 3/2. Sorted: 0.5, 1.0, 1.0, 1.5
    # j1=0.5, j2=1.0, j3=1.0 (top row of 6j)
    # j4=1.5 (bottom row of 6j)
    # Default intertwiner_spin will be min(non_zero_spins) = 0.5
    # So chosen_J_int = 0.5
    # 6j is {0.5, 1.0, 1.0 / 1.5, 0.5, 0.5}

    # Triads for {0.5, 1.0, 1.0 / 1.5, 0.5, 0.5}:
    # (0.5, 1.0, 1.0) -> sum=2.5 (NOT INTEGER) -> 6j should be 0 if strict, or Sympy handles.
    # Sympy's wigner_6j will return 0 if a triad sum is not an integer.
    # Let's test this:
    # from lqg_simulation.mathematics.wigner_symbols import calculate_wigner_6j
    # test_6j = calculate_wigner_6j(S(1)/2, S(1), S(1), S(3)/2, S(1)/2, S(1)/2)
    # print(f"Test 6j for example: {test_6j}") # Expected 0.0

    amp_nc_default_J = calculate_placeholder_vertex_amplitude(n_center, sn)
    print(f"Placeholder amplitude for NC (default J_int): {amp_nc_default_J}") # Expected 0.0

    # Try with a specified intertwiner spin, e.g., J_int = 1
    # Spins: 0.5, 1.0, 1.0, 1.5
    # j1=0.5, j2=1.0, j3=1.0
    # j4=1.5, chosen_J_int=1, chosen_J_int=1
    # 6j is {0.5, 1.0, 1.0 / 1.5, 1.0, 1.0}
    # Triads for {0.5, 1.0, 1.0 / 1.5, 1.0, 1.0}:
    # (0.5, 1.0, 1.0) -> sum=2.5 (NOT INTEGER) -> 6j should be 0.
    amp_nc_J1 = calculate_placeholder_vertex_amplitude(n_center, sn, intertwiner_spin=1)
    print(f"Placeholder amplitude for NC (J_int=1): {amp_nc_J1}") # Expected 0.0

    # To get a non-zero 6j, all 4 triads must have integer sums and satisfy triangle inequalities.
    # Example: {1,1,1,1,1,1} -> 1/6
    # Let's make links such that j1,j2,j3 = 1,1,1 and j4,J_int,J_int = 1,1,1
    sn_simple = SpinNetwork()
    n1s = sn_simple.add_node(node_name="N1s")
    n2s = sn_simple.add_node(node_name="N2s")
    n3s = sn_simple.add_node(node_name="N3s")
    n4s = sn_simple.add_node(node_name="N4s")
    ncs = sn_simple.add_node(node_name="NCs")

    sn_simple.add_link(ncs, n1s, spin_j=1) # j1
    sn_simple.add_link(ncs, n2s, spin_j=1) # j2
    sn_simple.add_link(ncs, n3s, spin_j=1) # j3
    sn_simple.add_link(ncs, n4s, spin_j=1) # j4 (used in 6j bottom row)

    # With intertwiner_spin = 1, we calculate {1,1,1 / 1,1,1}
    # val_6j = 1/6. chosen_J_int = 1.
    # Amplitude = (2*1+1) * (1/6) = 3 * 1/6 = 1/2 = 0.5
    amp_ncs_J1 = calculate_placeholder_vertex_amplitude(ncs, sn_simple, intertwiner_spin=1)
    print(f"Placeholder amplitude for NCs (J_int=1, all incident j=1): {amp_ncs_J1}") # Expected 0.5

    # Test non-4-valent node
    n_other = sn_simple.add_node(node_name="N_OTHER")
    sn_simple.add_link(ncs, n_other, spin_j=1) # NCs is now 5-valent
    amp_ncs_5valent = calculate_placeholder_vertex_amplitude(ncs, sn_simple, intertwiner_spin=1)
    print(f"Placeholder amplitude for NCs (5-valent): {amp_ncs_5valent}") # Expected 1.0 (trivial)

    # Test 3-valent node
    sn_3val = SpinNetwork()
    n1_3v = sn_3val.add_node("N1")
    n2_3v = sn_3val.add_node("N2")
    n3_3v = sn_3val.add_node("N3")
    nc_3v = sn_3val.add_node("NC_3V")
    sn_3val.add_link(nc_3v, n1_3v, 1)
    sn_3val.add_link(nc_3v, n2_3v, 1)
    sn_3val.add_link(nc_3v, n3_3v, 1)
    amp_nc_3v = calculate_placeholder_vertex_amplitude(nc_3v, sn_3val)
    print(f"Placeholder amplitude for NC_3V (3-valent): {amp_nc_3v}") # Expected 1.0

    # Test with default intertwiner spin when all incident spins are 0
    sn_zeros = SpinNetwork()
    nz1 = sn_zeros.add_node("NZ1")
    nz2 = sn_zeros.add_node("NZ2")
    nz3 = sn_zeros.add_node("NZ3")
    nz4 = sn_zeros.add_node("NZ4")
    ncz = sn_zeros.add_node("NCZ")
    sn_zeros.add_link(ncz, nz1, 0)
    sn_zeros.add_link(ncz, nz2, 0)
    sn_zeros.add_link(ncz, nz3, 0)
    sn_zeros.add_link(ncz, nz4, 0)
    # j1=0,j2=0,j3=0, j4=0. Default J_int = 1.
    # 6j is {0,0,0 / 0,1,1}. (0,0,0) sum=0 (int). (0,1,1) sum=2 (int). (0,0,1) sum=1 (int). (0,1,1) sum=2(int).
    # Value of {000,011} is 1/( (2*0+1)(2*1+1) ) = 1/3 if first row is (0,j,j) and second is (0,k,k)... no, this is specific formula.
    # {0 0 0} = (-1)^(j+k+l) / sqrt((2j+1)(2k+1)) for {0 j j; l k k} type structure... not quite.
    # {j1 j2 j3}
    # {j4 j5 j6}
    # If j1=0, then j2=j3 and j5=j6. Symbol becomes (-1)^(j2+j4+j5) / sqrt((2j2+1)(2j4+1)) * delta(j2,j4,j5) (triangle)
    # So for {0,0,0 / 0,1,1}: j1=0 -> j2=0, j3=0. j5=1, j6=1.
    # (-1)^(0+0+1) / sqrt((2*0+1)(2*0+1)) * delta(0,0,1)
    # = -1 / 1 * delta(0,0,1). (0,0,1) is a triangle. So result is -1.
    # val_6j = -1. chosen_J_int = 1.
    # Amplitude = (2*1+1) * (-1) = 3 * (-1) = -3.0
    amp_ncz_J_default = calculate_placeholder_vertex_amplitude(ncz, sn_zeros)
    print(f"Placeholder amplitude for NCZ (default J_int, all incident j=0): {amp_ncz_J_default}") # Expected -3.0

    # Test with S(0) intertwiner spin explicitly
    # 6j is {0,0,0 / 0,0,0}. Result is 1.0.
    # Amplitude = (2*0+1)*1.0 = 1.0
    amp_ncz_J0 = calculate_placeholder_vertex_amplitude(ncz, sn_zeros, intertwiner_spin=0)
    print(f"Placeholder amplitude for NCZ (J_int=0, all incident j=0): {amp_ncz_J0}") # Expected 1.0

    print("Completed amplitude examples.")

    print("\n--- Ponzano-Regge Spinfoam Amplitude Example ---")
    # Create a spin network with the topology of a tetrahedron
    sn_tetra = SpinNetwork()
    t_n1 = sn_tetra.add_node("T1")
    t_n2 = sn_tetra.add_node("T2")
    t_n3 = sn_tetra.add_node("T3")
    t_n4 = sn_tetra.add_node("T4")

    # 6 links forming a complete graph on 4 nodes (a tetrahedron)
    # Let's use spins that give a non-zero 6j symbol, e.g., all j=1
    sn_tetra.add_link(t_n1, t_n2, spin_j=1)
    sn_tetra.add_link(t_n1, t_n3, spin_j=1)
    sn_tetra.add_link(t_n1, t_n4, spin_j=1)
    sn_tetra.add_link(t_n2, t_n3, spin_j=1)
    sn_tetra.add_link(t_n2, t_n4, spin_j=1)
    sn_tetra.add_link(t_n3, t_n4, spin_j=1)

    # The 6j symbol {1,1,1; 1,1,1} is 1/6. This is the vertex amplitude.
    # The face amplitude for j=1 is (2*1+1) = 3. There are 6 faces.
    # The product of face amplitudes is 3^6 = 729.
    # The total amplitude is (1/6) * 729 = 121.5.
    total_amp, vertex_amp, face_prod = calculate_total_spinfoam_amplitude(sn_tetra)

    print(f"Ponzano-Regge vertex amplitude for a tetrahedron with all j=1 spins: {vertex_amp:.5f}")
    print(f"(Expected: 1/6 = {1/6:.5f})")
    print(f"Product of face amplitudes (6 faces with j=1, amp=2j+1=3): {face_prod:.1f}")
    print(f"(Expected: 3^6 = 729.0)")
    print(f"Total spinfoam amplitude: {total_amp:.2f}")
    print(f"(Expected: (1/6) * 729 = 121.50)")
