"""
Quantum Information Tools for Quantum Gravity Simulation Platform
- Entanglement entropy for spin networks
- Entanglement entropy for spinfoam boundaries
- (Future) Mutual information, holographic diagnostics, etc.
"""
import numpy as np

# Example: Von Neumann entropy for a reduced density matrix

def von_neumann_entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]  # filter numerical noise
    return -np.sum(eigvals * np.log(eigvals))

# --- Spin Network Entanglement ---
def calculate_spin_network_entanglement(network, region_nodes):
    """
    Compute entanglement entropy for a region (set of nodes) in a spin network.
    Args:
        network: SpinNetwork object
        region_nodes: list of node IDs defining the region
    Returns:
        entropy: float (Von Neumann entropy, placeholder)
    """
    # Placeholder: In real LQG, this requires constructing the reduced density matrix for the region.
    # Here, we use a toy model: entropy = log(number of boundary links)
    boundary_links = [l for l in network.links if (l.source in region_nodes) != (l.target in region_nodes)]
    n_boundary = len(boundary_links)
    if n_boundary == 0:
        return 0.0
    return np.log(n_boundary)

# --- Spinfoam Boundary Entanglement ---
def calculate_spinfoam_boundary_entanglement(spinfoam_complex, boundary_vertices):
    """
    Compute entanglement entropy for a set of boundary vertices in a spinfoam complex.
    Args:
        spinfoam_complex: SpinfoamComplex object
        boundary_vertices: list of vertex IDs
    Returns:
        entropy: float (Von Neumann entropy, placeholder)
    """
    # A face is considered a boundary face if any of its edge's vertices are in the boundary_vertices set
    boundary_vertex_set = set(boundary_vertices)
    boundary_faces = []
    for face in spinfoam_complex.faces:
        for edge_id in face.edges:
            edge = next((e for e in spinfoam_complex.edges if e.id == edge_id), None)
            if edge and (edge.vertex1 in boundary_vertex_set or edge.vertex2 in boundary_vertex_set):
                boundary_faces.append(face)
                break
    n_boundary = len(boundary_faces)
    if n_boundary == 0:
        return 0.0
    return np.log(n_boundary)

# --- Mutual Information (Spin Networks) ---
def calculate_spin_network_mutual_information(network, regionA_nodes, regionB_nodes):
    """
    Compute mutual information between two regions in a spin network (toy model).
    Args:
        network: SpinNetwork object
        regionA_nodes: list of node IDs for region A
        regionB_nodes: list of node IDs for region B
    Returns:
        mutual_info: float (placeholder)
    """
    # Toy model: mutual info = log(number of links connecting A and B)
    ab_links = [l for l in network.links if (l.source in regionA_nodes and l.target in regionB_nodes) or (l.source in regionB_nodes and l.target in regionA_nodes)]
    n_ab = len(ab_links)
    if n_ab == 0:
        return 0.0
    return np.log(n_ab)

# --- Mutual Information (Spinfoam Boundaries) ---
def calculate_spinfoam_boundary_mutual_information(spinfoam_complex, boundaryA_vertices, boundaryB_vertices):
    """
    Compute mutual information between two sets of boundary vertices in a spinfoam complex (toy model).
    Args:
        spinfoam_complex: SpinfoamComplex object
        boundaryA_vertices: list of vertex IDs for region A
        boundaryB_vertices: list of vertex IDs for region B
    Returns:
        mutual_info: float (placeholder)
    """
    # Toy model: mutual info = log(number of faces adjacent to both A and B)
    setA = set(boundaryA_vertices)
    setB = set(boundaryB_vertices)
    shared_faces = []
    for face in spinfoam_complex.faces:
        face_vertex_ids = set()
        for edge_id in face.edges:
            edge = next((e for e in spinfoam_complex.edges if e.id == edge_id), None)
            if edge:
                face_vertex_ids.add(edge.vertex1)
                face_vertex_ids.add(edge.vertex2)
        if face_vertex_ids & setA and face_vertex_ids & setB:
            shared_faces.append(face)
    n_shared = len(shared_faces)
    if n_shared == 0:
        return 0.0
    return np.log(n_shared)

# Future: Add holographic diagnostics, advanced quantum information tools, etc.
