import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass
import json

@dataclass
class SpinNode:
    """Represents a node in a spin network with quantum numbers and geometric properties."""
    
    id: str
    spin: float  # SU(2) representation label
    position: Tuple[float, float]  # 2D position for visualization
    intertwiners: Optional[List[float]] = None  # Intertwiner quantum numbers
    
    def __post_init__(self):
        if self.intertwiners is None:
            self.intertwiners = []
    
    def quantum_dimension(self) -> int:
        """Calculate the quantum dimension of the node."""
        return int(2 * self.spin + 1)
    
    def to_dict(self) -> Dict:
        """Convert node to dictionary for serialization."""
        return {
            'id': self.id,
            'spin': self.spin,
            'position': self.position,
            'intertwiners': self.intertwiners
        }

@dataclass
class SpinLink:
    """Represents a link in a spin network connecting two nodes."""
    
    source: str
    target: str
    spin: float  # SU(2) representation label for the link
    color: str = "red"  # For visualization
    
    def quantum_dimension(self) -> int:
        """Calculate the quantum dimension of the link."""
        return int(2 * self.spin + 1)
    
    def area_eigenvalue(self) -> float:
        """Calculate the area eigenvalue for this link in Planck units."""
        return np.sqrt(self.spin * (self.spin + 1))
    
    def to_dict(self) -> Dict:
        """Convert link to dictionary for serialization."""
        return {
            'source': self.source,
            'target': self.target,
            'spin': self.spin,
            'color': self.color
        }

class SpinNetwork:
    """Main class for representing and manipulating spin networks."""
    
    def __init__(self):
        self.nodes: List[SpinNode] = []
        self.links: List[SpinLink] = []
        self.graph = nx.Graph()
    
    def add_node(self, node: SpinNode):
        """Add a node to the spin network."""
        self.nodes.append(node)
        self.graph.add_node(node.id, spin=node.spin, position=node.position)
    
    def add_link(self, link: SpinLink):
        """Add a link to the spin network."""
        self.links.append(link)
        self.graph.add_edge(link.source, link.target, spin=link.spin)
    
    def remove_node(self, node_id: str):
        """Remove a node and all connected links."""
        # Remove links connected to this node
        self.links = [link for link in self.links 
                     if link.source != node_id and link.target != node_id]
        
        # Remove the node
        self.nodes = [node for node in self.nodes if node.id != node_id]
        
        # Update graph
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)
    
    def remove_link(self, source: str, target: str):
        """Remove a link between two nodes."""
        self.links = [link for link in self.links 
                     if not ((link.source == source and link.target == target) or 
                            (link.source == target and link.target == source))]
        
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
    
    def get_node(self, node_id: str) -> Optional[SpinNode]:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all neighbors of a node."""
        return list(self.graph.neighbors(node_id))
    
    def get_node_degree(self, node_id: str) -> int:
        """Get the degree (number of connections) of a node."""
        return self.graph.degree(node_id)
    
    def calculate_statistics(self) -> Dict:
        """Calculate various statistics of the spin network."""
        num_nodes = len(self.nodes)
        num_links = len(self.links)
        
        if num_nodes == 0:
            return {
                'num_nodes': 0,
                'num_links': 0,
                'avg_degree': 0,
                'total_spin': 0,
                'avg_spin': 0
            }
        
        degrees = [self.get_node_degree(node.id) for node in self.nodes]
        avg_degree = np.mean(degrees) if degrees else 0
        
        total_node_spin = sum(node.spin for node in self.nodes)
        total_link_spin = sum(link.spin for link in self.links)
        avg_node_spin = total_node_spin / num_nodes if num_nodes > 0 else 0
        
        return {
            'num_nodes': num_nodes,
            'num_links': num_links,
            'avg_degree': avg_degree,
            'total_node_spin': total_node_spin,
            'total_link_spin': total_link_spin,
            'avg_node_spin': avg_node_spin,
            'max_degree': max(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0
        }
    
    def validate_network(self) -> bool:
        """Validate that the network satisfies basic consistency conditions."""
        # Check that all links connect existing nodes
        node_ids = {node.id for node in self.nodes}
        
        for link in self.links:
            if link.source not in node_ids or link.target not in node_ids:
                return False
        
        # Check that spins are valid (non-negative, half-integer)
        for node in self.nodes:
            if node.spin < 0 or (2 * node.spin) % 1 != 0:
                return False
        
        for link in self.links:
            if link.spin < 0 or (2 * link.spin) % 1 != 0:
                return False
        
        return True
    
    def apply_move(self, move_type: str, **kwargs):
        """Apply a local move to the spin network."""
        if move_type == "change_spin":
            node_id = kwargs.get('node_id')
            new_spin = kwargs.get('new_spin')
            node = self.get_node(node_id)
            if node:
                node.spin = new_spin
        
        elif move_type == "add_node":
            node = SpinNode(
                id=kwargs.get('node_id'),
                spin=kwargs.get('spin', 1.0),
                position=kwargs.get('position', (0.0, 0.0))
            )
            self.add_node(node)
        
        elif move_type == "pachner_move":
            # Implement a simplified Pachner move
            self._apply_pachner_move(kwargs.get('vertices', []))
    
    def _apply_pachner_move(self, vertices: List[str]):
        """Apply a Pachner move (topological change) to the network."""
        # Simplified implementation of a 2-3 Pachner move
        if len(vertices) == 2:
            # Split an edge into two edges with a new vertex
            v1, v2 = vertices
            new_vertex_id = f"pachner_{len(self.nodes)}"
            
            # Remove old link
            self.remove_link(v1, v2)
            
            # Add new vertex at midpoint
            node1 = self.get_node(v1)
            node2 = self.get_node(v2)
            if node1 and node2:
                mid_pos = (
                    (node1.position[0] + node2.position[0]) / 2,
                    (node1.position[1] + node2.position[1]) / 2
                )
                new_node = SpinNode(new_vertex_id, 1.0, mid_pos)
                self.add_node(new_node)
                
                # Add new links
                self.add_link(SpinLink(v1, new_vertex_id, 1.0))
                self.add_link(SpinLink(new_vertex_id, v2, 1.0))
    
    def evolve_quantum(self, time_step: float = 0.1, hamiltonian_type: str = "spin_spin"):
        """Evolve the spin network according to a quantum Hamiltonian."""
        # Simplified quantum evolution
        if hamiltonian_type == "spin_spin":
            # Spin-spin interaction Hamiltonian
            for node in self.nodes:
                # Small random perturbation representing quantum fluctuations
                perturbation = 0.01 * time_step * np.random.randn()
                node.spin = max(0, node.spin + perturbation)
        
        elif hamiltonian_type == "area_volume":
            # Area-volume constraint evolution
            for link in self.links:
                perturbation = 0.01 * time_step * np.random.randn()
                link.spin = max(0, link.spin + perturbation)
    
    def calculate_total_volume(self) -> float:
        """Calculate the total volume of the spin network."""
        total_volume = 0
        for node in self.nodes:
            # Volume eigenvalue approximation
            neighbors = self.get_neighbors(node.id)
            neighbor_spins = []
            for neighbor_id in neighbors:
                for link in self.links:
                    if ((link.source == node.id and link.target == neighbor_id) or 
                        (link.source == neighbor_id and link.target == node.id)):
                        neighbor_spins.append(link.spin)
            
            if neighbor_spins:
                # Simplified volume calculation
                volume_contribution = np.sqrt(np.prod(neighbor_spins)) * node.spin
                total_volume += volume_contribution
        
        return total_volume
    
    def calculate_total_area(self) -> float:
        """Calculate the total area of the spin network."""
        return sum(link.area_eigenvalue() for link in self.links)
    
    def to_dict(self) -> Dict:
        """Convert the entire network to a dictionary for serialization."""
        return {
            'nodes': [node.to_dict() for node in self.nodes],
            'links': [link.to_dict() for link in self.links],
            'statistics': self.calculate_statistics()
        }
    
    def from_dict(self, data: Dict):
        """Load network from dictionary."""
        self.nodes = []
        self.links = []
        self.graph = nx.Graph()
        
        for node_data in data['nodes']:
            node = SpinNode(
                id=node_data['id'],
                spin=node_data['spin'],
                position=tuple(node_data['position']),
                intertwiners=node_data.get('intertwiners', [])
            )
            self.add_node(node)
        
        for link_data in data['links']:
            link = SpinLink(
                source=link_data['source'],
                target=link_data['target'],
                spin=link_data['spin'],
                color=link_data.get('color', 'red')
            )
            self.add_link(link)
    
    def save_to_file(self, filename: str):
        """Save the network to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load_from_file(self, filename: str):
        """Load the network from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        self.from_dict(data)
    
    @classmethod
    def generate_random(cls, num_nodes: int, connectivity: float = 0.3, 
                       max_spin: float = 2.0) -> 'SpinNetwork':
        """Generate a random spin network for testing and exploration."""
        network = cls()
        
        # Generate nodes
        for i in range(num_nodes):
            node = SpinNode(
                id=f"node_{i}",
                spin=random.uniform(0.5, max_spin),
                position=(
                    random.uniform(-5, 5),
                    random.uniform(-5, 5)
                )
            )
            network.add_node(node)
        
        # Generate links
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < connectivity:
                    link = SpinLink(
                        source=f"node_{i}",
                        target=f"node_{j}",
                        spin=random.uniform(0.5, max_spin)
                    )
                    network.add_link(link)
        
        return network
    
    @classmethod
    def generate_regular_lattice(cls, size: int, lattice_type: str = "square") -> 'SpinNetwork':
        """Generate a regular lattice spin network."""
        network = cls()
        
        if lattice_type == "square":
            # Generate square lattice
            for i in range(size):
                for j in range(size):
                    node = SpinNode(
                        id=f"node_{i}_{j}",
                        spin=1.0,
                        position=(i, j)
                    )
                    network.add_node(node)
            
            # Add horizontal links
            for i in range(size):
                for j in range(size - 1):
                    link = SpinLink(
                        source=f"node_{i}_{j}",
                        target=f"node_{i}_{j+1}",
                        spin=1.0
                    )
                    network.add_link(link)
            
            # Add vertical links
            for i in range(size - 1):
                for j in range(size):
                    link = SpinLink(
                        source=f"node_{i}_{j}",
                        target=f"node_{i+1}_{j}",
                        spin=1.0
                    )
                    network.add_link(link)
        
        elif lattice_type == "triangular":
            # Generate triangular lattice
            for i in range(size):
                for j in range(size):
                    node = SpinNode(
                        id=f"node_{i}_{j}",
                        spin=1.0,
                        position=(i + 0.5 * (j % 2), j * np.sqrt(3) / 2)
                    )
                    network.add_node(node)
            
            # Add links for triangular lattice
            for i in range(size):
                for j in range(size):
                    # Horizontal links
                    if j < size - 1:
                        link = SpinLink(
                            source=f"node_{i}_{j}",
                            target=f"node_{i}_{j+1}",
                            spin=1.0
                        )
                        network.add_link(link)
                    
                    # Diagonal links
                    if i < size - 1:
                        link = SpinLink(
                            source=f"node_{i}_{j}",
                            target=f"node_{i+1}_{j}",
                            spin=1.0
                        )
                        network.add_link(link)
        
        return network
