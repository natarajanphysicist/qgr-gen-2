import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import random
from core.wigner_symbols import WignerSymbols
from core.spin_networks import SpinNetwork
import warnings

@dataclass
class SpinfoamVertex:
    """Represents a vertex in a spinfoam complex."""
    
    id: str
    vertex_type: str  # "tetrahedron", "4-simplex", etc.
    spins: List[float]  # Associated spin values
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    intertwiners: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.intertwiners:
            # Initialize intertwiners based on vertex type
            if self.vertex_type == "tetrahedron":
                self.intertwiners = [1.0] * 4  # 4 intertwiners for tetrahedron
            elif self.vertex_type == "4-simplex":
                self.intertwiners = [1.0] * 5  # 5 intertwiners for 4-simplex
    
    def calculate_amplitude(self, model_type: str = "EPRL-FK", 
                          immirzi_parameter: float = 0.2375) -> complex:
        """Calculate the vertex amplitude for this vertex."""
        wigner_calc = WignerSymbols()
        
        if self.vertex_type == "tetrahedron" and model_type == "Ooguri":
            return self._calculate_ooguri_amplitude(wigner_calc)
        elif self.vertex_type == "4-simplex" and model_type == "EPRL-FK":
            return self._calculate_eprl_amplitude(wigner_calc, immirzi_parameter)
        elif self.vertex_type == "tetrahedron" and model_type == "Barrett-Crane":
            return self._calculate_barrett_crane_amplitude(wigner_calc)
        else:
            # Generic amplitude calculation
            return self._calculate_generic_amplitude(wigner_calc)
    
    def _calculate_ooguri_amplitude(self, wigner_calc: WignerSymbols) -> complex:
        """Calculate Ooguri model vertex amplitude (3D)."""
        if len(self.spins) < 6:
            warnings.warn("Insufficient spins for Ooguri amplitude calculation")
            return 0.0
        
        # Ooguri amplitude is essentially a 6j symbol
        amplitude = wigner_calc.calculate_6j(*self.spins[:6])
        
        # Apply quantum corrections
        quantum_dimension = np.prod([2*j + 1 for j in self.spins[:6]])
        
        return amplitude * np.sqrt(quantum_dimension)
    
    def _calculate_eprl_amplitude(self, wigner_calc: WignerSymbols, 
                                 immirzi_parameter: float) -> complex:
        """Calculate EPRL-FK model vertex amplitude (4D)."""
        if len(self.spins) < 15:
            warnings.warn("Insufficient spins for EPRL amplitude calculation")
            return 0.0
        
        # EPRL amplitude involves 15j symbols
        amplitude_15j = wigner_calc.calculate_15j(*self.spins[:15])
        
        # Apply Immirzi parameter corrections
        immirzi_factor = np.exp(1j * immirzi_parameter * sum(self.spins[:5]))
        
        # Quantum dimensions
        quantum_dimensions = [2*j + 1 for j in self.spins[:10]]
        normalization = np.sqrt(np.prod(quantum_dimensions))
        
        return amplitude_15j * immirzi_factor * normalization
    
    def _calculate_barrett_crane_amplitude(self, wigner_calc: WignerSymbols) -> complex:
        """Calculate Barrett-Crane model vertex amplitude."""
        if len(self.spins) < 6:
            warnings.warn("Insufficient spins for Barrett-Crane amplitude calculation")
            return 0.0
        
        # Barrett-Crane amplitude
        amplitude = wigner_calc.calculate_6j(*self.spins[:6])
        
        # Apply constraint factors
        constraint_factor = 1.0
        for j in self.spins[:6]:
            constraint_factor *= np.sqrt(2*j + 1)
        
        return amplitude * constraint_factor
    
    def _calculate_generic_amplitude(self, wigner_calc: WignerSymbols) -> complex:
        """Calculate a generic vertex amplitude."""
        if len(self.spins) < 3:
            return 1.0
        
        # Use product of 3j symbols as generic amplitude
        amplitude = 1.0
        for i in range(0, len(self.spins)-2, 3):
            if i+2 < len(self.spins):
                # Use zero magnetic quantum numbers for simplicity
                three_j = wigner_calc.calculate_3j(
                    self.spins[i], self.spins[i+1], self.spins[i+2],
                    0, 0, 0
                )
                amplitude *= three_j
        
        return amplitude
    
    def get_boundary_data(self) -> Dict:
        """Get boundary data for this vertex."""
        return {
            'vertex_id': self.id,
            'spins': self.spins,
            'intertwiners': self.intertwiners,
            'vertex_type': self.vertex_type
        }

@dataclass
class SpinfoamEdge:
    """Represents an edge in a spinfoam complex."""
    
    id: str
    vertex1: str
    vertex2: str
    face_spin: float
    area: float = 0.0
    
    def __post_init__(self):
        if self.area == 0.0:
            # Calculate area from spin
            self.area = np.sqrt(self.face_spin * (self.face_spin + 1))
    
    def calculate_edge_amplitude(self, model_type: str = "EPRL-FK") -> complex:
        """Calculate the edge amplitude."""
        if model_type == "EPRL-FK":
            # EPRL edge amplitude
            return (2 * self.face_spin + 1) * np.exp(1j * self.area)
        else:
            # Generic edge amplitude
            return np.sqrt(2 * self.face_spin + 1)

@dataclass
class SpinfoamFace:
    """Represents a face in a spinfoam complex."""
    
    id: str
    edges: List[str]
    spin: float
    area: float = 0.0
    
    def __post_init__(self):
        if self.area == 0.0:
            self.area = np.sqrt(self.spin * (self.spin + 1))
    
    def calculate_face_amplitude(self) -> complex:
        """Calculate the face amplitude."""
        # Face amplitude is typically just the quantum dimension
        return np.sqrt(2 * self.spin + 1)

class SpinfoamComplex:
    """Main class for representing spinfoam complexes."""
    
    def __init__(self):
        self.vertices: List[SpinfoamVertex] = []
        self.edges: List[SpinfoamEdge] = []
        self.faces: List[SpinfoamFace] = []
        self.model_type = "EPRL-FK"
        self.immirzi_parameter = 0.2375
    
    def add_vertex(self, vertex: SpinfoamVertex):
        """Add a vertex to the complex."""
        self.vertices.append(vertex)
    
    def add_edge(self, edge: SpinfoamEdge):
        """Add an edge to the complex."""
        self.edges.append(edge)
    
    def add_face(self, face: SpinfoamFace):
        """Add a face to the complex."""
        self.faces.append(face)
    
    def calculate_total_amplitude(self) -> complex:
        """Calculate the total amplitude of the spinfoam complex."""
        total_amplitude = 1.0
        
        # Vertex contributions
        for vertex in self.vertices:
            vertex_amplitude = vertex.calculate_amplitude(
                self.model_type, self.immirzi_parameter
            )
            total_amplitude *= vertex_amplitude
        
        # Edge contributions
        for edge in self.edges:
            edge_amplitude = edge.calculate_edge_amplitude(self.model_type)
            total_amplitude *= edge_amplitude
        
        # Face contributions
        for face in self.faces:
            face_amplitude = face.calculate_face_amplitude()
            total_amplitude *= face_amplitude
        
        return total_amplitude
    
    def calculate_partition_function(self, boundary_data: Dict) -> complex:
        """Calculate the partition function with given boundary data."""
        # Apply boundary conditions
        self._apply_boundary_conditions(boundary_data)
        
        # Calculate amplitude
        amplitude = self.calculate_total_amplitude()
        
        # Sum over internal degrees of freedom (simplified)
        partition_function = 0.0
        for spin_config in self._generate_spin_configurations():
            self._set_spin_configuration(spin_config)
            config_amplitude = self.calculate_total_amplitude()
            partition_function += abs(config_amplitude)**2
        
        return partition_function
    
    def _apply_boundary_conditions(self, boundary_data: Dict):
        """Apply boundary conditions to the complex."""
        # Simplified boundary condition application
        for vertex in self.vertices:
            vertex_id = vertex.id
            if vertex_id in boundary_data:
                vertex.spins = boundary_data[vertex_id].get('spins', vertex.spins)
                vertex.intertwiners = boundary_data[vertex_id].get('intertwiners', vertex.intertwiners)
    
    def _generate_spin_configurations(self) -> List[Dict]:
        """Generate possible spin configurations for summation."""
        # Simplified configuration generation
        configurations = []
        
        # Generate a few sample configurations
        for _ in range(10):  # Limited for computational efficiency
            config = {}
            for vertex in self.vertices:
                config[vertex.id] = {
                    'spins': [random.uniform(0.5, 2.0) for _ in vertex.spins],
                    'intertwiners': [random.uniform(0.5, 2.0) for _ in vertex.intertwiners]
                }
            configurations.append(config)
        
        return configurations
    
    def _set_spin_configuration(self, config: Dict):
        """Set the spin configuration for the complex."""
        for vertex in self.vertices:
            if vertex.id in config:
                vertex.spins = config[vertex.id]['spins']
                vertex.intertwiners = config[vertex.id]['intertwiners']
    
    def calculate_statistics(self) -> Dict:
        """Calculate statistics of the spinfoam complex."""
        return {
            'num_vertices': len(self.vertices),
            'num_edges': len(self.edges),
            'num_faces': len(self.faces),
            'total_spins': sum(len(v.spins) for v in self.vertices),
            'total_area': sum(e.area for e in self.edges),
            'avg_vertex_spins': np.mean([len(v.spins) for v in self.vertices]) if self.vertices else 0
        }
    
    def evolve_complex(self, time_step: float = 0.1):
        """Evolve the spinfoam complex in time."""
        for vertex in self.vertices:
            # Apply small perturbations to spins
            for i in range(len(vertex.spins)):
                perturbation = 0.01 * time_step * np.random.randn()
                vertex.spins[i] = max(0.5, vertex.spins[i] + perturbation)
        
        # Update edge areas
        for edge in self.edges:
            edge.area = np.sqrt(edge.face_spin * (edge.face_spin + 1))
    
    def calculate_observables(self) -> Dict:
        """Calculate physical observables from the spinfoam complex."""
        # Total volume
        total_volume = 0.0
        for vertex in self.vertices:
            if vertex.vertex_type == "tetrahedron":
                # Volume eigenvalue for tetrahedron
                volume_contribution = np.sqrt(np.prod(vertex.spins[:4]))
                total_volume += volume_contribution
            elif vertex.vertex_type == "4-simplex":
                # Volume eigenvalue for 4-simplex
                volume_contribution = np.sqrt(np.prod(vertex.spins[:5]))
                total_volume += volume_contribution
        
        # Total area
        total_area = sum(edge.area for edge in self.edges)
        
        # Curvature (simplified)
        curvature = 0.0
        for vertex in self.vertices:
            local_curvature = sum(vertex.spins) - len(vertex.spins)
            curvature += local_curvature
        
        return {
            'total_volume': total_volume,
            'total_area': total_area,
            'curvature': curvature,
            'euler_characteristic': len(self.vertices) - len(self.edges) + len(self.faces)
        }
    
    def generate_boundary_spin_network(self) -> SpinNetwork:
        """Generate a boundary spin network from the spinfoam complex."""
        spin_network = SpinNetwork()
        
        # Create nodes from vertices
        for vertex in self.vertices:
            from core.spin_networks import SpinNode
            node = SpinNode(
                id=vertex.id,
                spin=vertex.spins[0] if vertex.spins else 1.0,
                position=(vertex.position[0], vertex.position[1])
            )
            spin_network.add_node(node)
        
        # Create links from edges
        for edge in self.edges:
            from core.spin_networks import SpinLink
            link = SpinLink(
                source=edge.vertex1,
                target=edge.vertex2,
                spin=edge.face_spin
            )
            spin_network.add_link(link)
        
        return spin_network
    
    @classmethod
    def generate_complex(cls, num_vertices: int, model_type: str = "EPRL-FK", 
                        immirzi_parameter: float = 0.2375) -> 'SpinfoamComplex':
        """Generate a random spinfoam complex with valid spins for nonzero amplitudes."""
        import itertools
        complex_obj = cls()
        complex_obj.model_type = model_type
        complex_obj.immirzi_parameter = immirzi_parameter

        def valid_triangle(j1, j2, j3):
            return abs(j1-j2) <= j3 <= j1+j2 and (j1+j2+j3) % 1 == 0


        def generate_valid_6j():
            # Generate 6 spins that satisfy triangle inequalities for all 4 triangles in a tetrahedron
            # Use a deterministic approach to always find a valid set
            base = random.choice([0.5, 1.0, 1.5, 2.0])
            j = [base, base, base, base, base, base]
            return j

        def generate_valid_15j():
            # Generate 15 spins that are all the same, always valid for triangle inequalities
            base = random.choice([0.5, 1.0, 1.5, 2.0])
            j = [base] * 15
            return j

        # Generate vertices
        for i in range(num_vertices):
            if model_type == "EPRL-FK":
                vertex_type = "4-simplex"
                spins = generate_valid_15j()
            elif model_type == "Ooguri":
                vertex_type = "tetrahedron"
                spins = generate_valid_6j()
            else:
                vertex_type = "tetrahedron"
                spins = generate_valid_6j()

            vertex = SpinfoamVertex(
                id=f"vertex_{i}",
                vertex_type=vertex_type,
                spins=spins,
                position=(
                    random.uniform(-5, 5),
                    random.uniform(-5, 5),
                    random.uniform(-5, 5)
                )
            )
            complex_obj.add_vertex(vertex)

        # Generate edges
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                if random.random() < 0.3:  # 30% connectivity
                    edge = SpinfoamEdge(
                        id=f"edge_{i}_{j}",
                        vertex1=f"vertex_{i}",
                        vertex2=f"vertex_{j}",
                        face_spin=random.choice([0.5, 1.0, 1.5, 2.0])
                    )
                    complex_obj.add_edge(edge)

        # Generate faces (simplified)
        for i, edge in enumerate(complex_obj.edges):
            face = SpinfoamFace(
                id=f"face_{i}",
                edges=[edge.id],
                spin=edge.face_spin
            )
            complex_obj.add_face(face)

        return complex_obj
    def advanced_amplitude_analysis(self, n_samples: int = 20) -> dict:
        """Advanced research: sample amplitudes, log-amplitudes, and quantum fluctuations."""
        amplitudes = []
        for _ in range(n_samples):
            # Randomize spins (valid)
            for v in self.vertices:
                if v.vertex_type == "tetrahedron":
                    v.spins = [random.choice([0.5, 1.0, 1.5, 2.0]) for _ in range(6)]
                else:
                    v.spins = [random.choice([0.5, 1.0, 1.5, 2.0]) for _ in range(15)]
            amp = self.calculate_total_amplitude()
            amplitudes.append(amp)
        abs_amps = [abs(a) for a in amplitudes]
        log_amps = [np.log(abs(a)) if abs(a) > 0 else -np.inf for a in amplitudes]
        return {
            "amplitudes": amplitudes,
            "abs_amplitudes": abs_amps,
            "log_amplitudes": log_amps,
            "mean_abs": np.mean(abs_amps),
            "std_abs": np.std(abs_amps),
            "mean_log": np.mean([l for l in log_amps if l > -np.inf]) if any(l > -np.inf for l in log_amps) else -np.inf,
            "std_log": np.std([l for l in log_amps if l > -np.inf]) if any(l > -np.inf for l in log_amps) else 0.0
        }
    
    def to_dict(self) -> Dict:
        """Convert the complex to a dictionary for serialization."""
        return {
            'vertices': [
                {
                    'id': v.id,
                    'vertex_type': v.vertex_type,
                    'spins': v.spins,
                    'position': v.position,
                    'intertwiners': v.intertwiners
                } for v in self.vertices
            ],
            'edges': [
                {
                    'id': e.id,
                    'vertex1': e.vertex1,
                    'vertex2': e.vertex2,
                    'face_spin': e.face_spin,
                    'area': e.area
                } for e in self.edges
            ],
            'faces': [
                {
                    'id': f.id,
                    'edges': f.edges,
                    'spin': f.spin,
                    'area': f.area
                } for f in self.faces
            ],
            'model_type': self.model_type,
            'immirzi_parameter': self.immirzi_parameter
        }
