import numpy as np
from typing import List, Dict, Tuple, Optional
from core.spin_networks import SpinNetwork, SpinNode, SpinLink
from core.geometry import GeometricObservables
from core.wigner_symbols import WignerSymbols
import random

class SimpleEvolution:
    """Simple evolution examples for spin networks."""
    
    def __init__(self):
        self.wigner_calc = WignerSymbols()
        self.geometry = GeometricObservables()
    
    def run_evolution(self, network: SpinNetwork, time_steps: int = 50,
                     perturbation_strength: float = 0.1) -> Dict[str, any]:
        """
        Run a simple evolution of the spin network.
        
        Args:
            network: The spin network to evolve
            time_steps: Number of evolution steps
            perturbation_strength: Strength of quantum perturbations
            
        Returns:
            Dictionary containing evolution results
        """
        # Initialize storage for observables
        volume_evolution = []
        area_evolution = []
        curvature_evolution = []
        node_spin_evolution = []
        link_spin_evolution = []
        
        # Store initial state
        initial_volume = network.calculate_total_volume()
        initial_area = network.calculate_total_area()
        initial_curvature = self.geometry.calculate_curvature(network)
        
        # Track initial spins
        initial_node_spins = [node.spin for node in network.nodes]
        initial_link_spins = [link.spin for link in network.links]
        
        # Evolution loop
        for step in range(time_steps):
            # Apply quantum evolution
            self._apply_quantum_evolution(network, perturbation_strength)
            
            # Calculate observables
            current_volume = network.calculate_total_volume()
            current_area = network.calculate_total_area()
            current_curvature = self.geometry.calculate_curvature(network)
            
            # Store observables
            volume_evolution.append(current_volume)
            area_evolution.append(current_area)
            curvature_evolution.append(current_curvature)
            
            # Store spin evolution
            node_spins = [node.spin for node in network.nodes]
            link_spins = [link.spin for link in network.links]
            node_spin_evolution.append(np.mean(node_spins) if node_spins else 0)
            link_spin_evolution.append(np.mean(link_spins) if link_spins else 0)
            
            # Apply local moves occasionally
            if step % 10 == 0:
                self._apply_local_moves(network)
        
        return {
            'volume_evolution': volume_evolution,
            'area_evolution': area_evolution,
            'curvature_evolution': curvature_evolution,
            'node_spin_evolution': node_spin_evolution,
            'link_spin_evolution': link_spin_evolution,
            'initial_volume': initial_volume,
            'initial_area': initial_area,
            'initial_curvature': initial_curvature,
            'final_volume': volume_evolution[-1] if volume_evolution else 0,
            'final_area': area_evolution[-1] if area_evolution else 0,
            'final_curvature': curvature_evolution[-1] if curvature_evolution else 0,
            'time_steps': time_steps,
            'perturbation_strength': perturbation_strength
        }
    
    def _apply_quantum_evolution(self, network: SpinNetwork, strength: float):
        """Apply quantum evolution to the network."""
        # Evolve node spins
        for node in network.nodes:
            # Quantum fluctuations
            perturbation = strength * np.random.normal(0, 0.1)
            
            # Ensure spin remains valid (non-negative, half-integer)
            new_spin = node.spin + perturbation
            new_spin = max(0.5, new_spin)  # Minimum spin is 0.5
            new_spin = round(new_spin * 2) / 2  # Round to nearest half-integer
            
            node.spin = new_spin
        
        # Evolve link spins
        for link in network.links:
            # Quantum fluctuations
            perturbation = strength * np.random.normal(0, 0.1)
            
            # Ensure spin remains valid
            new_spin = link.spin + perturbation
            new_spin = max(0.5, new_spin)  # Minimum spin is 0.5
            new_spin = round(new_spin * 2) / 2  # Round to nearest half-integer
            
            link.spin = new_spin
    
    def _apply_local_moves(self, network: SpinNetwork):
        """Apply local moves to the network."""
        if not network.nodes:
            return
        
        # Randomly select a move type
        move_type = random.choice(['spin_change', 'node_addition', 'link_addition'])
        
        if move_type == 'spin_change':
            # Change a random node spin
            node = random.choice(network.nodes)
            new_spin = random.uniform(0.5, 3.0)
            new_spin = round(new_spin * 2) / 2
            node.spin = new_spin
        
        elif move_type == 'node_addition' and len(network.nodes) < 15:
            # Add a new node
            new_node = SpinNode(
                id=f"evolved_node_{len(network.nodes)}",
                spin=random.uniform(0.5, 2.0),
                position=(random.uniform(-5, 5), random.uniform(-5, 5))
            )
            network.add_node(new_node)
            
            # Connect to a random existing node
            if network.nodes:
                existing_node = random.choice(network.nodes[:-1])  # Don't connect to itself
                new_link = SpinLink(
                    source=existing_node.id,
                    target=new_node.id,
                    spin=random.uniform(0.5, 2.0)
                )
                network.add_link(new_link)
        
        elif move_type == 'link_addition' and len(network.nodes) >= 2:
            # Add a new link between existing nodes
            nodes = random.sample(network.nodes, 2)
            
            # Check if link already exists
            existing_link = any(
                (link.source == nodes[0].id and link.target == nodes[1].id) or
                (link.source == nodes[1].id and link.target == nodes[0].id)
                for link in network.links
            )
            
            if not existing_link:
                new_link = SpinLink(
                    source=nodes[0].id,
                    target=nodes[1].id,
                    spin=random.uniform(0.5, 2.0)
                )
                network.add_link(new_link)
    
    def run_constrained_evolution(self, network: SpinNetwork, time_steps: int = 50,
                                constraint_type: str = "volume") -> Dict[str, any]:
        """
        Run evolution with constraints (e.g., constant volume).
        
        Args:
            network: The spin network to evolve
            time_steps: Number of evolution steps
            constraint_type: Type of constraint to apply
            
        Returns:
            Dictionary containing evolution results
        """
        # Store initial constraint value
        if constraint_type == "volume":
            initial_constraint = network.calculate_total_volume()
        elif constraint_type == "area":
            initial_constraint = network.calculate_total_area()
        else:
            initial_constraint = 1.0
        
        # Evolution storage
        volume_evolution = []
        area_evolution = []
        constraint_violations = []
        
        for step in range(time_steps):
            # Apply evolution
            self._apply_quantum_evolution(network, 0.05)
            
            # Calculate current constraint value
            if constraint_type == "volume":
                current_constraint = network.calculate_total_volume()
            elif constraint_type == "area":
                current_constraint = network.calculate_total_area()
            else:
                current_constraint = 1.0
            
            # Apply constraint correction
            if current_constraint > 0:
                correction_factor = initial_constraint / current_constraint
                
                # Apply correction to all spins
                for node in network.nodes:
                    node.spin *= correction_factor**0.5
                
                for link in network.links:
                    link.spin *= correction_factor**0.5
            
            # Record observables
            volume_evolution.append(network.calculate_total_volume())
            area_evolution.append(network.calculate_total_area())
            
            # Record constraint violation
            final_constraint = network.calculate_total_volume() if constraint_type == "volume" else network.calculate_total_area()
            violation = abs(final_constraint - initial_constraint) / initial_constraint
            constraint_violations.append(violation)
        
        return {
            'volume_evolution': volume_evolution,
            'area_evolution': area_evolution,
            'constraint_violations': constraint_violations,
            'constraint_type': constraint_type,
            'initial_constraint': initial_constraint,
            'time_steps': time_steps
        }
    
    def run_thermal_evolution(self, network: SpinNetwork, temperature: float = 1.0,
                            time_steps: int = 50) -> Dict[str, any]:
        """
        Run thermal evolution at finite temperature.
        
        Args:
            network: The spin network to evolve
            temperature: Temperature in Planck units
            time_steps: Number of evolution steps
            
        Returns:
            Dictionary containing evolution results
        """
        # Thermal evolution storage
        energy_evolution = []
        entropy_evolution = []
        heat_capacity_evolution = []
        
        for step in range(time_steps):
            # Calculate current energy
            current_energy = self._calculate_energy(network)
            energy_evolution.append(current_energy)
            
            # Calculate entropy (simplified)
            current_entropy = self._calculate_entropy(network, temperature)
            entropy_evolution.append(current_entropy)
            
            # Calculate heat capacity
            if step > 0:
                dE = energy_evolution[step] - energy_evolution[step-1]
                dT = 0.01  # Small temperature change
                heat_capacity = dE / dT if dT > 0 else 0
                heat_capacity_evolution.append(heat_capacity)
            else:
                heat_capacity_evolution.append(0)
            
            # Apply thermal fluctuations
            self._apply_thermal_fluctuations(network, temperature)
        
        return {
            'energy_evolution': energy_evolution,
            'entropy_evolution': entropy_evolution,
            'heat_capacity_evolution': heat_capacity_evolution,
            'temperature': temperature,
            'time_steps': time_steps
        }
    
    def _calculate_energy(self, network: SpinNetwork) -> float:
        """Calculate total energy of the network."""
        # Simplified energy calculation
        energy = 0.0
        
        # Kinetic energy from node spins
        for node in network.nodes:
            energy += node.spin * (node.spin + 1)
        
        # Potential energy from link interactions
        for link in network.links:
            energy += 0.5 * link.spin * (link.spin + 1)
        
        # Interaction energy
        for i, node1 in enumerate(network.nodes):
            for j, node2 in enumerate(network.nodes[i+1:], i+1):
                # Check if nodes are connected
                connected = any(
                    (link.source == node1.id and link.target == node2.id) or
                    (link.source == node2.id and link.target == node1.id)
                    for link in network.links
                )
                
                if connected:
                    energy -= 0.1 * node1.spin * node2.spin
        
        return energy
    
    def _calculate_entropy(self, network: SpinNetwork, temperature: float) -> float:
        """Calculate entropy of the network."""
        # Simplified entropy calculation using spin multiplicities
        entropy = 0.0
        
        for node in network.nodes:
            # Multiplicity of spin state
            multiplicity = 2 * node.spin + 1
            
            # Boltzmann entropy
            if multiplicity > 1:
                entropy += np.log(multiplicity)
        
        # Temperature dependence
        entropy *= temperature
        
        return entropy
    
    def _apply_thermal_fluctuations(self, network: SpinNetwork, temperature: float):
        """Apply thermal fluctuations to the network."""
        # Thermal fluctuation strength
        thermal_strength = np.sqrt(temperature)
        
        for node in network.nodes:
            # Thermal perturbation
            perturbation = thermal_strength * np.random.normal(0, 0.1)
            
            # Apply with Boltzmann probability
            energy_change = abs(perturbation) * node.spin
            boltzmann_factor = np.exp(-energy_change / temperature)
            
            if np.random.random() < boltzmann_factor:
                new_spin = node.spin + perturbation
                new_spin = max(0.5, new_spin)
                new_spin = round(new_spin * 2) / 2
                node.spin = new_spin
        
        for link in network.links:
            # Thermal perturbation
            perturbation = thermal_strength * np.random.normal(0, 0.1)
            
            # Apply with Boltzmann probability
            energy_change = abs(perturbation) * link.spin
            boltzmann_factor = np.exp(-energy_change / temperature)
            
            if np.random.random() < boltzmann_factor:
                new_spin = link.spin + perturbation
                new_spin = max(0.5, new_spin)
                new_spin = round(new_spin * 2) / 2
                link.spin = new_spin
    
    def run_hamiltonian_evolution(self, network: SpinNetwork, 
                                hamiltonian_type: str = "spin_spin",
                                time_steps: int = 50) -> Dict[str, any]:
        """
        Run evolution under a specific Hamiltonian.
        
        Args:
            network: The spin network to evolve
            hamiltonian_type: Type of Hamiltonian to use
            time_steps: Number of evolution steps
            
        Returns:
            Dictionary containing evolution results
        """
        # Evolution storage
        energy_evolution = []
        magnetization_evolution = []
        correlation_evolution = []
        
        for step in range(time_steps):
            # Calculate current observables
            current_energy = self._calculate_hamiltonian_energy(network, hamiltonian_type)
            energy_evolution.append(current_energy)
            
            # Calculate magnetization
            total_magnetization = sum(node.spin for node in network.nodes)
            magnetization_evolution.append(total_magnetization)
            
            # Calculate spin-spin correlation
            correlation = self._calculate_spin_correlation(network)
            correlation_evolution.append(correlation)
            
            # Apply Hamiltonian evolution
            self._apply_hamiltonian_evolution(network, hamiltonian_type)
        
        return {
            'energy_evolution': energy_evolution,
            'magnetization_evolution': magnetization_evolution,
            'correlation_evolution': correlation_evolution,
            'hamiltonian_type': hamiltonian_type,
            'time_steps': time_steps
        }
    
    def _calculate_hamiltonian_energy(self, network: SpinNetwork, 
                                    hamiltonian_type: str) -> float:
        """Calculate energy under specific Hamiltonian."""
        energy = 0.0
        
        if hamiltonian_type == "spin_spin":
            # Nearest neighbor spin-spin interaction
            for link in network.links:
                source_node = network.get_node(link.source)
                target_node = network.get_node(link.target)
                
                if source_node and target_node:
                    energy -= source_node.spin * target_node.spin
        
        elif hamiltonian_type == "area_volume":
            # Area-volume constraint energy
            total_area = network.calculate_total_area()
            total_volume = network.calculate_total_volume()
            
            # Constraint energy
            energy = (total_area - total_volume)**2
        
        elif hamiltonian_type == "curvature":
            # Curvature energy
            curvature = self.geometry.calculate_curvature(network)
            energy = 0.5 * curvature**2
        
        return energy
    
    def _calculate_spin_correlation(self, network: SpinNetwork) -> float:
        """Calculate spin-spin correlation function."""
        if len(network.nodes) < 2:
            return 0.0
        
        total_correlation = 0.0
        count = 0
        
        for i, node1 in enumerate(network.nodes):
            for j, node2 in enumerate(network.nodes[i+1:], i+1):
                # Calculate correlation
                correlation = node1.spin * node2.spin
                total_correlation += correlation
                count += 1
        
        return total_correlation / count if count > 0 else 0.0
    
    def _apply_hamiltonian_evolution(self, network: SpinNetwork, 
                                   hamiltonian_type: str):
        """Apply evolution under specific Hamiltonian."""
        dt = 0.01  # Small time step
        
        if hamiltonian_type == "spin_spin":
            # Spin-spin interaction evolution
            for node in network.nodes:
                # Calculate effective field from neighbors
                effective_field = 0.0
                
                for link in network.links:
                    if link.source == node.id:
                        neighbor = network.get_node(link.target)
                        if neighbor:
                            effective_field += neighbor.spin
                    elif link.target == node.id:
                        neighbor = network.get_node(link.source)
                        if neighbor:
                            effective_field += neighbor.spin
                
                # Update spin
                node.spin += dt * effective_field * 0.1
                node.spin = max(0.5, node.spin)
                node.spin = round(node.spin * 2) / 2
        
        elif hamiltonian_type == "area_volume":
            # Area-volume constraint evolution
            total_area = network.calculate_total_area()
            total_volume = network.calculate_total_volume()
            
            # Constraint force
            constraint_force = 2 * (total_area - total_volume)
            
            # Apply force to all spins
            for node in network.nodes:
                node.spin -= dt * constraint_force * 0.01
                node.spin = max(0.5, node.spin)
                node.spin = round(node.spin * 2) / 2
            
            for link in network.links:
                link.spin -= dt * constraint_force * 0.01
                link.spin = max(0.5, link.spin)
                link.spin = round(link.spin * 2) / 2
