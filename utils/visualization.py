import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Tuple, Optional
from core.spin_networks import SpinNetwork, SpinNode, SpinLink
from core.spinfoam import SpinfoamComplex
import warnings

class SpinNetworkVisualizer:
    """Advanced visualization tools for spin networks."""
    
    def __init__(self):
        self.color_schemes = {
            'default': px.colors.qualitative.Set1,
            'viridis': px.colors.sequential.Viridis,
            'plasma': px.colors.sequential.Plasma,
            'rainbow': px.colors.qualitative.Pastel
        }
    
    def plot_2d_network(self, network: SpinNetwork, 
                       color_scheme: str = 'default',
                       show_spins: bool = True,
                       show_labels: bool = True) -> go.Figure:
        """
        Create a 2D visualization of the spin network.
        
        Args:
            network: The spin network to visualize
            color_scheme: Color scheme for visualization
            show_spins: Whether to show spin values
            show_labels: Whether to show node labels
            
        Returns:
            Plotly figure object
        """
        if not network.nodes:
            # Return empty plot with message
            fig = go.Figure()
            fig.add_annotation(
                text="No spin network to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create figure
        fig = go.Figure()
        
        # Plot links first (so they appear behind nodes)
        for link in network.links:
            source_node = network.get_node(link.source)
            target_node = network.get_node(link.target)
            
            if source_node and target_node:
                # Link line
                fig.add_trace(go.Scatter(
                    x=[source_node.position[0], target_node.position[0]],
                    y=[source_node.position[1], target_node.position[1]],
                    mode='lines',
                    line=dict(
                        color=self._get_spin_color(link.spin, color_scheme),
                        width=max(1, link.spin * 2)
                    ),
                    name=f'Link j={link.spin}',
                    hovertemplate=f'Link: {link.source} → {link.target}<br>Spin: {link.spin}<extra></extra>',
                    showlegend=False
                ))
                
                # Link spin label
                if show_spins:
                    mid_x = (source_node.position[0] + target_node.position[0]) / 2
                    mid_y = (source_node.position[1] + target_node.position[1]) / 2
                    
                    fig.add_trace(go.Scatter(
                        x=[mid_x],
                        y=[mid_y],
                        mode='text',
                        text=[f'j={link.spin}'],
                        textfont=dict(size=8, color='blue'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Plot nodes
        node_x = [node.position[0] for node in network.nodes]
        node_y = [node.position[1] for node in network.nodes]
        node_spins = [node.spin for node in network.nodes]
        node_colors = [self._get_spin_color(spin, color_scheme) for spin in node_spins]
        node_sizes = [max(10, spin * 8) for spin in node_spins]
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='black'),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Spin Value")
            ),
            text=[f'{node.id}<br>j={node.spin}' for node in network.nodes] if show_labels else None,
            textposition="middle center",
            textfont=dict(size=8, color='white'),
            name='Nodes',
            hovertemplate='Node: %{text}<br>Position: (%{x}, %{y})<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title='Spin Network Visualization',
            xaxis_title='X Position',
            yaxis_title='Y Position',
            showlegend=True,
            hovermode='closest',
            plot_bgcolor='white',
            width=800,
            height=600
        )
        
        return fig
    
    def plot_3d_network(self, network: SpinNetwork, 
                       color_scheme: str = 'default') -> go.Figure:
        """
        Create a 3D visualization of the spin network.
        
        Args:
            network: The spin network to visualize
            color_scheme: Color scheme for visualization
            
        Returns:
            Plotly figure object
        """
        if not network.nodes:
            fig = go.Figure()
            fig.add_annotation(
                text="No spin network to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create figure
        fig = go.Figure()
        
        # Extend 2D positions to 3D
        node_positions_3d = []
        for node in network.nodes:
            z_pos = node.spin  # Use spin value as Z coordinate
            node_positions_3d.append((node.position[0], node.position[1], z_pos))
        
        # Plot links
        for link in network.links:
            source_node = network.get_node(link.source)
            target_node = network.get_node(link.target)
            
            if source_node and target_node:
                source_pos = next(pos for i, pos in enumerate(node_positions_3d) 
                                if network.nodes[i].id == link.source)
                target_pos = next(pos for i, pos in enumerate(node_positions_3d) 
                                if network.nodes[i].id == link.target)
                
                fig.add_trace(go.Scatter3d(
                    x=[source_pos[0], target_pos[0]],
                    y=[source_pos[1], target_pos[1]],
                    z=[source_pos[2], target_pos[2]],
                    mode='lines',
                    line=dict(
                        color=self._get_spin_color(link.spin, color_scheme),
                        width=max(2, link.spin * 3)
                    ),
                    name=f'Link j={link.spin}',
                    showlegend=False
                ))
        
        # Plot nodes
        node_x = [pos[0] for pos in node_positions_3d]
        node_y = [pos[1] for pos in node_positions_3d]
        node_z = [pos[2] for pos in node_positions_3d]
        node_spins = [node.spin for node in network.nodes]
        node_colors = [self._get_spin_color(spin, color_scheme) for spin in node_spins]
        node_sizes = [max(5, spin * 4) for spin in node_spins]
        
        fig.add_trace(go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            text=[f'{node.id}' for node in network.nodes],
            textposition="middle center",
            name='Nodes',
            hovertemplate='Node: %{text}<br>Position: (%{x}, %{y}, %{z})<br>Spin: %{marker.color}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title='3D Spin Network Visualization',
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Spin Value',
                bgcolor='white'
            ),
            showlegend=True,
            width=800,
            height=600
        )
        
        return fig
    
    def plot_network_evolution(self, network_history: List[SpinNetwork], 
                              time_steps: List[float]) -> go.Figure:
        """
        Visualize the evolution of a spin network over time.
        
        Args:
            network_history: List of spin networks at different times
            time_steps: Corresponding time values
            
        Returns:
            Plotly figure object
        """
        if not network_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No network evolution data to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create subplots
        rows = int(np.ceil(len(network_history) / 3))
        cols = min(3, len(network_history))
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f'Time: {t:.2f}' for t in time_steps],
            specs=[[{'type': 'scatter'} for _ in range(cols)] for _ in range(rows)]
        )
        
        for i, (network, time) in enumerate(zip(network_history, time_steps)):
            row = i // cols + 1
            col = i % cols + 1
            
            if not network.nodes:
                continue
            
            # Plot nodes
            node_x = [node.position[0] for node in network.nodes]
            node_y = [node.position[1] for node in network.nodes]
            node_spins = [node.spin for node in network.nodes]
            
            fig.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers',
                    marker=dict(
                        size=[max(5, spin * 4) for spin in node_spins],
                        color=node_spins,
                        colorscale='Viridis',
                        showscale=False
                    ),
                    name=f'Nodes t={time:.2f}',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Plot links
            for link in network.links:
                source_node = network.get_node(link.source)
                target_node = network.get_node(link.target)
                
                if source_node and target_node:
                    fig.add_trace(
                        go.Scatter(
                            x=[source_node.position[0], target_node.position[0]],
                            y=[source_node.position[1], target_node.position[1]],
                            mode='lines',
                            line=dict(color='blue', width=1),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
        
        fig.update_layout(
            title='Spin Network Evolution',
            height=200 * rows,
            showlegend=False
        )
        
        return fig
    
    def plot_spinfoam_complex(self, complex_obj: SpinfoamComplex) -> go.Figure:
        """
        Visualize a spinfoam complex in 3D.
        
        Args:
            complex_obj: The spinfoam complex to visualize
            
        Returns:
            Plotly figure object
        """
        if not complex_obj.vertices:
            fig = go.Figure()
            fig.add_annotation(
                text="No spinfoam complex to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        fig = go.Figure()
        
        # Plot vertices
        vertex_x = [v.position[0] for v in complex_obj.vertices]
        vertex_y = [v.position[1] for v in complex_obj.vertices]
        vertex_z = [v.position[2] for v in complex_obj.vertices]
        
        # Color vertices by type
        vertex_colors = []
        for vertex in complex_obj.vertices:
            if vertex.vertex_type == "tetrahedron":
                vertex_colors.append("red")
            elif vertex.vertex_type == "4-simplex":
                vertex_colors.append("blue")
            else:
                vertex_colors.append("green")
        
        fig.add_trace(go.Scatter3d(
            x=vertex_x,
            y=vertex_y,
            z=vertex_z,
            mode='markers',
            marker=dict(
                size=8,
                color=vertex_colors,
                line=dict(width=1, color='black')
            ),
            name='Vertices',
            text=[f'{v.id} ({v.vertex_type})' for v in complex_obj.vertices],
            hovertemplate='Vertex: %{text}<br>Position: (%{x}, %{y}, %{z})<extra></extra>'
        ))
        
        # Plot edges
        for edge in complex_obj.edges:
            v1 = next(v for v in complex_obj.vertices if v.id == edge.vertex1)
            v2 = next(v for v in complex_obj.vertices if v.id == edge.vertex2)
            
            fig.add_trace(go.Scatter3d(
                x=[v1.position[0], v2.position[0]],
                y=[v1.position[1], v2.position[1]],
                z=[v1.position[2], v2.position[2]],
                mode='lines',
                line=dict(
                    color=self._get_spin_color(edge.face_spin, 'viridis'),
                    width=max(2, edge.face_spin * 2)
                ),
                name=f'Edge j={edge.face_spin}',
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title='Spinfoam Complex Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                bgcolor='white'
            ),
            showlegend=True,
            width=800,
            height=600
        )
        
        return fig
    
    def plot_wigner_symbol_landscape(self, symbol_type: str, 
                                   j_range: Tuple[float, float],
                                   resolution: int = 50) -> go.Figure:
        """
        Visualize the landscape of Wigner symbols.
        
        Args:
            symbol_type: Type of Wigner symbol ('3j', '6j', '9j')
            j_range: Range of j values to plot
            resolution: Grid resolution
            
        Returns:
            Plotly figure object
        """
        from core.wigner_symbols import WignerSymbols
        
        wigner_calc = WignerSymbols()
        
        if symbol_type == '3j':
            return self._plot_3j_landscape(wigner_calc, j_range, resolution)
        elif symbol_type == '6j':
            return self._plot_6j_landscape(wigner_calc, j_range, resolution)
        else:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Visualization for {symbol_type} symbols not implemented",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
    
    def _plot_3j_landscape(self, wigner_calc, j_range: Tuple[float, float], 
                          resolution: int) -> go.Figure:
        """Plot 3j symbol landscape."""
        j_values = np.linspace(j_range[0], j_range[1], resolution)
        
        # Create grid
        J1, J2 = np.meshgrid(j_values, j_values)
        Z = np.zeros_like(J1)
        
        # Calculate 3j symbols
        for i in range(resolution):
            for j in range(resolution):
                j1, j2 = J1[i, j], J2[i, j]
                j3 = abs(j1 - j2)  # Minimum allowed j3
                
                try:
                    symbol_value = wigner_calc.calculate_3j(j1, j2, j3, 0, 0, 0)
                    Z[i, j] = abs(symbol_value)
                except:
                    Z[i, j] = 0
        
        # Create surface plot
        fig = go.Figure(data=[go.Surface(
            x=J1, y=J2, z=Z,
            colorscale='Viridis',
            name='3j Symbol'
        )])
        
        fig.update_layout(
            title='3j Symbol Landscape',
            scene=dict(
                xaxis_title='j₁',
                yaxis_title='j₂',
                zaxis_title='|3j Symbol|'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def _plot_6j_landscape(self, wigner_calc, j_range: Tuple[float, float], 
                          resolution: int) -> go.Figure:
        """Plot 6j symbol landscape."""
        j_values = np.linspace(j_range[0], j_range[1], resolution)
        
        # Create grid for two varying parameters
        J1, J2 = np.meshgrid(j_values, j_values)
        Z = np.zeros_like(J1)
        
        # Fix other parameters
        j3 = j4 = j5 = j6 = 1.0
        
        # Calculate 6j symbols
        for i in range(resolution):
            for j in range(resolution):
                j1, j2 = J1[i, j], J2[i, j]
                
                try:
                    symbol_value = wigner_calc.calculate_6j(j1, j2, j3, j4, j5, j6)
                    Z[i, j] = abs(symbol_value)
                except:
                    Z[i, j] = 0
        
        # Create surface plot
        fig = go.Figure(data=[go.Surface(
            x=J1, y=J2, z=Z,
            colorscale='Plasma',
            name='6j Symbol'
        )])
        
        fig.update_layout(
            title='6j Symbol Landscape',
            scene=dict(
                xaxis_title='j₁',
                yaxis_title='j₂',
                zaxis_title='|6j Symbol|'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_quantum_geometry_spectrum(self, spectrum_data: Dict[str, List[float]]) -> go.Figure:
        """
        Visualize the discrete spectrum of quantum geometry.
        
        Args:
            spectrum_data: Dictionary containing area, volume, and length spectra
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Area Spectrum', 'Volume Spectrum', 'Length Spectrum'],
            specs=[[{'type': 'scatter'} for _ in range(3)]]
        )
        
        # Area spectrum
        if 'area_spectrum' in spectrum_data:
            area_values = spectrum_data['area_spectrum']
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(area_values))),
                    y=area_values,
                    mode='markers+lines',
                    name='Area',
                    marker=dict(color='red', size=8)
                ),
                row=1, col=1
            )
        
        # Volume spectrum
        if 'volume_spectrum' in spectrum_data:
            volume_values = spectrum_data['volume_spectrum']
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(volume_values))),
                    y=volume_values,
                    mode='markers+lines',
                    name='Volume',
                    marker=dict(color='blue', size=8)
                ),
                row=1, col=2
            )
        
        # Length spectrum
        if 'length_spectrum' in spectrum_data:
            length_values = spectrum_data['length_spectrum']
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(length_values))),
                    y=length_values,
                    mode='markers+lines',
                    name='Length',
                    marker=dict(color='green', size=8)
                ),
                row=1, col=3
            )
        
        fig.update_layout(
            title='Quantum Geometry Spectrum',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def _get_spin_color(self, spin: float, color_scheme: str) -> str:
        """Get color for a given spin value."""
        if color_scheme not in self.color_schemes:
            color_scheme = 'default'
        
        colors = self.color_schemes[color_scheme]
        
        # Map spin to color index
        color_index = int(spin * 2) % len(colors)
        return colors[color_index]
    
    def create_interactive_network_widget(self, network: SpinNetwork) -> go.Figure:
        """
        Create an interactive widget for exploring the spin network.
        
        Args:
            network: The spin network to make interactive
            
        Returns:
            Plotly figure with interactive elements
        """
        fig = self.plot_2d_network(network, show_spins=True, show_labels=True)
        
        # Add interactive features
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True] * len(fig.data)}],
                            label="Show All",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [i % 2 == 0 for i in range(len(fig.data))]}],
                            label="Nodes Only",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [i % 2 == 1 for i in range(len(fig.data))]}],
                            label="Links Only",
                            method="update"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )
        
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear"
            )
        )
        
        return fig
