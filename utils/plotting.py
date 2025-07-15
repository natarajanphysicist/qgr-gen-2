import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from core.spin_networks import SpinNetwork
from core.spinfoam import SpinfoamComplex
import warnings

class AdvancedPlotter:
    """Advanced plotting utilities for quantum gravity simulations."""
    
    def __init__(self):
        self.default_colors = px.colors.qualitative.Set1
        self.scientific_colors = px.colors.sequential.Viridis
        self.diverging_colors = px.colors.diverging.RdBu
    
    def plot_evolution_timeline(self, data: Dict[str, List[float]], 
                               title: str = "Evolution Timeline") -> go.Figure:
        """
        Plot multiple quantities evolving over time.
        
        Args:
            data: Dictionary with keys as quantity names and values as time series
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        colors = self.default_colors
        
        for i, (quantity, values) in enumerate(data.items()):
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode='lines+markers',
                name=quantity,
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time Step",
            yaxis_title="Value",
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_phase_space(self, x_data: List[float], y_data: List[float],
                        x_label: str = "X", y_label: str = "Y",
                        title: str = "Phase Space Plot") -> go.Figure:
        """
        Create a phase space plot.
        
        Args:
            x_data: X coordinates
            y_data: Y coordinates
            x_label: X axis label
            y_label: Y axis label
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Plot trajectory
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers',
            name='Trajectory',
            line=dict(color='blue', width=2),
            marker=dict(size=4, color='red')
        ))
        
        # Highlight start and end points
        if len(x_data) > 0:
            fig.add_trace(go.Scatter(
                x=[x_data[0]],
                y=[y_data[0]],
                mode='markers',
                name='Start',
                marker=dict(size=10, color='green', symbol='star')
            ))
            
            fig.add_trace(go.Scatter(
                x=[x_data[-1]],
                y=[y_data[-1]],
                mode='markers',
                name='End',
                marker=dict(size=10, color='red', symbol='square')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def plot_spectral_analysis(self, data: List[float], 
                              sampling_rate: float = 1.0,
                              title: str = "Spectral Analysis") -> go.Figure:
        """
        Plot frequency spectrum of time series data.
        
        Args:
            data: Time series data
            sampling_rate: Sampling rate
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if len(data) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient data for spectral analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Calculate FFT
        fft_data = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1/sampling_rate)
        
        # Take only positive frequencies
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = np.abs(fft_data[:len(fft_data)//2])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pos_freqs,
            y=pos_fft,
            mode='lines',
            name='Power Spectrum',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Frequency",
            yaxis_title="Amplitude",
            xaxis=dict(type='log'),
            yaxis=dict(type='log'),
            template='plotly_white'
        )
        
        return fig
    
    def plot_correlation_matrix(self, data: Dict[str, List[float]],
                               title: str = "Correlation Matrix") -> go.Figure:
        """
        Plot correlation matrix between different quantities.
        
        Args:
            data: Dictionary with quantity names and values
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if len(data) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Need at least 2 quantities for correlation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Quantity",
            yaxis_title="Quantity",
            template='plotly_white'
        )
        
        return fig
    
    def plot_distribution_comparison(self, data_sets: Dict[str, List[float]],
                                   title: str = "Distribution Comparison") -> go.Figure:
        """
        Compare distributions of different datasets.
        
        Args:
            data_sets: Dictionary with dataset names and values
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        colors = self.default_colors
        
        for i, (name, values) in enumerate(data_sets.items()):
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Histogram(
                x=values,
                name=name,
                opacity=0.7,
                nbinsx=30,
                marker_color=color
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Value",
            yaxis_title="Frequency",
            barmode='overlay',
            template='plotly_white'
        )
        
        return fig
    
    def plot_quantum_state_evolution(self, state_data: List[List[complex]],
                                    time_steps: List[float],
                                    title: str = "Quantum State Evolution") -> go.Figure:
        """
        Visualize the evolution of quantum states.
        
        Args:
            state_data: List of quantum states at different times
            time_steps: Corresponding time values
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if not state_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No quantum state data to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Calculate probabilities
        probabilities = []
        for state in state_data:
            probs = [abs(amplitude)**2 for amplitude in state]
            probabilities.append(probs)
        
        # Create subplot for each basis state
        num_states = len(state_data[0])
        fig = make_subplots(
            rows=1, cols=num_states,
            subplot_titles=[f'State |{i}⟩' for i in range(num_states)],
            shared_yaxes=True
        )
        
        colors = self.scientific_colors
        
        for i in range(num_states):
            state_probs = [probs[i] for probs in probabilities]
            
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=state_probs,
                    mode='lines+markers',
                    name=f'|{i}⟩',
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4)
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title=title,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def plot_energy_landscape(self, x_range: Tuple[float, float],
                             y_range: Tuple[float, float],
                             energy_function: callable,
                             resolution: int = 50,
                             title: str = "Energy Landscape") -> go.Figure:
        """
        Plot 2D energy landscape.
        
        Args:
            x_range: Range of x values
            y_range: Range of y values
            energy_function: Function that takes (x, y) and returns energy
            resolution: Grid resolution
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        # Create coordinate grids
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Calculate energy at each point
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                try:
                    Z[i, j] = energy_function(X[i, j], Y[i, j])
                except:
                    Z[i, j] = 0
        
        # Create surface plot
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            name='Energy Surface'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Energy'
            ),
            template='plotly_white'
        )
        
        return fig
    
    def plot_singularity_resolution(self, density_data: List[float],
                                   time_data: List[float],
                                   bounce_time: float = 0.0,
                                   title: str = "Singularity Resolution") -> go.Figure:
        """
        Plot singularity resolution with quantum bounce.
        
        Args:
            density_data: Energy density values
            time_data: Time values
            bounce_time: Time of quantum bounce
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Plot density evolution
        fig.add_trace(go.Scatter(
            x=time_data,
            y=density_data,
            mode='lines+markers',
            name='Energy Density',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))
        
        # Highlight bounce point
        if bounce_time in time_data:
            bounce_idx = time_data.index(bounce_time)
            bounce_density = density_data[bounce_idx]
            
            fig.add_trace(go.Scatter(
                x=[bounce_time],
                y=[bounce_density],
                mode='markers',
                name='Quantum Bounce',
                marker=dict(size=15, color='red', symbol='star')
            ))
        
        # Add vertical line at bounce
        fig.add_vline(
            x=bounce_time,
            line_dash="dash",
            line_color="red",
            annotation_text="Bounce Point"
        )
        
        # Add classical singularity line (hypothetical)
        fig.add_hline(
            y=max(density_data) * 2,
            line_dash="dot",
            line_color="gray",
            annotation_text="Classical Singularity"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Energy Density",
            template='plotly_white'
        )
        
        return fig
    
    def plot_spacetime_emergence(self, scale_data: List[float],
                                metric_data: List[float],
                                title: str = "Spacetime Emergence") -> go.Figure:
        """
        Plot the emergence of spacetime from discrete to continuous.
        
        Args:
            scale_data: Coarse-graining scale values
            metric_data: Effective metric values
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Plot emergence curve
        fig.add_trace(go.Scatter(
            x=scale_data,
            y=metric_data,
            mode='lines+markers',
            name='Effective Metric',
            line=dict(color='green', width=3),
            marker=dict(size=6)
        ))
        
        # Add discrete regime
        discrete_cutoff = min(scale_data) * 2
        fig.add_vrect(
            x0=min(scale_data),
            x1=discrete_cutoff,
            fillcolor="lightblue",
            opacity=0.3,
            annotation_text="Discrete Regime",
            annotation_position="top left"
        )
        
        # Add continuous regime
        fig.add_vrect(
            x0=discrete_cutoff,
            x1=max(scale_data),
            fillcolor="lightgreen",
            opacity=0.3,
            annotation_text="Continuous Regime",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Coarse-graining Scale",
            yaxis_title="Effective Metric",
            xaxis=dict(type='log'),
            template='plotly_white'
        )
        
        return fig
    
    def plot_holographic_correspondence(self, bulk_data: List[float],
                                      boundary_data: List[float],
                                      title: str = "Holographic Correspondence") -> go.Figure:
        """
        Plot bulk-boundary correspondence.
        
        Args:
            bulk_data: Bulk observable values
            boundary_data: Boundary observable values
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Scatter plot of bulk vs boundary
        fig.add_trace(go.Scatter(
            x=boundary_data,
            y=bulk_data,
            mode='markers',
            name='Bulk-Boundary Points',
            marker=dict(size=8, color='purple')
        ))
        
        # Add trend line
        if len(bulk_data) > 1 and len(boundary_data) > 1:
            z = np.polyfit(boundary_data, bulk_data, 1)
            p = np.poly1d(z)
            
            x_trend = np.linspace(min(boundary_data), max(boundary_data), 100)
            y_trend = p(x_trend)
            
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=y_trend,
                mode='lines',
                name='Correspondence',
                line=dict(color='red', dash='dash', width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Boundary Observable",
            yaxis_title="Bulk Observable",
            template='plotly_white'
        )
        
        return fig
    
    def plot_quantum_error_correction(self, fidelity_data: Dict[str, List[float]],
                                    title: str = "Quantum Error Correction") -> go.Figure:
        """
        Plot quantum error correction performance.
        
        Args:
            fidelity_data: Dictionary with fidelity evolution data
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (condition, fidelities) in enumerate(fidelity_data.items()):
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=list(range(len(fidelities))),
                y=fidelities,
                mode='lines+markers',
                name=condition,
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ))
        
        # Add fidelity threshold
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="gray",
            annotation_text="Fidelity Threshold"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Time Step",
            yaxis_title="Fidelity",
            template='plotly_white'
        )
        
        return fig
    
    def create_publication_figure(self, plot_function: callable,
                                 *args, **kwargs) -> go.Figure:
        """
        Create a publication-ready figure.
        
        Args:
            plot_function: Function to create the plot
            *args: Arguments for the plot function
            **kwargs: Keyword arguments for the plot function
            
        Returns:
            Publication-ready Plotly figure
        """
        fig = plot_function(*args, **kwargs)
        
        # Publication styling
        fig.update_layout(
            font=dict(size=14, family="Arial"),
            paper_bgcolor='white',
            plot_bgcolor='white',
            width=800,
            height=600,
            margin=dict(l=80, r=80, t=80, b=80)
        )
        
        # Update axes
        fig.update_xaxes(
            linecolor='black',
            linewidth=2,
            gridcolor='lightgray',
            gridwidth=1
        )
        
        fig.update_yaxes(
            linecolor='black',
            linewidth=2,
            gridcolor='lightgray',
            gridwidth=1
        )
        
        return fig
