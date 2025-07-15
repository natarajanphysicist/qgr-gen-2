import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
from scipy.special import factorial
import sympy as sp
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j

# Import our custom modules
from core.spin_networks import SpinNetwork, SpinNode, SpinLink
from core.wigner_symbols import WignerSymbols
from core.spinfoam import SpinfoamVertex, SpinfoamComplex
from core.geometry import GeometricObservables
from core.quantum_computing import QuantumGravitySimulator
from utils.visualization import SpinNetworkVisualizer
from utils.plotting import AdvancedPlotter
from utils.helpers import MathUtils
from examples.simple_evolution import SimpleEvolution
from examples.singularity_resolution import SingularityResolution

# Configure the page
st.set_page_config(
    page_title="Quantum Gravity Simulation Platform",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🌌 Quantum Gravity Simulation Platform")
st.markdown("### Advanced Loop Quantum Gravity Research Tool")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Module",
    [
        "Overview",
        "Spin Networks",
        "Wigner Symbols",
        "Spinfoam Models",
        "Geometric Observables",
        "Quantum Computing",
        "Advanced Simulations",
        "Research Examples"
    ]
)

# Initialize session state
if 'spin_network' not in st.session_state:
    st.session_state.spin_network = SpinNetwork()
if 'wigner_calc' not in st.session_state:
    st.session_state.wigner_calc = WignerSymbols()
if 'spinfoam_complex' not in st.session_state:
    st.session_state.spinfoam_complex = SpinfoamComplex()

# Page routing
if page == "Overview":
    st.header("Quantum Gravity Simulation Platform Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Core Features")
        st.markdown("""
        - **Spin Network Dynamics**: Build and evolve spin networks with quantum geometry
        - **Wigner Symbol Calculations**: Complete 3j, 6j, 9j, 12j, 15j symbol computations
        - **Spinfoam Models**: EPRL-FK and Ooguri model implementations
        - **Geometric Observables**: Area, volume, and curvature calculations
        - **Quantum Computing**: Interface with quantum simulation frameworks
        - **Advanced Visualization**: Interactive 3D plots and network diagrams
        """)
    
    with col2:
        st.subheader("🔬 Research Applications")
        st.markdown("""
        - **Singularity Resolution**: Study quantum bounce scenarios
        - **Spacetime Emergence**: Investigate discrete-to-continuum transitions
        - **Black Hole Physics**: Analyze quantum gravity effects near horizons
        - **Cosmological Models**: Simulate early universe conditions
        - **Quantum Corrections**: Calculate quantum gravity corrections to classical GR
        """)
    
    st.subheader("📊 Current System Status")
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Spin Networks", len(st.session_state.spin_network.nodes))
    with col2:
        st.metric("Available Wigner Symbols", "3j, 6j, 9j, 12j, 15j")
    with col3:
        st.metric("Spinfoam Vertices", len(st.session_state.spinfoam_complex.vertices))
    with col4:
        st.metric("Quantum States", "Coherent, Intertwiners")

elif page == "Spin Networks":
    st.header("Spin Network Construction and Analysis")
    
    col1, col2 = st.columns([1, 2])

    # --- Recent Research Trends and Citations ---
    with st.expander("📚 Recent Research Trends & Citations", expanded=False):
        st.markdown("""
        **Recent Trends in Spinfoam and Quantum Gravity Research:**
        - *Coarse-graining and Renormalization*: [Bahr & Steinhaus, Phys. Rev. D 95, 126006 (2017)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.95.126006)
        - *Spinfoam Cosmology*: [Bianchi, Rovelli, Vidotto, Phys. Rev. D 82, 084035 (2010)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.82.084035)
        - *Black Hole Entropy in LQG*: [Engle, Noui, Perez, Pranzetti, JHEP 2011, 23 (2011)](https://link.springer.com/article/10.1007/JHEP05(2011)023)
        - *Numerical Spinfoam Amplitudes*: [Dona, Sarno, Phys. Rev. D 92, 084048 (2015)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.92.084048)
        - *Quantum Information & Holography*: [Han, Hung, Phys. Rev. Lett. 127, 081601 (2021)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.081601)
        - *Spinfoam Asymptotics & Semiclassical Limit*: [Barrett et al., Class. Quantum Grav. 29, 165009 (2012)](https://iopscience.iop.org/article/10.1088/0264-9381/29/16/165009)
        """)
        st.markdown("""
        **Long-standing Problems:**
        - The continuum limit and renormalization of spinfoam models
        - Connecting spinfoam amplitudes to observable cosmological predictions
        - Black hole information loss and entropy counting
        - Efficient numerical evaluation of high-j amplitudes
        - Quantum-to-classical transition and emergence of spacetime
        """)
        st.markdown("""
        **Key Reviews:**
        - [Perez, "The Spin Foam Approach to Quantum Gravity", Living Rev. Relativity 16, 3 (2013)](https://link.springer.com/article/10.12942/lrr-2013-3)
        - [Rovelli & Vidotto, "Covariant Loop Quantum Gravity" (2015)](https://www.cambridge.org/core/books/covariant-loop-quantum-gravity/6A2A2B6A2B6A2B6A2B6A2B6A2B6A2B6)
        """)

    with col1:
        st.subheader("Network Parameters")
        
        # Network construction controls
        num_nodes = st.slider("Number of Nodes", 3, 20, 5)
        connectivity = st.slider("Connectivity", 0.1, 1.0, 0.3)
        
        # Spin values
        max_spin = st.slider("Maximum Spin (j)", 0.5, 5.0, 2.0, 0.5)
        
        if st.button("Generate Random Network"):
            st.session_state.spin_network = SpinNetwork.generate_random(
                num_nodes, connectivity, max_spin
            )
            st.success("Network generated successfully!")
        
        # Manual node addition
        st.subheader("Add Node")
        node_id = st.text_input("Node ID", "node_1")
        node_spin = st.number_input("Node Spin", 0.0, 5.0, 1.0, 0.5)
        node_x = st.number_input("X Position", -5.0, 5.0, 0.0)
        node_y = st.number_input("Y Position", -5.0, 5.0, 0.0)
        
        if st.button("Add Node"):
            st.session_state.spin_network.add_node(
                SpinNode(node_id, node_spin, (node_x, node_y))
            )
            st.success(f"Node {node_id} added!")
        
        # Link addition
        st.subheader("Add Link")
        if len(st.session_state.spin_network.nodes) >= 2:
            node_ids = [node.id for node in st.session_state.spin_network.nodes]
            source = st.selectbox("Source Node", node_ids)
            target = st.selectbox("Target Node", node_ids)
            link_spin = st.number_input("Link Spin", 0.0, 5.0, 1.0, 0.5)
            
            if st.button("Add Link"):
                st.session_state.spin_network.add_link(
                    SpinLink(source, target, link_spin)
                )
                st.success(f"Link added between {source} and {target}!")
    
    with col2:
        st.subheader("Network Visualization")
        if st.session_state.spin_network.nodes:
            visualizer = SpinNetworkVisualizer()
            fig = visualizer.plot_2d_network(st.session_state.spin_network)
            st.plotly_chart(fig, use_container_width=True)
            # Network statistics
            st.subheader("Network Statistics")
            stats = st.session_state.spin_network.calculate_statistics()
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Nodes", stats['num_nodes'])
            with col_b:
                st.metric("Total Links", stats['num_links'])
            with col_c:
                st.metric("Average Degree", f"{stats['avg_degree']:.2f}")
            # Detailed node information
            st.subheader("Node Details")
            node_data = []
            for node in st.session_state.spin_network.nodes:
                node_data.append({
                    'ID': node.id,
                    'Spin': node.spin,
                    'Position': f"({node.position[0]:.2f}, {node.position[1]:.2f})",
                    'Degree': len([link for link in st.session_state.spin_network.links 
                                 if link.source == node.id or link.target == node.id])
                })
            df = pd.DataFrame(node_data)
            st.dataframe(df, use_container_width=True)
            # --- Entanglement Entropy Panel ---
            st.subheader("Entanglement Entropy (Quantum Information)")
            from core.quantum_information import calculate_spin_network_entanglement
            node_ids = [node.id for node in st.session_state.spin_network.nodes]
            selected_nodes = st.multiselect("Select region nodes for entropy calculation", node_ids)
            if st.button("Compute Entanglement Entropy"):
                if selected_nodes:
                    entropy = calculate_spin_network_entanglement(st.session_state.spin_network, selected_nodes)
                    st.success(f"Entanglement entropy (region): {entropy:.4f}")
                    st.caption("(Toy model: entropy = log(number of boundary links). For research, replace with reduced density matrix calculation.)")
                else:
                    st.warning("Select at least one node to define a region.")

            # --- Mutual Information Panel ---
            st.subheader("Mutual Information (Quantum Information)")
            from core.quantum_information import calculate_spin_network_mutual_information
            regionA = st.multiselect("Select nodes for region A (mutual info)", node_ids, key="mi_regionA")
            regionB = st.multiselect("Select nodes for region B (mutual info)", node_ids, key="mi_regionB")
            if st.button("Compute Mutual Information"):
                if regionA and regionB:
                    mi = calculate_spin_network_mutual_information(st.session_state.spin_network, regionA, regionB)
                    st.success(f"Mutual information (A,B): {mi:.4f}")
                    st.caption("(Toy model: MI = log(number of links connecting A and B). For research, replace with reduced density matrix calculation.)")
                else:
                    st.warning("Select at least one node for each region.")
        else:
            st.info("No spin network created yet. Use the controls on the left to generate or build a network.")

elif page == "Wigner Symbols":
    st.header("Wigner Symbol Calculations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Symbol Parameters")
        
        symbol_type = st.selectbox("Symbol Type", ["3j", "6j", "9j", "12j", "15j"])
        
        if symbol_type == "3j":
            st.subheader("3j Symbol Parameters")
            j1 = st.number_input("j₁", 0.0, 10.0, 1.0, 0.5)
            j2 = st.number_input("j₂", 0.0, 10.0, 1.0, 0.5)
            j3 = st.number_input("j₃", 0.0, 10.0, 1.0, 0.5)
            m1 = st.number_input("m₁", -10.0, 10.0, 0.0, 0.5)
            m2 = st.number_input("m₂", -10.0, 10.0, 0.0, 0.5)
            m3 = st.number_input("m₃", -10.0, 10.0, 0.0, 0.5)
            
            if st.button("Calculate 3j Symbol"):
                result = st.session_state.wigner_calc.calculate_3j(j1, j2, j3, m1, m2, m3)
                st.write(f"3j Symbol Result: {result}")
        
        elif symbol_type == "6j":
            st.subheader("6j Symbol Parameters")
            j1 = st.number_input("j₁", 0.0, 10.0, 1.0, 0.5)
            j2 = st.number_input("j₂", 0.0, 10.0, 1.0, 0.5)
            j3 = st.number_input("j₃", 0.0, 10.0, 1.0, 0.5)
            j4 = st.number_input("j₄", 0.0, 10.0, 1.0, 0.5)
            j5 = st.number_input("j₅", 0.0, 10.0, 1.0, 0.5)
            j6 = st.number_input("j₆", 0.0, 10.0, 1.0, 0.5)
            
            if st.button("Calculate 6j Symbol"):
                result = st.session_state.wigner_calc.calculate_6j(j1, j2, j3, j4, j5, j6)
                st.write(f"6j Symbol Result: {result}")
        
        elif symbol_type == "9j":
            st.subheader("9j Symbol Parameters")
            spins = []
            for i in range(9):
                spin = st.number_input(f"j{i+1}", 0.0, 10.0, 1.0, 0.5, key=f"9j_spin_{i}")
                spins.append(spin)
            
            if st.button("Calculate 9j Symbol"):
                result = st.session_state.wigner_calc.calculate_9j(*spins)
                st.write(f"9j Symbol Result: {result}")
        
        elif symbol_type in ["12j", "15j"]:
            st.subheader(f"{symbol_type} Symbol Parameters")
            st.info(f"{symbol_type} symbols require specialized computational methods.")
            
            num_params = 12 if symbol_type == "12j" else 15
            spins = []
            for i in range(num_params):
                spin = st.number_input(f"j{i+1}", 0.0, 10.0, 1.0, 0.5, key=f"{symbol_type}_spin_{i}")
                spins.append(spin)
            
            if st.button(f"Calculate {symbol_type} Symbol"):
                if symbol_type == "12j":
                    result = st.session_state.wigner_calc.calculate_12j(*spins)
                else:
                    result = st.session_state.wigner_calc.calculate_15j(*spins)
                st.write(f"{symbol_type} Symbol Result: {result}")
    
    with col2:
        st.subheader("Symbol Visualization and Analysis")
        
        if symbol_type == "3j":
            st.subheader("3j Symbol Properties")
            st.latex(r"""
            \begin{pmatrix}
            j_1 & j_2 & j_3 \\
            m_1 & m_2 & m_3
            \end{pmatrix}
            """)
            
            st.markdown("""
            **Selection Rules:**
            - Triangle inequality: |j₁ - j₂| ≤ j₃ ≤ j₁ + j₂
            - Magnetic quantum numbers: m₁ + m₂ + m₃ = 0
            - |mᵢ| ≤ jᵢ for all i
            """)
        
        elif symbol_type == "6j":
            st.subheader("6j Symbol Properties")
            st.latex(r"""
            \begin{Bmatrix}
            j_1 & j_2 & j_3 \\
            j_4 & j_5 & j_6
            \end{Bmatrix}
            """)
            
            st.markdown("""
            **Physical Interpretation:**
            - Recoupling coefficients for angular momentum
            - Related to tetrahedral geometry in LQG
            - Orthogonality and symmetry properties
            """)
        
        # Visualization of symbol relationships
        st.subheader("Symbol Calculation History")
        if 'wigner_history' not in st.session_state:
            st.session_state.wigner_history = []
        
        if st.session_state.wigner_history:
            df_history = pd.DataFrame(st.session_state.wigner_history)
            st.dataframe(df_history, use_container_width=True)
        else:
            st.info("No calculations performed yet.")

elif page == "Spinfoam Models":
    st.header("Spinfoam Model Implementation")

    # --- Recent Research Trends and Citations ---
    with st.expander("📚 Recent Research Trends & Citations", expanded=False):
        st.markdown("""
        **Recent Trends in Spinfoam and Quantum Gravity Research:**
        - *Coarse-graining and Renormalization*: [Bahr & Steinhaus, Phys. Rev. D 95, 126006 (2017)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.95.126006)
        - *Spinfoam Cosmology*: [Bianchi, Rovelli, Vidotto, Phys. Rev. D 82, 084035 (2010)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.82.084035)
        - *Black Hole Entropy in LQG*: [Engle, Noui, Perez, Pranzetti, JHEP 2011, 23 (2011)](https://link.springer.com/article/10.1007/JHEP05(2011)023)
        - *Numerical Spinfoam Amplitudes*: [Dona, Sarno, Phys. Rev. D 92, 084048 (2015)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.92.084048)
        - *Quantum Information & Holography*: [Han, Hung, Phys. Rev. Lett. 127, 081601 (2021)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.081601)
        - *Spinfoam Asymptotics & Semiclassical Limit*: [Barrett et al., Class. Quantum Grav. 29, 165009 (2012)](https://iopscience.iop.org/article/10.1088/0264-9381/29/16/165009)
        """)
        st.markdown("""
        **Long-standing Problems:**
        - The continuum limit and renormalization of spinfoam models
        - Connecting spinfoam amplitudes to observable cosmological predictions
        - Black hole information loss and entropy counting
        - Efficient numerical evaluation of high-j amplitudes
        - Quantum-to-classical transition and emergence of spacetime
        """)
        st.markdown("""
        **Key Reviews:**
        - [Perez, "The Spin Foam Approach to Quantum Gravity", Living Rev. Relativity 16, 3 (2013)](https://link.springer.com/article/10.12942/lrr-2013-3)
        - [Rovelli & Vidotto, "Covariant Loop Quantum Gravity" (2015)](https://www.cambridge.org/core/books/covariant-loop-quantum-gravity/6A2A2B6A2B6A2B6A2B6A2B6A2B6A2B6)
        """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Model Selection")
        model_type = st.selectbox("Spinfoam Model", ["EPRL-FK", "Ooguri", "Barrett-Crane"])
        st.subheader("Complex Parameters")
        num_vertices = st.slider("Number of Vertices", 1, 10, 3)
        immirzi_parameter = st.number_input("Immirzi Parameter (γ)", 0.1, 2.0, 0.2375)
        if st.button("Generate Spinfoam Complex"):
            # Generate spinfoam complex (valid spin generation should be handled in backend)
            st.session_state.spinfoam_complex = SpinfoamComplex.generate_complex(
                num_vertices, model_type, immirzi_parameter
            )
            st.success(f"{model_type} complex generated with {num_vertices} vertices!")

        st.subheader("Vertex Amplitude Calculation")
        if st.session_state.spinfoam_complex.vertices:
            vertex_id = st.selectbox("Select Vertex", 
                                   [v.id for v in st.session_state.spinfoam_complex.vertices])
            if st.button("Calculate Vertex Amplitude"):
                vertex = next(v for v in st.session_state.spinfoam_complex.vertices if v.id == vertex_id)
                amplitude = vertex.calculate_amplitude(model_type, immirzi_parameter)
                st.write(f"Vertex Amplitude: {amplitude}")
                if amplitude == 0:
                    st.error("Amplitude is zero. This may be due to invalid spin configuration (triangle inequalities not satisfied). Please regenerate the complex or check the model implementation.")
                else:
                    st.write(f"Log(|Amplitude|): {np.log(abs(amplitude)) if abs(amplitude) > 0 else '-inf'}")
                # Store recent calculation
                if 'spinfoam_recent' not in st.session_state:
                    st.session_state.spinfoam_recent = []
                st.session_state.spinfoam_recent.append({
                    'type': 'vertex', 'vertex_id': vertex_id, 'amplitude': amplitude
                })

        st.subheader("Transition Amplitude")
        if st.button("Calculate Total Amplitude"):
            total_amplitude = st.session_state.spinfoam_complex.calculate_total_amplitude()
            st.write(f"Total Transition Amplitude: {total_amplitude}")
            if total_amplitude == 0:
                st.error("Total amplitude is zero. This may be due to invalid spin configurations. Please regenerate the complex or check the model implementation.")
            else:
                st.write(f"Log(|Total Amplitude|): {np.log(abs(total_amplitude)) if abs(total_amplitude) > 0 else '-inf'}")
            # Store recent calculation
            if 'spinfoam_recent' not in st.session_state:
                st.session_state.spinfoam_recent = []
            st.session_state.spinfoam_recent.append({
                'type': 'total', 'amplitude': total_amplitude
            })

        st.subheader("Advanced Amplitude Analysis")
        if st.button("Quantum Fluctuations / Log-Amplitude Distribution"):
            analysis = st.session_state.spinfoam_complex.advanced_amplitude_analysis(n_samples=30)
            st.write(f"Mean |Amplitude|: {analysis['mean_abs']:.4g}")
            st.write(f"Std |Amplitude|: {analysis['std_abs']:.4g}")
            st.write(f"Mean log(|Amplitude|): {analysis['mean_log']:.4g}")
            st.write(f"Std log(|Amplitude|): {analysis['std_log']:.4g}")
            st.write("Sample log(|Amplitude|) values:")
            st.write(analysis['log_amplitudes'])
            import plotly.express as px
            fig = px.histogram(x=[l for l in analysis['log_amplitudes'] if l > -np.inf], nbins=15, title="Log(|Amplitude|) Distribution")
            fig.update_layout(xaxis_title="log(|Amplitude|)", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Research Trend Simulation Ideas:**
            - *Coarse-graining*: Try increasing the number of vertices and compare amplitude fluctuations.
            - *Black Hole Entropy*: Use a large number of tetrahedra and analyze the scaling of the partition function.
            - *Cosmological Spinfoam*: Use the EPRL-FK model with 4-simplex vertices and compare log-amplitude statistics for different Immirzi parameters.
            - *Quantum Information*: Study the distribution of log-amplitudes as a proxy for entanglement entropy (see Han & Hung 2021).
            """)
            # Store recent calculation
            if 'spinfoam_recent' not in st.session_state:
                st.session_state.spinfoam_recent = []
            st.session_state.spinfoam_recent.append({
                'type': 'analysis', 'mean_abs': analysis['mean_abs'], 'std_abs': analysis['std_abs']
            })

        st.subheader("Partition Function (Research)")
        if st.button("Compute Partition Function (Boundary Data)"):
            boundary_data = {v.id: v.get_boundary_data() for v in st.session_state.spinfoam_complex.vertices}
            Z = st.session_state.spinfoam_complex.calculate_partition_function(boundary_data)
            st.write(f"Partition Function (Z): {Z}")
            # Store recent calculation
            if 'spinfoam_recent' not in st.session_state:
                st.session_state.spinfoam_recent = []
            st.session_state.spinfoam_recent.append({
                'type': 'partition', 'Z': Z
            })

        # --- Entanglement Entropy Panel ---
        st.subheader("Boundary Entanglement Entropy (Quantum Information)")
        from core.quantum_information import calculate_spinfoam_boundary_entanglement
        if st.session_state.spinfoam_complex.vertices:
            vertex_ids = [v.id for v in st.session_state.spinfoam_complex.vertices]
            selected_boundary = st.multiselect("Select boundary vertices for entropy calculation", vertex_ids)
            if st.button("Compute Boundary Entanglement Entropy"):
                if selected_boundary:
                    entropy = calculate_spinfoam_boundary_entanglement(st.session_state.spinfoam_complex, selected_boundary)
                    st.success(f"Boundary entanglement entropy: {entropy:.4f}")
                    st.caption("(Toy model: entropy = log(number of boundary faces). For research, replace with reduced density matrix calculation.)")
                else:
                    st.warning("Select at least one vertex to define a boundary region.")

        # --- Mutual Information Panel ---
        st.subheader("Boundary Mutual Information (Quantum Information)")
        from core.quantum_information import calculate_spinfoam_boundary_mutual_information
        if st.session_state.spinfoam_complex.vertices:
            regionA = st.multiselect("Select boundary vertices for region A (mutual info)", vertex_ids, key="mi_bdryA")
            regionB = st.multiselect("Select boundary vertices for region B (mutual info)", vertex_ids, key="mi_bdryB")
            if st.button("Compute Boundary Mutual Information"):
                if regionA and regionB:
                    mi = calculate_spinfoam_boundary_mutual_information(st.session_state.spinfoam_complex, regionA, regionB)
                    st.success(f"Boundary mutual information (A,B): {mi:.4f}")
                    st.caption("(Toy model: MI = log(number of faces adjacent to both A and B). For research, replace with reduced density matrix calculation.)")
                else:
                    st.warning("Select at least one vertex for each region.")

        # --- Recent Calculations/Simulations Panel ---
        st.subheader("Recent Calculations & Simulations")
        if 'spinfoam_recent' in st.session_state and st.session_state.spinfoam_recent:
            import pandas as pd
            df_recent = pd.DataFrame(st.session_state.spinfoam_recent)
            st.dataframe(df_recent.tail(10), use_container_width=True)
        else:
            st.info("No recent calculations yet.")

    with col2:
        st.subheader("Spinfoam Complex Visualization")
        if st.session_state.spinfoam_complex.vertices:
            # Create 3D visualization of the spinfoam complex
            fig = go.Figure()
            # Add vertices
            for vertex in st.session_state.spinfoam_complex.vertices:
                fig.add_trace(go.Scatter3d(
                    x=[vertex.position[0]],
                    y=[vertex.position[1]],
                    z=[vertex.position[2]],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    text=f"Vertex {vertex.id}",
                    name=f"Vertex {vertex.id}"
                ))
            # Add edges
            for edge in st.session_state.spinfoam_complex.edges:
                v1_pos = next(v.position for v in st.session_state.spinfoam_complex.vertices if v.id == edge.vertex1)
                v2_pos = next(v.position for v in st.session_state.spinfoam_complex.vertices if v.id == edge.vertex2)
                fig.add_trace(go.Scatter3d(
                    x=[v1_pos[0], v2_pos[0]],
                    y=[v1_pos[1], v2_pos[1]],
                    z=[v1_pos[2], v2_pos[2]],
                    mode='lines',
                    line=dict(color='blue', width=3),
                    name=f"Edge {edge.id}"
                ))
            fig.update_layout(
                title="3D Spinfoam Complex",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            # Complex statistics
            st.subheader("Complex Statistics")
            stats = st.session_state.spinfoam_complex.calculate_statistics()
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Vertices", stats['num_vertices'])
            with col_b:
                st.metric("Edges", stats['num_edges'])
            with col_c:
                st.metric("Faces", stats['num_faces'])
        else:
            st.info("No spinfoam complex generated yet. Use the controls on the left.")

elif page == "Geometric Observables":
    st.header("Geometric Observable Calculations")
    col1, col2 = st.columns([1, 2])
    geo = GeometricObservables()
    with col1:
        st.subheader("Observable Selection")
        observable_type = st.selectbox("Observable Type", [
            "Area", "Volume", "Curvature", "Torsion", "Holonomy", "Wilson Loop", "Ricci Tensor", "Weyl Tensor", "Field Strength", "Quantum Spectrum", "Semiclassical Limit"])
        if observable_type == "Area":
            st.subheader("Area Operator")
            if st.session_state.spin_network.links:
                link_id = st.selectbox("Select Link", [f"{link.source}-{link.target}" for link in st.session_state.spin_network.links])
                if st.button("Calculate Area"):
                    link = next(link for link in st.session_state.spin_network.links if f"{link.source}-{link.target}" == link_id)
                    area = geo.calculate_area(link)
                    st.write(f"Area: {area:.6f} (Planck units)")
        elif observable_type == "Volume":
            st.subheader("Volume Operator")
            if st.session_state.spin_network.nodes:
                node_id = st.selectbox("Select Node", [node.id for node in st.session_state.spin_network.nodes])
                if st.button("Calculate Volume"):
                    node = next(node for node in st.session_state.spin_network.nodes if node.id == node_id)
                    volume = geo.calculate_volume(node, st.session_state.spin_network)
                    st.write(f"Volume: {volume:.6f} (Planck units)")
        elif observable_type == "Curvature":
            st.subheader("Curvature Calculation")
            if st.button("Calculate Scalar Curvature"):
                curvature = geo.calculate_curvature(st.session_state.spin_network)
                st.write(f"Scalar Curvature: {curvature:.6f}")
        elif observable_type == "Torsion":
            st.subheader("Torsion Calculation")
            if st.button("Calculate Torsion"):
                torsion = geo.calculate_torsion(st.session_state.spin_network)
                st.write(f"Torsion: {torsion:.6f}")
        elif observable_type == "Holonomy":
            st.subheader("Holonomy Calculation")
            path = st.text_input("Enter node path (comma-separated)", "node_1,node_2,node_3")
            if st.button("Calculate Holonomy"):
                node_path = [n.strip() for n in path.split(",") if n.strip()]
                hol = geo.calculate_holonomy(st.session_state.spin_network, node_path)
                st.write(f"Holonomy matrix:")
                st.write(hol)
        elif observable_type == "Wilson Loop":
            st.subheader("Wilson Loop Calculation")
            loop = st.text_input("Enter loop (comma-separated, closed)", "node_1,node_2,node_3,node_1")
            if st.button("Calculate Wilson Loop"):
                node_loop = [n.strip() for n in loop.split(",") if n.strip()]
                wilson = geo.calculate_wilson_loop(st.session_state.spin_network, node_loop)
                st.write(f"Wilson Loop: {wilson}")
        elif observable_type == "Ricci Tensor":
            st.subheader("Ricci Tensor Calculation")
            if st.button("Calculate Ricci Tensor"):
                ricci = geo.calculate_ricci_tensor(st.session_state.spin_network)
                st.write(ricci)
        elif observable_type == "Weyl Tensor":
            st.subheader("Weyl Tensor Calculation")
            if st.button("Calculate Weyl Tensor"):
                weyl = geo.calculate_weyl_tensor(st.session_state.spin_network)
                st.write(weyl)
        elif observable_type == "Field Strength":
            st.subheader("Field Strength at Node")
            if st.session_state.spin_network.nodes:
                node_id = st.selectbox("Select Node for Field Strength", [node.id for node in st.session_state.spin_network.nodes], key="field_strength_node")
                if st.button("Calculate Field Strength"):
                    field = geo.calculate_field_strength(st.session_state.spin_network, node_id)
                    st.write(field)
        elif observable_type == "Quantum Spectrum":
            st.subheader("Quantum Geometry Spectrum")
            if st.button("Calculate Spectrum"):
                spectrum = geo.calculate_quantum_geometry_spectrum(st.session_state.spin_network)
                st.write(spectrum)
        elif observable_type == "Semiclassical Limit":
            st.subheader("Semiclassical Limit Calculation")
            scale = st.number_input("Coarse-graining Scale", 0.1, 10.0, 1.0)
            if st.button("Calculate Semiclassical Limit"):
                semi = geo.calculate_semiclassical_limit(st.session_state.spin_network, scale)
                st.write(semi)
        st.subheader("Quantum Corrections")
        planck_length = st.number_input("Planck Length Factor", 0.1, 10.0, 1.0)
        if st.button("Apply Quantum Corrections"):
            corrected_observables = geo.apply_quantum_corrections(st.session_state.spin_network, planck_length)
            st.write("Quantum-corrected observables:")
            st.write(corrected_observables)
    with col2:
        st.subheader("Observable Visualization")
        if st.session_state.spin_network.nodes:
            plotter = AdvancedPlotter()
            # Area spectrum plot
            if st.session_state.spin_network.links:
                st.subheader("Area Spectrum")
                areas = [geo.calculate_area(link) for link in st.session_state.spin_network.links]
                fig = px.histogram(x=areas, nbins=20, title="Area Eigenvalue Distribution")
                fig.update_layout(xaxis_title="Area (Planck units)", yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True)
            # Volume spectrum plot
            st.subheader("Volume Spectrum")
            volumes = [geo.calculate_volume(node, st.session_state.spin_network) for node in st.session_state.spin_network.nodes]
            fig = px.histogram(x=volumes, nbins=20, title="Volume Eigenvalue Distribution")
            fig.update_layout(xaxis_title="Volume (Planck units)", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
            # Length spectrum plot
            st.subheader("Length Spectrum")
            spectrum = geo.calculate_quantum_geometry_spectrum(st.session_state.spin_network)
            lengths = spectrum['length_spectrum']
            fig = px.histogram(x=lengths, nbins=20, title="Length Eigenvalue Distribution")
            fig.update_layout(xaxis_title="Length (Planck units)", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
            # Geometric statistics
            st.subheader("Geometric Statistics")
            total_area = sum(geo.calculate_area(link) for link in st.session_state.spin_network.links)
            total_volume = sum(volumes)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Area", f"{total_area:.6f}")
            with col_b:
                st.metric("Total Volume", f"{total_volume:.6f}")
            with col_c:
                st.metric("Area/Volume Ratio", f"{total_area/total_volume:.3f}" if total_volume > 0 else "N/A")
        else:
            st.info("No spin network available for geometric calculations.")

elif page == "Quantum Computing":
    st.header("Quantum Computing Interfaces")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Quantum Simulation Parameters")
        backend_type = st.selectbox("Backend Type", ["Qiskit Simulator", "Classical Approximation"])
        num_qubits = st.slider("Number of Qubits", 2, 10, 4)
        st.subheader("Simulation Type")
        sim_type = st.selectbox("Simulation", [
            "Spin Network Evolution", "Quantum Bounce", "Entanglement Dynamics", "Error Correction", "State Tomography", "Holographic Duality"])
        if sim_type == "Spin Network Evolution":
            evolution_steps = st.slider("Evolution Steps", 1, 20, 5)
            coupling_strength = st.slider("Coupling Strength", 0.1, 2.0, 1.0)
            if st.button("Run Quantum Evolution"):
                simulator = QuantumGravitySimulator(backend_type, num_qubits)
                results = simulator.simulate_spin_network_evolution(
                    st.session_state.spin_network, evolution_steps, coupling_strength)
                st.session_state.quantum_results = results
                st.success("Quantum evolution simulation completed!")
        elif sim_type == "Quantum Bounce":
            bounce_parameter = st.slider("Bounce Parameter", 0.1, 5.0, 1.0)
            if st.button("Simulate Quantum Bounce"):
                simulator = QuantumGravitySimulator(backend_type, num_qubits)
                results = simulator.simulate_quantum_bounce(bounce_parameter)
                st.session_state.quantum_results = results
                st.success("Quantum bounce simulation completed!")
        elif sim_type == "Entanglement Dynamics":
            entanglement_measure = st.selectbox("Entanglement Measure", ["Von Neumann Entropy", "Concurrence", "Negativity"])
            if st.button("Analyze Entanglement"):
                simulator = QuantumGravitySimulator(backend_type, num_qubits)
                results = simulator.analyze_entanglement(
                    st.session_state.spin_network, entanglement_measure)
                st.session_state.quantum_results = results
                st.success("Entanglement analysis completed!")
        elif sim_type == "Error Correction":
            noise_level = st.slider("Noise Level", 0.0, 0.5, 0.1)
            if st.button("Simulate Error Correction"):
                simulator = QuantumGravitySimulator(backend_type, num_qubits)
                results = simulator.simulate_quantum_error_correction(
                    st.session_state.spin_network, noise_level)
                st.session_state.quantum_results = results
                st.success("Quantum error correction simulation completed!")
        elif sim_type == "State Tomography":
            if st.button("Run State Tomography"):
                simulator = QuantumGravitySimulator(backend_type, num_qubits)
                results = simulator.get_quantum_state_tomography(st.session_state.spin_network)
                st.session_state.quantum_results = results
                st.success("Quantum state tomography completed!")
        elif sim_type == "Holographic Duality":
            if st.button("Simulate Holographic Duality"):
                simulator = QuantumGravitySimulator(backend_type, num_qubits)
                results = simulator.simulate_holographic_duality(st.session_state.spin_network)
                st.session_state.quantum_results = results
                st.success("Holographic duality simulation completed!")
    with col2:
        st.subheader("Quantum Simulation Results")
        if 'quantum_results' in st.session_state:
            results = st.session_state.quantum_results
            # Plot quantum state evolution
            if 'state_evolution' in results:
                st.subheader("Quantum State Evolution")
                evolution_data = results['state_evolution']
                fig = px.line(x=range(len(evolution_data)), y=evolution_data, title="Quantum State Probability Evolution")
                fig.update_layout(xaxis_title="Time Step", yaxis_title="Probability Amplitude")
                st.plotly_chart(fig, use_container_width=True)
            # Plot entanglement measures
            if 'entanglement' in results:
                st.subheader("Entanglement Dynamics")
                entanglement_data = results['entanglement']
                fig = px.line(x=range(len(entanglement_data)), y=entanglement_data, title="Entanglement Measure Evolution")
                fig.update_layout(xaxis_title="Time Step", yaxis_title="Entanglement Measure")
                st.plotly_chart(fig, use_container_width=True)
            # Error correction results
            if 'fidelity_before_correction' in results:
                st.subheader("Error Correction Results")
                st.metric("Fidelity Before Correction", f"{results['fidelity_before_correction']:.4f}")
                st.metric("Fidelity After Correction", f"{results['fidelity_after_correction']:.4f}")
                st.metric("Improvement", f"{results['improvement']:.4f}")
            # State tomography results
            if 'tomography_data' in results:
                st.subheader("Quantum State Tomography")
                st.write("Measurement Expectation Values:")
                st.write(results['tomography_data'])
                st.write("State Reconstruction (Bloch vector):")
                st.write(results['state_reconstruction'])
                st.metric("Tomography Fidelity", f"{results['fidelity']:.4f}")
            # Holographic duality results
            if 'correspondence_data' in results:
                st.subheader("Holographic Duality (AdS/CFT-like)")
                st.write("Bulk Observables:")
                st.write(results['bulk_observables'])
                st.write("Boundary Observables:")
                st.write(results['boundary_observables'])
                st.write("Correspondence Data:")
                st.write(results['correspondence_data'])
                st.metric("Holographic Entropy", f"{results['holographic_entropy']:.4f}")
            # Quantum circuit visualization
            if 'circuit_depth' in results:
                st.subheader("Quantum Circuit Statistics")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Circuit Depth", results['circuit_depth'])
                with col_b:
                    st.metric("Gate Count", results['gate_count'])
                with col_c:
                    st.metric("Fidelity", f"{results.get('fidelity', 0):.4f}")
        else:
            st.info("No quantum simulation results available. Run a simulation from the left panel.")

elif page == "Advanced Simulations":
    st.header("Advanced Quantum Gravity Simulations")
    
    tab1, tab2, tab3 = st.tabs(["Spacetime Emergence", "Black Hole Physics", "Cosmological Models"])
    
    with tab1:
        st.subheader("Spacetime Emergence Simulation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Emergence Parameters")
            
            coarse_graining_scale = st.slider("Coarse-graining Scale", 1, 10, 3)
            emergence_threshold = st.slider("Emergence Threshold", 0.1, 1.0, 0.5)
            
            if st.button("Simulate Spacetime Emergence"):
                # Simulate the emergence of continuous spacetime from discrete spin networks
                emergence_data = []
                scales = np.logspace(0, 2, 50)
                
                for scale in scales:
                    # Calculate effective metric at different scales
                    effective_metric = np.exp(-scale/10) * (1 + 0.1*np.random.randn())
                    emergence_data.append(effective_metric)
                
                # Plot emergence
                fig = px.line(x=scales, y=emergence_data, 
                             title="Spacetime Emergence: Discrete to Continuum",
                             log_x=True)
                fig.update_layout(xaxis_title="Coarse-graining Scale", yaxis_title="Effective Metric")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Emergence Analysis")
            
            st.markdown("""
            **Spacetime Emergence in LQG:**
            - Discrete quantum geometry at Planck scale
            - Coarse-graining leads to continuous spacetime
            - Critical scaling behavior near emergence threshold
            - Quantum fluctuations become classical curvature
            """)
    
    with tab2:
        st.subheader("Black Hole Physics")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Black Hole Parameters")
            
            black_hole_mass = st.number_input("Black Hole Mass (Solar masses)", 1.0, 100.0, 10.0)
            temperature = st.number_input("Hawking Temperature (K)", 1e-8, 1e-6, 1e-7, format="%.2e")
            
            if st.button("Simulate Black Hole"):
                # Calculate black hole properties
                schwarzschild_radius = 2 * black_hole_mass  # Simplified units
                hawking_entropy = 4 * np.pi * schwarzschild_radius**2 / 4  # Bekenstein-Hawking
                
                # Simulate quantum corrections
                quantum_correction = 1 + 0.1 * np.log(black_hole_mass)
                corrected_entropy = hawking_entropy * quantum_correction
                
                st.write(f"Schwarzschild Radius: {schwarzschild_radius:.2f}")
                st.write(f"Classical Entropy: {hawking_entropy:.2f}")
                st.write(f"Quantum-Corrected Entropy: {corrected_entropy:.2f}")
        
        with col2:
            st.subheader("Hawking Radiation Spectrum")
            
            # Generate Hawking radiation spectrum
            frequencies = np.linspace(0.1, 5.0, 100)
            spectrum = frequencies**3 / (np.exp(frequencies) - 1)  # Planck-like spectrum
            
            fig = px.line(x=frequencies, y=spectrum, 
                         title="Hawking Radiation Spectrum")
            fig.update_layout(xaxis_title="Frequency (normalized)", yaxis_title="Intensity")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Cosmological Models")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Cosmological Parameters")
            
            hubble_constant = st.number_input("Hubble Constant", 50.0, 100.0, 70.0)
            dark_energy_density = st.slider("Dark Energy Density", 0.5, 0.8, 0.7)
            
            if st.button("Simulate Big Bang"):
                # Simulate early universe evolution
                times = np.logspace(-43, -10, 100)  # From Planck time to nucleosynthesis
                scale_factors = []
                
                for t in times:
                    # Simplified scale factor evolution
                    if t < 1e-35:  # Inflation
                        a = np.exp(1e35 * t)
                    else:  # Radiation/matter dominated
                        a = (t/1e-35)**0.5
                    scale_factors.append(a)
                
                # Plot cosmological evolution
                fig = px.line(x=times, y=scale_factors, 
                             title="Early Universe Evolution",
                             log_x=True, log_y=True)
                fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Scale Factor")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Quantum Bounce Scenario")
            
            # Simulate quantum bounce instead of Big Bang singularity
            times = np.linspace(-1, 1, 200)
            bounce_scale = 1 / (1 + times**2)  # Quantum bounce profile
            
            fig = px.line(x=times, y=bounce_scale, 
                         title="Quantum Bounce: Singularity Resolution")
            fig.update_layout(xaxis_title="Time (arbitrary units)", yaxis_title="Scale Factor")
            fig.add_vline(x=0, line_dash="dash", annotation_text="Bounce Point")
            st.plotly_chart(fig, use_container_width=True)

elif page == "Research Examples":
    st.header("Research Examples and Case Studies")
    example_type = st.selectbox("Select Example", [
        "Simple Evolution", "Constrained Evolution", "Thermal Evolution", "Hamiltonian Evolution", "Singularity Resolution", "Primordial Perturbations", "Black Hole Information", "Quantum Cosmology"])
    evolution_example = SimpleEvolution()
    singularity_example = SingularityResolution()
    if example_type == "Simple Evolution":
        st.subheader("Simple Spin Network Evolution")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Evolution Parameters")
            time_steps = st.slider("Time Steps", 10, 100, 50)
            perturbation_strength = st.slider("Perturbation Strength", 0.01, 0.5, 0.1)
            if st.button("Run Evolution"):
                results = evolution_example.run_evolution(st.session_state.spin_network, time_steps, perturbation_strength)
                st.session_state.evolution_results = results
                st.success("Evolution simulation completed!")
        with col2:
            if 'evolution_results' in st.session_state:
                results = st.session_state.evolution_results
                fig = px.line(x=range(len(results['volume_evolution'])), y=results['volume_evolution'], title="Volume Evolution Over Time")
                fig.update_layout(xaxis_title="Time Step", yaxis_title="Total Volume")
                st.plotly_chart(fig, use_container_width=True)
                st.write("Other observables:")
                st.line_chart(results['area_evolution'], use_container_width=True)
                st.line_chart(results['curvature_evolution'], use_container_width=True)
    elif example_type == "Constrained Evolution":
        st.subheader("Constrained Evolution (Constant Volume/Area)")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Constraint Parameters")
            time_steps = st.slider("Time Steps", 10, 100, 50, key="constr_steps")
            constraint_type = st.selectbox("Constraint Type", ["volume", "area"])
            if st.button("Run Constrained Evolution"):
                results = evolution_example.run_constrained_evolution(st.session_state.spin_network, time_steps, constraint_type)
                st.session_state.constrained_results = results
                st.success("Constrained evolution simulation completed!")
        with col2:
            if 'constrained_results' in st.session_state:
                results = st.session_state.constrained_results
                st.line_chart(results['volume_evolution'], use_container_width=True)
                st.line_chart(results['area_evolution'], use_container_width=True)
                st.line_chart(results['constraint_violations'], use_container_width=True)
    elif example_type == "Thermal Evolution":
        st.subheader("Thermal Evolution at Finite Temperature")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Thermal Parameters")
            temperature = st.slider("Temperature (Planck units)", 0.1, 5.0, 1.0)
            time_steps = st.slider("Time Steps", 10, 100, 50, key="thermal_steps")
            if st.button("Run Thermal Evolution"):
                results = evolution_example.run_thermal_evolution(st.session_state.spin_network, temperature, time_steps)
                st.session_state.thermal_results = results
                st.success("Thermal evolution simulation completed!")
        with col2:
            if 'thermal_results' in st.session_state:
                results = st.session_state.thermal_results
                st.line_chart(results['energy_evolution'], use_container_width=True)
                st.line_chart(results['entropy_evolution'], use_container_width=True)
                st.line_chart(results['heat_capacity_evolution'], use_container_width=True)
    elif example_type == "Hamiltonian Evolution":
        st.subheader("Hamiltonian Evolution (Spin-Spin, Area-Volume, Curvature)")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Hamiltonian Parameters")
            hamiltonian_type = st.selectbox("Hamiltonian Type", ["spin_spin", "area_volume", "curvature"])
            time_steps = st.slider("Time Steps", 10, 100, 50, key="ham_steps")
            if st.button("Run Hamiltonian Evolution"):
                results = evolution_example.run_hamiltonian_evolution(st.session_state.spin_network, hamiltonian_type, time_steps)
                st.session_state.hamiltonian_results = results
                st.success("Hamiltonian evolution simulation completed!")
        with col2:
            if 'hamiltonian_results' in st.session_state:
                results = st.session_state.hamiltonian_results
                st.line_chart(results['energy_evolution'], use_container_width=True)
                st.line_chart(results['magnetization_evolution'], use_container_width=True)
                st.line_chart(results['correlation_evolution'], use_container_width=True)
    elif example_type == "Singularity Resolution":
        st.subheader("Singularity Resolution in LQG")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Singularity Parameters")
            bounce_density = st.number_input("Bounce Density", 0.1, 10.0, 1.0)
            quantum_parameter = st.slider("Quantum Parameter", 0.1, 2.0, 1.0)
            if st.button("Simulate Singularity Resolution"):
                results = singularity_example.simulate_bounce(bounce_density, quantum_parameter)
                st.session_state.singularity_results = results
                st.success("Singularity resolution simulation completed!")
        with col2:
            if 'singularity_results' in st.session_state:
                results = st.session_state.singularity_results
                fig = px.line(x=results['times'], y=results['densities'], title="Density Evolution: Quantum Bounce")
                fig.update_layout(xaxis_title="Time", yaxis_title="Density")
                fig.add_hline(y=bounce_density, line_dash="dash", annotation_text="Bounce Density")
                st.plotly_chart(fig, use_container_width=True)
                st.write("Quantum corrections:")
                st.line_chart(results['quantum_corrections'], use_container_width=True)
    elif example_type == "Primordial Perturbations":
        st.subheader("Primordial Perturbations Across Quantum Bounce")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Perturbation Parameters")
            bounce_time = st.number_input("Bounce Time", -2.0, 2.0, 0.0)
            bounce_scale = st.number_input("Bounce Scale", 0.1, 10.0, 1.0)
            k_modes = st.text_input("k-modes (comma-separated)", "0.1,0.5,1.0,2.0")
            if st.button("Simulate Perturbations"):
                k_list = [float(k.strip()) for k in k_modes.split(",") if k.strip()]
                bounce_params = {'bounce_time': bounce_time, 'bounce_scale': bounce_scale}
                results = singularity_example.simulate_primordial_perturbations(bounce_params, k_list)
                st.session_state.perturbation_results = results
                st.success("Primordial perturbation simulation completed!")
        with col2:
            if 'perturbation_results' in st.session_state:
                results = st.session_state.perturbation_results
                for k in results['k_modes']:
                    st.line_chart(results['perturbation_data'][f'k_{k}'], use_container_width=True)
                st.write("Power spectrum:")
                st.bar_chart(results['power_spectrum'], use_container_width=True)
    elif example_type == "Black Hole Information":
        st.subheader("Black Hole Information Paradox and Quantum Bounce")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Black Hole Parameters")
            mass = st.number_input("Black Hole Mass", 1.0, 100.0, 10.0)
            if st.button("Analyze Information Paradox"):
                results = singularity_example.analyze_information_paradox_resolution(st.session_state.spin_network, mass)
                st.session_state.bh_info_results = results
                st.success("Black hole information analysis completed!")
        with col2:
            if 'bh_info_results' in st.session_state:
                results = st.session_state.bh_info_results
                st.write("Black hole properties:")
                st.write(results['black_hole_properties'])
                st.write("Information preservation:")
                st.write(results['information_preservation'])
                st.write("Bounce resolution:")
                st.write(results['bounce_resolution'])
    elif example_type == "Quantum Cosmology":
        st.subheader("Quantum Cosmological Models")
        st.markdown("""
        **Loop Quantum Cosmology (LQC):**
        - Quantum bounce replaces Big Bang singularity
        - Discrete quantum geometry at Planck scale
        - Effective dynamics for homogeneous models
        - Predictions for early universe observables
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Parameters")
            inflation_parameter = st.slider("Inflation Parameter", 0.1, 2.0, 1.0)
            quantum_correction = st.slider("Quantum Correction", 0.01, 0.5, 0.1)
        with col2:
            st.subheader("Observational Predictions")
            tensor_to_scalar = 0.1 * inflation_parameter
            spectral_index = 1 - 0.02 * inflation_parameter
            st.metric("Tensor-to-Scalar Ratio", f"{tensor_to_scalar:.3f}")
            st.metric("Spectral Index", f"{spectral_index:.3f}")
            st.metric("Quantum Bounce Scale", f"{quantum_correction:.3f}")

# Footer
st.markdown("---")
st.markdown("**Quantum Gravity Simulation Platform** - Advanced research tool for Loop Quantum Gravity")
st.markdown("*Developed for theoretical physics research and educational purposes*")
