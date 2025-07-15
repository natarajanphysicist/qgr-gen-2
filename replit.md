# Quantum Gravity Simulation Platform

## Overview

This is a comprehensive quantum gravity simulation platform built with Python and Streamlit, focusing on Loop Quantum Gravity (LQG) research. The platform provides tools for simulating spin networks, calculating Wigner symbols, exploring spinfoam models, and studying geometric observables in quantum gravity.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **Layout**: Multi-page application with sidebar navigation
- **Visualization**: Plotly for interactive plots and NetworkX for graph visualizations
- **User Interface**: Streamlit components for parameter input and results display

### Backend Architecture
- **Core Engine**: Python-based scientific computing stack
- **Modular Design**: Separate modules for different aspects of quantum gravity
- **Mathematical Computing**: NumPy, SciPy, and SymPy for numerical and symbolic calculations
- **Quantum Computing**: Optional Qiskit integration for quantum simulations

### Data Storage Solutions
- **In-Memory Processing**: All calculations performed in memory
- **Serialization**: JSON support for spin network data structures
- **No Persistent Database**: Current implementation uses session-based storage

## Key Components

### 1. Spin Networks (`core/spin_networks.py`)
- **SpinNode**: Represents quantum nodes with SU(2) spin labels
- **SpinLink**: Represents connections between nodes with quantum numbers
- **SpinNetwork**: Main container class managing collections of nodes and links
- **Functionality**: Area and volume calculations, network manipulation

### 2. Wigner Symbols (`core/wigner_symbols.py`)
- **3j Symbols**: Clebsch-Gordan coefficients for angular momentum coupling
- **6j Symbols**: Racah coefficients for three-body coupling
- **9j Symbols**: Nine-j symbols for complex angular momentum recoupling
- **Caching**: LRU cache for performance optimization

### 3. Spinfoam Models (`core/spinfoam.py`)
- **SpinfoamVertex**: Represents vertices in spinfoam complexes
- **SpinfoamComplex**: Container for spinfoam structures
- **Amplitude Calculations**: EPRL-FK, Ooguri, and Barrett-Crane models
- **Vertex Types**: Support for tetrahedra and 4-simplices

### 4. Geometric Observables (`core/geometry.py`)
- **Area Calculations**: Quantum area eigenvalues from spin network links
- **Volume Calculations**: Volume eigenvalues from spin network nodes
- **Curvature**: Geometric curvature calculations
- **Physical Units**: Planck scale conversions

### 5. Quantum Computing Integration (`core/quantum_computing.py`)
- **QuantumGravitySimulator**: Interface for quantum simulations
- **Qiskit Integration**: Optional quantum circuit simulations
- **Classical Fallback**: Classical approximations when quantum hardware unavailable
- **Spin Evolution**: Quantum evolution of spin network states

### 6. Visualization Tools (`utils/visualization.py`, `utils/plotting.py`)
- **SpinNetworkVisualizer**: 2D and 3D network visualization
- **AdvancedPlotter**: Scientific plotting utilities
- **Interactive Plots**: Plotly-based interactive visualizations
- **Multiple Color Schemes**: Various visualization styles

### 7. Research Examples (`examples/`)
- **SimpleEvolution**: Basic spin network evolution simulations
- **SingularityResolution**: Big Bang singularity resolution models
- **Quantum Bounce**: Bounce cosmology implementations

## Data Flow

### 1. Input Processing
- User defines spin network parameters through Streamlit interface
- Spin and geometric quantum numbers are validated
- Network topology is constructed from node and link specifications

### 2. Calculation Pipeline
- Wigner symbols are calculated using SymPy for exact results
- Spinfoam amplitudes are computed using appropriate models
- Geometric observables are derived from quantum numbers
- Evolution simulations process networks through time steps

### 3. Output Generation
- Results are formatted for visualization
- Interactive plots are generated using Plotly
- Numerical results are displayed in structured format
- Export capabilities for further analysis

## External Dependencies

### Core Scientific Libraries
- **NumPy**: Numerical computing and array operations
- **SciPy**: Scientific computing and special functions
- **SymPy**: Symbolic mathematics for exact calculations
- **Matplotlib**: Basic plotting capabilities
- **Plotly**: Interactive visualization framework

### Quantum Computing (Optional)
- **Qiskit**: Quantum circuit simulation and hardware access
- **Graceful Degradation**: Classical fallback when unavailable

### Web Framework
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NetworkX**: Graph theory and network analysis

### Mathematical Tools
- **sympy.physics.wigner**: Wigner symbol calculations
- **scipy.special**: Special mathematical functions
- **functools.lru_cache**: Performance optimization

## Deployment Strategy

### Local Development
- **Environment**: Python 3.7+ with scientific computing stack
- **Dependencies**: Managed through pip/conda
- **Development Server**: Streamlit development server

### Production Deployment
- **Platform**: Streamlit Cloud or similar hosting
- **Containerization**: Docker support for consistent deployment
- **Resource Requirements**: CPU-intensive calculations require adequate compute resources

### Scalability Considerations
- **Caching**: LRU caching for expensive calculations
- **Memory Management**: In-memory processing with garbage collection
- **Performance**: Optimized numerical algorithms for large networks

### Configuration Management
- **Parameters**: Configurable through Streamlit interface
- **Models**: Switchable between different quantum gravity models
- **Visualization**: Customizable plotting parameters

## Research Applications

### Current Capabilities
- Spin network construction and analysis
- Wigner symbol calculations up to 9j symbols
- Basic spinfoam model implementations
- Geometric observable calculations
- Simple evolution simulations

### Future Extensions
- 12j and 15j symbol calculations
- Advanced spinfoam models (EPRL-FK, FK)
- Quantum computer integration
- Curved spacetime emergence studies
- Cosmological applications

### Performance Optimization
- Symbolic-to-numerical conversion pipelines
- Efficient caching strategies
- Parallel processing capabilities
- Memory-efficient data structures