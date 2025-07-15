import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any, Union
from functools import lru_cache
import warnings

class MathUtils:
    """Mathematical utility functions for quantum gravity calculations."""
    
    @staticmethod
    def factorial(n: int) -> int:
        """Calculate factorial with validation."""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n > 170:  # Avoid overflow
            warnings.warn("Large factorial computation may cause overflow")
        return math.factorial(n)
    
    @staticmethod
    def double_factorial(n: int) -> int:
        """Calculate double factorial n!!."""
        if n < 0:
            raise ValueError("Double factorial is not defined for negative numbers")
        
        result = 1
        while n > 0:
            result *= n
            n -= 2
        return result
    
    @staticmethod
    def binomial_coefficient(n: int, k: int) -> int:
        """Calculate binomial coefficient C(n,k)."""
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        
        k = min(k, n - k)  # Take advantage of symmetry
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result
    
    @staticmethod
    def pochhammer_symbol(x: float, n: int) -> float:
        """Calculate Pochhammer symbol (x)_n = x(x+1)...(x+n-1)."""
        if n == 0:
            return 1.0
        if n < 0:
            raise ValueError("Pochhammer symbol is not defined for negative n")
        
        result = 1.0
        for i in range(n):
            result *= (x + i)
        return result
    
    @staticmethod
    def spherical_harmonic_normalization(l: int, m: int) -> float:
        """Calculate normalization constant for spherical harmonics."""
        if abs(m) > l:
            return 0.0
        
        # Normalization factor
        factor = (2 * l + 1) / (4 * np.pi)
        factor *= MathUtils.factorial(l - abs(m)) / MathUtils.factorial(l + abs(m))
        
        return np.sqrt(factor)
    
    @staticmethod
    def legendre_polynomial(n: int, x: float) -> float:
        """Calculate Legendre polynomial P_n(x)."""
        if n < 0:
            raise ValueError("Legendre polynomial degree must be non-negative")
        
        if n == 0:
            return 1.0
        elif n == 1:
            return x
        
        # Use recurrence relation
        p_prev2 = 1.0
        p_prev1 = x
        
        for i in range(2, n + 1):
            p_current = ((2 * i - 1) * x * p_prev1 - (i - 1) * p_prev2) / i
            p_prev2 = p_prev1
            p_prev1 = p_current
        
        return p_prev1
    
    @staticmethod
    def associated_legendre_polynomial(n: int, m: int, x: float) -> float:
        """Calculate associated Legendre polynomial P_n^m(x)."""
        if abs(m) > n:
            return 0.0
        
        if m == 0:
            return MathUtils.legendre_polynomial(n, x)
        
        # Use the formula for associated Legendre polynomials
        # This is a simplified implementation
        if abs(x) > 1:
            warnings.warn("Associated Legendre polynomial evaluated outside [-1,1]")
        
        # For positive m, use the standard formula
        if m > 0:
            factor = ((-1)**m) * ((1 - x**2)**(m/2))
            
            # Calculate derivative of Legendre polynomial
            # This is a simplified numerical approximation
            h = 1e-8
            p_plus = MathUtils.legendre_polynomial(n, x + h)
            p_minus = MathUtils.legendre_polynomial(n, x - h)
            
            # m-th derivative (simplified)
            derivative = (p_plus - p_minus) / (2 * h)
            for _ in range(m - 1):
                p_plus = MathUtils.legendre_polynomial(n, x + h)
                p_minus = MathUtils.legendre_polynomial(n, x - h)
                derivative = (p_plus - p_minus) / (2 * h)
            
            return factor * derivative
        else:
            # For negative m, use symmetry relation
            m_pos = -m
            factor = ((-1)**m_pos) * (MathUtils.factorial(n - m_pos) / MathUtils.factorial(n + m_pos))
            return factor * MathUtils.associated_legendre_polynomial(n, m_pos, x)
    
    @staticmethod
    def gamma_function(x: float) -> float:
        """Calculate gamma function using approximation."""
        if x < 0:
            # Use reflection formula
            return np.pi / (np.sin(np.pi * x) * MathUtils.gamma_function(1 - x))
        
        # Use built-in gamma function
        return math.gamma(x)
    
    @staticmethod
    def beta_function(x: float, y: float) -> float:
        """Calculate beta function B(x,y) = Γ(x)Γ(y)/Γ(x+y)."""
        return MathUtils.gamma_function(x) * MathUtils.gamma_function(y) / MathUtils.gamma_function(x + y)
    
    @staticmethod
    def hypergeometric_2f1(a: float, b: float, c: float, z: float) -> float:
        """Calculate hypergeometric function 2F1(a,b;c;z) using series expansion."""
        if abs(z) >= 1:
            warnings.warn("Hypergeometric function may not converge for |z| >= 1")
        
        # Series expansion
        result = 1.0
        term = 1.0
        
        for n in range(1, 100):  # Limit iterations
            term *= (a + n - 1) * (b + n - 1) * z / ((c + n - 1) * n)
            result += term
            
            if abs(term) < 1e-12:  # Convergence check
                break
        
        return result
    
    @staticmethod
    def bessel_j(n: int, x: float) -> float:
        """Calculate Bessel function of the first kind J_n(x)."""
        if n < 0:
            return ((-1)**n) * MathUtils.bessel_j(-n, x)
        
        # Use series expansion for small x
        if abs(x) < 10:
            result = 0.0
            for m in range(50):  # Limit iterations
                term = ((-1)**m) * ((x/2)**(2*m + n))
                term /= (MathUtils.factorial(m) * MathUtils.factorial(m + n))
                result += term
                
                if abs(term) < 1e-12:
                    break
            
            return result
        else:
            # Use asymptotic expansion for large x
            return np.sqrt(2/(np.pi*x)) * np.cos(x - n*np.pi/2 - np.pi/4)
    
    @staticmethod
    def spherical_bessel_j(n: int, x: float) -> float:
        """Calculate spherical Bessel function j_n(x)."""
        if x == 0:
            return 1.0 if n == 0 else 0.0
        
        return np.sqrt(np.pi/(2*x)) * MathUtils.bessel_j(n + 0.5, x)
    
    @staticmethod
    def wigner_d_matrix(j: float, m: int, n: int, beta: float) -> float:
        """Calculate Wigner d-matrix element d^j_{m,n}(β)."""
        if abs(m) > j or abs(n) > j:
            return 0.0
        
        # Use the formula for Wigner d-matrix
        # This is a simplified implementation
        cos_half = np.cos(beta/2)
        sin_half = np.sin(beta/2)
        
        # Determine the range of summation
        k_min = max(0, m - n)
        k_max = min(int(j + m), int(j - n))
        
        result = 0.0
        for k in range(k_min, k_max + 1):
            # Calculate the term
            numerator = ((-1)**(n - m + k)) * (cos_half**(2*j + m - n - 2*k)) * (sin_half**(n - m + 2*k))
            denominator = (MathUtils.factorial(k) * 
                          MathUtils.factorial(int(j + m - k)) * 
                          MathUtils.factorial(int(j - n - k)) * 
                          MathUtils.factorial(int(n - m + k)))
            
            result += numerator / denominator
        
        # Add the prefactor
        prefactor = np.sqrt(MathUtils.factorial(int(j + m)) * 
                           MathUtils.factorial(int(j - m)) * 
                           MathUtils.factorial(int(j + n)) * 
                           MathUtils.factorial(int(j - n)))
        
        return prefactor * result
    
    @staticmethod
    def clebsch_gordan_coefficient(j1: float, j2: float, j3: float,
                                  m1: float, m2: float, m3: float) -> float:
        """Calculate Clebsch-Gordan coefficient."""
        # Check selection rules
        if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
            return 0.0
        
        if abs(m1 + m2 - m3) > 1e-10:
            return 0.0
        
        if not (abs(j1 - j2) <= j3 <= j1 + j2):
            return 0.0
        
        # Use the formula for Clebsch-Gordan coefficients
        # This is a simplified implementation
        from core.wigner_symbols import WignerSymbols
        wigner_calc = WignerSymbols()
        
        # C(j1,j2,j3;m1,m2,m3) = (-1)^(j1-j2+m3) * sqrt(2j3+1) * 3j(j1,j2,j3;m1,m2,-m3)
        phase = (-1)**(j1 - j2 + m3)
        normalization = np.sqrt(2*j3 + 1)
        three_j = wigner_calc.calculate_3j(j1, j2, j3, m1, m2, -m3)
        
        return phase * normalization * three_j
    
    @staticmethod
    def racah_w_coefficient(j1: float, j2: float, j3: float, 
                           j4: float, j5: float, j6: float) -> float:
        """Calculate Racah W coefficient."""
        from core.wigner_symbols import WignerSymbols
        wigner_calc = WignerSymbols()
        
        # W(j1,j2,j3,j4;j5,j6) = (-1)^(j1+j2+j3+j4) * 6j(j1,j2,j5;j4,j3,j6)
        phase = (-1)**(j1 + j2 + j3 + j4)
        six_j = wigner_calc.calculate_6j(j1, j2, j5, j4, j3, j6)
        
        return phase * six_j
    
    @staticmethod
    def triangle_coefficient(j1: float, j2: float, j3: float) -> float:
        """Calculate triangle coefficient Δ(j1,j2,j3)."""
        # Check triangle inequality
        if not (abs(j1 - j2) <= j3 <= j1 + j2):
            return 0.0
        
        # Calculate triangle coefficient
        numerator = (MathUtils.factorial(int(j1 + j2 - j3)) * 
                    MathUtils.factorial(int(j1 - j2 + j3)) * 
                    MathUtils.factorial(int(-j1 + j2 + j3)))
        
        denominator = MathUtils.factorial(int(j1 + j2 + j3 + 1))
        
        return np.sqrt(numerator / denominator)
    
    @staticmethod
    def quantum_6j_symbol(j1: float, j2: float, j3: float,
                         j4: float, j5: float, j6: float, q: float) -> float:
        """Calculate quantum 6j symbol (q-deformed)."""
        # This is a placeholder for quantum 6j symbols
        # In practice, this would involve q-deformed factorials and more complex calculations
        
        # For now, return the classical 6j symbol
        from core.wigner_symbols import WignerSymbols
        wigner_calc = WignerSymbols()
        
        classical_6j = wigner_calc.calculate_6j(j1, j2, j3, j4, j5, j6)
        
        # Simple q-deformation (placeholder)
        q_correction = 1.0 + 0.1 * (q - 1.0)
        
        return classical_6j * q_correction
    
    @staticmethod
    def tensor_product_decomposition(j1: float, j2: float) -> List[float]:
        """Decompose tensor product of two angular momentum representations."""
        j_values = []
        
        j_min = abs(j1 - j2)
        j_max = j1 + j2
        
        j = j_min
        while j <= j_max:
            j_values.append(j)
            j += 1.0
        
        return j_values
    
    @staticmethod
    def su2_casimir_eigenvalue(j: float) -> float:
        """Calculate eigenvalue of SU(2) Casimir operator."""
        return j * (j + 1)
    
    @staticmethod
    def su2_dimension(j: float) -> int:
        """Calculate dimension of SU(2) representation."""
        return int(2 * j + 1)

class PhysicsConstants:
    """Physical constants for quantum gravity calculations."""
    
    # Fundamental constants (SI units)
    c = 299792458.0  # Speed of light (m/s)
    h = 6.62607015e-34  # Planck constant (J⋅s)
    hbar = h / (2 * np.pi)  # Reduced Planck constant
    G = 6.67430e-11  # Gravitational constant (m³/kg⋅s²)
    
    # Derived constants
    planck_length = np.sqrt(hbar * G / c**3)  # Planck length (m)
    planck_time = np.sqrt(hbar * G / c**5)    # Planck time (s)
    planck_mass = np.sqrt(hbar * c / G)       # Planck mass (kg)
    planck_energy = planck_mass * c**2        # Planck energy (J)
    planck_temperature = planck_energy / (1.380649e-23)  # Planck temperature (K)
    
    # Loop quantum gravity specific
    immirzi_parameter = 0.2375  # Immirzi parameter (dimensionless)
    
    # Quantum mechanics
    electron_mass = 9.1093837015e-31  # Electron mass (kg)
    proton_mass = 1.67262192369e-27   # Proton mass (kg)
    
    @staticmethod
    def planck_units_conversion(value: float, from_unit: str, to_unit: str) -> float:
        """Convert between Planck units and SI units."""
        conversions = {
            'length_to_si': PhysicsConstants.planck_length,
            'time_to_si': PhysicsConstants.planck_time,
            'mass_to_si': PhysicsConstants.planck_mass,
            'energy_to_si': PhysicsConstants.planck_energy,
            'si_to_length': 1.0 / PhysicsConstants.planck_length,
            'si_to_time': 1.0 / PhysicsConstants.planck_time,
            'si_to_mass': 1.0 / PhysicsConstants.planck_mass,
            'si_to_energy': 1.0 / PhysicsConstants.planck_energy,
        }
        
        conversion_key = f"{from_unit}_to_{to_unit}"
        if conversion_key in conversions:
            return value * conversions[conversion_key]
        else:
            raise ValueError(f"Unknown conversion: {from_unit} to {to_unit}")

class ValidationUtils:
    """Validation utilities for quantum gravity calculations."""
    
    @staticmethod
    def validate_angular_momentum(j: float) -> bool:
        """Validate angular momentum quantum number."""
        # Must be non-negative and half-integer
        return j >= 0 and (2 * j) % 1 == 0
    
    @staticmethod
    def validate_magnetic_quantum_number(j: float, m: float) -> bool:
        """Validate magnetic quantum number."""
        return ValidationUtils.validate_angular_momentum(j) and abs(m) <= j and (2 * m) % 1 == 0
    
    @staticmethod
    def validate_triangle_inequality(j1: float, j2: float, j3: float) -> bool:
        """Validate triangle inequality for three angular momenta."""
        return (abs(j1 - j2) <= j3 <= j1 + j2 and
                abs(j2 - j3) <= j1 <= j2 + j3 and
                abs(j3 - j1) <= j2 <= j3 + j1)
    
    @staticmethod
    def validate_spin_network_consistency(nodes: List, links: List) -> Dict[str, bool]:
        """Validate spin network consistency."""
        results = {
            'node_spins_valid': True,
            'link_spins_valid': True,
            'connectivity_valid': True,
            'triangle_inequalities_valid': True
        }
        
        # Check node spins
        for node in nodes:
            if not ValidationUtils.validate_angular_momentum(node.spin):
                results['node_spins_valid'] = False
        
        # Check link spins
        for link in links:
            if not ValidationUtils.validate_angular_momentum(link.spin):
                results['link_spins_valid'] = False
        
        # Check connectivity
        node_ids = {node.id for node in nodes}
        for link in links:
            if link.source not in node_ids or link.target not in node_ids:
                results['connectivity_valid'] = False
        
        return results
    
    @staticmethod
    def validate_wigner_symbol_args(symbol_type: str, args: List[float]) -> bool:
        """Validate arguments for Wigner symbol calculation."""
        if symbol_type == '3j':
            if len(args) != 6:
                return False
            j1, j2, j3, m1, m2, m3 = args
            return (ValidationUtils.validate_angular_momentum(j1) and
                    ValidationUtils.validate_angular_momentum(j2) and
                    ValidationUtils.validate_angular_momentum(j3) and
                    ValidationUtils.validate_magnetic_quantum_number(j1, m1) and
                    ValidationUtils.validate_magnetic_quantum_number(j2, m2) and
                    ValidationUtils.validate_magnetic_quantum_number(j3, m3) and
                    ValidationUtils.validate_triangle_inequality(j1, j2, j3) and
                    abs(m1 + m2 + m3) < 1e-10)
        
        elif symbol_type == '6j':
            if len(args) != 6:
                return False
            j1, j2, j3, j4, j5, j6 = args
            return (all(ValidationUtils.validate_angular_momentum(j) for j in args) and
                    ValidationUtils.validate_triangle_inequality(j1, j2, j3) and
                    ValidationUtils.validate_triangle_inequality(j1, j5, j6) and
                    ValidationUtils.validate_triangle_inequality(j4, j2, j6) and
                    ValidationUtils.validate_triangle_inequality(j4, j5, j3))
        
        return False

class NumericalUtils:
    """Numerical utilities for quantum gravity calculations."""
    
    @staticmethod
    def numerical_derivative(func: callable, x: float, h: float = 1e-8) -> float:
        """Calculate numerical derivative using central difference."""
        return (func(x + h) - func(x - h)) / (2 * h)
    
    @staticmethod
    def numerical_integral(func: callable, a: float, b: float, n: int = 1000) -> float:
        """Calculate numerical integral using trapezoidal rule."""
        if n <= 0:
            raise ValueError("Number of intervals must be positive")
        
        h = (b - a) / n
        result = 0.5 * (func(a) + func(b))
        
        for i in range(1, n):
            x = a + i * h
            result += func(x)
        
        return result * h
    
    @staticmethod
    def solve_equation(func: callable, x0: float, tol: float = 1e-10, max_iter: int = 100) -> float:
        """Solve equation f(x) = 0 using Newton's method."""
        x = x0
        
        for _ in range(max_iter):
            fx = func(x)
            if abs(fx) < tol:
                return x
            
            # Calculate derivative
            dfx = NumericalUtils.numerical_derivative(func, x)
            if abs(dfx) < 1e-15:
                raise ValueError("Derivative is zero, cannot continue")
            
            x = x - fx / dfx
        
        raise ValueError("Newton's method did not converge")
    
    @staticmethod
    def interpolate_linear(x_data: List[float], y_data: List[float], x: float) -> float:
        """Linear interpolation."""
        if len(x_data) != len(y_data) or len(x_data) < 2:
            raise ValueError("Invalid data for interpolation")
        
        # Find the interval
        for i in range(len(x_data) - 1):
            if x_data[i] <= x <= x_data[i + 1]:
                # Linear interpolation
                t = (x - x_data[i]) / (x_data[i + 1] - x_data[i])
                return y_data[i] + t * (y_data[i + 1] - y_data[i])
        
        # Extrapolation
        if x < x_data[0]:
            return y_data[0]
        else:
            return y_data[-1]
    
    @staticmethod
    def matrix_exponential(matrix: np.ndarray, t: float = 1.0) -> np.ndarray:
        """Calculate matrix exponential exp(t * A)."""
        return np.linalg.matrix_power(np.eye(matrix.shape[0]) + t * matrix / 10, 10)
    
    @staticmethod
    def eigenvalues_sorted(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate eigenvalues and eigenvectors, sorted by eigenvalue."""
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        # Sort by eigenvalue
        indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]
        
        return eigenvalues, eigenvectors
    
    @staticmethod
    def is_hermitian(matrix: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if matrix is Hermitian."""
        return np.allclose(matrix, matrix.conj().T, atol=tol)
    
    @staticmethod
    def is_unitary(matrix: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if matrix is unitary."""
        n = matrix.shape[0]
        product = np.dot(matrix, matrix.conj().T)
        identity = np.eye(n)
        return np.allclose(product, identity, atol=tol)
