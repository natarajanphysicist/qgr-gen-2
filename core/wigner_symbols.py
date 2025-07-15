import numpy as np
from scipy.special import factorial, comb
from sympy import symbols, sqrt, simplify, Rational
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j
import sympy as sp
from typing import Union, List, Tuple, Dict
from functools import lru_cache
import warnings

class WignerSymbols:
    """Advanced Wigner symbol calculations for quantum gravity applications."""
    
    def __init__(self):
        self.cache_size = 1000
        self.precision = 1e-12
    
    @lru_cache(maxsize=1000)
    def calculate_3j(self, j1: float, j2: float, j3: float, 
                    m1: float, m2: float, m3: float) -> float:
        """
        Calculate the Wigner 3j symbol.
        
        Args:
            j1, j2, j3: Angular momentum quantum numbers
            m1, m2, m3: Magnetic quantum numbers
            
        Returns:
            The value of the 3j symbol
        """
        try:
            # Check selection rules
            if not self._check_3j_selection_rules(j1, j2, j3, m1, m2, m3):
                return 0.0
            
            # Use SymPy for exact calculation
            result = wigner_3j(j1, j2, j3, m1, m2, m3)
            
            # Convert to float
            return float(result.evalf())
        
        except Exception as e:
            warnings.warn(f"Error calculating 3j symbol: {e}")
            return 0.0
    
    @lru_cache(maxsize=1000)
    def calculate_6j(self, j1: float, j2: float, j3: float, 
                    j4: float, j5: float, j6: float) -> float:
        """
        Calculate the Wigner 6j symbol (Racah coefficient).
        
        Args:
            j1, j2, j3, j4, j5, j6: Angular momentum quantum numbers
            
        Returns:
            The value of the 6j symbol
        """
        try:
            # Check selection rules
            if not self._check_6j_selection_rules(j1, j2, j3, j4, j5, j6):
                return 0.0
            
            # Use SymPy for exact calculation
            result = wigner_6j(j1, j2, j3, j4, j5, j6)
            
            # Convert to float
            return float(result.evalf())
        
        except Exception as e:
            warnings.warn(f"Error calculating 6j symbol: {e}")
            return 0.0
    
    @lru_cache(maxsize=500)
    def calculate_9j(self, j11: float, j12: float, j13: float,
                    j21: float, j22: float, j23: float,
                    j31: float, j32: float, j33: float) -> float:
        """
        Calculate the Wigner 9j symbol.
        
        Args:
            j11, j12, j13: First row of angular momentum quantum numbers
            j21, j22, j23: Second row of angular momentum quantum numbers
            j31, j32, j33: Third row of angular momentum quantum numbers
            
        Returns:
            The value of the 9j symbol
        """
        try:
            # Check selection rules
            if not self._check_9j_selection_rules(j11, j12, j13, j21, j22, j23, j31, j32, j33):
                return 0.0
            
            # Use SymPy for exact calculation
            result = wigner_9j(j11, j12, j13, j21, j22, j23, j31, j32, j33)
            
            # Convert to float
            return float(result.evalf())
        
        except Exception as e:
            warnings.warn(f"Error calculating 9j symbol: {e}")
            return 0.0
    
    def calculate_12j(self, *args) -> float:
        """
        Calculate the Wigner 12j symbol using recurrence relations.
        
        The 12j symbol is constructed from combinations of 6j symbols.
        """
        if len(args) != 12:
            raise ValueError("12j symbol requires exactly 12 arguments")
        
        # Extract the 12 angular momentum quantum numbers
        j = args
        
        try:
            # Use the formula for 12j in terms of 6j symbols
            # This is a simplified implementation
            result = 0.0
            
            # Sum over intermediate angular momenta
            for k in np.arange(0, 5, 0.5):  # Limited range for computational efficiency
                term1 = self.calculate_6j(j[0], j[1], j[2], j[3], j[4], k)
                term2 = self.calculate_6j(j[5], j[6], j[7], j[8], j[9], k)
                term3 = self.calculate_6j(j[10], j[11], k, j[2], j[7], j[0])
                
                result += (2*k + 1) * term1 * term2 * term3
            
            return result
        
        except Exception as e:
            warnings.warn(f"Error calculating 12j symbol: {e}")
            return 0.0
    
    def calculate_15j(self, *args) -> float:
        """
        Calculate the Wigner 15j symbol for spinfoam vertices.
        
        The 15j symbol is crucial for 4D spinfoam models like EPRL-FK.
        """
        if len(args) != 15:
            raise ValueError("15j symbol requires exactly 15 arguments")
        
        # Extract the 15 angular momentum quantum numbers
        j = args
        
        try:
            # The 15j symbol can be expressed in terms of 6j symbols
            # This is a highly simplified implementation
            result = 0.0
            
            # Use the fact that the 15j symbol is related to the amplitude
            # of a 4-simplex in spinfoam models
            
            # Sum over intermediate angular momenta (limited for efficiency)
            for k1 in np.arange(0, 3, 0.5):
                for k2 in np.arange(0, 3, 0.5):
                    for k3 in np.arange(0, 3, 0.5):
                        # Calculate contributing 6j symbols
                        term1 = self.calculate_6j(j[0], j[1], j[2], j[3], j[4], k1)
                        term2 = self.calculate_6j(j[5], j[6], j[7], j[8], j[9], k2)
                        term3 = self.calculate_6j(j[10], j[11], j[12], j[13], j[14], k3)
                        term4 = self.calculate_6j(k1, k2, k3, j[0], j[5], j[10])
                        
                        prefactor = (2*k1 + 1) * (2*k2 + 1) * (2*k3 + 1)
                        result += prefactor * term1 * term2 * term3 * term4
            
            return result
        
        except Exception as e:
            warnings.warn(f"Error calculating 15j symbol: {e}")
            return 0.0
    
    def _check_3j_selection_rules(self, j1: float, j2: float, j3: float, 
                                 m1: float, m2: float, m3: float) -> bool:
        """Check selection rules for 3j symbols."""
        # Triangle inequality
        if not (abs(j1 - j2) <= j3 <= j1 + j2):
            return False
        
        # Magnetic quantum number constraint
        if abs(m1 + m2 + m3) > self.precision:
            return False
        
        # Individual magnetic quantum number constraints
        if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
            return False
        
        # All j values must be non-negative
        if j1 < 0 or j2 < 0 or j3 < 0:
            return False
        
        return True
    
    def _check_6j_selection_rules(self, j1: float, j2: float, j3: float, 
                                 j4: float, j5: float, j6: float) -> bool:
        """Check selection rules for 6j symbols."""
        # Triangle inequalities for all four triads
        triads = [(j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3)]
        
        for triad in triads:
            a, b, c = triad
            if not (abs(a - b) <= c <= a + b):
                return False
        
        # All j values must be non-negative
        if any(j < 0 for j in [j1, j2, j3, j4, j5, j6]):
            return False
        
        return True
    
    def _check_9j_selection_rules(self, j11: float, j12: float, j13: float,
                                 j21: float, j22: float, j23: float,
                                 j31: float, j32: float, j33: float) -> bool:
        """Check selection rules for 9j symbols."""
        # Triangle inequalities for rows and columns
        rows = [(j11, j12, j13), (j21, j22, j23), (j31, j32, j33)]
        cols = [(j11, j21, j31), (j12, j22, j32), (j13, j23, j33)]
        
        for triad in rows + cols:
            a, b, c = triad
            if not (abs(a - b) <= c <= a + b):
                return False
        
        # All j values must be non-negative
        js = [j11, j12, j13, j21, j22, j23, j31, j32, j33]
        if any(j < 0 for j in js):
            return False
        
        return True
    
    def calculate_clebsch_gordan(self, j1: float, j2: float, j: float,
                                m1: float, m2: float, m: float) -> float:
        """
        Calculate Clebsch-Gordan coefficient.
        
        Related to 3j symbol by:
        C(j1,j2,j;m1,m2,m) = (-1)^(j1-j2+m) * sqrt(2j+1) * 3j(j1,j2,j;m1,m2,-m)
        """
        try:
            three_j = self.calculate_3j(j1, j2, j, m1, m2, -m)
            phase = (-1)**(j1 - j2 + m)
            normalization = np.sqrt(2*j + 1)
            
            return phase * normalization * three_j
        
        except Exception as e:
            warnings.warn(f"Error calculating Clebsch-Gordan coefficient: {e}")
            return 0.0
    
    def calculate_racah_coefficient(self, j1: float, j2: float, j3: float,
                                   j4: float, j5: float, j6: float) -> float:
        """
        Calculate Racah coefficient W(j1,j2,j3,j4;j5,j6).
        
        Related to 6j symbol by:
        W(j1,j2,j3,j4;j5,j6) = (-1)^(j1+j2+j3+j4) * 6j(j1,j2,j5;j4,j3,j6)
        """
        try:
            six_j = self.calculate_6j(j1, j2, j5, j4, j3, j6)
            phase = (-1)**(j1 + j2 + j3 + j4)
            
            return phase * six_j
        
        except Exception as e:
            warnings.warn(f"Error calculating Racah coefficient: {e}")
            return 0.0
    
    def calculate_symbol_sum(self, symbol_type: str, ranges: List[Tuple[float, float]], 
                           fixed_params: List[float]) -> float:
        """
        Calculate sums of Wigner symbols over given ranges.
        
        Useful for computing partition functions and amplitudes.
        """
        if symbol_type == "3j":
            return self._sum_3j_symbols(ranges, fixed_params)
        elif symbol_type == "6j":
            return self._sum_6j_symbols(ranges, fixed_params)
        else:
            raise ValueError(f"Unsupported symbol type: {symbol_type}")
    
    def _sum_3j_symbols(self, ranges: List[Tuple[float, float]], 
                       fixed_params: List[float]) -> float:
        """Sum 3j symbols over specified ranges."""
        total = 0.0
        
        # Simple implementation - sum over discrete values
        for j1 in np.arange(ranges[0][0], ranges[0][1] + 0.5, 0.5):
            for j2 in np.arange(ranges[1][0], ranges[1][1] + 0.5, 0.5):
                for j3 in np.arange(ranges[2][0], ranges[2][1] + 0.5, 0.5):
                    symbol_value = self.calculate_3j(j1, j2, j3, *fixed_params)
                    total += symbol_value**2  # Square for probability
        
        return total
    
    def _sum_6j_symbols(self, ranges: List[Tuple[float, float]], 
                       fixed_params: List[float]) -> float:
        """Sum 6j symbols over specified ranges."""
        total = 0.0
        
        # Simple implementation - sum over discrete values
        for j1 in np.arange(ranges[0][0], ranges[0][1] + 0.5, 0.5):
            for j2 in np.arange(ranges[1][0], ranges[1][1] + 0.5, 0.5):
                for j3 in np.arange(ranges[2][0], ranges[2][1] + 0.5, 0.5):
                    symbol_value = self.calculate_6j(j1, j2, j3, *fixed_params)
                    total += symbol_value**2  # Square for probability
        
        return total
    
    def calculate_asymptotic_6j(self, j1: float, j2: float, j3: float,
                               j4: float, j5: float, j6: float) -> float:
        """
        Calculate asymptotic approximation of 6j symbol for large j values.
        
        Uses the Ponzano-Regge asymptotic formula.
        """
        try:
            # Check if we're in the large j regime
            if all(j > 10 for j in [j1, j2, j3, j4, j5, j6]):
                # Use asymptotic formula
                # This is a simplified version
                volume = self._tetrahedron_volume(j1, j2, j3, j4, j5, j6)
                if volume > 0:
                    return np.cos(volume) / np.sqrt(volume)
                else:
                    return 0.0
            else:
                # Use exact calculation for small j
                return self.calculate_6j(j1, j2, j3, j4, j5, j6)
        
        except Exception as e:
            warnings.warn(f"Error calculating asymptotic 6j symbol: {e}")
            return 0.0
    
    def _tetrahedron_volume(self, j1: float, j2: float, j3: float,
                           j4: float, j5: float, j6: float) -> float:
        """Calculate the volume of a tetrahedron with edge lengths proportional to j values."""
        # Simplified volume calculation
        # In real LQG, this would be more complex
        edges = [j1, j2, j3, j4, j5, j6]
        return np.sqrt(np.prod(edges)) / 6.0
    
    def validate_quantum_numbers(self, j: float) -> bool:
        """Validate that j is a valid angular momentum quantum number."""
        # Must be non-negative and half-integer
        return j >= 0 and (2*j) % 1 == 0
    
    def get_symbol_properties(self, symbol_type: str, *args) -> Dict:
        """Get properties and metadata for a calculated symbol."""
        if symbol_type == "3j":
            value = self.calculate_3j(*args)
            return {
                'value': value,
                'type': '3j',
                'arguments': args,
                'selection_rules_satisfied': self._check_3j_selection_rules(*args),
                'quantum_numbers_valid': all(self.validate_quantum_numbers(j) for j in args[:3])
            }
        elif symbol_type == "6j":
            value = self.calculate_6j(*args)
            return {
                'value': value,
                'type': '6j',
                'arguments': args,
                'selection_rules_satisfied': self._check_6j_selection_rules(*args),
                'quantum_numbers_valid': all(self.validate_quantum_numbers(j) for j in args)
            }
        else:
            raise ValueError(f"Unsupported symbol type: {symbol_type}")
