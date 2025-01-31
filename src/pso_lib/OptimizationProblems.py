import numpy as np
import math
from abc import ABC, abstractmethod

class OptimizationProblem(ABC):
    """
    Base class for optimization problems
    """
    def __init__(self, boundaries: np.array, optimum: np.array, initialization: np.array):
        """Constructor to initialise internal variables"""
        self.dimension_size = boundaries.shape[0]
        self.boundaries = boundaries
        self.optimum = optimum
        self.optimum_fitness = None
        self.initialization_bounds = initialization

    @abstractmethod
    def evaluate(self, position: np.array) -> float:
        """Abstract method for evaluating position with an equation"""
        pass

    def in_bounds(self, position: np.array) -> bool:
        """Checks if position is within bounds of the array"""
        for i in range(self.dimension_size):
            if position[i] < self.boundaries[i][0] or position[i] > self.boundaries[i][1]:
                return False
            
        return True


class AckleyProblem(OptimizationProblem):
    """
    The AckleyProblem class is derived from the optimzation class and implements the ackley function.
    The Ackley problem is a classical optimization problem with many local optima, making it suitable for
    testing PSO algorithms to balance exploration and exploitation.
    """
    def __init__(self, dimension_size: int):
        """Initialise the problem's variables"""
        # Define boundaries and optimum of problem
        boundaries = np.tile([-30, 30], (dimension_size, 1))
        initialization = np.tile([16, 30], (dimension_size, 1))
        optimum = np.tile(0, (dimension_size, 1))
        self.inverse_dimension_size = 1 / dimension_size
        super().__init__(boundaries, optimum, initialization)
        # Calculate optimal fitness
        self.optimum_fitness = self.evaluate(optimum)
        

    def evaluate(self, position: np.array) -> float:        
        """
        Evaluate takes a particle as a parameter and returns the ackley function value
        of its position: -20 * exp{-0.2√(1/D *∑Di=1 (xi^2)} - exp{(1/D)*∑Di=1 (cos (2πxi))} + 20 + e
        """
        if self.dimension_size != position.size:
            raise ValueError("Particle's position of incorrect size")
        
        # Function calculating cos(2 * pi * number)
        def cos_two_pi(number: float) -> float:
            return np.cos(2 * np.pi * number)

        # Square each element in array
        squared_array = np.square(position)
        # Summation sequence of squared elements
        square_sum = np.sum(squared_array)

        # Create array where each element is x->cos(2*pi*x) from particle position
        cos_two_pi_array = np.cos(2 * np.pi * position)
        # Summation sequence of cos(2*pi*element)
        cos_sum = np.sum(cos_two_pi_array)

        # With all values attained, calculate ackley function
        ackley_value = -20 * math.exp(-0.2 * math.sqrt(self.inverse_dimension_size * square_sum))
        ackley_value = ackley_value - math.exp(self.inverse_dimension_size * cos_sum) + 20 + math.e

        return ackley_value