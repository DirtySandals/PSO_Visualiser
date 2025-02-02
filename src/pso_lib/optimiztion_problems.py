import numpy as np
import math
from abc import ABC, abstractmethod
from typing import List

class OptimizationProblem(ABC):
    """
    Base class for optimization problems
    """
    def __init__(self, boundaries: List[float], dimension_size: int):
        """Constructor to initialise internal variables"""
        self.dimension_size = dimension_size
        self.boundaries = np.tile(boundaries, (dimension_size, 1))

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
        boundaries = [-30, 30]
        super().__init__(boundaries, dimension_size)
        self.inverse_dimension_size = 1 / dimension_size

    def evaluate(self, position: np.array) -> float:   
        """
        Evaluate takes a particle as a parameter and returns the ackley function value
        of its position: -20 * exp{-0.2√(1/D *∑Di=1 (xi^2)} - exp{(1/D)*∑Di=1 (cos (2πxi))} + 20 + e
        """
        if self.dimension_size != position.size:
            raise ValueError("Particle's position of incorrect size")

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

class SphereParabola(OptimizationProblem):
    """
    The Sphere/Parabola class is derived from the optimzation class and implements the sphere/parabola function.
    The problem is fairly easy for PSO algorithms due to its monotonic nature stemming from its optima: [0, 0, ...]
    """
    def __init__(self, dimension_size: int):
        """Initialise the problem's variables"""
        # Define boundaries and optimum of problem
        boundaries = [-100, 100]
        super().__init__(boundaries, dimension_size)

    def evaluate(self, position: np.array) -> float:        
        """
        Evaluate takes a particle as a parameter and returns the sphere/parabola function value
        of its position: ∑Di=1 (xi^2)
        """
        if self.dimension_size != position.size:
            raise ValueError("Particle's position of incorrect size")

        squared_array = np.square(position)

        function_val = np.sum(squared_array)

        return function_val

class Schwefel(OptimizationProblem):
    """
    The Schwefel class is derived from the optimzation class and implements the schwefel 1.2 function.
    The schwefel function is monotonic stemming from a shallow optima about: [0, 0, ...]
    """
    def __init__(self, dimension_size: int):
        """Initialise the problem's variables"""
        # Define boundaries and optimum of problem
        boundaries = [-100, 100]
        super().__init__(boundaries, dimension_size)

    def evaluate(self, position: np.array) -> float:        
        """
        Evaluate takes a particle as a parameter and returns the schwefel function value
        of its position: ∑Di=1((∑Dj=1 (xj))^2)
        """
        if self.dimension_size != position.size:
            raise ValueError("Particle's position of incorrect size")

        cumsum_arr = np.cumsum(position)
        squared_arr = np.square(cumsum_arr)

        function_val = np.sum(squared_arr)

        return function_val

class GeneralisedRosenbrock(OptimizationProblem):
    """
    The GeneralisedRosenbrock class is derived from the optimzation class and implements the rosenbrock function.
    The problem also referred to as the valley or banana function, is a unimodal function with a global minima in
    a narrow, parabolic valley. Convergence to valley is easy, but towards global minima is difficult
    """
    def __init__(self, dimension_size: int):
        """Initialise the problem's variables"""
        # Define boundaries and optimum of problem
        boundaries = [-2.048, 2.048]
        super().__init__(boundaries, dimension_size)

    def evaluate(self, position: np.array) -> float:        
        """
        Evaluate takes a particle as a parameter and returns the generalised rosenbrock function value
        of its position: ∑(D-1)i=1(100(x(i+1) - xi^2)^2 + (xi - 1)^2)
        """
        if self.dimension_size != position.size:
            raise ValueError("Particle's position of incorrect size")

        def transform(index, value):
            index = index[0]
            t_value = 100 * (position[index + 1] - value ** 2) ** 2
            t_value = t_value + (value - 1) ** 2
            return t_value

        transformed_values = np.array([transform(i, v) for i, v in np.ndenumerate(position[:self.dimension_size - 1])])

        function_val = np.sum(transformed_values)

        return function_val

class GeneralisedSchwefel(OptimizationProblem):
    """
    The GeneralisedSchwefel class is derived from the optimzation class and implements the generalised schwefel function.
    The generalised schwefel function is complex with many local minima with a global minima at: [420.9687, ..., 420.9687]
    """
    def __init__(self, dimension_size: int):
        """Initialise the problem's variables"""
        # Define boundaries and optimum of problem
        boundaries = [-500, 500]
        super().__init__(boundaries, dimension_size)


    def evaluate(self, position: np.array) -> float:        
        """
        Evaluate takes a particle as a parameter and returns the generalised schwefel function value
        of its position: ∑Di=1(xi * sin(sqrt(xi)))
        """
        if self.dimension_size != position.size:
            raise ValueError("Particle's position of incorrect size")

        sqrt_arr = np.sqrt(np.abs(position))
        sin_arr = np.sin(sqrt_arr)
        result_arr = position * sin_arr

        function_val = 418.9829 * self.dimension_size - np.sum(result_arr)

        return function_val

class GeneralisedRastrigin(OptimizationProblem):
    """
    The GeneralisedRastrigin class is derived from the optimzation class and implements the rastrigin function.
    The problem has many local minima with steep gradients, with global minima at: [0, 0, ...]
    """
    def __init__(self, dimension_size: int):
        """Initialise the problem's variables"""
        # Define boundaries and optimum of problem
        boundaries = [-5.12, 5.12]
        super().__init__(boundaries, dimension_size)

    def evaluate(self, position: np.array) -> float:        
        """
        Evaluate takes a particle as a parameter and returns the generalised rastrigin function value
        of its position: ∑Di=1(xi^2 - 10 * cos(2*pi*xi) + 10)
        """
        if self.dimension_size != position.size:
            raise ValueError("Particle's position of incorrect size")

        squared_arr = np.square(position)

        cos_arr = -10 * np.cos(2 * np.pi * position)

        return_arr = squared_arr + cos_arr + 10

        function_val = np.sum(return_arr)

        return function_val

class GeneralisedGriewank(OptimizationProblem):
    """
    The GeneralisedGriewank class is derived from the optimzation class and implements the griewank function.
    The problem has many local minima, and global minima at: [0, 0, ...]
    """
    def __init__(self, dimension_size: int):
        """Initialise the problem's variables"""
        # Define boundaries and optimum of problem
        boundaries = [-600, 600]
        super().__init__(boundaries, dimension_size)

    def evaluate(self, position: np.array) -> float:        
        """
        Evaluate takes a particle as a parameter and returns the generalised grienwank function value
        of its position: (1/4000)∑Di=1(xi^2) - ∏Di=1(cos(xi/sqrt(i))) + 1
        """
        if self.dimension_size != position.size:
            raise ValueError("Particle's position of incorrect size")

        indicies = np.arange(1, len(position) + 1)

        square_arr = (1/4000) * np.square(position)
        prod_arr = -1 * np.prod(np.cos(position / np.sqrt(indicies)))

        function_val = np.sum(square_arr) + np.sum(prod_arr) + 1

        return function_val

class SixHumpCamelBack(OptimizationProblem):
    """
    The SixHumpCamelBack class is derived from the optimzation class and implements the six-hump camel back function.
    The problem has 6 local minima, with 2 global at: [0.0898, -0.7126] and [-0.0898, 0.7126]
    """
    def __init__(self):
        """Initialise the problem's variables"""
        # Define boundaries and optimum of problem
        boundaries = [-5, 5]
        super().__init__(boundaries, 2)

    def evaluate(self, position: np.array) -> float:        
        """
        Evaluate takes a particle as a parameter and returns the six-hump camel-back function value
        of its position: 4x1^2 - 2.1x1^4 + (1/3)x1^6 + x1x2 - 4x2^2 + 4x2^4
        """
        if self.dimension_size != position.size:
            raise ValueError("Particle's position of incorrect size")

        x1 = position[0]
        x2 = position[1]

        function_val  = 4 * (x1 ** 2) - 2.1 * (x1 ** 4) + (1/3) * (x1 ** 6)
        function_val += x1 * x2
        function_val += -4 * (x2 ** 2) + 4 * (x2 ** 4)

        return function_val

class GoldsteinPrice(OptimizationProblem):
    """
    The GoldsteinPrice class is derived from the optimzation class and implements the goldstein-price function.
    The function has several local minima, with global minima at: [0, -1]. However, the basin of attraction towards
    the global minima is shallow making it difficult to converge about the global minima
    """
    def __init__(self):
        """Initialise the problem's variables"""
        # Define boundaries and optimum of problem
        boundaries = [-2, 2]
        super().__init__(boundaries, 2)

    def evaluate(self, position: np.array) -> float:        
        """
        Evaluate takes a particle as a parameter and returns the goldstein price function value
        of its position: { 1 + (x1 + x2 + 1)^2 * (19 -14x1 + 3x1^2 - 14x2 + 6x1x2 + 3x2^2) }
        * { 30 + (2x1 - 3x2)^2 * (18 - 32x1 + 12x1^2 + 48x2 - 36x1x2 + 27x2^2) } 
        """
        if self.dimension_size != position.size:
            raise ValueError("Particle's position of incorrect size")

        x1 = position[0]
        x2 = position[1]

        prod1 = 1 + ((x1 + x2 + 1) ** 2) * (19 - 14 * x1 + 3 * (x1 ** 2) - 14 * x2 + 6 * x1 * x2 + 3 * (x2 ** 2))
        prod2 = 30 + ((2 * x1 - 3 * x2) ** 2) * (18 - 32 * x1 + 12 * (x1 ** 2) + 48 * x2 - 36 * x1 * x2 + 27 * (x2 ** 2))

        function_val = prod1 * prod2

        return function_val