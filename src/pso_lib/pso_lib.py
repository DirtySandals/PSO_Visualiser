import numpy as np
import sys
import random
from typing import Optional
from .OptimizationProblems import *
from abc import ABC, abstractmethod

class Particle:
    """
    Class for encapsulating a particle, having its own position, velocity, fitness,
    best position and best fitness. Also includes a pointer to the neighbourhood it is apart of
    """
    def __init__(self, dimension_size: int):
        """Constructor accepts the dimension size and initialises variables"""
        self.position = np.empty(dimension_size)
        self.velocity = np.empty(dimension_size)
        self.fitness = None
        self.best_position = None
        self.best_fitness = None
        self.neighbourhood = None
    
    def update(self, new_fitness: float):
        """
        Accepts new fitness value and updates its fitness
        If fitness is better than best fitness, update its best position and fitness
        and call its neighbourhood to potentially update its 'best' variables
        """
        self.fitness = new_fitness

        if self.best_fitness is None or self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()
            self.neighbourhood.update_fitness(self)
        
    def __repr__(self):
        """Function to print particle's position and velocity"""
        return f"Particle(position: {self.position}, velocity: {self.velocity})"


class Topology(ABC):
    """Base class for topologies/neighbourhoods"""
    def __init__(self, particles, dimension_size: int):
        """Constructor to initialise variables"""
        self.particles = particles
        self.best_position = None
        self.best_fitness = None
        self.dimension_size = dimension_size
        self.topology_size = len(particles)

    @staticmethod
    @abstractmethod
    def init_topology(swarm):
        """Abstract static method to initialise a swarm's particles' topology"""
        pass

    @abstractmethod
    def update_fitness(self, particle):
        """
        Abstract method which accepts a particle and updates the best position and fitness of the
        topologies it is apart of
        """
        # Check and update the neighborhood's best position and fitness
        for i in range(self.topology_size):
            cur_topology = self.particles[i].neighbourhood
            # If cur_topology's best fitness is none or less than particle's fitness, update it's 'best' variables
            if cur_topology.best_fitness is None or particle.fitness < cur_topology.best_fitness:
                cur_topology.best_fitness = particle.fitness
                cur_topology.best_position = particle.position.copy()
        

class Gbest(Topology):
    """
    Gbest topology derived class implements the global topology
    where all particles are connected
    """
    def __init__(self, particles, dimension_size: int):
        """Call base class constructor"""
        super().__init__(particles, dimension_size)

    @staticmethod
    def init_topology(swarm):
        """
        Initialise the topology of a swarm. Since Gbest is global, each particle share
        the same neighbourhood
        """
        topology = Gbest(swarm.particles, swarm.dimension)
        for i in range(swarm.population_size):
            swarm.particles[i].neighbourhood = topology

    def update_fitness(self, particle):
        """Since only one topology in the whole swarm, only update the 'best' variables of this topology"""
        if self.best_fitness is None or particle.fitness < self.best_fitness:
            self.best_fitness = particle.fitness
            self.best_position = particle.position.copy()
        

class Lbest(Topology):
    """
    Lbest class implements the lBest topology, where each particle has a neighbourhood
    containing themselves and two other particles.
    """
    def __init__(self, particles, dimension_size: int):
        """Call base class constructor"""
        super().__init__(particles, dimension_size)

    @staticmethod
    def init_topology(swarm):
        """
        Initialise the topology of a swarm by iterating through particles and create topology
        for each particle containing themselves and the particle before and after itself
        """
        for i in range(swarm.population_size):
            # limits are the indexes of adjacent particles
            lower_limit = i - 1
            upper_limit = (i + 1) % swarm.population_size
            # neighbourhood particles include itself and two adjacent in list
            particle_list = [swarm.particles[lower_limit], swarm.particles[i], swarm.particles[upper_limit]]
            neighbourhood = Lbest(particle_list, swarm.dimension)
            swarm.particles[i].neighbourhood = neighbourhood

    def update_fitness(self, particle: Particle):
        """Update fitness using default topology method"""
        super().update_fitness(particle)
        

class Star(Topology):
    """
    Select a random particle as the star. The star has every particle in its neighborhood.
    Every other particle has only itself and the star in its neighborhood.
    """
    def __init__(self, particles, dimension_size: int):
        """Call base class constructor"""
        super().__init__(particles, dimension_size)

    @staticmethod
    def init_topology(swarm):
        """
        Initialise topology by randomly choosing a particle as the 'center', then creating topologies
        of particles connecting to it and itself
        """
        # Randomly select center of swarm
        star_particle = random.choice(swarm.particles)
        # Center's neighbourhood is global
        star_particle.neighbourhood = Star(swarm.particles, swarm.dimension)

        # For each particle other than center, initialise neighbourhood with itself and center
        for particle in swarm.particles:
            if particle == star_particle:
                continue
            particle.neighbourhood = Star([particle, star_particle], swarm.dimension)

    def update_fitness(self, particle: Particle):
        """Update fitness using default topology method"""
        super().update_fitness(particle)
        

class Random50(Topology):
    def __init__(self, particles, dimension_size: int):
        """Call base class constructor"""
        super().__init__(particles, dimension_size)

    @staticmethod
    def init_topology(swarm):
        """Initialise a swarm by having each particle connect randomly to half the swarm's particles"""
        n = len(swarm.particles)
        half_n = n // 2  # Randomly select half of the particles

        for particle in swarm.particles:
            particles_without_particle = [x for x in swarm.particles if x != particle]
            # Sample half_n - 1 of the swarm particles (need 1 for particle itself)
            neighborhood = random.sample(particles_without_particle, max(half_n - 1, 0))
            # append itself to list
            neighborhood.append(particle)
            # Create topology of randomly selected particles
            particle.neighbourhood = Random50(neighborhood, swarm.dimension)

    def update_fitness(self, particle: Particle):
        """Update fitness using default topology method"""
        super().update_fitness(particle)


class Swarm:
    """Swarm class acts as a container for all the particles"""
    def __init__(self, population_size: int, problem: OptimizationProblem):
        """Initialise variables"""
        self.particles = []
        self.dimension = problem.dimension_size
        self.population_size = population_size
        self.problem = problem
        self.initialize_population()

    def initialize_population(self):
        """Initialise each particle of specified dimension size"""
        for _ in range(self.population_size):
            self.particles.append(Particle(self.dimension))    
    
    def calculate_fitness(self):
        """Calculate fitness of each particle (used for initialisation pruposes)"""
        for i in range(self.population_size):
            new_fitness = self.problem.evaluate(self.particles[i].position)
            self.particles[i].update(new_fitness)
            
    # The following functions are used in graphing
    def get_center_of_mass(self):
        """Get function for mean position of swarm's particles"""
        positions = np.array([particle.position for particle in self.particles])
        return np.mean(positions, axis=0)

    def get_std(self, center_of_mass):
        """Get function for standard deviation of the positions of the particles"""
        positions = np.array([particle.position for particle in self.particles])
        return np.std(np.linalg.norm(positions - center_of_mass, axis=1))

    def get_mean_velocity_length(self):
        """Get function for the mean velocity of the particles"""
        velocities = np.array([particle.velocity for particle in self.particles])
        return np.mean(np.linalg.norm(velocities, axis=1))


class ParticleInitializer:
    """Class which contains static methods to initialize the position/velocity of swarm's particles"""
    @staticmethod
    def zero_velocity(swarm: Swarm):
        """Initialise swarm with zero velocity"""
        for particle in swarm.particles:
            particle.velocity = np.zeros(swarm.dimension)

    @staticmethod
    def uniform_random_velocity(swarm: Swarm, scale: float):
        """
        Initialise swarm with uniform random velocity bound by difference between random vector
        and current position, all multiplied by a scale float
        """
        # Obtain the maximum and minimum values for each dimensions
        min = swarm.problem.boundaries[:, 0]
        max = swarm.problem.boundaries[:, 1]
        for particle in swarm.particles:
            particle.velocity = np.random.uniform(low=min, high=max, size=swarm.dimension)
            particle.velocity -= particle.position
            particle.velocity *= scale

    @staticmethod
    def uniform_random_positions(swarm: Swarm, problem: OptimizationProblem):
        """Initialise swarm with uniform random position within problem bounds"""
        # Obtain the maximum and minimum values for each dimensions
        max = problem.boundaries[:, 1]
        min = problem.boundaries[:, 0]
        for particle in swarm.particles:            
            particle.position = np.random.uniform(low=min, high=max)

    @staticmethod
    def initial_bounds_uniform_positions(swarm: Swarm, problem: OptimizationProblem):
        """Initialise swarm with uniform random position within problem initial bounds"""
        # Obtain the maximum and minimum values for each dimensions
        max = problem.initialization_bounds[:, 1]
        min = problem.initialization_bounds[:, 0]
        for particle in swarm.particles:            
            particle.position = np.random.uniform(low=min, high=max)


class Optimizer(ABC):
    """
    Base class for PSO algorithms, for a given OptimizationProblem
    """
    def __init__(self, problem: OptimizationProblem, population_size: int, inertia: bool):
        """Initialise problem and swarm"""
        self.problem = problem
        self.swarm = Swarm(population_size, problem)
        self.inertia = False
        self.c1 = 2.05
        self.c2 = 2.05
        self.constriction_factor = 0.72984
        self.w_max = 0.9  # Initial inertia weight
        self.w_min = 0.4  # Final inertia weight
        self.n = 0.1      # Modulation index

    @abstractmethod
    def optimize(self, max_generations: int, use_inertia_weight: bool, early_stopping_tolerance: int=2000):
        """ Optimization process for PSO """
        self.init_metrics(max_generations)

        best_generation = 0
        best_found_fitness = sys.maxsize
        generation = 0

        for generation in range(max_generations):
            if (use_inertia_weight):
                inertia_weight = self.calculate_nonlinear_inertia_weight(generation, max_generations)
                self.optimize_generation(inertia_weight)
            else:
                self.optimize_generation()
            # Update metrics
            self.track_metrics(generation)

            if self.min_fitness < best_found_fitness:
                best_generation = generation
                best_found_fitness = self.min_fitness

            if (generation - best_generation) > early_stopping_tolerance:
                print(f"Early stopping at generation {generation + 1}")
                self.pad_stats(max_generations, generation)
                break
    
    def calculate_nonlinear_inertia_weight(self, t, T):
        """ Calculates the inertia weight using non-linear adjustment """
        return ((T - t) / T) ** self.n * (self.w_min - self.w_max) + self.w_max
    
    @abstractmethod
    def update_velocity_position(self, particle: Particle, inertia_weight: Optional[float]=None):
        pass

    @abstractmethod
    def optimize_generation(self, inertia_weight: Optional[float]=None):
        """ Single generation optimization with inertia weight adjustment """
        for particle in self.swarm.particles:
            self.update_velocity_position(particle, inertia_weight)
            new_fitness = self.problem.evaluate(particle.position)
            if self.problem.in_bounds(particle.position):
                self.min_fitness = min(self.min_fitness, new_fitness)
                particle.update(new_fitness)

    def init_metrics(self, max_generations: int):
        self.best_fitness_values = np.empty(max_generations)
        self.std = np.empty(max_generations)
        self.mean_velocity_lengths = np.empty(max_generations)
        self.min_fitness = min(particle.fitness for particle in self.swarm.particles)

    """
    Used to track metrics used for graphing
    Currently prints each generations results, could modify this to save to an excel if needed
    """
    def track_metrics(self, generation: int):
        """ Tracks metrics used for analysis and graphing """
        best_fitness = self.min_fitness
        center_of_mass = self.swarm.get_center_of_mass()
        std = self.swarm.get_std(center_of_mass)
        mean_velocity_length = self.swarm.get_mean_velocity_length()

        self.best_fitness_values[generation] = best_fitness
        self.std[generation] = std
        self.mean_velocity_lengths[generation] = mean_velocity_length

    def pad_stats(self, max_generations: int, current_generation: int):
        """ Pads the tracked metrics with NaN for unfinished generations """
        for i in range(max_generations - (current_generation + 1)):
            index = current_generation + 1 + i
            self.best_fitness_values[index] = np.nan
            self.std[index] = np.nan
            self.mean_velocity_lengths[index] = np.nan
            
class StandardPSO(Optimizer):
    """ 
    Class is used for problem 6,7,9
    """
    def __init__(self, problem: OptimizationProblem, population_size: int):
        super().__init__(problem, population_size)

        # Initialise the swarm's positions
        ParticleInitializer.uniform_random_positions(self.swarm, self.problem)
        #Initialise the swarm's velocities
        ParticleInitializer.uniform_random_velocity(self.swarm, 0.5)
        Lbest.init_topology(self.swarm)

        self.swarm.calculate_fitness()

    def optimize(self, max_generations: int, early_stopping_tolerance=2000):
        """ Optimization process for Standard PSO """
        super().optimize(max_generations, False, early_stopping_tolerance)

    def optimize_generation(self, inertia_weight: Optional[float]=None):
        """ Single generation optimization for Standard PSO """
        super().optimize_generation(None)

    def update_velocity_position(self, particle: Particle, inertia_weight: Optional[float]=None):
        """ Updates particle velocity and position """
        rand1 = np.random.rand(self.problem.dimension_size)
        rand2 = np.random.rand(self.problem.dimension_size)

        cognitive_component = self.c1 * rand1 * (particle.best_position - particle.position)
        social_component = self.c2 * rand2 * (particle.neighbourhood.best_position - particle.position)

        new_velocity = self.constriction_factor * (particle.velocity + cognitive_component + social_component)
        particle.velocity = new_velocity
        particle.position += particle.velocity
    
class InertiaWeightPSO(Optimizer):
    """ 
    Class is used for problem 8.
    Uses non-linear inertia weight
    """
    def __init__(self, problem: OptimizationProblem, population_size: int, topology_type: Topology):
        super().__init__(problem, population_size)

        ParticleInitializer.uniform_random_positions(self.swarm, self.problem)
        ParticleInitializer.uniform_random_velocity(self.swarm, 0.5)
        topology_type.init_topology(self.swarm)

        self.swarm.calculate_fitness()

    def optimize(self, max_generations: int, early_stopping_tolerance=1e-8):
        """ Optimization process with inertia weight adjustment """
        super().optimize(max_generations, True, early_stopping_tolerance)

    def optimize_generation(self, inertia_weight: Optional[float]=None):
        """ Single generation optimization with inertia weight adjustment """
        super().optimize_generation(inertia_weight)

    def update_velocity_position(self, particle: Particle, inertial_weight: float):
        """ Updates particle velocity and position """
        rand1 = np.random.rand(self.problem.dimension_size)
        rand2 = np.random.rand(self.problem.dimension_size)

        cognitive_component = self.c1 * rand1 * (particle.best_position - particle.position)
        social_component = self.c2 * rand2 * (particle.neighbourhood.best_position - particle.position)

        new_velocity = inertial_weight * particle.velocity + cognitive_component + social_component
        particle.velocity = new_velocity
        particle.position += particle.velocity