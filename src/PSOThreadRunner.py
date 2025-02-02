from pso_lib import pso_lib
from pso_lib import OptimizationProblems as OP
from pso_lib import *
import numpy as np
import queue
from typing import List
import threading
import time

class PSOThreadRunner:
    def __init__(self):
        self.pso_thread = None
        self.pso = None
        self.early_stopping_tolerance = 500
        self.particles = None
        self.particle_queue = queue.Queue()
        self.last_frame_time = time.time_ns()
        self.frame_diff = 10
        self.num_frames = 0

    def load_alg(self, pso: Optimizer):
        self.pso = pso

        pop_size = pso.swarm.population_size

        self.particles = None

    def run_pso(self, max_generations: int):
        if self.pso is None:
            raise TypeError("pso algorithm has not been loaded.")

        if self.pso_thread is not None:
            self.stop()

        self.pso_thread = threading.Thread(target=self.pso.optimize(
            max_generations, 
            track_stats=False, 
            export_particles=self.update_particles,
            early_stopping_tolerance=self.early_stopping_tolerance
        ))
        self.pso_thread.run()

    def stop(self):
        if self.pso_thread is None:
            return
        
        if self.pso_thread is None:
            return
        
        if not self.pso_thread.is_alive():
            return
        
        self.pso.stop()
        self.pso_thread.join()
        self.pso_thread = None

    def update_particles(self, particles: List[Particle]):
        new_particles = np.empty((len(particles), 2))

        for index, particle in np.ndenumerate(particles):
            new_particles[index] = particle.position.copy()

        self.particle_queue.put(new_particles)
        self.num_frames += 1

    def get_particles(self):
        if self.num_frames == 0:
            if self.particles is None:
                return None
            else:
                return self.particles.copy()
        
        time_now = time.time_ns()
        time_diff = (time_now - self.last_frame_time) // 1_000_000

        if time_diff > self.frame_diff:
            self.last_frame_time = time_now
            self.num_frames -= 1
            self.particles = self.particle_queue.get()
        elif self.particles is None:
            return None
        
        return self.particles.copy()