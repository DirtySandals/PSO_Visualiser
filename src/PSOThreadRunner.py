from pso_lib import *
import numpy as np
import queue
from typing import List
import threading
from functools import partial
import time

class PSOThreadRunner:
    """
    PSOThreadRunner runs the pso_lib Optimizer in a separate thread
    and controls the output of particle generations using a queue of 'frames'
    """
    def __init__(self):
        # Init variables
        self.pso_thread = None
        self.pso = None
        self.early_stopping_tolerance = 100
        self.particles = None
        self.particle_queue = queue.Queue()
        self.last_frame_time = time.time_ns()
        self.frame_diff = 10
        self.num_frames = 0

    def load_alg(self, pso: Optimizer):
        """Accepts an algorithm to use"""
        self.pso = pso
        self.update_particles(self.pso.swarm.particles) # Create initial particles positions frame

    def run_pso(self, max_generations: int):
        """Runs the PSO algorithm"""
        if self.pso is None:
            raise TypeError("pso algorithm has not been loaded.")

        # If algorithm is already running, stop it and go again
        if self.pso_thread is not None:
            self.stop()
        # Run algorithm in parallel
        self.pso_thread = threading.Thread(
            target=partial(
                self.pso.optimize,
                max_generations,
                track_stats=False,
                export_particles=self.update_particles,
                early_stopping_tolerance=self.early_stopping_tolerance
            )
        )
        self.pso_thread.start()

    def stop(self):
        """
        Stops the PSO algorithm from running and joins respective
        thread
        """
        # Reinitialise variables
        self.particles = None
        self.num_frames = 0
        self.particle_queue = queue.Queue()

        if self.pso_thread is None:
            return
        
        if self.pso_thread is None:
            return
        
        if not self.pso_thread.is_alive():
            return
        
        self.pso.stop() # Stop algorithm
        self.pso_thread.join() # Kill thread
        self.pso_thread = None

    def update_particles(self, particles: List[Particle]):
        """Add new generation of particle positions to queue"""
        new_particles = np.empty((len(particles), 2))

        for index, particle in np.ndenumerate(particles):
            new_particles[index] = particle.position.copy()

        self.particle_queue.put(new_particles)
        self.num_frames += 1

    def get_particles(self):
        """
        Getter function for particles but controls the frequency in
        in which generations can be accessed
        """
        # If no new generations in queue, use current generation
        if self.num_frames == 0:
            if self.particles is None:
                return None
            else:
                return self.particles.copy()

        # Calculate time since last 'frame' taken
        time_now = time.time_ns()
        time_diff = (time_now - self.last_frame_time) // 1_000_000
        # If time has been long enough since last 'frame'
        if time_diff > self.frame_diff:
            self.last_frame_time = time_now
            self.num_frames -= 1
            self.particles = self.particle_queue.get()
        elif self.particles is None:
            return None

        return self.particles.copy()