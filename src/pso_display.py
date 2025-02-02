from pso_lib import *
import numpy as np
import matplotlib.cm as cm
import pygame

def scale_particles(particles: List, boundaries: List, heatmap_length: int, left: int, top: int):
    """
    This function accepts a list of particles, a list of boundaries for the particles, the width/height
    of the heatmap, as well as its left-most and top-most pixel and outputs a list of new particle positions. 
    scale_particles scales the positions of said particles to the size of the heatmap, whilst making sure the
    out-of-bounds particles remain out of sight
    """
    # Calculate center of heatmap
    center_x = left + heatmap_length // 2
    center_y = top + heatmap_length // 2

    # Scale points to fit in graph exactly
    scale_x = (heatmap_length // 2) / boundaries[0][1]
    scale_y = (heatmap_length // 2) / boundaries[1][1]
    # Scale and translate each point to fit in graph
    for i in range(len(particles)):
        particles[i][0] = center_x + (particles[i][0]) * scale_x     
        particles[i][1] = center_y + (particles[i][1]) * scale_y
        # If particle out-of-bounds, ensure it is out of view
        if abs(particles[i][0] - center_x) > heatmap_length // 2:
            particles[i][0] = -100
        elif abs(particles[i][1] - center_y) > heatmap_length // 2:
            particles[i][1] = -100

    return particles

def generate_heatmap(pygame: pygame, problem: OptimizationProblem, length: int, divisions: int):
    """
    This function accepts pygame to generate surface, problem to model heatmap, length for width/height,
    and divisions for the resolution of the heatmap. This function generates a heatmap for a problem
    with minimal values being cold colours and maximal values being warm colours.
    """
    # Identify boundaries of problem
    min = problem.boundaries[0][0]
    max = problem.boundaries[0][1]
    # Create two arrays with x and y values between boundaries with 'divisions' elements
    x = np.linspace(min, max, divisions)
    y = np.linspace(min, max, divisions)
    # Creates two 2D arrays X and Y with repeated rows/columns of their lowercase counterparts
    X, Y = np.meshgrid(x, y)
    # Create a 2D array of values derived from problem equation.
    Z = np.array([problem.evaluate(np.array([X[i][j], Y[i][j]])) for i in range(X.shape[0]) for j in range(X.shape[1])])
    Z = Z.reshape(X.shape) # Ensure heatmap becomes (divisions, divisions)

    Z_min, Z_max = Z.min(), Z.max()
    Z_norm = (Z - Z_min) / (Z_max - Z_min)  # Normalize between 0 and 1
    
    # Convert Z values into a colormap
    colormap = cm.viridis
    colors = (colormap(Z_norm)[:, :, :3] * 255).astype(np.uint8)  # RGB values

    # Convert the color grid into a Pygame surface
    heatmap_surface = pygame.surfarray.make_surface(colors.transpose(1, 0, 2))
    scaled_surface = pygame.transform.scale(heatmap_surface, (length, length))

    return scaled_surface