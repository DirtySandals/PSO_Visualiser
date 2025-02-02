from pso_lib import *
import numpy as np
import matplotlib.cm as cm
import pygame

def scale_particles(particles, boundaries, heatmap_length: int, left: int, top: int):
    center_x = left + heatmap_length // 2
    center_y = top + heatmap_length // 2

    # Scale points to fit in graph exactly
    scale_x = (heatmap_length // 2) / boundaries[0][1]
    scale_y = (heatmap_length // 2) / boundaries[1][1]
    # Scale and translate each point to fit in graph
    for i in range(len(particles)):
        particles[i][0] = center_x + (particles[i][0]) * scale_x     
        particles[i][1] = center_y + (particles[i][1]) * scale_y
        if abs(particles[i][0] - center_x) > heatmap_length // 2:
            particles[i][0] = -100
        elif abs(particles[i][1] - center_y) > heatmap_length // 2:
            particles[i][1] = -100

    return particles

def generate_heatmap(pygame: pygame, problem: OptimizationProblem, length: int, divisions: int, filename: str):
    min = problem.boundaries[0][0]
    max = problem.boundaries[0][1]
    x = np.linspace(min, max, divisions)
    y = np.linspace(min, max, divisions)

    X, Y = np.meshgrid(x, y)

    
    Z = np.array([problem.evaluate(np.array([X[i][j], Y[i][j]])) for i in range(X.shape[0]) for j in range(X.shape[1])])
    Z = Z.reshape(X.shape)

    Z_min, Z_max = Z.min(), Z.max()
    Z_norm = (Z - Z_min) / (Z_max - Z_min)  # Normalize between 0 and 1
    
    # Convert Z values into a colormap
    colormap = cm.viridis  # Change colormap if needed
    colors = (colormap(Z_norm)[:, :, :3] * 255).astype(np.uint8)  # RGB values

    # Convert the color grid into a Pygame surface
    heatmap_surface = pygame.surfarray.make_surface(colors.transpose(1, 0, 2))
    scaled_surface = pygame.transform.scale(heatmap_surface, (length, length))
    pygame.image.save(scaled_surface, filename)
    return scaled_surface