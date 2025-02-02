import pygame
import sys
import os
from numpy import add
from enum import Enum
from Button import Button
from pso_lib import *
from PSOThreadRunner import PSOThreadRunner
from pso_display import *

# Initialise pygame and screen variables
pygame.init()

WIDTH, HEIGHT = 800, 600 # Dimensions of screen

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PSO Solver")

background = (81, 213, 224) # Background colour

clock = pygame.time.Clock()

pso_runner = PSOThreadRunner()

# Global font
current_directory = os.path.dirname(os.path.abspath(__file__))
font_path =  os.path.join(current_directory, "./assets/font.ttf")

# Map of functions to be used in PSO
problems = {
    "Parabola": SphereParabola(2),
    "Schwefel 1.2": Schwefel(2),
    "Generalised Rosenbrock": GeneralisedRosenbrock(2),
    "Generalised Schwefel": GeneralisedSchwefel(2),
    "Generalised Rastrigin": GeneralisedRastrigin(2),
    "Ackley Problem": AckleyProblem(2),
    "Generalised Griewank": GeneralisedGriewank(2),
    "Six-Hump Camel-Back": SixHumpCamelBack(),
    "Goldstein-Price": GoldsteinPrice()
}

# Map of topology options
topology_options = {
    "GBest": Gbest,
    "LBest": Lbest,
    "Star": Star,
    "Random50": Random50
}

# Back button used by most pages
back_button = Button(
    image=None,
    pos=(WIDTH // 10, HEIGHT // 10),
    text_input="Back",
    font=pygame.font.Font(font_path, 16),
    base_color=(0, 0, 0),
    hovering_color="White"
)

# Particle characteristics
particle_color = (255, 0, 0)
particle_radius = 3

# Equation heatmap characteristics
heatmap_side_length = 450
heatmap_center_x = WIDTH // 2
heatmap_center_y = HEIGHT // 2
heatmap_left = heatmap_center_x - (heatmap_side_length // 2)
heatmap_top = heatmap_center_y - (heatmap_side_length // 2)

# Quit function to exit game gracefully
def quit_gui() -> None:
    pygame.quit()
    sys.exit()
    
# Displays screen for showing algorithm work    
def display_alg(problem: OptimizationProblem, heatmap_filename: str, standard_pso: bool, pop_size: str, topology: str="") -> None:
    start_button = Button(
        image=None,
        pos=(WIDTH // 2, 550),
        text_input="Start Algorithm",
        font=pygame.font.Font(font_path, 24),
        base_color=(0, 0, 0),
        hovering_color="White"
    )
    
    pop_size = int(pop_size) # Convert to int

    # Initialise PSO algorithm
    pso = None

    if standard_pso:
        pso = StandardPSO(problem, pop_size)            
    else:
        topology = topology_options[topology]
        pso = InertiaWeightPSO(problem, pop_size, topology)

    pso_runner.load_alg(pso)
    
    # Retrieve heatmap from assets
    heatmap_path = os.path.join(current_directory, heatmap_filename)
    heatmap = pygame.image.load(heatmap_path)

    while True:
        screen.fill(background)
        
        mouse_pos = pygame.mouse.get_pos()      

        back_button.changeColor(mouse_pos)
        back_button.update(screen)

        start_button.changeColor(mouse_pos)
        start_button.update(screen)

        screen.blit(heatmap, (heatmap_left, heatmap_top))

        particles = pso_runner.get_particles() # Get generation of particles
    
        if particles is not None:
            # Scale particles to heatmap
            particles = scale_particles(particles, problem.boundaries, heatmap_side_length, heatmap_left, heatmap_top)
            # Draw all particles
            for particle in particles:
                pygame.draw.circle(screen, particle_color, (particle[0], particle[1]), particle_radius)

        # Handle clickable events
        for event in pygame.event.get():
            # Quit game
            if event.type == pygame.QUIT:
                quit_gui()
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Go back to previous screen and stop process from running algorithm
                if back_button.checkForInput(mouse_pos):
                    pso_runner.stop()
                    return
                # Start the algorithm when button pressed
                if start_button.checkForInput(mouse_pos):
                    pso_runner.stop()
                    # If pso already running, restart it
                    if pso is not None:
                        if standard_pso:
                            pso = StandardPSO(problem, pop_size)            
                        else:
                            pso = InertiaWeightPSO(problem, pop_size, topology)

                        pso_runner.load_alg(pso)
                    
                    pso_runner.run_pso(10000)
        
        clock.tick(60)
        pygame.display.update()
        
# Allows user to choose genetic algorithm they want to use on their problem
def configure_algorithm(problem: OptimizationProblem, heatmap_filename: str) -> None:
    # Handle clicking on button
    def handle_option_click(options, mouse_pos, current_value):
        for option in options:
            if option["button"].checkForInput(mouse_pos):
                return option["value"]
        
        return current_value
    # Enum type for what algorithm type is chosen
    class AlgorithmType(Enum):
        Standard = 0
        IW = 1
        
    selected_algorithm = None
    
    # Default algorithm
    selected_topology = "GBest"
    selected_population_size = "50"

    # Create grid layout for options
    option_row = 300
    option_cols = [250, 550]

    option_cell = []
    
    for col in option_cols:
        option_cell.append((col, option_row))

    # Intialise options
    subtitle_option_separation = 32
    
    option_font = pygame.font.Font(font_path, 16)

    topologies = []
    topology_types = ["GBest", "LBest", "Star", "Random50"]

    population_sizes = []
    sizes = ["25", "50", "100", "200"]

    for index, (topology, pop_size) in enumerate(zip(topology_types, sizes)):
        topologies.append({
            "button": Button(
                        image=None,
                        pos=tuple(add(option_cell[0], (0, subtitle_option_separation + 24 * index))),
                        text_input=topology,
                        font=option_font,
                        base_color=(0, 0, 0),
                        hovering_color="White"
                    ),
            "value": topology
        })
        
        population_sizes.append({
            "button": Button(
                        image=None,
                        pos=tuple(add(option_cell[1], (0, subtitle_option_separation + 24 * index))),
                        text_input=pop_size,
                        font=option_font,
                        base_color=(0, 0, 0),
                        hovering_color="White"
                    ),
            "value": pop_size
        })

    # Title of page
    select_font = pygame.font.Font(font_path, 38)
    select_text = select_font.render("Select Algorithm", True, (0, 0, 0), None)
    select_rect = select_text.get_rect(center=(WIDTH // 2, WIDTH // 4))

    # Algorithm choice buttons
    alg_choice_font = pygame.font.Font(font_path, 24)

    standard_button = Button(
            image=None,
            pos=(WIDTH // 2, select_rect.centery + select_rect.height + 50),
            text_input="Standard Algorithm",
            font=alg_choice_font,
            base_color=(0, 0, 0),
            hovering_color="White"
    )
    
    iw_button = Button(
            image=None,
            pos=(WIDTH // 2, select_rect.centery + select_rect.height + 90),
            text_input="Inertia Weight Algorithm",
            font=alg_choice_font,
            base_color=(0, 0, 0),
            hovering_color="White"
    )

    # Button to next page
    next_button = Button(
            image=None,
            pos=(WIDTH // 2, HEIGHT * 6 // 7),
            text_input="Next",
            font=pygame.font.Font(font_path, 24),
            base_color=(0, 0, 0),
            hovering_color="White"
    )
    
    while True:
        screen.fill(background)
        
        mouse_pos = pygame.mouse.get_pos()      

        back_button.changeColor(mouse_pos)
        back_button.update(screen)
        
        if selected_algorithm is AlgorithmType.IW:
            next_button.changeColor(mouse_pos)
            next_button.update(screen)
        # Display algorithm select
        if selected_algorithm is None:
            screen.blit(select_text, select_rect)
            
            standard_button.changeColor(mouse_pos)
            standard_button.update(screen)

            iw_button.changeColor(mouse_pos)
            iw_button.update(screen)
        # Display custom algorithm options   
        elif selected_algorithm is AlgorithmType.IW:
            custom_font = pygame.font.Font(font_path, 32)
            custom_title = custom_font.render("Custom Algorithm", True, (0, 0, 0), None)
            custom_rect = custom_title.get_rect(center=(WIDTH // 2, HEIGHT // 6))

            screen.blit(custom_title, custom_rect)
            
            subtitle_font = pygame.font.Font(font_path, 20)
            
            topology_text = subtitle_font.render("Topology", True, (0, 0, 0), None)
            topology_rect = topology_text.get_rect(center=option_cell[0])
            
            screen.blit(topology_text, topology_rect)
            # If option selected, change color to red, else color is black
            for option in topologies:
                if option["value"] == selected_topology:
                    option["button"].base_color = (255, 0, 0)
                else:
                    option["button"].base_color = (0, 0, 0)
                    
                option["button"].changeColor(mouse_pos)
                option["button"].update(screen)

            population_text = subtitle_font.render("Population Size", True, (0, 0, 0), None)
            population_rect = population_text.get_rect(center=option_cell[1])
            
            screen.blit(population_text, population_rect)
            # If option selected, change color to red, else color is black
            for option in population_sizes:
                if option["value"] == selected_population_size:
                    option["button"].base_color = (255, 0, 0)
                else:
                    option["button"].base_color = (0, 0, 0)

                option["button"].changeColor(mouse_pos)
                option["button"].update(screen)
            

        for event in pygame.event.get():
            # Quit game
            if event.type == pygame.QUIT:
                quit_gui()
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Go back or go back to seletion menu
                if back_button.checkForInput(mouse_pos):
                    if selected_algorithm is not None:
                        selected_algorithm = None
                    else:
                        return
                # Select algorithm
                if selected_algorithm is None:
                    if standard_button.checkForInput(mouse_pos):
                        selected_algorithm = AlgorithmType.Standard
                        display_alg(problem, heatmap_filename, True, "200")
                        return
                    elif iw_button.checkForInput(mouse_pos):
                        selected_algorithm = AlgorithmType.IW
                # Select option
                elif selected_algorithm is AlgorithmType.IW:
                    if next_button.checkForInput(mouse_pos):
                        display_alg(problem, heatmap_filename, False, selected_population_size, selected_topology)
                        return
                    
                    selected_topology = handle_option_click(topologies, mouse_pos, selected_topology)
                    selected_population_size = handle_option_click(population_sizes, mouse_pos, selected_population_size)
        
        clock.tick(60)
        pygame.display.update()
        
# Select standard optimisation problems
def select_equation() -> None:
    # Title and buttons
    title_font = pygame.font.Font(font_path, 32)
    menu_title = title_font.render("Select A Problem", True, (0, 0, 0), None)
    menu_title_rect = menu_title.get_rect(center=(WIDTH // 2, HEIGHT // 5))
    
    button_font = pygame.font.Font(font_path, 22)
    buttons = []
    button_start_height = menu_title_rect.centery + 60
    button_gap = button_font.get_height() * 1.5

    # Create list of buttons for each optimisation problem
    for index, name in enumerate(problems):
        buttons.append(Button(
            image=None,
            pos=(WIDTH // 2, button_start_height + button_gap * index),
            text_input=name,
            font=button_font,
            base_color=(0, 0, 0),
            hovering_color="White"
        ))
        
    while True:
        screen.fill(background)
        
        mouse_pos = pygame.mouse.get_pos()
        
        back_button.changeColor(mouse_pos)
        back_button.update(screen)

        screen.blit(menu_title, menu_title_rect)
        
        # Display buttons
        for button in buttons:
            button.changeColor(mouse_pos)
            button.update(screen)

        for event in pygame.event.get():
            # Quit game
            if event.type == pygame.QUIT:
                quit_gui()
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Go back
                if back_button.checkForInput(mouse_pos):
                    return
                # Detect clicking on button
                for button in buttons:
                    if button.checkForInput(mouse_pos):
                        problem = problems[button.text_input]
                        file_path = f"./assets/{button.text_input}.png" # File path to heatmap png file
                        # Go to next page to configure algorithm
                        configure_algorithm(problem, file_path)
                        break
        
        clock.tick(60)
        pygame.display.update()
        
# Main menu for application
def main_menu() -> None:
    # Init buttons
    button_font = pygame.font.Font(font_path, 20)
    
    std_eqn_button = Button(
        image=None,
        pos=(WIDTH // 2, HEIGHT // 1.5),
        text_input="Start",
        font=button_font,
        base_color=(0, 0, 0),
        hovering_color="White"
    )

    quit_button = Button(
            image=None,
            pos=(WIDTH // 10, HEIGHT // 10),
            text_input="Quit",
            font=pygame.font.Font(font_path, 16),
            base_color=(0, 0, 0),
            hovering_color="Red"
    )
    
    while True:
        screen.fill(background)
        
        mouse_pos = pygame.mouse.get_pos()
        # Quit button to go back
        quit_button.changeColor(mouse_pos)
        quit_button.update(screen)
        # Display title
        title_font = pygame.font.Font(font_path, 24)
        menu_title_1 = title_font.render("Particle Swarm Optimisation", True, (0, 0, 0), None)
        menu_title_rect_1 = menu_title_1.get_rect(center=(WIDTH // 2, HEIGHT // 2.5))
        
        screen.blit(menu_title_1, menu_title_rect_1)

        menu_title_2 = title_font.render("2D Equation Solver", True, (0, 0, 0), None)
        menu_title_rect_2 = menu_title_2.get_rect(center=(WIDTH // 2, (HEIGHT // 2.5) + menu_title_1.get_height() + 25))
        
        screen.blit(menu_title_2, menu_title_rect_2)

        # Display button
        std_eqn_button.changeColor(mouse_pos)
        std_eqn_button.update(screen)

        try:
            for event in pygame.event.get():
                # Quit game
                if event.type == pygame.QUIT:
                    quit_gui()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Load instance from file
                    if std_eqn_button.checkForInput(mouse_pos):
                        select_equation()
                    # Quit game
                    if quit_button.checkForInput(mouse_pos):
                        quit_gui()
        except SystemError as e:
            print(e)

        clock.tick(60)
        pygame.display.update()

# If main file, run application
if __name__ == "__main__":        
    main_menu()