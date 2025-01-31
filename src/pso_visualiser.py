import pygame
import sys
from numpy import add
from enum import Enum
from typing import List, Tuple
from Button import Button
from pso_lib import pso_lib
from pso_lib import OptimizationProblems as op
# Initialise pygame and screen variables
pygame.init()

WIDTH, HEIGHT = 800, 600

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TSP")

background = (81, 213, 224)

clock = pygame.time.Clock()

font_path = "./assets/font.ttf"

problems = {
    "Ackley Problem": op.AckleyProblem
}

back_button = Button(
    image=None,
    pos=(WIDTH // 10, HEIGHT // 10),
    text_input="Back",
    font=pygame.font.Font(font_path, 16),
    base_color=(0, 0, 0),
    hovering_color="White"
)

point_color = (255, 0, 0)
point_radius = 5

graph_width = WIDTH * 0.60
graph_height = HEIGHT * 0.60
graph_center_x = WIDTH // 2
graph_center_y = HEIGHT // 2
graph_color = (255, 255, 255)

graph = pygame.Rect(graph_center_x - graph_width // 2, graph_center_y - graph_height // 2, graph_width, graph_height)

# Quit function to exit game gracefully
def quit_gui() -> None:
    global process
    process.cleanup()
    pygame.quit()
    sys.exit()
    
# Displays screen for showing algorithm work    
def display_alg(points: List[Tuple[int, int]], inverover: bool, mutator: str, crossover: str, 
                selector: str, pop_size: str) -> None:
    start_button = Button(
        image=None,
        pos=(WIDTH // 2, HEIGHT * 6 // 7),
        text_input="Start Algorithm",
        font=pygame.font.Font(font_path, 24),
        base_color=(0, 0, 0),
        hovering_color="White"
    )
    
    line_color = (0, 0, 0)
    line_width = int(point_radius * 1.3)

    while True:
        screen.fill(background)
        
        mouse_pos = pygame.mouse.get_pos()      

        back_button.changeColor(mouse_pos)
        back_button.update(screen)
        
        pygame.draw.rect(screen, graph_color, graph)
        
        # Draw hamiltonian cycle path between points via their indexes
        if len(process.best_route) > 0:
            for i, city in enumerate(process.best_route):
                index = process.best_route[i]
                next_index = process.best_route[(i + 1) % len(process.best_route)]
                
                pygame.draw.line(screen, line_color, points[index - 1], points[next_index - 1], line_width)
        
        # Draw problem points        
        for point in points:
            pygame.draw.circle(screen, point_color, point, point_radius)

        start_button.changeColor(mouse_pos)
        start_button.update(screen)

            
        # Handle clickable events
        for event in pygame.event.get():
            # Quit game
            if event.type == pygame.QUIT:
                quit_gui()
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Go back to previous screen and stop process from running algorithm
                if back_button.checkForInput(mouse_pos):
                    process.stop()
                    return
                # Start the algorithm when button pressed
                if start_button.checkForInput(mouse_pos):
                    process.stop()
                    process.start_ga(inverover, 20000, pop_size, mutator, crossover, selector)
        
        clock.tick(60)
        pygame.display.update()
        
# Allows user to choose genetic algorithm they want to use on their problem
def configure_algorithm(points: List[Tuple[int, int]]) -> None:
    # Handle clicking on button
    def handle_option_click(options, mouse_pos, current_value):
        for option in options:
            if option["button"].checkForInput(mouse_pos):
                return option["value"]
        
        return current_value
    # Enum type for what algorithm type is chosen
    class AlgorithmType(Enum):
        InverOver = 0
        Custom = 1
        
    selected_algorithm = None
    
    # Default algorithm
    selected_topology = "inversion"
    selected_population_size = "50"
    # Create grid layout for options
    option_rows = [200, 350]
    option_cols = [250, 550]

    option_cell = []
    
    for row in option_rows:
        for col in option_cols:
            option_cell.append((col, row))

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
    
    custom_button = Button(
            image=None,
            pos=(WIDTH // 2, select_rect.centery + select_rect.height + 90),
            text_input="Custom Algorithm",
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
        
        if selected_algorithm is AlgorithmType.Custom:
            next_button.changeColor(mouse_pos)
            next_button.update(screen)
        # Display algorithm select
        if selected_algorithm is None:
            screen.blit(select_text, select_rect)
            
            standard_button.changeColor(mouse_pos)
            standard_button.update(screen)

            custom_button.changeColor(mouse_pos)
            custom_button.update(screen)
        # Display custom algorithm options   
        elif selected_algorithm is AlgorithmType.Custom:
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
                if option["value"] == selected_mutator:
                    option["button"].base_color = (255, 0, 0)
                else:
                    option["button"].base_color = (0, 0, 0)
                    
                option["button"].changeColor(mouse_pos)
                option["button"].update(screen)

            population_text = subtitle_font.render("Population Size", True, (0, 0, 0), None)
            population_rect = population_text.get_rect(center=option_cell[3])
            
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
                        selected_algorithm = AlgorithmType.InverOver
                        display_alg(points, True, None, None, None, None)
                        return
                    elif custom_button.checkForInput(mouse_pos):
                        selected_algorithm = AlgorithmType.Custom
                # Select option
                elif selected_algorithm is AlgorithmType.Custom:
                    if next_button.checkForInput(mouse_pos):
                        display_alg(points, False, selected_mutator, selected_crossover, selected_selector, selected_population_size)
                        return
                    
                    selected_topology = handle_option_click(topologies, mouse_pos, selected_topology)
                    selected_population_size = handle_option_click(population_sizes, mouse_pos, selected_population_size)
        
        clock.tick(60)
        pygame.display.update()
        
# Allows user to create an instance by clicking on screen
def make_equation() -> None:
    equation_str = ""

    solve_button = Button(
        image=None,
        pos=(WIDTH // 2, HEIGHT - 60),
        text_input="Solve Problem",
        font=pygame.font.Font(font_path, 24),
        base_color=(0, 0, 0),
        hovering_color="White"
    )
    # Warnings to display
    show_empty_warning = False
    warning_font = pygame.font.Font(font_path, 12)
    warning_pos = (WIDTH // 2, HEIGHT - 100)
    warning_color = (255, 0, 0)
    
    while True:
        screen.fill(background)
        
        mouse_pos = pygame.mouse.get_pos()

        back_button.changeColor(mouse_pos)
        back_button.update(screen)
        
        pygame.draw.rect(screen, graph_color, graph)
        
        # Display Warning logic
        if len(equation_str) == 0:
            pass
            
        if show_empty_warning:
            warning = warning_font.render("You Must Have More Than Three Entries", True, warning_color, None)
            warning_rect = warning.get_rect(center=warning_pos)
            
            screen.blit(warning, warning_rect) 

        solve_button.changeColor(mouse_pos)
        solve_button.update(screen)

        for event in pygame.event.get():
            # Quit game
            if event.type == pygame.QUIT:
                quit_gui()
            elif event.type == pygame.KEYUP:
                equation_str += pygame.key.name(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Back button
                if back_button.checkForInput(mouse_pos):
                    return
                # Solve button, go to algorithm select page
                if solve_button.checkForInput(mouse_pos):
                    # If 3 or less points, too little amount of cities, warn user
                    if len(equation_str) == 0:
                        show_empty_warning = True
                    # Progress to next page
                    else:
                        equation = process.load_instance(equation_str)
                        configure_algorithm(equation)
        
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
    for i, instance in enumerate(problems):
        buttons.append(Button(
            image=None,
            pos=(WIDTH // 2, button_start_height + button_gap * i),
            text_input=instance,
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
                        # Load file on process
                        process.load_file(button.text_input)
                        # Retrieve coords from file
                        instance_coords = parse_instance(path)
                        # Scale coords to fit on screen
                        instance_coords = instance_scaler(instance_coords, graph_center_x, graph_center_y, graph_width, graph_height)
                        # Go to next page to configure algorithm
                        configure_algorithm(instance_coords)
                        break
        
        clock.tick(60)
        pygame.display.update()
        
# Main menu for application
def main_menu() -> None:
    # Init buttons
    button_font = pygame.font.Font(font_path, 24)
    
    file_instance_button = Button(
        image=None,
        pos=(WIDTH // 2, HEIGHT // 2),
        text_input="Solve Standard Equations",
        font=button_font,
        base_color=(0, 0, 0),
        hovering_color="White"
    )

    custom_instance_button = Button(
        image=None,
        pos=(file_instance_button.x_pos, file_instance_button.y_pos + file_instance_button.font.get_height() * 2),
        text_input="Solve Custom Equation",
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
            hovering_color="White"
    )
    
    while True:
        screen.fill(background)
        
        mouse_pos = pygame.mouse.get_pos()
        # Quit button to go back
        quit_button.changeColor(mouse_pos)
        quit_button.update(screen)
        # Display title
        title_font = pygame.font.Font(font_path, 42)
        menu_title_1 = title_font.render("Particle Swarm Optimisation", True, (0, 0, 0), None)
        menu_title_rect_1 = menu_title_1.get_rect(center=(WIDTH // 2, HEIGHT // 4))
        
        screen.blit(menu_title_1, menu_title_rect_1)

        menu_title_2 = title_font.render("2D Equation Solver", True, (0, 0, 0), None)
        menu_title_rect_2 = menu_title_2.get_rect(center=(WIDTH // 2, (HEIGHT // 4) + menu_title_1.get_height()))
        
        screen.blit(menu_title_2, menu_title_rect_2)
        # Display buttons
        for button in [file_instance_button, custom_instance_button]:
            button.changeColor(mouse_pos)
            button.update(screen)
        try:
            for event in pygame.event.get():
                # Quit game
                if event.type == pygame.QUIT:
                    quit_gui()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Load instance from file
                    if file_instance_button.checkForInput(mouse_pos):
                        select_equation()
                    # Create instance on screen
                    if custom_instance_button.checkForInput(mouse_pos):
                        make_equation()
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