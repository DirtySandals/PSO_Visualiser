<h1>TSPSolver and TSPGUI</h1>
<h2>Overview</h2>
This repository provides a GUI using the pygame library to visualise Particle Swarm Optimisation 
(PSO) on a variety of standard test functions. The PSO library within pso_lib.py utilises PSO to 
explore within the bounds of an equation's search space to identify the global minima. Each function 
is provided with an accurate color-gradient map to illustrate the minimal values (cold colours), and 
maximal values (warm colours).

Users are given the option to choose the following equations:

- Parabola
- Schwefel 1.2
- Generalised Rosenbrock
- Generalised Schwefel
- Generalised Rastrigin
- Ackley Problem
- Generalised Griewank
- Six-Hump Camel-Back
- Goldstein-Price

All of the aforementioned functions are modelled in the optimization_problems.py file.

After selecting an equation, a PSO algorithm can be selected from the pso_lib.

<h3>Standard Algorithm</h3>
The Standard Algorithm is modelled from "Defining a Standard for Particle Swarm
Optimization" by Daniel Bratton (2007). In this paper, Bratton describes a PSO
algorithm to set a 'standard' in the emerging field of PSO.

<h3>Inertia Weight Algorithm</h3>
In "Major Advances in Particle Swarm Optimization: Theory, Analysis, and Application"
Essam H. Houssein (2021), Houssein et al describes a PSO algorithm in Section 3.3.1 of 
non-linear inertia weight adjustment when updating a particle's velocity. In this repository,
the Inertia Weight Algorithm attempts to replicate the described PSO.

<h4>Customisation</h4>
The Inertia Weight Algorithm is able to be partially customised with a list of Topologies illustrated 
by Houssein et al, and a range of population sizes to set the number of particles present.

<h3>Display</h3>
After selecting an algorithm, the color-gradient map of the chosen equation is displayed while the 
particles are painted on top of the map. Once the 'Start Algorithm' button is pressed, the PSO
algorithm runs while the particle positions are shown on screen. To ensure algorithm is not too
fast for viewing, the generations produced from the algorithm queued.

<h2>Running pso_visualiser</h2>
To run the GUI, first install the required dependencies with:

```
pip install -r requirements.txt
```

Next, the GUI needs a compile executable to run as a subprocess.

Finally run the script with:
```
python src/pso_visualiser.py
```