# conformal-BDNK-1p1-tx
Python simulation for conformal fluids in BDNK hydrodynamics for 1+1 (t,x) Minkowski spacetime.

Solves the following conservation law numerically

dq/dt + df/dx = s

and visualizes, stores, organizes the data of each simulation. 

# required libraries
numpy, matplotlib, os, numba


# How to run this code?
set up a directory in which you would like to run this simulation, then from Terminal/Command Prompt write

python<version> main.py

# What does this code do?
- Running the main.py will run the simulation that outputs an array containg time-space data (t,x) of the
conserved quantities in the state q, the flux f, and the source s.
- The post.py module will also be called and output all the variables contained within q, f, s, like element of the energy-momentum tensor, the temperature-flow product vector, and its gradients. Other physical variables of interest may also be calculated like energy desnity, velocity, lorentz factor.
- These variables will be subsequently animated using animator.py.

# Workflow of the simulation.
The main workhorse of this simulation is organized in the following way:

1. config.py plugs into parameters.py. Config is the only place
2. parameters.py plugs into several modules, like initiailize.py, evolver.py. This carries initial and constant variables throughout the code. There is a dictionary generated in parameters.py that is used to organize the simulation parameters. 
3. initialize.py creates the initial state q from parameters.py
4. bdnk.py the main output from this file are the flux and source arrays. Within bdnk.py are also all the bdnk hydro functions that calculate and deliver the variables necessary to create the flux and source outputs.
5. kurganovtadmor.py - the main output from this concerning the simulation is the "Hflux" function, which gives the numerical flux according to the KT algorithm. This also contains the building blocks of the numerical flux, which includes the numerical spatial gradient and grid shifting of the state q.
6. evolver.py - This is the bulk of the simulation. This contains the class "Evolve" which contains all relevant objects used to complete the Kurganov-Tadmor-Total-Variation-Diminshing-Runge-Kutta-2 scheme. It keeps track of simulation time and energy-momentum changes from the initial data at each time step. It also runs post.py (post processing) calculations on the completed simulation as well as the animator.py to visualize the time and space evolution of relevant variables.
7. data_manager.py - this loads, saves, and checks simulation data/specification. All data is saved as .npy files. Dictionaries are saved as .json. Energy-momentum conservation and simulation configuration are saved as .txt files. Each simulation must be given a name and this will be saved in the sims directory. This will eventually contain all simulation data, the variables calcualted from post.py and the animations from animator.py.
8. main.py - run this to simulate the fluid.

# What you can change:
The only place where the user *must* interface with the simulation is in config.py, here you can change the simulation metadata, the physical parameters, and the problem types.
