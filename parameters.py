import numpy as np

import config
import kurganovtadmor as kt


#--------------------
# FUNCTIONS 
#--------------------
# =========================================================================

#--------------------
# Functions for initial data
#--------------------
def fermidirac(x, x0, fL, fR, Delta):
    return fR + (fL-fR)/(1 + np.exp( (x-x0)/Delta ) )

def gaussian(x,x0,fmax,fmin,width):
    return fmax * np.exp(-(x-x0)**2/width**2) + fmin

def gevrey(x,fmin,fmax):    
    return np.where(np.abs(x) < 10, fmax * np.exp(100 / (x**2 - 100)) + fmin, fmin)
    # division by zero warning

def shock(x,x0,fL,fR): 
    u = np.zeros(x.shape)
    
    for j in range(len(x)):

        if x[j] <= x0:
            u[j] = fL
            
        if x[j] > x0:
            u[j] = fR
            
    return u

def negative_gaussian(x,x0, f0, fscale, width):
    return f0 - fscale * np.exp(-(x-x0)**2/width**2)

# SPECIAL CASE FUNCTIONS, called outside of evolver stage only 
# ~~~~~~~~~~~~~~~~
def make_temp(ep, ep_coeff):
    return (ep/ep_coeff)**(1/4)

def make_gaussian_dTdx(A_ep, ep, x,Delta, ep_coeff ):
    return - ( A_ep *  x * np.exp(-x**2/Delta**2) ) / (2 * Delta**2 * ep_coeff * (ep/ep_coeff)**(3/4))

# ~~~~~~~~~~~~~~~~

#--------------------
# hydro frame functions
#--------------------
def hydro_frame(a1:float, a2:float):

    # a1 = 5.0 # must be greater than 4
    # constant = 4.0 # must be greater than 3
    # a2 = constant * a1 / (a1-1)

    # maximum characteristic velocity for BDNK
    cplus = np.sqrt(( a1 * (2+a2) + 2* np.sqrt(a1 * (a1 + a1*a2 + a2**2)) ) / (3*a1*a2))

    if (a1 >= 4) and (a2 >= 3 * a1 / (a1-1)) and (0 <= cplus**2 <= 1):
        return a1, a2, cplus
    else: 
        print("a1 >=4, a1 = ", a1)
        print("a2 >= " + str(3 * a1 / (a1-1)) + ", a2 = " + str( a2) )
        print("c_+ <= 1, c_+ = ", cplus )
        print("invalid a1 and/or a2")

def hydro_frame_constant_cplus(cplus:float, a1:float):

    # a1_min = 4/(3/cplus**2-1)**2

    a2 = 12 * a1*cplus**2 / (-4 + a1 - 6 * a1 * cplus**2 + 9*a1*cplus**4)

    if (a1 >= 4) and (a2 >= 3 * a1 / (a1-1)) and (0 <= cplus**2 <= 1):
        return a1, a2, cplus
    else:
        print("a1 >=4, a1 = ", a1)
        print("a2 >= " + str(3 * a1 / (a1-1)) + ", a2 = " + str( a2) )
        print("c_+ <= 1, c_+ = ", cplus )
        print("invalid c_+ and/or a1")

#--------------------

# GLOBAL VARIABLES IN PARAMS, CALLED THROUGHOUT CODE
# imports from config file
# ========================
#problem
#--------------------
problem = config.problem
problem_list = config.problem_list
#--------------------

# grid 
#--------------------
cCFL = config.cCFL
hx = config.hx
N = config.N
iterations = config.iterations
#--------------------

# viscosity
#--------------------
Neta = config.Neta
#--------------------

# conformal EoS
#--------------------
ep_coeff = config.ep_coeff
#--------------------

# Kurganov-Tadmor - minmod_limiter
#--------------------
theta_kt = config.theta_kt
#--------------------

# metadata 
# ==========================
scheme = config.scheme
# alarms on or off?
alarms = config.alarms
# automatically pass checks and prints and generate file names for cluster?
sucomp = config.sucomp
# animation bounds?
anim_bounds = config.anim_bounds
# zoom in on hydro frame animations?
anim_zoom = config.anim_zoom
# ==========================


# initial conditions
#--------------------
epL = config.epL
epR = config.epR
vL = config.vL
vR = config.vR
Delta = config.Delta
#--------------------

# Making the grid
# Cartesian Mesh creation (t,x)
#--------------------
ht = config.ht
    
# initial time
t0 = 0.0
# grid specs in x
xpos = np.arange(hx,N+hx,hx) # xpos = [+hx,...,N]
xneg = -xpos[::-1] # reverse "::-1" and negative "-"

# the whole grid
if config.x0 == True:
    x = np.concatenate((xneg, [0], xpos)) # number of cells always odd and centered at 0
else:
    x = np.concatenate((xneg, xpos)) # even number of cells, SKIPS 0

xL = x[0]
xR = x[-1]
xM = x[ int((len(x)-1)/2) ]
Ncells = len(x)

# time grid
t_array = np.arange(ht,config.iterations+ht,ht)

# indexing, need two ghost points for KT numerical gradients
Km2 = np.arange(0, Ncells - 4)
Km1 = np.arange(1, Ncells - 3)
K   = np.arange(2, Ncells - 2)
Kp1 = np.arange(3, Ncells - 1)
Kp2 = np.arange(4, Ncells)

#python list for access
indexing = [Km2, Km1, K, Kp1, Kp2]

#--------------------

# Useful information
#--------------------
units = "GeV"
invunits = "1/GeV"
hbarc = 0.1973
cs = 1.0 / np.sqrt(3.0) # conformal speed of sound
spacetime = "cartesian"
#--------------------

# BDNK parameter variables
#--------------------
bulk_scalar = 0.0 # conformal
etaovers = Neta * 1/(4 * np.pi)# eta / s, a measure of viscosity
# 15.6 for a gas of massless quarks and gluons
shear_coeff =  (4/3) * ep_coeff *  etaovers 
# shear_coeff =  4/3 * (ep_coeff)**(1/2)* etaovers # WRONG

# HYDRO FRAMES
a1, a2, cplus = hydro_frame(config.a1,config.a2)

frame = config.frame
#--------------------

# Global sim information variables
#--------------------

# Dictionary which contains global variables that will never change and may used throughout the sim
#--------------------
param_dict = {"cs": cs, "a": cplus, "theta_kt": theta_kt, "hx": hx, "ht": ht, "cCFL": cCFL, \
            "a1": a1, "a2": a2, "frame": frame, "etaovers": etaovers, "ep_coeff": ep_coeff, "shear_coeff": shear_coeff}


#Dictionary which contains global variables that will never change, is unique to the problem, 
# and will only be used in POST-PROCESSING or INITIALIZATION
#--------------------
post_dict = {"hbarc": hbarc, "cs": cs, "epL": epL, "epR": epR, "vL": vL, "vR": vR, "Delta": Delta, "Ncells": Ncells, \
            "xL": xL, "xR": xR, "xM": xM, "t0": t0, "iterations": iterations, "spacetime": spacetime, "problem": problem}

# unique string to name sim folder
#--------------------
unique = "bdnk_" + str(problem) \
+ "_(epL,epR)=(" + str(epL) + ","+ str(epR) + ")"\
+ "_(vL,vR)=(" + str(vL) + "," + str(vR) + ")" \
+ "_a1=" + str(round(a1,2)) + "_a2=" + str(round(a2,2)) \
+ "_etaovs=" + str(round(etaovers,3)) + "_Delta=" + str(Delta) \
+ "_hx=" + str(hx) + "_ht=" + str(round(ht,3)) \
+ '_Ncells=' + str(Ncells) + "_grid=[" + str(round(xL,3)) + "," + str(round(xR,3)) + "]" \
+ "_c+=" + str(round(cplus, 2)) \
+ "_"+ scheme

# unique text file inside the sim folder
#--------------------
sim_config = [f"Simulation Configuration:", \
            f"This code simulates the {problem} problem in {spacetime} spacetime using a {scheme} scheme.", \
            f"Spatial resolution of hx = {hx} {invunits} on a grid = [{xL}, {xR}] {invunits}.", \
            f"Temporal resolution of ht = {ht} {invunits}.", \
            f" ------------------------- ", \
            f"All parameters: {param_dict}", \
            f"{post_dict}" ]
            # f"Scales: hx = {hx}, l_eta = [{l_eta_min0}, {l_eta_max0}], Knudsen = [{KN_min0, KN_max0}]" ]

# print here as a check
#--------------------
specs = "spacetime = " + str(spacetime) + ", problem = " + str(problem) + ", alarms = " + str(alarms) + "\n" \
    + "(epL, epR) = (" + str(epL) + ", " + str(epR) + ")" + \
    ", (vL, vR) = (" + str(vL) + ", " + str(vR) + ")" \
    + ", shock width Delta = " + str(Delta) + "\n" \
    + "theta_kt = " + str(theta_kt) + ", hx = " + str(hx) + ", ht = " + str(round(ht,3)) \
    + ', Ntime = ' + str(config.iterations) + ", tgrid = [" + str(t_array[0]) + "," + str(t_array[-1]) + "]" +', Nspace = ' + str(Ncells) + ", xgrid = [" + str(xL) + "," + str(xR) + "]" + "\n" \
    + f"hydro frames: a1 = {a1} and a2 = {a2}, c_+ = {cplus}, viscosity: eta/s = " + str(round(etaovers,3))
#--------------------

print(specs)

# version ran
version = problem +"_"+ scheme

