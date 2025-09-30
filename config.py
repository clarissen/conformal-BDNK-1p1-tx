import sys
# problem type      0           1           2           3               4               5             6
problem_list = ["gaussian", "ep-shock", "v-shock", "v-gaussian", "ep-shocktube", "ep-v-gaussian", "ep-v-fd"]
problem = problem_list[1]

# scheme
scheme_list =  ["kt-minmod-tvdrk2", "kt-superbee-tvdrk2", "kt-vanleer-tvdrk2", "kt-mc-tvdrk2"]
scheme = scheme_list[0]

# mesh
# ==========================
x0 = True # include x = 0 in grid?
cCFL = 0.5 # courant factor <=0.5

hx = 0.0125 # step sizes (0.2,0.1,0.05,0.025,0.0125,0.0125/2,0.0125/4,0.0125/8)
ht = 0.5*hx # for now make this a fixed (SMALL) number independent of hx so that time steps line up in convergence test
if ht > cCFL*hx:
    print("CFL Condition not met. Exiting program.")
    sys.exit()

N = 75  # length of grid
iterations = N/(ht) 
tmod = int(1/ht)
# ==========================

# viscosity
# let's test {1,3,6}
Neta = 3 # multiple of 1/4pi for eta/s

# conformal alpha in ep = alpha * T^4
# ep_coeff = 5.26379 # - Teaney
ep_coeff = 10 # - Princeton
# ep_coeff = 15.6268736 # - Vicente

# hydro frames
# ==========================
choose_cplus_a1 = False
frame = 3 # 0,1,2,3

if choose_cplus_a1 == True: 

    if frame == 0: # B - Princeton
        cplus = 0.999
        a1 = 25/4 
        # a2 = 25/7

    if frame == 1: # A - Princeton
        cplus = 0.86
        a1 = 25/4 
        # a2 = 75/7
    
    if frame == 2: 
        cplus = 0.77
        a1 = 25/2
        # a2 = 25

    if frame == 3:
        cplus = 0.70
        a1 = 25
        # a2 = 75

    if frame == 4: # A - Princeton
        cplus = 0.85
        a1 = 25/2 
        # a2 = 25/3

else:

    if frame == 0: # B - Princeton
        # cplus = 0.99
        a1 = 25/4 
        a2 = 25/7

    if frame == 1: 
        # cplus = 0.86
        a1 = 25/4 
        a2 = 75/7
    
    if frame == 2: 
        # cplus = 0.77
        a1 = 25/2
        a2 = 25

    if frame == 3:
        # cplus = 0.70
        a1 = 25
        a2 = 75

    if frame == 4: # A - Princeton
        # cplus = 0.85
        a1 = 25/2 
        a2 = 25/3


 
# ==========================

# Kurganov-Tadmor
theta_kt = 1.0

# metadata 
# ==========================
# alarms on or off?
alarms = False
# automatically pass INITIAL checks and prints and generate file names for automatic runs (sucomputer cluster)
sucomp = True
# animation bounds?
anim_bounds = True
# zoom in on hydro frame animations?
anim_zoom = False
# ==========================

# Initial conditions
# ==========================
if problem == problem_list[0]:
    epL = 0.40 # epLarger
    epR = 0.1 # * 64 #epsmalleR
    vL = 0.0
    vR = vL
    Delta = 5.0 # width, WILL BE SQUARED

# smooth shocktube epsilon
if problem == problem_list[1]:
    epL = 1.3
    epR = 0.3
    vL = 0.0
    vR = vL
    Delta = 1.0 # 0.2 

# smooth shocktube velocity
if problem == problem_list[2]:
    # constant background eps
    epL = 1.0
    epR = epL
    # L/R distribution in v
    vL = 0.999
    vR = 1/(3*vL)
    # distribution gradient width
    Delta = 2.0

# negative gaussian in velocity
if problem == problem_list[3]:
    epL = 1.0
    epR = epL
    vL = 0.9 # max
    vR = 0.01 # height below vL
    Delta = 5.0

# shocktube in epsilon
if problem == problem_list[4]:
    epL = 0.4
    epR = 0.1
    vL = 0.0
    vR = vL
    Delta = 0.0

# gaussian in epsilon and velocity
if problem == problem_list[5]:
    epL = 0.4 # epLarger
    epR = 0.1 #epsmalleR
    vL = 0.3
    vR = 0.0
    Delta = 5.0 # width, WILL BE SQUARED

# smooth stairs (shock) in ep and v with jump conditions
if problem == problem_list[6]:
    epL = 1.0
    vL = 0.9
    vR = 1/(3*vL)
    epR = epL * (9*vL**2 - 1)/(3*(1-vL**2))
    Delta = 1.0 

# ==========================
