import numpy as np

import config
import bdnk as hydro
from numba import njit, jit

#--------------------
# OUTPUT functions
#--------------------
# =========================================================================

#KT-numerical flux
def Hflux(qM, qP, cplus, it):

    # choosing the maximum local propagation speed
    # a = cplus
    a = 1.0

    return 0.5 * (hydro.get_flux(qM, it) + hydro.get_flux(qP, it)) - 0.5 * a * (qP - qM)

# =========================================================================

#--------------------
# KT ALGORITHM functions
#--------------------
# =========================================================================

# q plus (positive direction of numerical cell)
def qp(qL, qM, qR, theta, hx):
    return qM - 0.5 * hx * qx(qL, qM, qR, theta, hx)

# q minus (negative direction of numerical cell)
def qm(qL, qM, qR, theta, hx):
    return qM + 0.5 * hx * qx(qL, qM, qR, theta, hx)

# NUMERICAL GRADIENTS
# ===================


# FLUX LIMITERS
# ==============
@njit
def phi_minmod3(r, theta):
    # classical 2‑point minmod limiter
    return minmod3(theta*r, 0.5*(1.0+r), theta)

@njit
def phi_superbee(r):
    b = 1.0 # [1,2]
    # Superbee: max(0, min(2r,1), min(r,2))
    return np.maximum(0.0, np.minimum(b*r, 1.0), np.minimum(r, b))

# @njit
def phi_van_leer(r):
    # van Leer: (r+|r|)/(1+|r|)
    return (r + abs(r)) / (1.0 + abs(r))

@njit
def phi_mc(r):
    # Monotonized Central: min(2r, (1+r)/2, 2)
    return np.minimum(2.0*r, np.minimum((1.0+r)/2.0, 2.0))
# ==============

@njit
def minmod(a,b):
    return (np.sign(a)+np.sign(b)) * np.minimum(np.abs(a), np.abs(b)) * 0.5 

@njit
def minmod3(a,b,c):
    return minmod(a, minmod(b,c))

# numerical dq/dx. q here is the state but this can be applied to any multidimensional array 
# of type: dim x grid length+/-int

# @njit
# def qx(qL, qM, qR, theta, hx):
#     return minmod3(theta * (qM - qL) / hx, 0.5   * (qR - qL) / hx, theta * (qR - qM) / hx)

if config.scheme == "kt-minmod-tvdrk2":
    limiter_type = 0
if config.scheme == "kt-superbee-tvdrk2":
    limiter_type = 1
if config.scheme == "kt-vanleer-tvdrk2":
    limiter_type = 2
if config.scheme == "kt-mc-tvdrk2":
    limiter_type = 3

tol = 1e-12

def qx(qL, qM, qR, theta, hx):
    dqL = (qM - qL) / hx
    dqR = (qR - qM) / hx
    r   = dqL / (dqR + tol)

    if limiter_type == 0:
        phi = phi_minmod3(r,theta)
    elif limiter_type == 1:
        phi = phi_superbee(r)
    elif limiter_type == 2:
        phi = phi_van_leer(r)
    elif limiter_type == 3:
        phi = phi_mc(r)
    else:
        print("invalid limiter_type")

    return phi * dqR

# ^
# these two functions should produce the same thing
# v
def qx_old(qL, qM, qR, theta, hx):

    return np.vectorize(minmod3)(
        theta * (qM - qL) / hx, # a
        0.5   * (qR - qL) / hx, # b
        theta * (qR - qM) / hx  # c
    )

# generic spatial derivative using KT
def general_ddx(vec, theta, hx):

    # maybe this isn't good?
    # extending outer values so dvec/dx can take "non shifted" KT gradients
    vec = np.append(vec, vec[-1])
    vec = np.insert(vec,0,vec[0])

    # setting up indexing 
    N = len(vec) 

    k = np.arange(1, N - 1)
    km1 = np.arange(0, N - 2)
    kp1 = np.arange(2, N)

    # creating left, right, middle slices of vec
    vecM = vec[k]
    vecL = vec[km1]
    vecR = vec[kp1]

    return qx(vecL, vecM, vecR, theta, hx)


# ==============