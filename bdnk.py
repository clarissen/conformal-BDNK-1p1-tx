import numpy as np
from numba import njit

import parameters as params
from data_manager import upgrade_precision_if_needed

# import data_manager as data_m

#--------------------
# generic objects or operations
#--------------------
# =========================================================================
# converts rank 2 tensor indices (a,b) to a row-flattened vector index (c)
# using matrix representation dimension = dim (e.g. metric gmn is r2 with dim = 4)
@njit
def flat_index(dim, row, col): # OK
    return dim * row + col

# flat spacetime metric, row-flattened
gmn = np.array([-1,0,0,0, 
                0,1,0,0, 
                0,0,1,0, 
                0,0,0,1])
g_mn = gmn # raised = lowered

# gamma factor
@njit
def gamma(v): # OK
    return 1/np.sqrt(1-v**2)

@njit
def down(var, g_mn, i, j): # dim = 4 for metric always
    return g_mn[flat_index(4,i,j)] * var

@njit
def up(var, gmn, i, j):
    return gmn[flat_index(4,i,j)] * var

@njit
def force_where_zero(arr):
    for i in range(len(arr)):
        if abs(arr[i]) < 1e-16:
            arr[i] = 0
    return arr

# =========================================================================

#--------------------
# OUTPUT functions
#--------------------
# =========================================================================

def get_flux(q, it): # OK

    # importing parameters that should never change between tests
    ep_coeff, shear_coeff, bulk_scalar, cs = params.ep_coeff, params.shear_coeff, params.bulk_scalar, params.cs

    # parameters subject to change between tests, ! make sure these are being properly updated
    a1, a2, etaovers = params.a1, params.a2, params.etaovers

    # compute args for Tmn, Xmn here
    # ------------------------------
    # from C_m vector
    C_m = get_C_m(q) #
    Cm = get_Cm(C_m) #
    T = get_temp(C_m, Cm) #
    um = get_um(Cm,T) #
    u_m = get_u_m(C_m,T) # 
    Dmn = get_Dmn(um) #

    ep = get_ep(ep_coeff, T) #
    P = get_P(ep) #
    shear_scalar = get_shearscalar(T, shear_coeff) #

    # transport coefficients 
    tau_ep = get_tau_ep(T, a1, etaovers) #
    tau_P = tau_ep / 3.0 #
    tau_Q = get_tau_Q(T, a2, etaovers) #

    Tmn0 = get_Tmn_2x2_ideal(Dmn, um, ep, P) 

    X_mn = get_X_mn_2x2(q, Tmn0, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn, it)
    Tmn = get_Tmn_2x2(q, Tmn0, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn, X_mn)

    # ------------------------------

    f = np.zeros(q.shape)

    f[0] = q[1] 
    f[1] = Tmn[flat_index(2,1,1)]
    f[4] = - X_mn[flat_index(2,0,1)]
    f[5] = - X_mn[flat_index(2,0,0)]

    # if params.problem == "gaussian":
    #     epL, Delta = params.post_dict['epL'], params.post_dict['Delta']
    #     if it == 0:
    #         f[0] = q[1] 
    #         f[1] = P
    #         f[4] = np.zeros(len(T))
    #         f[5] = params.make_gaussian_dTdx(epL,ep,x_,Delta,ep_coeff)

    return f


def get_src(q, it):

    # importing parameters that should never change between tests
    ep_coeff, shear_coeff, bulk_scalar, cs = params.ep_coeff, params.shear_coeff, params.bulk_scalar, params.cs

    # parameters subject to change between tests, ! make sure these are being properly updated
    a1, a2, etaovers = params.a1, params.a2, params.etaovers

    # compute args for Tmn, Xmn here
    # ------------------------------
    # from C_m vector
    C_m = get_C_m(q)
    Cm = get_Cm(C_m)
    T = get_temp(C_m, Cm)
    um = get_um(Cm,T)
    u_m = get_u_m(C_m,T)
    Dmn = get_Dmn(um)

    ep = get_ep(ep_coeff, T)
    P = get_P(ep)
    shear_scalar = get_shearscalar(T, shear_coeff)

    # transport coefficients 
    tau_ep = get_tau_ep(T, a1, etaovers) 
    tau_P = tau_ep / 3.0
    tau_Q = get_tau_Q(T, a2, etaovers)

    Tmn0 = get_Tmn_2x2_ideal(Dmn, um, ep, P)

    X_mn = get_X_mn_2x2(q, Tmn0, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn, it)

    # ------------------------------

    s = np.zeros(q.shape)

    s[2] = X_mn[flat_index(2,0,0)]
    s[3] = X_mn[flat_index(2,0,1)]

    # if params.problem == "gaussian":
    #     if it == 0:
    #         s[2] = np.zeros(len(T))
    #         epL, Delta = params.post_dict['epL'], params.post_dict['Delta'] 
    #         s[3] = - params.make_gaussian_dTdx(epL,ep,x_,Delta,ep_coeff)

    return s

# =========================================================================

#--------------------
# BDNK hydro functions
#--------------------
# =========================================================================
# where else can we use @njit?

@njit
def get_tau_ep(temp, a1, etaovers): # OK
    return a1 * etaovers / temp

@njit
def get_tau_Q(temp, a2, etaovers): # OK
    return 3.0 * a2 * etaovers / temp

@njit
def get_shearscalar(temp, shear_coeff): # OK
    return shear_coeff * temp ** 3

@njit 
def get_ep(ep_coeff, temp): # OK
    return ep_coeff * temp ** 4

@njit
def get_P(ep): # OK
    return ep/3

@njit
def get_temp(C_m, Cm): # OK 
    return np.sqrt( np.abs( - C_m[0] * Cm[0] - C_m[1] * Cm[1] ) )
    # return np.sqrt( - C_m[0] * Cm[0] - C_m[1] * Cm[1] ) 

# four vector C_m = (C_0, C_i), maybe avoid this and just call C_0 = q[2], C_1 = q[3] where necessary?

def get_C_m(q): # OK
    return np.array(  [q[2], q[3], np.zeros(len(q[0])), np.zeros(len(q[0])) ] )

def get_Cm(C_m): # OK
    return np.array( [up(C_m[0],gmn, 0,0), up(C_m[1],gmn, 1,1), C_m[2], C_m[3]  ] )

@njit
def get_um(Cm, T): # OK
    return Cm / T

@njit
def get_u_m(C_m,T): # OK
    return C_m/T

@njit
def get_v(um): # OK
    return um[1]/um[0]

# r2 tensor, flattened for speed
def get_Dmn(um): # OK
    ones = np.ones(len(um[0]))
    zeros = np.zeros(len(um[0]))

    return np.array([ -ones + um[0]*um[0],        um[1]*um[0],   zeros, zeros,
                              um[1]*um[0], ones + um[1]*um[1],   zeros, zeros,
                                    zeros,              zeros,    ones, zeros,
                                    zeros,              zeros,   zeros,  ones ] )

# rank 4 projector, called by element given with indices i1,i2,i3,i4
def get_Dmnab(i1,i2,i3,i4, Dmn): # OK

    return 0.5 * ( Dmn[flat_index(4,i1, i3)] * Dmn[flat_index(4,i2, i4)] \
                    + Dmn[flat_index(4,i1, i4)] * Dmn[flat_index(4,i2, i3)] ) \
                    - ( Dmn[flat_index(4,i3, i4)]* Dmn[flat_index(4,i1, i2)] ) / 3.0

# treating the necessary elements as a 2x2 matrix, flattened for speed
def get_Tmn_2x2_ideal(Dmn, um, ep, P): # OK

    # T^mn = ep u^m u^n + P D^mn

    # [ T^00, T^01, 
    #   T^10, T^11 ]
    return np.array( [ep * um[0] * um[0] + P * Dmn[flat_index(4,0,0)], ep * um[0] * um[1] + P * Dmn[flat_index(4,0,1)]  , \
                    ep * um[1] * um[0] + P * Dmn[flat_index(4,1,0)],  ep * um[1] * um[1] + P * Dmn[flat_index(4,1,1)]] )

# rank 4 tensor Hmnab, called by element given with indices i1,i2,i3,i4
def get_Hmnab(i1,i2,i3,i4, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, Dmn):

    term1 = (tau_ep * (ep + P) * T / cs**2.0 ) * um[i1] * um[i2] * (um[i3] * um[i4] \
                + cs**2.0 * Dmn[flat_index(4,i3, i4)] )

    term2 = (tau_P * (ep + P) * T / cs**2.0 ) * Dmn[flat_index(4,i1, i2)] * (um[i3] * um[i4] \
                + cs**2.0 * Dmn[flat_index(4,i3, i4)])

    term3 = - bulk_scalar * T * Dmn[flat_index(4,i1, i2)] * Dmn[flat_index(4,i3, i4)]

    term4 = - 2 * shear_scalar * T * get_Dmnab(i1,i2,i3,i4,Dmn)

    term5 = tau_Q * T * (ep + P) * (um[i1]*um[i4] * Dmn[flat_index(4,i3, i2)] \
                + um[i2] * um[i4] * Dmn[flat_index(4,i1, i3)]+ um[i1] * um[i3] * Dmn[flat_index(4,i4, i2)] \
                + um[i2] * um[i3] * Dmn[flat_index(4,i1, i4)] )

    return term1 + term2 + term3 + term4 + term5

def get_Fmna(i1,i2,i3, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn): # OK

    Fmna = np.zeros(len(T))

    for j in range(0,4):
        Fmna += (2 * u_m[j] / (T**2) ) * get_Hmnab(i1,i2,i3, j, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, Dmn) 

    return Fmna
        

# treating the necessary elements as a 2x2 matrix, flattened for speed
def get_Tmn_2x2(q, Tmn0, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn, X_mn): # OK

    hot = np.zeros(len(T))

    for j in [0,1]: # first summed index [0,1,2,3], but only need elements [0,1] bc 
        for k in [0,1]: # second summed index
            hot += get_Hmnab(1,1,j,k, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, Dmn) \
                            * X_mn[flat_index(2,j,k)] / (T ** 2) \
                    + get_Fmna(1,1, j, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn) \
                            * um[k]* X_mn[flat_index(2,j,k)]

    T11 = Tmn0[flat_index(2,1,1)] + hot

    # [T00, T01,
    # T01, T11 ])

    return np.array( [q[0], q[1],
                      q[1], T11 ])


#@upgrade_precision_if_needed
def get_Ymnab(i1,i2,i3,i4, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn):

    return get_Hmnab(i1,i2,i3,i4, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, Dmn) \
            + T**2 * get_Fmna(i1,i2,i3, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn) \
                * um[i4]

#@upgrade_precision_if_needed
def get_Y_mnab_inv(i1,i2,i3,i4, Y0000, Y0100, Y0001, Y0101):

    # how to diagonalize a 2x2 matrix with certain rank 4 tensor elements for this problem

    # this will be a problem for v->1
    # factor = 1/ (Y0101 * Y0000 - Y0100 * Y0001 ) # putting a tol < 1e-10 here doesn't seem to do much, gaussian problem
    # print("factor max = " + str(np.max(factor)))
    # print("factor min = " + str(np.min(factor)))

    # elements of Y_inv assumed to be lowered, e.g. Yinv_{mnab}

    #Yinv_0000
    if (i1 == 0) and (i2 == 0) and (i3 == 0) and (i4 == 0): # GOOD
        return 1/ (Y0101 * Y0000 - Y0100 * Y0001 ) * Y0101

    #Yinv_0001
    if (i1 == 0) and (i2 == 0) and (i3 == 0) and (i4 == 1): # GOOD
        return - 1/ (Y0101 * Y0000 - Y0100 * Y0001 ) * Y0001

    #Yinv_0100
    if (i1 == 0) and (i2 == 1) and (i3 == 0) and (i4 == 0): # GOOD
        return - 1/ (Y0101 * Y0000 - Y0100 * Y0001 ) * Y0100

    #Yinv_0101
    if (i1 == 0) and (i2 == 1) and (i3 == 0) and (i4 == 1): # GOOD
        return 1/ (Y0101 * Y0000 - Y0100 * Y0001 ) * Y0000

    else:
        print('unnecessary or invalid indices for Y_mnab_inv')

#@upgrade_precision_if_needed
def get_X_mn_2x2(q, Tmn0, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn, it):

    X_11 = q[4]
    X_10 = q[5]

    T00 = q[0]
    T01 = q[1]

    T00_ideal = Tmn0[flat_index(2,0,0)]
    T01_ideal = Tmn0[flat_index(2,0,1)]

    # used in matrix inversion, computing Y_inv
    Y0000 = get_Ymnab(0,0,0,0, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn)
    Y0100 = get_Ymnab(0,1,0,0, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn)
    Y0001 = get_Ymnab(0,0,0,1, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn)
    Y0101 = get_Ymnab(0,1,0,1, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn)
    
    # data_m.save_single(Y0000, "Y0000")
    # data_m.save_single(Y0100, "Y0100")
    # data_m.save_single(Y0001, "Y0001")
    # data_m.save_single(Y0101, "Y0101")

    # needed elements to compute X_mn
    Y0111 = get_Ymnab(0,1,1,1, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn)
    Y0010 = get_Ymnab(0,0,1,0, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn)
    Y0011 = get_Ymnab(0,0,1,1, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn)
    Y0110 = get_Ymnab(0,1,1,0, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn)

    # X_0n = Yinv_0n00 * (T**2 * (T00 - T00_ideal) - (Y0010*X_10 + Y0011*X_11))
    #         Yinv_0n01 *(T**2 * (T01 - T01_ideal) - (Y0110*X_10 + Y0111*X_11))

    # Z00 = Y0010*X_10 + Y0011*X_11
    # Z01 = Y0110*X_10 + Y0111*X_11

    W00 = T**2 * (T00 - T00_ideal) - (Y0010*X_10 + Y0011*X_11)
    W01 = T**2 * (T01 - T01_ideal) - (Y0110*X_10 + Y0111*X_11)

    X_00 = (get_Y_mnab_inv(0,0,0,0, Y0000, Y0100, Y0001, Y0101) * W00 \
            + get_Y_mnab_inv(0,0,0,1, Y0000, Y0100, Y0001, Y0101) * W01)

    X_01 = get_Y_mnab_inv(0,1,0,0, Y0000, Y0100, Y0001, Y0101)  * W00 \
            + get_Y_mnab_inv(0,1,0,1, Y0000, Y0100, Y0001, Y0101) * W01
    
    if it == 0:
        X_00 = force_where_zero(X_00)
        X_01 = force_where_zero(X_01)

    return np.array([X_00, X_01, \
                     X_10, X_11 ])

# functions only used in post
# ~~~~~~~~~~~~~~~~

def get_Kn_u(etaovers, T, v, um, X_x0, X_xx):

    return um[0]**2 * ( etaovers * X_xx / T **2 - v * etaovers * X_x0 / T**2)

def get_Kn_T(etaovers, T, v, um, X_x0, X_xx):

    return um[0] * (etaovers * X_x0 / T **2 - v * etaovers * X_xx / T **2 )


def get_Anorm(T, um, Dmn, X_00, X_0x, X_x0, X_xx, ep, P, etaovers, a1):

    A = np.zeros(len(T))

    X_mn = np.array([X_00, X_0x, X_x0, X_xx])

    tau_ep = a1 * etaovers / T

    for j in [0,1]:
        for k in [0,1]:
            A += 1/T * 4 * ep * tau_ep * (-um[j]*um[k] + Dmn[flat_index(4,j,k)]/3) * X_mn[flat_index(2,j,k)]

    return A/(ep+P)

def get_Qnorm(T, um, Dmn, X_00, X_0x, X_x0, X_xx, ep, P, etaovers, a2):

    Q0 = np.zeros(len(T))
    Q1 = np.zeros(len(T))

    X_mn = np.array([X_00, X_0x, X_x0, X_xx])

    tau_Q = 3 * a2 * etaovers / T

    for j in [0,1]:
        for k in [0,1]:
            Q0 += 1/T * tau_Q * (ep + P) * (um[j] * Dmn[flat_index(4,0,k)] - um[k] * Dmn[flat_index(4,0,j)] ) \
                                            * X_mn[flat_index(2,j,k)]

            Q1 += 1/T * tau_Q * (ep + P) * (um[j] * Dmn[flat_index(4,1,k)] - um[k] * Dmn[flat_index(4,1,j)] ) \
                                            * X_mn[flat_index(2,j,k)]
            
    return (Q0 * down(Q0,gmn,0,0) + Q1 * down(Q1, gmn, 1, 1) )/ (ep + P)**2 

def get_Qnorm_sqrt(T, um, Dmn, X_00, X_0x, X_x0, X_xx, ep, P, etaovers, a2):

    Q0 = np.zeros(len(T))
    Q1 = np.zeros(len(T))

    X_mn = np.array([X_00, X_0x, X_x0, X_xx])

    tau_Q = 3 * a2 * etaovers / T

    for j in [0,1]:
        for k in [0,1]:
            Q0 += 1/T * tau_Q * (ep + P) * (um[j] * Dmn[flat_index(4,0,k)] - um[k] * Dmn[flat_index(4,0,j)] ) \
                                            * X_mn[flat_index(2,j,k)]

            Q1 += 1/T * tau_Q * (ep + P) * (um[j] * Dmn[flat_index(4,1,k)] - um[k] * Dmn[flat_index(4,1,j)] ) \
                                            * X_mn[flat_index(2,j,k)]
            
    return np.sqrt((Q0 * down(Q0,gmn,0,0) + Q1 * down(Q1, gmn, 1, 1) )/ (ep + P)**2 )

def get_Qnorm_sqrt_post(Qnorm_tx):
    Qnorm_sqrt_tx = []
    for i in range(0,len(Qnorm_tx)):
        Qnorm_sqrt_tx.append(np.sqrt(Qnorm_tx[i]))

    return np.array(Qnorm_sqrt_tx)

# =========================================================================




def get_Anorm_OLD(T, Dmn, X_00, X_0x, X_x0, X_xx, ep, P, ep_coeff, a1, etaovers): # NEEDS CHECKING

    A = np.zeros(len(T))

    X_mn = np.array([X_00, X_0x, X_x0, X_xx])

    for j in [0,1]:
        for k in [0,1]:
            A += 4 * ep_coeff * a1 * etaovers * T**2 * gmn[flat_index(4,j,k)] * X_mn[flat_index(2,j,k)] \
                - (2/3) * Dmn[flat_index(4,j,k)] * X_mn[flat_index(2,j,k)]
    return A/(ep+P)

# I THINK THIS IS WRONG
def get_Qnorm_OLD(T, u_m, um, Dmn, X_00, X_0x, X_x0, X_xx, ep, P, ep_coeff, a2, etaovers): # NEEDS CHECKING

    Q_0 = np.zeros(len(T))
    Q_1 = np.zeros(len(T))

    X_mn = np.array([X_00, X_0x, X_x0, X_xx])

    for j in [0,1]:
        for k in [0,1]:
            Q_0 += 4 * ep_coeff * a2 * etaovers * T**2 * ( um[j] * X_mn[flat_index(2,j,0)] \
                                                          + u_m[0] * um[j] * um[k] * X_mn[flat_index(2,j,k)] \
                                                          + Dmn[flat_index(4,k,0)] * um[j] * X_mn[flat_index(2,k,j)] )

            Q_1 += 4 * ep_coeff * a2 * etaovers * T**2 * ( um[j] * X_mn[flat_index(2,j,1)] \
                                                          + u_m[1] * um[j] * um[k] * X_mn[flat_index(2,j,k)] \
                                                          - Dmn[flat_index(4,k,1)] * um[j] * X_mn[flat_index(2,k,j)] )

    return np.sqrt( np.abs(Q_0 * up(Q_0,gmn,0,0) + Q_1 * up(Q_1, gmn, 1, 1)) )/ (ep + P)