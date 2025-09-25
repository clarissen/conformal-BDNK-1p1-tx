import numpy as np

# homemade modules
import parameters as params
import bdnk as hydro
from bdnk import gmn, g_mn
import kurganovtadmor as kt
import data_manager as data_m

# =========================================================================

#--------------------
# initialize fluid state functions
#--------------------
# =========================================================================

def init_ep_shock(post_dict, param_dict, x):
    epL = post_dict["epL"]
    epR = post_dict["epR"]
    vL = post_dict["vL"]
    Delta = post_dict["Delta"]
    cs = post_dict["cs"]
    x0 = post_dict["xM"]

    ep_coeff = param_dict["ep_coeff"]
    shear_coeff = param_dict["shear_coeff"]
    theta_kt = param_dict["theta_kt"]
    hx = param_dict["hx"]

    # ep, v, u^mu, T
    ep = params.fermidirac(x,x0,epL,epR,Delta)
    v = vL * np.ones(len(x))

    ut = hydro.gamma(v) * np.ones(len(x))
    ux = v * hydro.gamma(v) * np.ones(len(x))

    u_t = hydro.down(ut, g_mn, 0, 0)
    u_x = hydro.down(ux, g_mn, 1, 1)

    T = params.make_temp(ep, ep_coeff)

    # ep, u^mu(v) gradients
    dxep = kt.general_ddx(ep, theta_kt, hx)

    dxu_x = kt.general_ddx(u_x, theta_kt, hx)
    dxu_t = kt.general_ddx(u_t, theta_kt, hx)

    dtu_x = 1/ut * ( (u_t**2 * dxu_x - ux**2 * dxu_x - ux * dxep/(4*ep))*u_x/(3+ 2*u_x**2) - ux*dxu_x-dxep/(4*ep) )
    dtu_t = - ux * dtu_x / ut

    # creating the state, q
    # ---------------------
    C_t = T * u_t
    C_x = T * u_x

    # numerical computations of X_mn = partial_m C_n
    X_xx = kt.general_ddx(C_x, theta_kt, hx)
    X_xt = kt.general_ddx(C_t, theta_kt, hx)

    # Euler-like ICs for Tmn require shear elements (o00 = o_00, o0x = - o_0x)
    # ott = (4/3 - 2/3*u_t**2)*dtu_t + u_t*u_x *dxu_t-(1+u_t**2)*dxu_x/3
    # otx = -0.5*(-u_t*u_x*dtu_t/3 + (u_t*u_x/3 + u_x**2)*dxu_x +(1-u_t**2)*dtu_x + dxu_t )
    # print('ott min/max = '+ str(min(ott)) +"/" + str(max(ott)) )

    ott = - 2/3*(ut**2 - 1) * dtu_t + u_t * u_x * dxu_t - (ut**2 - 1) * dxu_x/3
    print('ott min/max = '+ str(min(ott)) +"/" + str(max(ott)) )
    otx = u_t*u_x*dtu_t/6 + dtu_x * (ut**2 - 1)/2 - dxu_x * u_t * u_x /6 - ut**2 * dxu_t/2

    # elements of Tmn 
    Ttt = ep*(ut**2) + ep*(ut**2 + -1)/3 - 2 * hydro.get_shearscalar(T,shear_coeff) * ott
    Ttx = 4/3*ep*ut*ux - 2 * hydro.get_shearscalar(T,shear_coeff) * otx
    # ---------------------
    return np.array([Ttt, Ttx, C_t, C_x, X_xx, X_xt])


def init_v_shock(post_dict, param_dict, x):
    epL = post_dict["epL"]
    epR = post_dict["epR"]
    vL = post_dict["vL"]
    vR = post_dict["vR"]
    Delta = post_dict["Delta"]
    cs = post_dict["cs"]
    x0 = post_dict["xM"]

    ep_coeff = param_dict["ep_coeff"]
    shear_coeff = param_dict["shear_coeff"]
    theta_kt = param_dict["theta_kt"]
    hx = param_dict["hx"]

    # ep, v, u^mu, T
    ep = epL * np.ones(len(x))
    v = params.fermidirac(x,x0,vL,vR,Delta)

    ut = hydro.gamma(v) * np.ones(len(x))
    ux = v * hydro.gamma(v) * np.ones(len(x))

    u_t = hydro.down(ut, g_mn, 0, 0)
    u_x = hydro.down(ux, g_mn, 1, 1)

    T = params.make_temp(ep, ep_coeff)

    # ep, u^mu(v) gradients
    # dxep = kt.general_ddx(ep, theta_kt, hx)
    dxep = np.zeros(len(x))

    dxu_x = kt.general_ddx(u_x, theta_kt, hx) # dxux = dxu_x
    dxu_t = kt.general_ddx(u_t, theta_kt, hx)

    dtu_x = 1/ut * ( (u_t**2 * dxu_x - ux**2 * dxu_x - ux * dxep/(4*ep))*u_x/(3+ 2*u_x**2) - ux*dxu_x-dxep/(4*ep) )
    dtu_t = - ux * dtu_x / ut

    # creating the state, q
    # ---------------------
    C_t = T * u_t
    C_x = T * u_x

    # numerical computations of X_mn = partial_m C_n
    X_xx = kt.general_ddx(C_x, theta_kt, hx)
    X_xt = kt.general_ddx(C_t, theta_kt, hx)

    # Euler-like ICs for Tmn require shear elements
    ott = - 2/3*(ut**2 - 1) * dtu_t + u_t * u_x * dxu_t - (ut**2 - 1) * dxu_x/3
    print('ott min/max = '+ str(min(ott)) +"/" + str(max(ott)) )
    otx = u_t*u_x*dtu_t/6 + dtu_x * (ut**2 - 1)/2 - dxu_x * u_t * u_x /6 - ut**2 * dxu_t/2

    # elements of Tmn 
    Ttt = ep*(4*(ut**2)/3 -1/3) - 2 * hydro.get_shearscalar(T,shear_coeff) * ott
    Ttx = 4/3*ep*ut*ux - 2 * hydro.get_shearscalar(T,shear_coeff) * otx
    # ---------------------

    return np.array([Ttt, Ttx, C_t, C_x, X_xx, X_xt])

def init_ep_v_fd(post_dict, param_dict, x):
    epL = post_dict["epL"]
    epR = post_dict["epR"]
    vL = post_dict["vL"]
    vR = post_dict["vR"]
    Delta = post_dict["Delta"]
    cs = post_dict["cs"]
    x0 = post_dict["xM"]

    ep_coeff = param_dict["ep_coeff"]
    shear_coeff = param_dict["shear_coeff"]
    theta_kt = param_dict["theta_kt"]
    hx = param_dict["hx"]

    # ep, v, u^mu, T
    ep = params.fermidirac(x,x0,epL,epR,Delta)
    v = params.fermidirac(x,x0,vL,vR,Delta)

    ut = hydro.gamma(v) * np.ones(len(x))
    ux = v * hydro.gamma(v) * np.ones(len(x))

    u_t = hydro.down(ut, g_mn, 0, 0)
    u_x = hydro.down(ux, g_mn, 1, 1)

    T = params.make_temp(ep, ep_coeff)

    # ep, u^mu(v) gradients
    dxep = kt.general_ddx(ep, theta_kt, hx)

    dxu_x = kt.general_ddx(u_x, theta_kt, hx) # dxux = dxu_x
    dxu_t = kt.general_ddx(u_t, theta_kt, hx)

    dtu_x = 1/ut * ( (u_t**2 * dxu_x - ux**2 * dxu_x - ux * dxep/(4*ep))*u_x/(3+ 2*u_x**2) - ux*dxu_x-dxep/(4*ep) )
    dtu_t = - ux * dtu_x / ut

    # creating the state, q
    # ---------------------
    C_t = T * u_t
    C_x = T * u_x

    # numerical computations of X_mn = partial_m C_n
    X_xx = kt.general_ddx(C_x, theta_kt, hx)
    X_xt = kt.general_ddx(C_t, theta_kt, hx)

    # Euler-like ICs for Tmn require shear elements
    ott = - 2/3*(ut**2 - 1) * dtu_t + u_t * u_x * dxu_t - (ut**2 - 1) * dxu_x/3
    print('ott min/max = '+ str(min(ott)) +"/" + str(max(ott)) )
    otx = u_t*u_x*dtu_t/6 + dtu_x * (ut**2 - 1)/2 - dxu_x * u_t * u_x /6 - ut**2 * dxu_t/2

    # elements of Tmn 
    Ttt = ep*(4*(ut**2)/3 -1/3) - 2 * hydro.get_shearscalar(T,shear_coeff) * ott
    Ttx = 4/3*ep*ut*ux - 2 * hydro.get_shearscalar(T,shear_coeff) * otx
    # ---------------------

    return np.array([Ttt, Ttx, C_t, C_x, X_xx, X_xt])


def init_gaussian(post_dict, param_dict, x):
    epR = post_dict["epR"]
    epL = post_dict["epL"]
    vL = post_dict["vL"]
    ep_coeff = param_dict["ep_coeff"]
    shear_coeff = param_dict["shear_coeff"]
    theta_kt = param_dict["theta_kt"]
    hx = param_dict["hx"]
    a2 = param_dict["a2"]
    etaovers = param_dict["etaovers"]

    x0 = post_dict["xM"]

    # ep, v, u^mu, T
    Delta = post_dict["Delta"] # 1/GeV
    ep = params.gaussian(x,0,epL, epR, Delta)
    T = params.make_temp(ep, ep_coeff)

    ut = np.ones(len(x))
    ux = np.zeros(len(x))

    u_t = hydro.down(ut, g_mn, 0, 0)
    u_x = hydro.down(ux, g_mn, 1, 1)

    # gradients of T, u^mu(v)
    dTdx = params.make_gaussian_dTdx(epL,ep,x,Delta,ep_coeff)
    dTdx_num = kt.general_ddx(T, theta_kt, hx)

    dxu_x = kt.general_ddx(u_x, theta_kt, hx)
    dxu_t = kt.general_ddx(u_t, theta_kt, hx)

    # creating the state, q
    # ---------------------
    C_t = T * u_t
    C_x = T * u_x

    # numerical computations of X_mn = partial_m C_n
    # X_xx = kt.general_ddx(C_x, theta_kt, hx)
    # X_xt = kt.general_ddx(C_t, theta_kt, hx)
    X_xx = np.zeros(len(T))
    X_xt = - dTdx 

    # elements of Tmn 
    Ttt = ep
    Ttx = np.zeros(len(T))
    # ---------------------
    q = np.array([Ttt, Ttx, C_t, C_x, X_xx, X_xt])

    # BCs
    q[:,-1] = q[:,-2] = q[:,-3] 
    q[:,0] = q[:,1] = q[:,2]

    data_m.save_single(q, "q_init")
    data_m.save_single(x, "x")
    data_m.save_single(T, "T_init")
    data_m.save_single([a2, 0], "a2_init")
    data_m.save_single([etaovers, 0], "etaovers_init")
    data_m.save_single(ep, "ep_init")
    data_m.save_single(dTdx, "dTdx_init")

    save_key_variables(q)

    return q

def init_v_gaussian(post_dict, param_dict, x):
    epL = post_dict["epL"]
    epR = post_dict["epR"]
    vL = post_dict["vL"]
    vR = post_dict["vR"]
    Delta = post_dict["Delta"]
    cs = post_dict["cs"]
    x0 = post_dict["xM"]

    ep_coeff = param_dict["ep_coeff"]
    shear_coeff = param_dict["shear_coeff"]
    theta_kt = param_dict["theta_kt"]
    hx = param_dict["hx"]

    # ep, v, u^mu, T
    ep = epL * np.ones(len(x))
    v = params.negative_gaussian(x,x0,vL,vR,Delta)

    ut = hydro.gamma(v)
    ux = v * hydro.gamma(v)

    u_t = hydro.down(ut, g_mn, 0, 0)
    u_x = hydro.down(ux, g_mn, 1, 1)

    T = params.make_temp(ep, ep_coeff)

    # ep, u^mu(v) gradients
    # dxep = kt.general_ddx(ep, theta_kt, hx)
    dxep = np.zeros(len(x))

    dxu_x = kt.general_ddx(u_x, theta_kt, hx) # dxux = dxu_x
    dxu_t = kt.general_ddx(u_t, theta_kt, hx)

    dtu_x = 1/ut * ( (u_t**2 * dxu_x - ux**2 * dxu_x - ux * dxep/(4*ep))*u_x/(3+ 2*u_x**2) - ux*dxu_x-dxep/(4*ep) )
    dtu_t = - ux * dtu_x / ut

    # creating the state, q
    # ---------------------
    C_t = T * u_t
    C_x = T * u_x

    # numerical computations of X_mn = partial_m C_n
    X_xx = kt.general_ddx(C_x, theta_kt, hx)
    X_xt = kt.general_ddx(C_t, theta_kt, hx)

    # Euler-like ICs for Tmn require shear elements
    ott = - 2/3*(ut**2 - 1) * dtu_t + u_t * u_x * dxu_t - (ut**2 - 1) * dxu_x/3
    print('ott min/max = '+ str(min(ott)) +"/" + str(max(ott)) )
    otx = u_t*u_x*dtu_t/6 + dtu_x * (ut**2 - 1)/2 - dxu_x * u_t * u_x /6 - ut**2 * dxu_t/2

    # elements of Tmn 
    Ttt = ep*(4*(ut**2)/3 -1/3) - 2 * hydro.get_shearscalar(T,shear_coeff) * ott
    Ttx = 4/3*ep*ut*ux - 2 * hydro.get_shearscalar(T,shear_coeff) * otx
    # ---------------------

    return np.array([Ttt, Ttx, C_t, C_x, X_xx, X_xt])  

def init_ep_shocktube(post_dict, param_dict, x):
    epL = post_dict["epL"]
    epR = post_dict["epR"]
    vL = post_dict["vL"]
    Delta = post_dict["Delta"]
    cs = post_dict["cs"]
    x0 = post_dict["xM"]

    ep_coeff = param_dict["ep_coeff"]
    shear_coeff = param_dict["shear_coeff"]
    theta_kt = param_dict["theta_kt"]
    hx = param_dict["hx"]

    # ep, v, u^mu, T
    ep = params.shock(x,x0,epL,epR)
    v = vL * np.ones(len(x))

    ut = np.ones(len(x))
    ux = np.zeros(len(x))

    u_t = hydro.down(ut, g_mn, 0, 0)
    u_x = hydro.down(ux, g_mn, 1, 1)

    T = params.make_temp(ep, ep_coeff)

    # ep, u^mu(v) gradients
    dxep = kt.general_ddx(ep, theta_kt, hx)

    dxu_x = kt.general_ddx(u_x, theta_kt, hx)
    dxu_t = kt.general_ddx(u_t, theta_kt, hx)

    dtu_x = 1/ut * ( (u_t**2 * dxu_x - ux**2 * dxu_x - ux * dxep/(4*ep))*u_x/(3+ 2*u_x**2) - ux*dxu_x-dxep/(4*ep) )
    dtu_t = - ux * dtu_x / ut

    # creating the state, q
    # ---------------------
    C_t = T * u_t
    C_x = T * u_x

    # numerical computations of X_mn = partial_m C_n
    X_xx = kt.general_ddx(C_x, theta_kt, hx)
    X_xt = kt.general_ddx(C_t, theta_kt, hx)

    # Euler-like ICs for Tmn require shear elements (o00 = o_00, o0x = - o_0x)
    # ott = (4/3 - 2/3*u_t**2)*dtu_t + u_t*u_x *dxu_t-(1+u_t**2)*dxu_x/3
    # otx = -0.5*(-u_t*u_x*dtu_t/3 + (u_t*u_x/3 + u_x**2)*dxu_x +(1-u_t**2)*dtu_x + dxu_t )
    # print('ott min/max = '+ str(min(ott)) +"/" + str(max(ott)) )

    ott = - 2/3*(ut**2 - 1) * dtu_t + u_t * u_x * dxu_t - (ut**2 - 1) * dxu_x/3
    print('ott min/max = '+ str(min(ott)) +"/" + str(max(ott)) )
    otx = u_t*u_x*dtu_t/6 + dtu_x * (ut**2 - 1)/2 - dxu_x * u_t * u_x /6 - ut**2 * dxu_t/2

    # elements of Tmn 
    Ttt = ep*(ut**2) + ep*(ut**2 + -1)/3 - 2 * hydro.get_shearscalar(T,shear_coeff) * ott
    Ttx = 4/3*ep*ut*ux - 2 * hydro.get_shearscalar(T,shear_coeff) * otx
    # ---------------------
    return np.array([Ttt, Ttx, C_t, C_x, X_xx, X_xt])

def init_ep_v_gaussian(post_dict, param_dict, x):
    epL = post_dict["epL"]
    epR = post_dict["epR"]
    vL = post_dict["vL"]
    vR = post_dict["vR"]
    Delta = post_dict["Delta"]
    cs = post_dict["cs"]
    x0 = post_dict["xM"]

    ep_coeff = param_dict["ep_coeff"]
    shear_coeff = param_dict["shear_coeff"]
    theta_kt = param_dict["theta_kt"]
    hx = param_dict["hx"]

    # ep, v, u^mu, T
    ep = params.gaussian(x,0,epL,epR,Delta)
    v = params.gaussian(x,0,vL,vR,Delta)

    ut = hydro.gamma(v)
    ux = v * hydro.gamma(v)

    u_t = hydro.down(ut, g_mn, 0, 0)
    u_x = hydro.down(ux, g_mn, 1, 1)

    T = params.make_temp(ep, ep_coeff)

    # ep, u^mu(v) gradients
    dxep = kt.general_ddx(ep, theta_kt, hx)

    dxu_x = kt.general_ddx(u_x, theta_kt, hx) # dxux = dxu_x
    dxu_t = kt.general_ddx(u_t, theta_kt, hx)

    dtu_x = 1/ut * ( (u_t**2 * dxu_x - ux**2 * dxu_x - ux * dxep/(4*ep))*u_x/(3+ 2*u_x**2) - ux*dxu_x-dxep/(4*ep) )
    dtu_t = - ux * dtu_x / ut

    # creating the state, q
    # ---------------------
    C_t = T * u_t
    C_x = T * u_x

    # numerical computations of X_mn = partial_m C_n
    X_xx = kt.general_ddx(C_x, theta_kt, hx)
    X_xt = kt.general_ddx(C_t, theta_kt, hx)

    # Euler-like ICs for Tmn require shear elements
    ott = - 2/3*(ut**2 - 1) * dtu_t + u_t * u_x * dxu_t - (ut**2 - 1) * dxu_x/3
    print('ott min/max = '+ str(min(ott)) +"/" + str(max(ott)) )
    otx = u_t*u_x*dtu_t/6 + dtu_x * (ut**2 - 1)/2 - dxu_x * u_t * u_x /6 - ut**2 * dxu_t/2

    # elements of Tmn 
    Ttt = ep*(4*(ut**2)/3 -1/3) - 2 * hydro.get_shearscalar(T,shear_coeff) * ott
    Ttx = 4/3*ep*ut*ux - 2 * hydro.get_shearscalar(T,shear_coeff) * otx
    # ---------------------
    return np.array([Ttt, Ttx, C_t, C_x, X_xx, X_xt])

#--------------------
# initialize output
#--------------------
def initial_conditions(problem):

    if problem == params.problem_list[0]:
        return init_gaussian(params.post_dict, params.param_dict, params.x)
        
    if problem == params.problem_list[1]:
        return init_ep_shock(params.post_dict, params.param_dict, params.x)
    
    if problem == params.problem_list[2]:
        return init_v_shock(params.post_dict, params.param_dict, params.x)
    
    if problem == params.problem_list[3]:
        return init_v_gaussian(params.post_dict, params.param_dict, params.x)

    if problem == params.problem_list[4]:
        return init_ep_shocktube(params.post_dict, params.param_dict, params.x)  
    
    if problem == params.problem_list[5]:
        return init_ep_v_gaussian(params.post_dict, params.param_dict, params.x)
    
    if problem == params.problem_list[6]:
        return init_ep_v_fd(params.post_dict, params.param_dict, params.x)

    
# =========================================================================



#--------------------
# EXTRAS
#--------------------
def save_key_variables(q):

        # importing parameters that should never change between tests
    ep_coeff, shear_coeff, bulk_scalar, cs = params.ep_coeff, params.shear_coeff, params.bulk_scalar, params.cs

    # parameters subject to change between tests, ! make sure these are being properly updated
    a1, a2, etaovers = params.a1, params.a2, params.etaovers

    # compute args for Tmn, Xmn here
    # ------------------------------
    # from C_m vector
    C_m = hydro.get_C_m(q) #
    Cm = hydro.get_Cm(C_m) #
    T = hydro.get_temp(C_m, Cm) #
    um = hydro.get_um(Cm,T) #
    u_m = hydro.get_u_m(C_m,T) # 
    Dmn = hydro.get_Dmn(um) #

    ep = hydro.get_ep(ep_coeff, T) #
    P = hydro.get_P(ep) #
    shear_scalar = hydro.get_shearscalar(T, shear_coeff) #

    # transport coefficients 
    tau_ep = hydro.get_tau_ep(T, a1, etaovers) #
    tau_P = tau_ep / 3.0 #
    tau_Q = hydro.get_tau_Q(T, a2, etaovers) #

    Tmn0 = hydro.get_Tmn_2x2_ideal(Dmn, um, ep, P)
    data_m.save_single(Tmn0, "Tmn0")

    X_mn = hydro.get_X_mn_2x2(q, Tmn0, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn, 0)
    data_m.save_single(X_mn, "X_mn")


    # used in matrix inversion, computing Y_inv
    Y0000 = hydro.get_Ymnab(0,0,0,0, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn)
    Y0100 = hydro.get_Ymnab(0,1,0,0, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn)
    Y0001 = hydro.get_Ymnab(0,0,0,1, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn)
    Y0101 = hydro.get_Ymnab(0,1,0,1, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn)
    
    data_m.save_single(Y0000, "Y0000")
    data_m.save_single(Y0100, "Y0100")
    data_m.save_single(Y0001, "Y0001")
    data_m.save_single(Y0101, "Y0101")

    # needed elements to compute X_mn
    Y0111 = hydro.get_Ymnab(0,1,1,1, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn)
    Y0010 = hydro.get_Ymnab(0,0,1,0, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn)
    Y0011 = hydro.get_Ymnab(0,0,1,1, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn)
    Y0110 = hydro.get_Ymnab(0,1,1,0, tau_ep, tau_P, tau_Q, bulk_scalar, shear_scalar, cs, T, ep, P, um, u_m, Dmn)

    data_m.save_single(Y0111, "Y0111")
    data_m.save_single(Y0010, "Y0010")
    data_m.save_single(Y0011, "Y0011")
    data_m.save_single(Y0110, "Y0110")

    Y_0000_inv = hydro.get_Y_mnab_inv(0,0,0,0, Y0000, Y0100, Y0001, Y0101)
    Y_0001_inv = hydro.get_Y_mnab_inv(0,0,0,1, Y0000, Y0100, Y0001, Y0101)
    Y_0100_inv = hydro.get_Y_mnab_inv(0,1,0,0, Y0000, Y0100, Y0001, Y0101)
    Y_0101_inv = hydro.get_Y_mnab_inv(0,1,0,1, Y0000, Y0100, Y0001, Y0101)

    data_m.save_single(Y_0000_inv, "Y_0000_inv")
    data_m.save_single(Y_0001_inv, "Y_0001_inv")
    data_m.save_single(Y_0100_inv, "Y_0100_inv")
    data_m.save_single(Y_0101_inv, "Y_0101_inv")

#--------------------
# SCRATCH
#--------------------
def init_gevrey(post_dict, param_dict, x):
    epmin = post_dict["epR"]
    epmax = post_dict["epL"]
    vL = post_dict["vL"]
    ep_coeff = param_dict["ep_coeff"]
    shear_coeff = param_dict["shear_coeff"]
    theta_kt = param_dict["theta_kt"]
    hx = param_dict["hx"]
    a2 = param_dict["a2"]
    etaovers = param_dict["etaovers"]

    x0 = post_dict["xM"]

    # ep, v, u^mu, T
    ep = params.gevrey(x,epmin,epmax)

    # v = vL * np.ones(len(x))
    # ut = hydro.gamma(v) * np.ones(len(x))
    # ux = v * hydro.gamma(v) * np.ones(len(x))
    ut = np.ones(len(x))
    ux = np.zeros(len(x))

    u_t = hydro.down(ut, g_mn, 0, 0)
    u_x = hydro.down(ux, g_mn, 1, 1)

    T = hydro.make_temp(ep, ep_coeff)

    # gradients of ep, u^mu(v) 
    # dxep = kt.general_ddx(ep, theta_kt, hx)

    # gradients of T, u^mu(v)
    dTdx = kt.general_ddx(T, theta_kt, hx)

    dxu_x = kt.general_ddx(u_x, theta_kt, hx)
    dxu_t = kt.general_ddx(u_t, theta_kt, hx)

    # creating the state, q
    # ---------------------
    C_t = T * u_t
    C_x = T * u_x

    # numerical computations of X_mn = partial_m C_n
    # X_xx = kt.general_ddx(C_x, theta_kt, hx)
    # X_xt = kt.general_ddx(C_t, theta_kt, hx)
    X_xx = np.zeros(len(T))
    X_xt = - dTdx

    tau_Q = hydro.get_tau_Q(T, a2, etaovers)
    # elements of Tmn 
    Ttt = ep
    # Ttx = 1/3 * tau_Q * dxep
    Ttx = np.zeros(len(T))
    # ---------------------
    q = np.array([Ttt, Ttx, C_t, C_x, X_xx, X_xt])

    # BCs
    q[:,-1] = q[:,-2] = q[:,-3] 
    q[:,0] = q[:,1] = q[:,2]

    data_m.save_single(q, "q_init")
    data_m.save_single(x, "x")
    data_m.save_single(T, "T_init")
    data_m.save_single([a2, 0], "a2_init")
    data_m.save_single([etaovers, 0], "etaovers_init")
    data_m.save_single(ep, "ep_init")
    data_m.save_single(dTdx, "dTdx_init")

    save_key_variables(q)


    return q

def init_gaussian_NOTBDNK(post_dict, param_dict, x):
    epmin = post_dict["epR"]
    epmax = post_dict["epL"]
    vL = post_dict["vL"]
    ep_coeff = param_dict["ep_coeff"]
    shear_coeff = param_dict["shear_coeff"]
    theta_kt = param_dict["theta_kt"]
    hx = param_dict["hx"]
    a2 = param_dict["a2"]
    etaovers = param_dict["etaovers"]

    x0 = post_dict["xM"]

    # ep, v, u^mu, T
    width = 5 # 1/GeV
    ep = params.gaussian(x,x0,epmax,epmin,width)

    # v = vL * np.ones(len(x))
    # ut = hydro.gamma(v) * np.ones(len(x))
    # ux = v * hydro.gamma(v) * np.ones(len(x))
    ut = np.ones(len(x))
    ux = np.zeros(len(x))

    u_t = hydro.down(ut, g_mn, 0, 0)
    u_x = hydro.down(ux, g_mn, 1, 1)

    T = hydro.make_temp(ep, ep_coeff)

    # gradients of ep, u^mu(v) 
    dxep = kt.general_ddx(ep, theta_kt, hx)

    dxu_x = kt.general_ddx(u_x, theta_kt, hx)
    dxu_t = kt.general_ddx(u_t, theta_kt, hx)

    # creating the state, q
    # ---------------------
    C_t = T * u_t
    C_x = T * u_x

    # numerical computations of X_mn = partial_m C_n
    X_xx = kt.general_ddx(C_x, theta_kt, hx)
    X_xt = kt.general_ddx(C_t, theta_kt, hx)

    tau_Q = hydro.get_tau_Q(T, a2, etaovers)
    # elements of Tmn 
    Ttt = ep
    # Ttx = 1/3 * tau_Q * dxep
    Ttx = a2 * etaovers / T * dxep
    # ---------------------
    q = np.array([Ttt, Ttx, C_t, C_x, X_xx, X_xt])

    # BCs
    q[:,-1] = q[:,-2] = q[:,-3] 
    q[:,0] = q[:,1] = q[:,2]

    data_m.save_single(q, "q_init")
    data_m.save_single(x, "x")
    data_m.save_single(T, "T_init")
    data_m.save_single([a2, 0], "a2_init")
    data_m.save_single([etaovers, 0], "etaovers_init")
    data_m.save_single(ep, "ep_init")
    data_m.save_single(dxep, "dxep_init")

    save_key_variables(q)

    print("!#$^@$%&@#$%!#$^&@$%")

    return q

def init_gaussian_momentum(post_dict, param_dict, x):

    epR = post_dict["epR"]
    epL = post_dict["epL"]
    vL = post_dict["vL"]
    ep_coeff = param_dict["ep_coeff"]
    shear_coeff = param_dict["shear_coeff"]
    theta_kt = param_dict["theta_kt"]
    hx = param_dict["hx"]
    a2 = param_dict["a2"]
    etaovers = param_dict["etaovers"]

    x0 = post_dict["xM"]

    # ep, v, u^mu, T
    Delta = post_dict["Delta"] # 1/GeV
    ep = params.gaussian(x,0,epL, epR, Delta)
    T = params.make_temp(ep, ep_coeff)

    ut = np.ones(len(x))
    ux = np.zeros(len(x))

    u_t = hydro.down(ut, g_mn, 0, 0)
    u_x = hydro.down(ux, g_mn, 1, 1)

    # gradients of T, u^mu(v)
    dTdx = params.make_gaussian_dTdx(epL,ep,x,Delta,ep_coeff)
    dTdx_num = kt.general_ddx(T, theta_kt, hx)

    dxu_x = kt.general_ddx(u_x, theta_kt, hx)
    dxu_t = kt.general_ddx(u_t, theta_kt, hx)

    tau_Q = hydro.get_tau_Q(T,a2,etaovers)

    # creating the state, q
    # ---------------------
    C_t = T * u_t
    C_x = T * u_x

    X_xx = np.zeros(len(T))
    X_xt = - dTdx 

    # elements of Tmn 
    Ttt = ep
    Ttx = tau_Q * (4*ep/3) * dTdx / T
    # ---------------------
    q = np.array([Ttt, Ttx, C_t, C_x, X_xx, X_xt])

    # BCs
    q[:,-1] = q[:,-2] = q[:,-3] 
    q[:,0] = q[:,1] = q[:,2]

    data_m.save_single(q, "q_init")
    data_m.save_single(x, "x")
    data_m.save_single(T, "T_init")
    data_m.save_single([a2, 0], "a2_init")
    data_m.save_single([etaovers, 0], "etaovers_init")
    data_m.save_single(ep, "ep_init")
    data_m.save_single(dTdx, "dTdx_init")

    save_key_variables(q)

    return q

def init_shocktube(post_dict, param_dict, x):
    epL = post_dict["epL"]
    epR = post_dict["epR"]
    vL = post_dict["vL"]
    vR = post_dict["vR"]
    Delta = post_dict["Delta"]
    cs = post_dict["cs"]
    x0 = post_dict["xM"]

    ep_coeff = param_dict["ep_coeff"]
    shear_coeff = param_dict["shear_coeff"]
    theta_kt = param_dict["theta_kt"]
    hx = param_dict["hx"]

    ep = params.shock(x,x0,epL,epR)

    ut = np.ones(len(x))
    ux = np.zeros(len(x))

    u_t = hydro.down(ut, g_mn, 0, 0)
    u_x = hydro.down(ux, g_mn, 1, 1)

    T = params.make_temp(ep, ep_coeff)

    dTdx = kt.general_ddx(T, theta_kt, hx)

    dxu_x = kt.general_ddx(u_x, theta_kt, hx)
    dxu_t = kt.general_ddx(u_t, theta_kt, hx)

    # creating the state, q
    # ---------------------
    C_t = T * u_t
    C_x = T * u_x

    # numerical computations of X_mn = partial_m C_n
    # X_xx = kt.general_ddx(C_x, theta_kt, hx)
    # X_xt = kt.general_ddx(C_t, theta_kt, hx)
    X_xx = np.zeros(len(T))
   #vX_xt = - dTdx
    X_xt = np.zeros(len(T)) # = \partial_x C_t ~ \partial_x T

    # elements of Tmn 
    Ttt = ep
    # Ttx = 1/3 * tau_Q * dxep
    Ttx = np.zeros(len(T))
    # ---------------------
    q = np.array([Ttt, Ttx, C_t, C_x, X_xx, X_xt])

    # BCs
    q[:,-1] = q[:,-2] = q[:,-3] 
    q[:,0] = q[:,1] = q[:,2]

    return q
#--------------------