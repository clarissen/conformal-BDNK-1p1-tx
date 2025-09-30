import numpy as np
import json

# homemade modules
import bdnk as hydro
import data_manager as data_m
import kurganovtadmor as kt

def calculate_vars(sim_name):
    # importing simulation data
    param_dict, post_dict, x, t_arr, qtx, ftx, stx = data_m.load_sim(sim_name, False)

    # loading dict 
    ep_coeff = param_dict["ep_coeff"]
    a1 = param_dict["a1"]
    a2 = param_dict["a2"]

    print("performing post-simulation calculations to obtain more physical variables...")

    theta_kt = param_dict["theta_kt"]
    hx = param_dict["hx"]
    etaovers = param_dict["etaovers"]

    # post-processing calculations
    # states
    T00_tx = qtx[:,0]
    T0x_tx = qtx[:,1]
    C_0_tx = qtx[:,2]
    C_x_tx = qtx[:,3]
    X_xx_tx = qtx[:,4]
    X_x0_tx = qtx[:,5]

    # fluxes/sources
    Txx_tx = ftx[:,1]
    X_00_tx = stx[:,2]
    X_0x_tx = stx[:,3]

    # # states
    # T00_tx = qtx[::tmod,0]
    # T0x_tx = qtx[::tmod,1]
    # C_0_tx = qtx[::tmod,2]
    # C_x_tx = qtx[::tmod,3]
    # X_xx_tx = qtx[::tmod,4]
    # X_x0_tx = qtx[::tmod,5]

    # # fluxes/sources
    # Txx_tx = ftx[::tmod,1]
    # X_00_tx = stx[::tmod,2]
    # X_0x_tx = stx[::tmod,3]

    # scalars
    Ttx = []
    eptx = []
    vtx = []
    Ptx = []

    Anormtx = []
    Qnormsqrttx = []

    # gradients    
    depdx_tx = []
    dvdx_tx = []

    # vectors

    # C_m = [C_0, C_x, C_y=0, C_z=0]
    C_m_tx = []
    Cm_tx = []
    um_tx = []
    u_m_tx = []

    # tensors
    Dmn_tx = []

    # Knudsens
    Kn_u_tx = [] # velocity dependent Kn
    Kn_T_tx = [] # temperature dependent Kn

    # in order of variable obtaining 

    for i in range(0,len(t_arr)):
        C_m_tx.append(hydro.get_C_m(qtx[i]))
        Cm_tx.append(hydro.get_Cm(C_m_tx[i]))

        Ttx.append(hydro.get_temp(C_m_tx[i], Cm_tx[i]))
        eptx.append(hydro.get_ep(ep_coeff, Ttx[i]) )
        Ptx.append(hydro.get_P(eptx[i]))
        depdx_tx.append(kt.general_ddx(eptx[i], theta_kt, hx))

        um_tx.append(hydro.get_um(Cm_tx[i], Ttx[i]))
        u_m_tx.append(hydro.get_u_m(C_m_tx[i], Ttx[i]))
        vtx.append(hydro.get_v(um_tx[i]) )
        dvdx_tx.append(kt.general_ddx(vtx[i], theta_kt, hx))

        # viscous corrections
        Dmn_tx.append(hydro.get_Dmn(um_tx[i]))
        Anormtx.append(hydro.get_Anorm(Ttx[i],um_tx[i], Dmn_tx[i], X_00_tx[i], X_0x_tx[i], X_x0_tx[i], X_xx_tx[i], eptx[i], Ptx[i], etaovers, a1))
        Qnormsqrttx.append(hydro.get_Qnorm_sqrt(Ttx[i],um_tx[i], Dmn_tx[i], X_00_tx[i], X_0x_tx[i], X_x0_tx[i], X_xx_tx[i], eptx[i], Ptx[i], etaovers, a2))
    
        Kn_u_tx.append(hydro.get_Kn_u(etaovers, Ttx[i], vtx[i], um_tx[i], X_x0_tx[i], X_xx_tx[i] ) )
        Kn_T_tx.append(hydro.get_Kn_T(etaovers, Ttx[i], vtx[i], um_tx[i], X_x0_tx[i], X_xx_tx[i] ) )


    um_tx = np.array(um_tx)
    u0_tx = um_tx[:,0]
    # ux_tx = um_tx[:,1]


    # scalars
    Ttx = np.array(Ttx)
    eptx = np.array(eptx)
    vtx = np.array(vtx)
    Ptx = np.array(Ptx)

    # gradients    
    depdx_tx = np.array(depdx_tx)
    dvdx_tx = np.array(dvdx_tx)

    # vectors

    # C_m = [C_0, C_x, C_y=0, C_z=0]

            # paper variables
    variables = ["T00(t,x)", T00_tx, "T0x(t,x)", T0x_tx, "Kn_u(t,x)", Kn_u_tx, "Kn_T(t,x)", Kn_T_tx, 
            "Anorm(t,x)", Anormtx, "Qnorm_sqrt(t,x)", Qnormsqrttx,
            
            "Txx(t,x)", Txx_tx, "ep(t,x)", eptx, \
            "depdx(t,x)", depdx_tx, "v(t,x)", vtx, "dvdx(t,x)", dvdx_tx, "P(t,x)", Ptx, \
                
            # more variables
            "C_0(t,x)", C_0_tx, "C_x(t,x)", C_x_tx, "X_xx(t,x)", X_xx_tx, "X_x0(t,x)", X_x0_tx,
    
             "X_00(t,x)", X_00_tx, "X_0x(t,x)", X_0x_tx, "u0(t,x)", u0_tx ]

    # variables = ["v(t,x)", vtx, "ep(t,x)", eptx, "P(t,x)", Ptx]
    print("...calculations complete.")
    data_m.save_npy_list(sim_name, variables, "vars")

    return variables
