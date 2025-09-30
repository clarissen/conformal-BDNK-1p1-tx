import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

import data_manager as data_m

#--------------------
# Functions
#--------------------
# =========================================================================
def get_common_grid():

    return 

def get_L2_errors(phi_coarse, phi_medium, phi_fine, hx_list, ht_list):
     
    """ Computes discrete L2 norm between two fields defined on the same grid. """

    hx_coarse, hx_medium, hx_fine = hx_list
    ht_coarse, ht_medium, ht_fine = ht_list

    # ratio in x grids ( should be 2)
    rx_cm = int(round(hx_coarse/hx_medium))
    rx_cf = int(round(hx_coarse/hx_fine))

    # ratio in t grids ( should be 1)
    # rt_cm = int(round(ht_coarse/ht_medium))
    # rt_cf = int(round(ht_coarse/ht_fine))
    
    # delta_cm = (phi_coarse - phi_medium[::rt_cm,::rx_cm])
    # delta_mf = (phi_medium[::rt_cm,::rx_cm] - phi_fine[::rt_cf,::rx_cf])

    # time is already aligned bc i only save mod 1/ht
    delta_cm = (phi_coarse - phi_medium[:,::rx_cm])
    delta_mf = (phi_medium[:,::rx_cm] - phi_fine[:,::rx_cf])
    print(delta_cm.shape)

    # print('rx_cm, rx_cf, rt_cm, rt_cf:')
    # print(rx_cm, rx_cf, rt_cm, rt_cf)

    a = 0 # 4
    b = -1 # -5
    print("delta_cm.shape", delta_cm.shape)
    print("delta_cm.shape[:,a:b]", delta_cm[:,a:b].shape)

    L2_cm = np.sqrt(np.sum( delta_cm[:,a:b]**2, axis=1) * hx_coarse )
    L2_mf = np.sqrt(np.sum( delta_mf[:,a:b]**2, axis=1) * hx_coarse )
    return L2_cm, L2_mf

def get_L1_errors(phi_coarse, phi_medium, phi_fine, hx_list, ht_list):
     
    """ Computes discrete L1 norm between two fields defined on the same grid. """

    hx_coarse, hx_medium, hx_fine = hx_list
    ht_coarse, ht_medium, ht_fine = ht_list

    # ratio in x grids ( should be 2)
    rx_cm = int(round(hx_coarse/hx_medium))
    rx_cf = int(round(hx_coarse/hx_fine))

    # ratio in t grids ( should be 1)
    # rt_cm = int(round(ht_coarse/ht_medium))
    # rt_cf = int(round(ht_coarse/ht_fine))
    
    # delta_cm = (phi_coarse - phi_medium[::rt_cm,::rx_cm])
    # delta_mf = (phi_medium[::rt_cm,::rx_cm] - phi_fine[::rt_cf,::rx_cf])

    # time is already aligned bc i only save mod 1/ht
    delta_cm = (phi_coarse - phi_medium[:,::rx_cm])
    delta_mf = (phi_medium[:,::rx_cm] - phi_fine[:,::rx_cf])

    # print('rx_cm, rx_cf, rt_cm, rt_cf:')
    # print(rx_cm, rx_cf, rt_cm, rt_cf)

    L1_cm = np.sum( np.abs(delta_cm), axis=1) * hx_coarse 
    L1_mf = np.sum( np.abs(delta_mf), axis=1) * hx_coarse
    return L1_cm, L1_mf

def get_convergence_Q(L2_coarse_medium, L2_medium_fine):
    tol = 0
    return L2_coarse_medium/( L2_medium_fine + tol)

def get_convergence_Q_log2(L2_coarse_medium, L2_medium_fine):
    tol = 0
    return np.log2(L2_coarse_medium/(L2_medium_fine + tol) )

def plot_convergence(sim_name, problem, t_array, hx_list, ht_list, Q_list, var_name_list, var_label_list, frame, etaovers, Q_type):
    
    plt.rcParams.update({
    # "axes.titlesize": 16,     # subplot titles
    "axes.labelsize": 14,     # x and y labels
    # "xtick.labelsize": 12,    # tick labels
    # "ytick.labelsize": 12,
    "legend.fontsize": 10
    })

    sim_loc = "./sims/" + sim_name
    gct_loc = sim_loc + "/gct/"
    
    if not os.path.exists(gct_loc ):
        os.makedirs(gct_loc)

    hx_c, hx_m, hx_f = hx_list
    ht_c, ht_m, ht_f = ht_list

    linestyles = ['k', 'b', 'm', 'r', 'g']
    
    plt.figure() # not plotting early times

    for i in range(0, len(Q_list)):
        plt.plot(t_array[2:], Q_list[i][2:], linestyles[i], label = var_label_list[i])
        if Q_type == 'L1_log2':
            np.save(gct_loc+ Q_type + "_" + var_name_list[i], Q_list[i])
    
    plt.xlabel(r"$t$ (1/GeV)")
    plt.ylabel(r"Q(t)")

    plt.ylim(0, 2)

    plt.legend(loc = 'lower left')

    # plt.savefig(gct_loc + Q_type + "_" +problem+ "_frame=" + str(frame) +"_hx="+str(hx_c)+","+str(hx_m)+","+str(hx_f)+"_etas="+str(etaovers)+"_"+var_name_list[0]+"_"+var_name_list[1]+".pdf")
    plt.savefig(gct_loc + Q_type +".pdf")

    print("saved under: " + gct_loc)

# =========================================================================


#--------------------
# Dataframe
#--------------------
# =========================================================================
data = {
    # CONVENTION 
    #                       list
    # name of list : [coarse sim, medium sim, fine sim]


    # EULER

    "ep-shock, euler, ep=(1.3,0.3), hx = 0.05,0.025,0.0125, ht = 0.5hx, minmod3, Delta=1.0": ["euler/euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.0_theta_kt=1.0_alarms=False_sucomp=True",
                                                                                              "euler/euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.025_ht=0.0125_Ncells=6001_grid=[-75.0,75.0]_c+=0.0_theta_kt=1.0_alarms=False_sucomp=True",
                                                                                              "euler/euler_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0,0)_Delta=1.0_hx=0.0125_ht=0.00625_Ncells=12001_grid=[-75.0,75.0]_c+=0.0_theta_kt=1.0_alarms=False_sucomp=True"
                                                                                               ],

    # Gaussian 
    # ---------------

    # TEANEY ICs

    # higher resolution
    'gaussian, frame 2, eta/s=1/4pi, ep=(A=0.48,d=0.12), hx = 0.025,0.0125,0.00625, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_gaussian_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_gaussian_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                            ], 

    'gaussian, frame 2, eta/s=3/4pi, ep=(A=0.48,d=0.12), hx = 0.025,0.0125,0.00625, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_gaussian_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_gaussian_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                            ],                                                                                                                                                                                                                                                                                          
    
    
    'gaussian, frame 2, eta/s=3/4pi, ep=(A=0.48,d=0.12), hx = 0.0125,0.00625,0.00625/2, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                                "bdnk_gaussian_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                                "bdnk_gaussian_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                                ],

    'gaussian, frame 2, eta/s=3/4pi, ep=(A=0.48,d=0.12), hx = 0.0125,0.00625,0.00625/2, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                                "bdnk_gaussian_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                                "bdnk_gaussian_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.0015625_ht=0.001_Ncells=96001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                                ],

    # 3/4pi
    # d = 0.12
    'gaussian, frame 2, eta/s=3/4pi, ep=(A=0.48,d=0.12), hx = 0.2,0.1,0.05, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.2_ht=0.1_Ncells=751_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_gaussian_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.1_ht=0.05_Ncells=1501_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_gaussian_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                            ],

    # d = 0.24
    # coarse
    'gaussian, frame 2, eta/s=3/4pi, ep=(A=0.48,d=0.24), hx = 0.2,0.1,0.05, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.24)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.2_ht=0.1_Ncells=751_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_gaussian_(epL,epR)=(0.48,0.24)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.1_ht=0.05_Ncells=1501_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_gaussian_(epL,epR)=(0.48,0.24)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                            ],
    # -----
    'gaussian, frame 2, eta/s=3/4pi, ep=(A=0.48,d=0.24), hx = 0.05,0.05/2,0.05/4, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.24)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.24)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.24)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          ],
    
    'gaussian, frame 2, eta/s=3/4pi, ep=(A=0.48,d=0.24), hx = 0.05/2,0.05/4,0.05/8, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.24)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_gaussian_(epL,epR)=(0.48,0.24)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_gaussian_(epL,epR)=(0.48,0.24)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                            ],

    'gaussian, frame 2, eta/s=3/4pi, ep=(A=0.48,d=0.24), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.24)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                             "bdnk_gaussian_(epL,epR)=(0.48,0.24)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                             "bdnk_gaussian_(epL,epR)=(0.48,0.24)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                             ],

    # d = 0.36
    'gaussian, frame 2, eta/s=3/4pi, ep=(A=0.48,d=0.36), hx = 0.2,0.1,0.05, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.36)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.2_ht=0.1_Ncells=751_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_gaussian_(epL,epR)=(0.48,0.36)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.1_ht=0.05_Ncells=1501_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_gaussian_(epL,epR)=(0.48,0.36)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                            ],

    # d = 0.48
    'gaussian, frame 2, eta/s=3/4pi, ep=(A=0.48,d=0.48), hx = 0.05,0.05/2,0.05/4, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          ],

    'gaussian, frame 2, eta/s=3/4pi, ep=(A=0.48,d=0.48), hx = 0.05/2,0.05/4,0.05/8, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                          ],                                                                                                         

    'gaussian, frame 2, eta/s=3/4pi, ep=(A=0.48,d=0.48), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                          ],

    #eta/s = 6/4pi
    # d = 0.48
    'gaussian, frame 2, eta/s=6/4pi, ep=(A=0.48,d=0.48), hx = 0.05,0.05/2,0.05/4, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                          ],

    'gaussian, frame 2, eta/s=6/4pi, ep=(A=0.48,d=0.48), hx = 0.05/2,0.05/4,0.05/8, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                          ],

    'gaussian, frame 2, eta/s=6/4pi, ep=(A=0.48,d=0.48), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.48)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                          ],

    # d = 0.96
    'gaussian, frame 2, eta/s=6/4pi, ep=(A=0.48,d=0.96), hx = 0.05,0.05/2,0.05/4, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.96)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.96)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.96)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          ],

    'gaussian, frame 2, eta/s=6/4pi, ep=(A=0.48,d=0.96), hx = 0.05/2,0.05/4,0.05/8, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.96)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.96)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.96)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          ],   

    'gaussian, frame 2, eta/s=6/4pi, ep=(A=0.48,d=0.96), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,0.96)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.96)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,0.96)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                          ],                                                                                                       

    # d = 1.92
    'gaussian, frame 2, eta/s=6/4pi, ep=(A=0.48,d=1.92), hx = 0.05,0.05/2,0.05/4, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,1.92)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,1.92)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,1.92)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                          ],

    'gaussian, frame 2, eta/s=6/4pi, ep=(A=0.48,d=1.92), hx = 0.05/2,0.05/4,0.05/8, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,1.92)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,1.92)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,1.92)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                          ],

    'gaussian, frame 2, eta/s=6/4pi, ep=(A=0.48,d=1.92), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,1.92)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,1.92)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_gaussian_(epL,epR)=(0.48,1.92)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.477_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                          ],

    # eta/s = 20/4pi
    'gaussian, frame 2, eta/s=20/4pi, ep=(A=0.48,d=3.84), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,3.84)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=1.592_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                              "bdnk_gaussian_(epL,epR)=(0.48,3.84)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=1.592_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                              "bdnk_gaussian_(epL,epR)=(0.48,3.84)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=1.592_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                              ],

    'gaussian, frame 2, eta/s=20/4pi, ep=(A=0.48,d=7.68), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.48,7.68)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=1.592_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                              "bdnk_gaussian_(epL,epR)=(0.48,7.68)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=1.592_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                              "bdnk_gaussian_(epL,epR)=(0.48,7.68)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=1.592_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                              ],

    # PRINCETON ICs
    # eta/s = 1/4pi
    # d= 0.1
    'gaussian, frame 1, eta/s=1/4pi, ep=(A=0.4,d=0.1), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.4,0.1)_(vL,vR)=(0.0,0.0)_a1=6.25_a2=3.57_etaovs=0.08_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=1.0_kt-minmod-tvdrk2",
                                                                                                           "bdnk_gaussian_(epL,epR)=(0.4,0.1)_(vL,vR)=(0.0,0.0)_a1=6.25_a2=3.57_etaovs=0.08_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=1.0_kt-minmod-tvdrk2",
                                                                                                           "bdnk_gaussian_(epL,epR)=(0.4,0.1)_(vL,vR)=(0.0,0.0)_a1=6.25_a2=3.57_etaovs=0.08_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=1.0_kt-minmod-tvdrk2"
                                                                                                           ],

    'gaussian, frame 2, eta/s=1/4pi, ep=(A=0.4,d=0.1), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.4,0.1)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                           "bdnk_gaussian_(epL,epR)=(0.4,0.1)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                           "bdnk_gaussian_(epL,epR)=(0.4,0.1)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                           ],

    'gaussian, frame 3, eta/s=1/4pi, ep=(A=0.4,d=0.1), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.4,0.1)_(vL,vR)=(0.0,0.0)_a1=25_a2=25_etaovs=0.08_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.74_kt-minmod-tvdrk2",
                                                                                                            "bdnk_gaussian_(epL,epR)=(0.4,0.1)_(vL,vR)=(0.0,0.0)_a1=25_a2=25_etaovs=0.08_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.74_kt-minmod-tvdrk2",
                                                                                                            "bdnk_gaussian_(epL,epR)=(0.4,0.1)_(vL,vR)=(0.0,0.0)_a1=25_a2=25_etaovs=0.08_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.74_kt-minmod-tvdrk2"
                                                                                                            ],


    # eta/s = 3/4pi
    # d=0.1
    'gaussian, frame 2, eta/s=3/4pi, ep=(A=0.4,d=0.1), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.4,0.1)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                             "bdnk_gaussian_(epL,epR)=(0.4,0.1)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                             "bdnk_gaussian_(epL,epR)=(0.4,0.1)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                             ],


    # d=4*0.1
    'gaussian, frame 2, eta/s=3/4pi, ep=(A=0.4,d=4*0.1), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.4,0.4)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                             "bdnk_gaussian_(epL,epR)=(0.4,0.4)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                             "bdnk_gaussian_(epL,epR)=(0.4,0.4)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                             ],                                                                                        

    # eta/s = 20/4pi
    # d= 0.1
    'gaussian, frame 2, eta/s=20/4pi, ep=(A=0.4,d=0.1), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.4,0.1)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=1.592_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_gaussian_(epL,epR)=(0.4,0.1)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=1.592_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_gaussian_(epL,epR)=(0.4,0.1)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=1.592_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                            ],
    
    # d= 64 * 0.1
    'gaussian, frame 2, eta/s=20/4pi, ep=(A=0.4,d=64*0.1), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3': ["bdnk_gaussian_(epL,epR)=(0.4,6.4)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=1.592_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                               "bdnk_gaussian_(epL,epR)=(0.4,6.4)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=1.592_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                               "bdnk_gaussian_(epL,epR)=(0.4,6.4)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=1.592_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                               ],

    # ---------------

    # smooth shock epsilon
    # ---------------

    # Delta = 0.2 
    'ep-shock, frame 2, eta/s=1/4pi, ep=(A=0.48,d=0.12), hx = 0.2,0.1,0.05, ht = 0.5hx, minmod3, Delta=0.2': ["bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=0.2_hx=0.2_ht=0.1_Ncells=751_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                   "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=0.2_hx=0.1_ht=0.05_Ncells=1501_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                   "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=0.2_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                   ],

    'ep-shock, frame 2, eta/s=1/4pi, ep=(A=0.48,d=0.12), hx = 0.05/2,0.05/4,0.05/8, ht = 0.5hx, minmod3, Delta=0.2': ["bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=0.2_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                              "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=0.2_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                              "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=0.2_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                              ],

    
    # Delta = 1
    'ep-shock, frame 2, eta/s=1/4pi, ep=(A=0.48,d=0.12), hx = 0.2,0.1,0.05, ht = 0.5hx, minmod3, Delta=1': ["bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.2_ht=0.1_Ncells=751_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                              "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.1_ht=0.05_Ncells=1501_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                              "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                              ],

    'ep-shock, frame 2, eta/s=1/4pi, ep=(A=0.48,d=0.12), hx = 0.1,0.05,0.025, ht = 0.5hx, minmod3, Delta=1': ["bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.1_ht=0.05_Ncells=1501_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                              "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                              ],

    'ep-shock, frame 2, eta/s=1/4pi, ep=(A=0.48,d=0.12), hx = 0.05,0.025,0.0125 ht = 0.5hx, minmod3, Delta=1': ["bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                              "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                              "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                              ],                                                                                                        

    'ep-shock, frame 2, eta/s=1/4pi, ep=(A=0.48,d=0.12), hx = 0.025,0.0125,0.00625, ht = 0.5hx, minmod3': ["bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=0.2_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=0.2_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=0.2_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                            ],

    # 'ep-shock, frame 2, eta/s=1/4pi, ep=(A=0.48,d=0.12), hx = 0.025,0.0125,0.0125/2, ht = 0.5hx, minmod3': ["bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=0.2_hx=0.025_ht=0.013_Ncells=6001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
    #                                                                                                         "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=0.2_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
    #                                                                                                         "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=0.2_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
    #                                                                                                         ],

    # Delta = 1, coarse, minmod
    'ep-shock, frame 2, eta/s=1/4pi, ep=(A=0.48,d=0.12), hx = 0.2,0.1,0.05, ht = 0.5hx, minmod3, Delta=1': ["bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=1.0_hx=0.2_ht=0.1_Ncells=751_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=1.0_hx=0.1_ht=0.05_Ncells=1501_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=1.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                            ],
                                                                                                            
    # Delta = 1, coarse, superbee
    'ep-shock, frame 2, eta/s=1/4pi, ep=(A=0.48,d=0.12), hx = 0.2,0.1,0.05, ht = 0.5hx, superbee, Delta=1': ["bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=1.0_hx=0.2_ht=0.1_Ncells=751_grid=[-75.0,75.0]_c+=0.85_kt-superbee-tvdrk2",
                                                                                                             "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=1.0_hx=0.1_ht=0.05_Ncells=1501_grid=[-75.0,75.0]_c+=0.85_kt-superbee-tvdrk2",
                                                                                                             "bdnk_ep-shock_(epL,epR)=(0.48,0.12)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.239_Delta=1.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-superbee-tvdrk2"
                                                                                                             ],
    # Searching for Convergence

    "ep-shock, frame 2, eta/s=1/4pi, ep=(0.5,0.45), hx = 0.05/4,0.05/4,0.05/16, ht = 0.5hx, minmod3, Delta=1.0": ["bdnk_ep-shock_(epL,epR)=(0.5,0.45)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                                 "bdnk_ep-shock_(epL,epR)=(0.5,0.45)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                                 "bdnk_ep-shock_(epL,epR)=(0.5,0.45)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                                 ],

    "ep-shock, frame 2, eta/s=1/4pi, ep=(0.5,0.45), hx = 0.05/4,0.05/4,0.05/16, ht = 0.5hx, minmod3, Delta=2.0": ["bdnk_ep-shock_(epL,epR)=(0.5,0.45)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=2.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                                  "bdnk_ep-shock_(epL,epR)=(0.5,0.45)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=2.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                                  "bdnk_ep-shock_(epL,epR)=(0.5,0.45)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=2.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                                  ],                                                                                        

    # EXTREME
    "ep-shock, frame 1, eta/s=1/4pi, ep=(1.3,0.3), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5hx, minmod3, Delta=1.0": ["bdnk_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0.0,0.0)_a1=6.25_a2=3.57_etaovs=0.08_Delta=1.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=1.0_kt-minmod-tvdrk2",
                                                                                                              "bdnk_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0.0,0.0)_a1=6.25_a2=3.57_etaovs=0.08_Delta=1.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=1.0_kt-minmod-tvdrk2",
                                                                                                              "bdnk_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0.0,0.0)_a1=6.25_a2=3.57_etaovs=0.08_Delta=1.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=1.0_kt-minmod-tvdrk2"

    ],

    "ep-shock, frame 2, eta/s=1/4pi, ep=(1.3,0.3), hx = 0.05*4,0.05*2,0.05, ht = 0.5hx, minmod3, Delta=1.0": ["bdnk_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.2_ht=0.1_Ncells=751_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                              "bdnk_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.1_ht=0.05_Ncells=1501_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                              "bdnk_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
         ],

    "ep-shock, frame 2, eta/s=1/4pi, ep=(1.3,0.3), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5hx, minmod3, Delta=1.0": ["bdnk_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                                 "bdnk_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                                 "bdnk_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0.0,0.0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=1.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                                 ],

    "ep-shock, frame 3, eta/s=1/4pi, ep=(1.3,0.3), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5hx, minmod3, Delta=1.0": ["bdnk_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0.0,0.0)_a1=25_a2=25_etaovs=0.08_Delta=1.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.74_kt-minmod-tvdrk2",
                                                                                                                 "bdnk_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0.0,0.0)_a1=25_a2=25_etaovs=0.08_Delta=1.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.74_kt-minmod-tvdrk2",
                                                                                                                 "bdnk_ep-shock_(epL,epR)=(1.3,0.3)_(vL,vR)=(0.0,0.0)_a1=25_a2=25_etaovs=0.08_Delta=1.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.74_kt-minmod-tvdrk2"
                                                                                                                 ],

    # ---------------

    # velocity gaussian
    # ---------------

    'v-gaussian, frame 2, eta/s=1/4pi, v=(A=0.6,d=0.9), hx = 0.05,0.025,0.00125, ht = 0.5hx, minmod3': ["bdnk_v-gaussian_(epL,epR)=(0.5,0.5)_(vL,vR)=(0.9,0.6)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.05_ht=0.025_Ncells=6001_grid=[-150.0,150.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_v-gaussian_(epL,epR)=(0.5,0.5)_(vL,vR)=(0.9,0.6)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.025_ht=0.013_Ncells=12001_grid=[-150.0,150.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_v-gaussian_(epL,epR)=(0.5,0.5)_(vL,vR)=(0.9,0.6)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.0125_ht=0.006_Ncells=24001_grid=[-150.0,150.0]_c+=0.85_kt-minmod-tvdrk2"],

    'v-gaussian, frame 2, eta/s=1/4pi, v=(A=0.6,d=0.9), hx = 0.05,0.025,0.00125, ht = 0.5hx, minmod3': ["bdnk_v-gaussian_(epL,epR)=(0.5,0.5)_(vL,vR)=(0.9,0.6)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.025_ht=0.013_Ncells=12001_grid=[-150.0,150.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                           "bdnk_v-gaussian_(epL,epR)=(0.5,0.5)_(vL,vR)=(0.9,0.6)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.0125_ht=0.006_Ncells=24001_grid=[-150.0,150.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                           "bdnk_v-gaussian_(epL,epR)=(0.5,0.5)_(vL,vR)=(0.9,0.6)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.00625_ht=0.003_Ncells=48001_grid=[-150.0,150.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                           ],

    'v-gaussian, frame 2, eta/s=1/4pi, v=(A=0.3,d=0.6), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5hx, minmod3': ["bdnk_v-gaussian_(epL,epR)=(0.5,0.5)_(vL,vR)=(0.6,0.3)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                           "bdnk_v-gaussian_(epL,epR)=(0.5,0.5)_(vL,vR)=(0.6,0.3)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                           "bdnk_v-gaussian_(epL,epR)=(0.5,0.5)_(vL,vR)=(0.6,0.3)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                           ],

    'v-gaussian, frame 2, eta/s=1/4pi, v=(A=0.05,d=0.1), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5hx, minmod3': ["bdnk_v-gaussian_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.1,0.05)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                           "bdnk_v-gaussian_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.1,0.05)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                           "bdnk_v-gaussian_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.1,0.05)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                           ],

    'v-gaussian, frame 2, eta/s=1/4pi, v=(A=0.0,d=0.1), hx = 0.05*4,0.05*2,0.05, ht = 0.5hx, minmod3': ["bdnk_v-gaussian_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.1,0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.2_ht=0.1_Ncells=751_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                        "bdnk_v-gaussian_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.1,0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.1_ht=0.05_Ncells=1501_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                        "bdnk_v-gaussian_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.1,0)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                        ],

    'v-gaussian, frame 2, eta/s=1/4pi, v=(A=0.001,d=0.1), hx = 0.05*4,0.05*2,0.05, ht = 0.5hx, minmod3': ["bdnk_v-gaussian_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.1,0.001)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.2_ht=0.1_Ncells=751_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_v-gaussian_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.1,0.001)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.1_ht=0.05_Ncells=1501_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                          "bdnk_v-gaussian_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.1,0.001)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                          ],


    'v-gaussian, frame 2, eta/s=1/4pi, v=(A=0.01,d=0.9), hx = 0.05*4,0.05*2,0.05, ht = 0.5hx, minmod3': ["bdnk_v-gaussian_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.9,0.01)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.2_ht=0.1_Ncells=751_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                        "bdnk_v-gaussian_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.9,0.01)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.1_ht=0.05_Ncells=1501_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                        "bdnk_v-gaussian_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.9,0.01)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                        ], 

    'v-gaussian, frame 2, eta/s=1/4pi, v=(A=0.01,d=0.9), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5hx, minmod3': ["bdnk_v-gaussian_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.9,0.01)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_v-gaussian_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.9,0.01)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                            "bdnk_v-gaussian_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.9,0.01)_a1=12.5_a2=8.33_etaovs=0.08_Delta=5.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                            ],                                                                                                       
    
    # smooth velocity shock

    'v-shock, frame 2, eta/s = 1/4pi, v=(0.999, 1/3vL), hx = 0.05*4,0.05*2,0.05, ht = 0.5hx, minmod3': ["bdnk_v-shock_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.999,0.33366700033366703)_a1=12.5_a2=8.33_etaovs=0.08_Delta=2.0_hx=0.2_ht=0.1_Ncells=751_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                        "bdnk_v-shock_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.999,0.33366700033366703)_a1=12.5_a2=8.33_etaovs=0.08_Delta=2.0_hx=0.1_ht=0.05_Ncells=1501_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                        "bdnk_v-shock_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.999,0.33366700033366703)_a1=12.5_a2=8.33_etaovs=0.08_Delta=2.0_hx=0.05_ht=0.025_Ncells=3001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                        ],

    'v-shock, frame 2, eta/s = 1/4pi, v=(0.999, 1/3vL), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5hx, minmod3': ["bdnk_v-shock_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.999,0.33366700033366703)_a1=12.5_a2=8.33_etaovs=0.08_Delta=2.0_hx=0.0125_ht=0.006_Ncells=12001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                           "bdnk_v-shock_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.999,0.33366700033366703)_a1=12.5_a2=8.33_etaovs=0.08_Delta=2.0_hx=0.00625_ht=0.003_Ncells=24001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2",
                                                                                                           "bdnk_v-shock_(epL,epR)=(1.0,1.0)_(vL,vR)=(0.999,0.33366700033366703)_a1=12.5_a2=8.33_etaovs=0.08_Delta=2.0_hx=0.003125_ht=0.002_Ncells=48001_grid=[-75.0,75.0]_c+=0.85_kt-minmod-tvdrk2"
                                                                                                           ],
}
# =========================================================================

#--------------------
# Loading 
#--------------------
# =========================================================================


df = pd.DataFrame(data)
runs = ['gaussian', 'ep-shock', 'v-gaussian']

run = input("ep shock or gaussian or v shock or gaussian w momentum? type g, e, vg, v: ")

if run == 'g':
    df_type = 'gaussian, frame 2, eta/s=20/4pi, ep=(A=0.4,d=64*0.1), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3'
    # df_type = 'gaussian, frame 2, eta/s=20/4pi, ep=(A=0.4,d=0.1), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3'
    # df_type = 'gaussian, frame 2, eta/s=3/4pi, ep=(A=0.4,d=0.1), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3'
    # df_type = 'gaussian, frame 2, eta/s=3/4pi, ep=(A=0.4,d=4*0.1), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3'
    # df_type = 'gaussian, frame 2, eta/s=1/4pi, ep=(A=0.4,d=0.1), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3'
    # df_type = 'gaussian, frame 1, eta/s=1/4pi, ep=(A=0.4,d=0.1), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3'
    # df_type = 'gaussian, frame 3, eta/s=1/4pi, ep=(A=0.4,d=0.1), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5*hx, minmod3'
    sim_c = df.loc[0, df_type]
    sim_m = df.loc[1, df_type]
    sim_f = df.loc[2, df_type]
    problem = 'gaussian'

if run == 'e':
    # df_type = "ep-shock, frame 1, eta/s=1/4pi, ep=(1.3,0.3), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5hx, minmod3, Delta=1.0"
    # df_type = "ep-shock, frame 2, eta/s=1/4pi, ep=(1.3,0.3), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5hx, minmod3, Delta=1.0"
    # df_type = "ep-shock, frame 3, eta/s=1/4pi, ep=(1.3,0.3), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5hx, minmod3, Delta=1.0"
    df_type = "ep-shock, euler, ep=(1.3,0.3), hx = 0.05,0.025,0.0125, ht = 0.5hx, minmod3, Delta=1.0"
    sim_c = df.loc[0, df_type]
    sim_m = df.loc[1, df_type]
    sim_f = df.loc[2, df_type]
    problem = 'ep-shock'

if run == 'vg':
    # df_type = 'v-gaussian, frame 2, eta/s=1/4pi, v=(A=0.001,d=0.1), hx = 0.05*4,0.05*2,0.05, ht = 0.5hx, minmod3'
    df_type = 'v-gaussian, frame 2, eta/s=1/4pi, v=(A=0.01,d=0.9), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5hx, minmod3'
    sim_c = df.loc[0, df_type]
    sim_m = df.loc[1, df_type]
    sim_f = df.loc[2, df_type]
    problem = 'v-gaussian'

if run == 'v':
    # df_type = 'v-shock, frame 2, eta/s = 1/4pi, v=(0.999, 1/3vL), hx = 0.05*4,0.05*2,0.05, ht = 0.5hx, minmod3'
    df_type = 'v-shock, frame 2, eta/s = 1/4pi, v=(0.999, 1/3vL), hx = 0.05/4,0.05/8,0.05/16, ht = 0.5hx, minmod3'
    sim_c = df.loc[0, df_type]
    sim_m = df.loc[1, df_type]
    sim_f = df.loc[2, df_type]
    problem = 'v-shock'

# loading data
# ===============================
sim_names = [sim_c, sim_m, sim_f]

# Global Variables may be used in convergence test
# coarse data
param_dict_c, post_dict_c, x_c, t_c, qtx_c, ftx_c, stx_c = data_m.load_sim(sim_names[0], False)
vars_c, var_names_c = data_m.load_vars(sim_names[0], False) 
# medium data
param_dict_m, post_dict_m, x_m, t_m, qtx_m, ftx_m, stx_m = data_m.load_sim(sim_names[1], False)
vars_m, var_names_m = data_m.load_vars(sim_names[1], False)
# fine data
param_dict_f, post_dict_f, x_f, t_f, qtx_f, ftx_f, stx_f = data_m.load_sim(sim_names[2], False)
vars_f, var_names_f = data_m.load_vars(sim_names[2], False)

t_list = [t_c, t_m, t_f]
print("time")
print(t_list[0])
print(t_list[1])
print(t_list[2])

x_list = [x_c, x_m, x_f]
print("space")
print(x_list[0])
print(x_list[1])
print(x_list[2])

# resolutions
hx_c = param_dict_c["hx"]
hx_m = param_dict_m["hx"]
hx_f = param_dict_f["hx"]
hx_list = [hx_c, hx_m, hx_f]

ht_c = param_dict_c["ht"]
ht_m = param_dict_m["ht"]
ht_f = param_dict_f["ht"]
ht_list = [ht_c, ht_m, ht_f]

cCFL = param_dict_c["cCFL"]
frame = param_dict_c["frame"]
etaovers = round(param_dict_c["etaovers"],3)

epL = post_dict_c["epL"]
epR = post_dict_c["epR"]

# ===============================

# General plotting parameters
var_name_list = ["T00(t,x)","T0x(t,x)", "Txx(t,x)", "ep(t,x)", "v(t,x)"]
var_label_list = [r"$T^{00}(t,x)$", r"$T^{0x}(t,x)$", r"$T^{xx}(t,x)$", r"$\varepsilon(t,x)$", r"$v(t,x)$"]
t_array = t_c

# T00, T0x, ep, v
# -------------------
# coarse
T00_tx_c = qtx_c[:,0]
T0x_tx_c = qtx_c[:,1]
ep_tx_c = vars_c[15]
v_tx_c = vars_c[19]
Txx_tx_c = vars_c[13]

# medium
T00_tx_m = qtx_m[:,0]
T0x_tx_m = qtx_m[:,1]
ep_tx_m = vars_m[15]
v_tx_m = vars_m[19]
Txx_tx_m = vars_m[13]

# fine
T00_tx_f = qtx_f[:,0]
T0x_tx_f = qtx_f[:,1]
ep_tx_f = vars_f[15]
v_tx_f = vars_f[19]
Txx_tx_f = vars_f[13]

# -------------------

# L2 stuff
# ---------
# T00
T00_L2_cm, T00_L2_mf = get_L2_errors(T00_tx_c, T00_tx_m, T00_tx_f, hx_list, ht_list)
Q_T00_log2_L2 = get_convergence_Q_log2(T00_L2_cm, T00_L2_mf)
Q_T00_L2 = get_convergence_Q(T00_L2_cm, T00_L2_mf)

# T0x 
T0x_L2_cm, T0x_L2_mf = get_L2_errors(T0x_tx_c, T0x_tx_m, T0x_tx_f, hx_list, ht_list)
Q_T0x_log2_L2 = get_convergence_Q_log2(T0x_L2_cm, T0x_L2_mf)
Q_T0x_L2 = get_convergence_Q(T0x_L2_cm, T0x_L2_mf)

# ep
ep_L2_cm, ep_L2_mf = get_L2_errors(ep_tx_c, ep_tx_m, ep_tx_f, hx_list, ht_list)
Q_ep_log2_L2 = get_convergence_Q_log2(ep_L2_cm, ep_L2_mf)
Q_ep_L2 = get_convergence_Q(ep_L2_cm, ep_L2_mf)

# v
v_L2_cm, v_L2_mf = get_L2_errors(v_tx_c, v_tx_m, v_tx_f, hx_list, ht_list)
Q_v_log2_L2 = get_convergence_Q_log2(v_L2_cm, v_L2_mf)
Q_v_L2 = get_convergence_Q(v_L2_cm, v_L2_mf)

# Txx
Txx_L2_cm, Txx_L2_mf = get_L2_errors(Txx_tx_c, Txx_tx_m, Txx_tx_f, hx_list, ht_list)
Q_Txx_log2_L2 = get_convergence_Q_log2(Txx_L2_cm, Txx_L2_mf)
Q_Txx_L2 = get_convergence_Q(Txx_L2_cm, Txx_L2_mf)

# all 
Q_list_log2_L2 = [Q_T00_log2_L2, Q_T0x_log2_L2, Q_Txx_log2_L2, Q_ep_log2_L2, Q_v_log2_L2]
Q_list_L2 = [Q_T00_L2, Q_T0x_L2, Q_Txx_L2, Q_v_L2]

# plotting
plot_convergence(sim_f, problem, t_array, hx_list, ht_list, Q_list_L2, var_name_list, var_label_list, frame, etaovers, "L2")
plot_convergence(sim_f, problem, t_array, hx_list, ht_list, Q_list_log2_L2, var_name_list, var_label_list, frame, etaovers, "L2_log2")

# ---------

# L1 stuff
# ---------
# T00
T00_L1_cm, T00_L1_mf = get_L1_errors(T00_tx_c, T00_tx_m, T00_tx_f, hx_list, ht_list)
Q_T00_log2_L1 = get_convergence_Q_log2(T00_L1_cm, T00_L1_mf)
Q_T00_L1 = get_convergence_Q(T00_L1_cm, T00_L1_mf)

# T0x 
T0x_L1_cm, T0x_L1_mf = get_L1_errors(T0x_tx_c, T0x_tx_m, T0x_tx_f, hx_list, ht_list)
Q_T0x_log2_L1 = get_convergence_Q_log2(T0x_L1_cm, T0x_L1_mf)
Q_T0x_L1 = get_convergence_Q(T0x_L1_cm, T0x_L1_mf)

# ep
ep_L1_cm, ep_L1_mf = get_L1_errors(ep_tx_c, ep_tx_m, ep_tx_f, hx_list, ht_list)
Q_ep_log2_L1 = get_convergence_Q_log2(ep_L1_cm, ep_L1_mf)
Q_ep_L1 = get_convergence_Q(ep_L1_cm, ep_L1_mf)

# v
v_L1_cm, v_L1_mf = get_L1_errors(v_tx_c, v_tx_m, v_tx_f, hx_list, ht_list)
Q_v_log2_L1 = get_convergence_Q_log2(v_L1_cm, v_L1_mf)
Q_v_L1 = get_convergence_Q(v_L1_cm, v_L1_mf)

# Txx
Txx_L1_cm, Txx_L1_mf = get_L1_errors(Txx_tx_c, Txx_tx_m, Txx_tx_f, hx_list, ht_list)
Q_Txx_log2_L1 = get_convergence_Q_log2(Txx_L1_cm, Txx_L1_mf)
Q_Txx_L1 = get_convergence_Q(Txx_L1_cm, Txx_L1_mf)

# all 
Q_list_log2_L1 = [Q_T00_log2_L1, Q_T0x_log2_L1, Q_Txx_log2_L1, Q_ep_log2_L1, Q_v_log2_L1]
Q_list_L1 = [Q_T00_L1, Q_T0x_L1, Q_Txx_L1, Q_v_L1]

# plotting
plot_convergence(sim_f, problem, t_array, hx_list, ht_list, Q_list_L1, var_name_list, var_label_list, frame, etaovers, "L1")
plot_convergence(sim_f, problem, t_array, hx_list, ht_list, Q_list_log2_L1, var_name_list, var_label_list, frame, etaovers, "L1_log2")
# ---------
# =========================================================================