import numpy as np
import matplotlib.pyplot as plt
import bdnk as hydro

import os

from matplotlib.animation import PillowWriter
plt.rcParams['font.size'] = 13 
plt.rcParams["figure.figsize"] = (9,7)

import data_manager as data_m
import parameters as params

# def animate(sim_name, var_name, var_label, var_data):

#     linestyles1 = ['k-.', 'r', 'b', 'g', 'm', 'm:', 'y:']
#     linestyles2 = ['k', 'r-.', 'b--', 'g:', 'm-.', 'm-.', 'y:']

#     fig,ax = plt.subplots()

#     metadata = dict(title = 'Movie', artist = 'nclar')
#     writer = PillowWriter(fps = 15, metadata = metadata) # fps here

#     param_dict, post_dict, x, t_arr, qtx, ftx, stx = data_m.load_sim(sim_name, False)
#     print("making animations...")

#     theta_kt = param_dict["theta_kt"]
#     hx = param_dict["hx"] 
#     frame = param_dict["frame"]

#     anim1 = var_data
#     ylabel = var_label
#     ylabel0 = ylabel + " (t=0)"

#     with writer.saving(fig, data_m.path_anims(sim_name, var_name+".gif"), dpi=100):
            
#         # for i in range(0,len(t_arr), int(1/hx)): # play with last entry to skip time, speed up animations
#         for i in range(0,len(t_arr)):

#             ax.clear()
#             ax.plot(x, anim1[0], linestyles1[0], label = ylabel0)
#             ax.plot(x, anim1[i], linestyles1[2], label = ylabel)   

#             title = "Frame " + str(frame) + r", $t($1/GeV$) = $" + str('%.2f'%(t_arr[i]))

#             ax.set_xlabel(r"$x$ " + "(1/GeV)")
#             ax.set_ylabel(ylabel)
#             ax.set_xlim([x[0], x[-1]])

#             if params.anim_bounds == True:
#                 # adapative y boundaries
#                 if np.min(anim1[:]) < 0:
#                     ymin = (np.min(anim1[:]) + 0.2 * np.min(anim1[:]) )
#                 if np.min(anim1[:]) > 0:
#                     ymin = (np.min(anim1[:]) - 0.2 * np.min(anim1[:]) )
#                 if np.max(anim1[:]) > 0:
#                     ymax = (np.max(anim1[:]) + 0.2 * np.max(anim1[:]))
#                 if np.max(anim1[:]) <0:  
#                     ymax = (np.max(anim1[:]) - 0.2 * np.max(anim1[:]))

#                 if np.min(anim1[:]) == 0:
#                     # ymin = (np.min(anim1[:]) - 0.5 * np.abs(np.max(anim1[:]) ))
#                     ymid = (np.abs(np.min(anim1[:])) +  np.abs(np.max(anim1[:]) ))/2
#                     ymin = np.min(anim1[:]) - 0.2 *ymid

#                 if np.max(anim1[:]) == 0:
#                     # ymax = (np.max(anim1[:]) + 0.5 * np.abs(np.min(anim1[:]) ))
#                     ymid = (np.abs(np.min(anim1[:])) +  np.abs(np.max(anim1[:]) ))/2
#                     ymax = np.max(anim1[:]) + 0.2 * ymid



#                 ax.set_ylim( [ymin, ymax ] ) 

#             ax.set_title(title)
#             ax.legend()
#             ax.legend(loc = 'upper right')

#             writer.grab_frame()

#             plt.tight_layout()
        

#         plt.savefig(data_m.path_plots(sim_name, var_name+".pdf"))

#         print("animation for " + str(var_name) + " completed.")


#         # vars_animate, var_names = data_m.load_vars(sim_name, False)

#         # animator.animate_all(sim_name, var_names, vars_animate)

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np

def animate(sim_name, var_name, var_label, var_data):
    linestyles_anim = ['k-.', 'b']  # for the movie (initial + evolving variable)

    fig, ax = plt.subplots()

    metadata = dict(title='Movie', artist='nclar')
    writer = PillowWriter(fps=15, metadata=metadata)

    param_dict, post_dict, x, t_arr, qtx, ftx, stx = data_m.load_sim(sim_name, False)
    print("making animations...")

    hx = param_dict["hx"] 
    frame = param_dict["frame"]

    anim1 = var_data
    ylabel = var_label
    ylabel0 = ylabel + " (t=0)"

    # Frames we want to save on one final plot
    save_frames = [0, round(len(t_arr)*.5), round(len(t_arr)*.9)]
    save_times = [ f"{t_arr[save_frames[0]]:.2f}", f"{t_arr[save_frames[1]]:.2f}", f"{t_arr[save_frames[2]]:.2f}" ]

    # === Make the .gif ===
    with writer.saving(fig, data_m.path_anims(sim_name, var_name + ".gif"), dpi=100):
        for i in range(len(t_arr)):
            ax.clear()
            ax.plot(x, anim1[0], linestyles_anim[0], label=ylabel0)
            ax.plot(x, anim1[i], linestyles_anim[1], label=ylabel)   

            title = f"Frame {frame}, $t$ (1/GeV) = {t_arr[i]:.2f}"
            ax.set_xlabel(r"$x$ (1/GeV)")
            ax.set_ylabel(ylabel)
            ax.set_xlim([x[0], x[-1]])

            if params.anim_bounds:
                ymin = np.min(anim1[:])
                ymax = np.max(anim1[:])
                if ymin < 0:
                    ymin = ymin + 0.2*ymin
                else:
                    ymin = ymin - 0.2*np.abs(ymin)
                if ymax > 0:
                    ymax = ymax + 0.2*ymax
                else:
                    ymax = ymax - 0.2*np.abs(ymax)
                ax.set_ylim([ymin, ymax])

            ax.set_title(title)
            ax.legend(loc='upper right')

            writer.grab_frame()
            plt.tight_layout()

    # === After animation: make one combined plot ===
    fig_combined, axc = plt.subplots()

    linestyles = [':', '--', '-']   # dotted → dashed → solid
    alphas     = [0.3, 0.6, 1.0]   # lighter → darker

    for ls, alpha, fidx in zip(linestyles, alphas, save_frames):
        axc.plot(
            x, anim1[fidx],
            linestyle=ls, color='b', alpha=alpha,
            label=f"{ylabel} (t={t_arr[fidx]:.1f})"
        )

    axc.set_xlabel(r"$x$ (1/GeV)")
    axc.set_ylabel(ylabel)
    axc.legend()
    # axc.set_title(f"{var_name}: selected frames {save_frames}")

    plt.tight_layout()
    plt.savefig(data_m.path_plots(sim_name, var_name + ".pdf"), bbox_inches="tight")
    plt.close(fig_combined)

    print(f"animation for {var_name} completed, saved combined PDF with times (1/GeV): {save_times}.")


def animate_all(sim_name):
    vars_animate, var_names = data_m.load_vars(sim_name, False)
    for i in range(0, len(vars_animate), 2):
        animate(sim_name, var_names[i], vars_animate[i], vars_animate[i+1])

def animate_hydro_frames(sim1,sim2,sim3,simtype):

    # ====================================================

    # unpacking from sim 1
    param_dict1, post_dict1, x1, t_arr1, qtx1, ftx1, stx1 = data_m.load_sim(sim1, False)

    vars_animate1, var_names1 = data_m.load_vars(sim1, False)
    Kn_u_tx_1 = vars_animate1[5]
    Kn_T_tx_1 = vars_animate1[7]
    A_tx_1 = vars_animate1[9]
    Q_tx_1 = vars_animate1[11]
    # Qq_tx_1 = hydro.get_Qnorm_sqrt_post(Q_tx_1)

    a1_1 = param_dict1["a1"]
    a2_1 = param_dict1["a2"]
    frame1 = param_dict1["frame"]

    T00_tx_1 = qtx1[:,0]
    T0x_tx_1 = qtx1[:,1]
    Txx_tx_1  = ftx1[:,1]

    # sim1_frame_label = r"$a_1$ = "+str(a1_1) +r", $a_2$ = "+str(round(a2_1,2))
    sim1_frame_label = r"$a_1,a_2 $ = "+str(a1_1) +","+str(round(a2_1,3))

    # ~~~~~~~~~~~~~~~~~~~~~
    # these should be the same across ALL
    hx = param_dict1['hx']
    cplus = param_dict1["a"]
    etaovers = param_dict1['etaovers']
    epL = post_dict1["epL"]
    epR = post_dict1["epR"]
    vL = post_dict1["vL"]
    vR = post_dict1["vR"]
    # ~~~~~~~~~~~~~~~~~~~~~

    # unpacking from sim 2
    param_dict2, post_dict2, x2, t_arr2, qtx2, ftx2, stx2 = data_m.load_sim(sim2, False)

    vars_animate2, var_names2 = data_m.load_vars(sim2, False)
    Kn_u_tx_2 = vars_animate2[5]
    Kn_T_tx_2 = vars_animate2[7]
    A_tx_2 = vars_animate2[9]
    Q_tx_2 = vars_animate2[11]
    # Qq_tx_2 = hydro.get_Qnorm_sqrt_post(Q_tx_2)

    a1_2 = param_dict2["a1"]
    a2_2 = param_dict2["a2"]
    frame2 = param_dict2["frame"]

    T00_tx_2 = qtx2[:,0]
    T0x_tx_2 = qtx2[:,1]
    Txx_tx_2 = ftx2[:,1]

    sim2_frame_label = r"$a_1,a_2 $ = "+str(a1_2) +","+str(round(a2_2,2))

    # unpacking from sim 3
    param_dict3, post_dict3, x3, t_arr3, qtx3, ftx3, stx3 = data_m.load_sim(sim3, False)

    vars_animate3, var_names3 = data_m.load_vars(sim3, False)
    Kn_u_tx_3 = vars_animate3[5]
    Kn_T_tx_3 = vars_animate3[7]
    A_tx_3 = vars_animate3[9]
    Q_tx_3 = vars_animate3[11]
    # Qq_tx_3 = hydro.get_Qnorm_sqrt_post(Q_tx_3)

    a1_3 = param_dict3["a1"]
    a2_3 = param_dict3["a2"]
    frame3 = param_dict3["frame"]

    T00_tx_3 = qtx3[:,0]
    T0x_tx_3 = qtx3[:,1]
    Txx_tx_3 = ftx3[:,1]

    sim3_frame_label = r"$a_1,a_2 $ = "+str(a1_3) +","+str(round(a2_3,2))

    # categorizing by variables
    T00_ylabel = r"$T^{00}$ (GeV$^4$)"
    T0x_ylabel = r"$T^{0x}$ (GeV$^4$)"
    Txx_ylabel = r"$T^{xx}$ (GeV$^4$)"
    Kn_u_ylabel = r"Kn$_u$"
    Kn_T_ylabel = r"Kn$_T$"
    A_ylabel = r"$A/(\varepsilon + P)$"
    Q_ylabel = r"$\sqrt{Q_\mu Q^\mu}/(\varepsilon + P)$"
    # Qq_ylabel = r"$\sqrt{Q_\mu Q^\mu}/(\varepsilon + P)$"

    # var_labels = [T00_ylabel, T0x_ylabel, Txx_ylabel, Kn_u_ylabel, Kn_T_ylabel, A_ylabel, Q_ylabel, Qq_ylabel]

    T00_name = "T00(t,x)"
    T0x_name = "T0x(t,x)"
    Txx_name = "Txx(t,x)"
    Kn_u_name = "Kn_u(t,x)"
    Kn_T_name = "Kn_T(t,x)"
    A_name = "Anorm(t,x)"
    Q_name = "Qnorm_sqrt(t,x)"
    # Qq_name = "Qnorm_sqrt(t,x)"

    T00_anims = [T00_tx_1, T00_tx_2, T00_tx_3]
    T0x_anims = [T0x_tx_1, T0x_tx_2, T0x_tx_3]
    Txx_anims = [Txx_tx_1, Txx_tx_2, Txx_tx_3]
    Kn_u_anims = [Kn_u_tx_1, Kn_u_tx_2, Kn_u_tx_3]
    Kn_T_anims = [Kn_T_tx_1, Kn_T_tx_2, Kn_T_tx_3]
    A_anims = [A_tx_1, A_tx_2, A_tx_3]
    Q_anims = [Q_tx_1, Q_tx_2, Q_tx_3]
    # Qq_anims = [Qq_tx_1, Qq_tx_2, Qq_tx_3]

    frame_labels = [sim1_frame_label, sim2_frame_label, sim3_frame_label]

    # ====================================================

    path = "./hydroframes/"+simtype+"_frames="+str(frame1)+"," +str(frame2)+","+str(frame3)+"_etaovers="+str(round(etaovers,3))+"_ep=("+str(epL)+","+str(epR) \
        +")_v=("+str(vL)+","+str(vR)+ ")_hx="+str(hx)+"/"
    if not os.path.exists(path):
        os.makedirs(path)

    print("making animations...")

    frames = [frame1, frame2, frame3]

    # T00
    animate_and_save_3(path, T00_name, T00_anims, frame_labels, T00_ylabel, x1, t_arr1, hx, cplus, etaovers, epL, vL, frames)
    # T0x
    animate_and_save_3(path, T0x_name, T0x_anims, frame_labels, T0x_ylabel, x1, t_arr1, hx, cplus, etaovers, epL, vL, frames)
    # Txx
    animate_and_save_3(path, Txx_name, Txx_anims, frame_labels, Txx_ylabel, x1, t_arr1, hx, cplus, etaovers, epL, vL, frames)

    # Kn_u
    animate_and_save_3(path, Kn_u_name, Kn_u_anims, frame_labels, Kn_u_ylabel, x1, t_arr1, hx, cplus, etaovers, epL, vL, frames)
    # Kn_T
    animate_and_save_3(path, Kn_T_name, Kn_T_anims, frame_labels, Kn_T_ylabel, x1, t_arr1, hx, cplus, etaovers, epL, vL, frames)
    # A norm
    animate_and_save_3(path, A_name, A_anims, frame_labels, A_ylabel, x1, t_arr1, hx, cplus, etaovers, epL, vL, frames)
    # Q norm sqrt
    animate_and_save_3(path, Q_name, Q_anims, frame_labels, Q_ylabel, x1, t_arr1, hx, cplus, etaovers, epL, vL, frames)
    # # Q norm sqrt
    # animate_and_save_3(path, Qq_name, Qq_anims, frame_labels, Qq_ylabel, x1, t_arr1, hx, cplus, etaovers, epL, vL, frames)


def animate_and_save_3(path:str, name:str, anims:list, animlabels:list, var_label, x, t_arr, hx, cplus, etaovers, epL, vL, frames):

    anim1, anim2, anim3, = anims
    label1, label2, label3 = animlabels

    frame1, frame2, frame3 = frames

    linestyles1 = ['k-.', 'r', 'b', 'g', 'm', 'm:', 'y:']
    linestyles2 = ['k', 'r-.', 'b--', 'g:', 'm-.', 'm-.', 'y:']

    fig,ax = plt.subplots()

    metadata = dict(title = 'Movie', artist = 'nclar')
    writer = PillowWriter(fps = 15, metadata = metadata) # fps here

    # ANIMATION 
    # ========================================
    with writer.saving(fig, path+name+".gif", dpi=100):

        # for i in range(0,len(t_arr), int(1/hx)): # play with last entry to skip time, speed up animations
        for i in range(0,len(t_arr)):

            ax.clear()
            ax.plot(x, anim1[i], linestyles2[1], label = label1)  
            ax.plot(x, anim2[i], linestyles2[2], label = label2) 
            ax.plot(x, anim3[i], linestyles2[3], label = label3)  

            title = r"$t($1/GeV$) = $" + str('%.2f'%(t_arr[i]) + r", $c_+$ = " + str(cplus)) + r", $\eta/s$ = " +str(etaovers) \
                    + r", $\epsilon_{L}$ = "+str(epL) + r", $v_{L}$ = "+str(vL) + r"$h_x$ = " +str(hx)

            ax.set_xlabel(r"$x$ " + "(1/GeV)")
            ax.set_ylabel(var_label)
            ax.set_xlim([x[0], x[-1]])

            if params.anim_bounds == True:
                # adapative y boundaries
                if np.min(anim1[:]) < 0:
                    ymin = (np.min(anim1[:]) + 0.2 * np.min(anim1[:]) )
                if np.min(anim1[:]) > 0:
                    ymin = (np.min(anim1[:]) - 0.2 * np.min(anim1[:]) )
                if np.max(anim1[:]) > 0:
                    ymax = (np.max(anim1[:]) + 0.2 * np.max(anim1[:]))
                if np.max(anim1[:]) < 0:  
                    ymax = (np.max(anim1[:]) - 0.2 * np.max(anim1[:]))

                if np.min(anim1[:]) == 0:
                    # ymin = (np.min(anim1[:]) - 0.5 * np.abs(np.max(anim1[:]) ))
                    ymid = (np.abs(np.min(anim1[:])) +  np.abs(np.max(anim1[:]) ))/2
                    ymin = np.min(anim1[:]) - 0.5 *ymid

                if np.max(anim1[:]) == 0:
                    # ymax = (np.max(anim1[:]) + 0.5 * np.abs(np.min(anim1[:]) ))
                    ymid = (np.abs(np.min(anim1[:])) +  np.abs(np.max(anim1[:]) ))/2
                    ymax = np.max(anim1[:]) + 0.5 * ymid

                ax.set_ylim( [ymin, ymax ] )
                
            if params.anim_zoom == True:
                ymin = 0
                ymax = 0.5
                
                ax.set_ylim( [ymin, ymax ] )

            ax.set_title(title)
            ax.legend(prop={"size": 10}, loc = 'upper right')

            writer.grab_frame()

            # ========================================

            if i == 0:
                anim1_0 = anim1[i]
                anim2_0 = anim2[i]
                anim3_0 = anim3[i]
                t_0 = round(t_arr[i],2)


            if i == 40:
                anim1_mid = anim1[i]
                anim2_mid = anim2[i]
                anim3_mid = anim3[i]
                t_mid = round(t_arr[i],2)

            if i == 140:
                anim1_end = anim1[i]
                anim2_end = anim2[i]
                anim3_end = anim3[i]
                t_end = round(t_arr[i],2)

        # PLOTTING 
        # -------------------
        fig,ax = plt.subplots(1)
        # plt.tight_layout()

        # initial time
        ax.plot(x, anim1_0, 'r:', alpha = 0.25) #, label = label1)  
        ax.plot(x, anim2_0, 'k:', alpha = 0.25) #, label = label2) 
        ax.plot(x, anim3_0, 'c:', alpha = 0.25) #, label = label3)

        # middle time
        ax.plot(x, anim1_mid, 'r-.', alpha = 0.5) #, label = label1)  
        ax.plot(x, anim2_mid, 'k-.', alpha = 0.5) #, label = label2) 
        ax.plot(x, anim3_mid, 'c-.', alpha = 0.5) # , label = label3)

        # end time

        ax.plot(x, anim1_end, 'r--', alpha = 0.9, label = r"$F_1$")  
        ax.plot(x, anim2_end, 'k--', alpha = 0.9, label = r"$F_2$") 
        ax.plot(x, anim3_end, 'c--', alpha = 0.9, label = r"$F_3$")

        ax.set_xlabel(r"$x$ " + "(1/GeV)")
        ax.set_ylabel(var_label)
        ax.set_xlim([x[0], x[-1]])
        ax.legend()
        

        #plt.title("t = ("+str(t_0) + ", " + str( t_mid) + ", " + str(t_end) + ") (1/GeV)" )
        print("snapshots of t = ("+str(t_0) + ", " + str( t_mid) + ", " + str(t_end) + ") (1/GeV)")
        path2 = path+"/paper/"
        if not os.path.exists(path2):
            os.makedirs(path2)
        plt.savefig(path2+name+".pdf")
        # -------------------


        times1, anims1 = plot_from_anim(anim1_0, anim1_mid, anim1_end, t_0, t_mid, t_end, x, var_label, path, name, label1)
        times2, anims2 = plot_from_anim(anim2_0, anim2_mid, anim2_end, t_0, t_mid, t_end, x, var_label, path, name, label2)
        times3, anims3 = plot_from_anim(anim3_0, anim3_mid, anim3_end, t_0, t_mid, t_end, x, var_label, path, name, label3)

        hydro_frame_norm(x, times1, anims1, anims2, path, name, var_label)

        print("animation for " + name + " " + str(var_label) + " completed.")

def plot_from_anim(anim_0, anim_mid, anim_end, t_0, t_mid, t_end, x, var_label, path, name, label):
        
        fig,ax = plt.subplots()

        ax.plot(x, anim_0, 'b:', alpha = 0.25, label = r"$t =$ "+str(t_0)+" (1/GeV)", )
        ax.plot(x, anim_mid, 'b-.', alpha = 0.5, label = r"$t =$ "+str(t_mid)+" (1/GeV)")
        ax.plot(x, anim_end, 'b', alpha = 1.0, label = r"$t =$ "+str(t_end)+" (1/GeV)")

        ax.set_xlabel(r"$x$ " + "(1/GeV)")
        ax.set_ylabel(var_label)
        ax.set_xlim([x[0], x[-1]]) 

        # ax.legend(fontsize='small')
        ax.legend(prop={"size": 10}, loc = 'upper right')

        plt.savefig(path+name+label+".pdf")

        times = [t_0, t_mid, t_end]
        anims = [anim_0, anim_mid, anim_end]
        return times, anims

def hydro_frame_norm(x, times, anims1, anims2, path, name, var_label):
    
        t_0, t_mid, t_end = times

        # delta = |(Tmn(a_1-small) - Tmn(a_1-big))/Tmn(a_1-small)|

        delta_0 = np.abs((anims1[0] - anims2[0]))
        delta_mid = np.abs((anims1[1] - anims2[1]))
        delta_end = np.abs((anims1[2] - anims2[2]))

        fig,ax = plt.subplots()

        ax.plot(x, delta_0, 'b:', alpha = 0.25, label = r"$t =$ "+str(t_0)+" (1/GeV)", )
        ax.plot(x, delta_mid, 'b-.', alpha = 0.5, label = r"$t =$ "+str(t_mid)+" (1/GeV)")
        ax.plot(x, delta_end, 'b', alpha = 1.0, label = r"$t =$ "+str(t_end)+" (1/GeV)")

        ax.set_xlabel(r"$x$ " + "(1/GeV)")
        ax.set_ylabel(r"$\delta$"+var_label) #* FIX UNITS
        ax.set_xlim([x[0], x[-1]]) 

        # ax.legend(fontsize='small')
        ax.legend(prop={"size": 10}, loc = 'upper right')

        plt.savefig(path+name+"delta-in-time.pdf")


def plot_from_animate(i:int, t_arr:np.array, x:np.array, anim:np.array, label:str, var_label:str, path:str, name:str):

    if i == 0:
        anim_0 = anim[i]
        t_0 = round(t_arr[i],2)


    if i == 40:
        anim_mid = anim[i]
        t_mid = round(t_arr[i],2)

    if i == 140:
        anim_end = anim[i]
        t_end = round(t_arr[i],2)

        # -------------------
        fig,ax = plt.subplots(1)
        # plt.tight_layout()

        # initial time
        ax.plot(x, anim_0, 'b:', alpha = 0.25, label = label)  

        # middle time
        ax.plot(x, anim_mid, 'b-.', alpha = 0.5, label = label)  

        # end time
        ax.plot(x, anim_end, 'b--', alpha = 0.9, label = label)  

        ax.set_xlabel(r"$x$ " + "(1/GeV)")
        ax.set_ylabel(var_label)
        ax.set_xlim([x[0], x[-1]])


        plt.title("t = ("+str(t_0) + ", " + str( t_mid) + ", " + str(t_end) + ") (1/GeV)" )
        plt.savefig(path+name+".pdf")

# sim_like_princeton = 
# def compare_princeton():

#     princeton_path = "./princeton/datafiles"


# if params.sucomp == False:
#     sim_name = data_m.load_check("sims")
#     vars, var_names = data_m.load_vars(sim_name, False)
