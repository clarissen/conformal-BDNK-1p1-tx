import numpy as np
import os 
import json 
import sys


# PRECISION
# -------------------------------------
def upgrade_precision_if_needed(func):
    def wrapper(*args, **kwargs):
        new_args = []
        # Get float64 limits for later use
        finfo64 = np.finfo(np.float64)
        
        for arg in args:
            # Check if the argument is a numpy array of type float64
            if isinstance(arg, np.ndarray) and arg.dtype == np.float64:
                # Check if any element exceeds the maximum or is below the minimum (tiny) positive normalized value
                if (np.any(np.abs(arg) > finfo64.max) or 
                    np.any((np.abs(arg) < finfo64.tiny) & (arg != 0))):
                    # print("Upgrading an array from float64 to float128 due to precision limits")
                    arg = arg.astype(np.float128)
            new_args.append(arg)
        
        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray) and value.dtype == np.float64:
                if (np.any(np.abs(value) > finfo64.max) or 
                    np.any((np.abs(value) < finfo64.tiny) & (value != 0))):
                    # print(f"Upgrading keyword argument '{key}' from float64 to float128 due to precision limits")
                    value = value.astype(np.float128)
            new_kwargs[key] = value
        
        return func(*new_args, **new_kwargs)
    
    return wrapper
# -------------------------------------

# CHECKS AND SAVES
# -------------------------------------
def sim_check(specs, sucomp):
    print("you are about to run a sim with specifications...")
    print("-------------")
    print(specs)
    print("-------------")
    if sucomp == True:
        return
    else:
        inp = input("is this correct? type y to continue or n to stop: ")
        if inp == "y":
            pass
        else: 
            sys.exit("code stopped.") 
    

def save_check(dir, unique, sucomp):
    print("saving data in /"+ str(dir) +"/ directory. ")
    if sucomp == True:
        return unique
    else:
        autonaming = input("auto-generate name? y or n: ")
        if autonaming == "y":
            return unique
        else:
            return input('please name the /' + str(dir) + "/ folder: ")

    # inp = input("save data in " + str(dir) + " directory? ")
    # if inp == "n":
    #     pass
    # else:
    #     name = input('please name the ' + str(dir) + "folder: ")
    #     return name

def save_single(file, file_name):
    path = "./tests/"
    # creates dir if dir not made
    if not os.path.exists(path):
        os.makedirs(path)

    np.save(path + file_name, file)
    np.savetxt(path + file_name, file)
    # with open(path+file_name, 'w') as f:
    #     for line in file:
    #         f.write(str(line))
    #         f.write("\n")

    print("file saved under: " + path + file_name)

# saves all necessary simulation objects
def save_sim(sim_name, jsonfiles, npyfiles, txtfiles):
    path = "./sims/"
    new_path = path + sim_name
    # creates dir if dir not made
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    save_loc = new_path + "/"

    # using json, save dicts
    for i in range(0,len(jsonfiles),2):
        with open(save_loc + jsonfiles[i], 'w') as f:
                json.dump(jsonfiles[i+1], f, indent=4)

    # saving npy files
    for i in range(0,len(npyfiles),2): 
        np.save(save_loc + npyfiles[i], npyfiles[i+1])
    
    # saving txtfiles
    for i in range(0,len(txtfiles),2):
         with open(save_loc + txtfiles[i], 'w') as f:
              for line in txtfiles[i+1]:
                   f.write(str(line))
                   f.write("\n")

    print("sim files saved under: " + save_loc)

def save_npy_list(sim_name, npyfiles, dir):

    save_loc = "./sims/" + sim_name + "/" + dir +"/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    for i in range(0,len(npyfiles),2): 
        np.save(save_loc + npyfiles[i], npyfiles[i+1])

    print("vars files saved under: " + save_loc )

def path_anims(sim_name, anim_file):
    save_loc = "./sims/" + sim_name + "/anims/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    return save_loc + anim_file

def path_plots(sim_name, plot_file):
    save_loc = "./sims/" + sim_name + "/anims/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    return save_loc + plot_file
    
# -------------------------------------


# POST PROCESSING
# -------------------------------------

def load_check(dir):
    print("loading data from /sim/ directory. ")
    name = input('please give the name of the /' + str(dir) + "/ folder: ")
    return name

def load_sim(sim_name, printt):
    path = "./sims/"
    load_loc = path + sim_name + "/"
    if printt == True:
        print("loading files from " + load_loc)
        specs = np.load(load_loc + "specs.npy")
        version = np.load(load_loc + "version.npy")
        print("-------------")
        print(specs)
        print("version: ", version)
        print("-------------")

    with open(load_loc+"param_dict.json", 'r') as f:
        param_dict = json.load(f)
    with open(load_loc+"post_dict.json", 'r') as f:
        post_dict = json.load(f)

    x = np.load(load_loc + "x.npy")
    t_arr = np.load(load_loc + "t_arr.npy")
    qtx = np.load(load_loc + "qtx.npy")
    ftx = np.load(load_loc + "ftx.npy")
    stx = np.load(load_loc + "stx.npy")

    return param_dict, post_dict, x, t_arr, qtx, ftx, stx

def load_vars(sim_name, print):
    path = "./sims/"
    load_loc_sims = path + sim_name + "/"
    load_loc = path + sim_name + "/vars/"
    if print == True:
        print("loading files from " + load_loc_sims)
        specs = np.load(load_loc_sims + "specs.npy")
        version = np.load(load_loc_sims + "version.npy")
        print("-------------")
        print(specs)
        print("version: ", version)
        print("-------------")

    var_files = ["T00(t,x).npy", "T0x(t,x).npy", "Kn_u(t,x).npy", "Kn_T(t,x).npy", "Anorm(t,x).npy", "Qnorm_sqrt(t,x).npy", \
                 
                 "Txx(t,x).npy", "ep(t,x).npy", "depdx(t,x).npy", "v(t,x).npy", "dvdx(t,x).npy", "P(t,x).npy", \
                
                 "C_0(t,x).npy", "C_x(t,x).npy", "X_xx(t,x).npy", "X_x0(t,x).npy", "X_00(t,x).npy", "X_0x(t,x).npy", "u0(t,x).npy", ]
    
    T00_tx = np.load(load_loc + var_files[0])
    T0x_tx = np.load(load_loc + var_files[1])
    Kn_u_tx = np.load(load_loc + var_files[2])
    Kn_T_tx = np.load(load_loc + var_files[3])
    Anormtx = np.load(load_loc + var_files[4])
    Qnormsqrttx = np.load(load_loc + var_files[5])
    Txx_tx = np.load(load_loc + var_files[6])
    eptx = np.load(load_loc + var_files[7])
    depdx_tx = np.load(load_loc + var_files[8])
    vtx = np.load(load_loc + var_files[9])
    dvdx_tx = np.load(load_loc + var_files[10])
    Ptx = np.load(load_loc + var_files[11])
    C_0_tx = np.load(load_loc + var_files[12])
    C_x_tx = np.load(load_loc + var_files[13])
    X_xx_tx = np.load(load_loc + var_files[14])
    X_x0_tx = np.load(load_loc + var_files[15])
    X_00_tx = np.load(load_loc + var_files[16])
    X_0x_tx = np.load(load_loc + var_files[17])
    u0_tx = np.load(load_loc + var_files[18])



    vars_animate = [r"$T^{00}$ (GeV$^4$)", T00_tx, r"$T^{0x}$ (GeV$^4$)", T0x_tx, r"Kn_$u$", Kn_u_tx, r"Kn_$T$", Kn_T_tx, 
                    
                    r"$A/(\varepsilon + P)$", Anormtx, r"$\sqrt{Q_\mu Q^\mu}/(\varepsilon + P)$", Qnormsqrttx,
                
                    r"$T^{xx}$ (GeV$^4$)", Txx_tx, r"$\epsilon$ (GeV$^4$)", eptx, \
                    
            r"$\partial_x \epsilon $", depdx_tx, r"$v $", vtx, r"$\partial_x v $", dvdx_tx, r"$P$ (GeV$^4$)", Ptx, \
            # more variables
            r"$C_0$", C_0_tx, r"$C_x$", C_x_tx, r"$X_{xx}$", X_xx_tx, r"$X_{x0}$", X_x0_tx, r"$X_{00}$", X_00_tx, r"$X_{0x}$", X_0x_tx, r"$u^0$", u0_tx ]
    
    var_names = ["T00(t,x)", T00_tx, "T0x(t,x)", T0x_tx, "Kn_u(t,x)", Kn_u_tx, "Kn_T(t,x)", Kn_T_tx, 
            "Anorm(t,x)", Anormtx, "Qnorm_sqrt(t,x)", Qnormsqrttx,
            
            "Txx(t,x)", Txx_tx, "ep(t,x)", eptx, \
            "depdx(t,x)", depdx_tx, "v(t,x)", vtx, "dvdx(t,x)", dvdx_tx, "P(t,x)", Ptx, \
                
            # more variables
            "C_0(t,x)", C_0_tx, "C_x(t,x)", C_x_tx, "X_xx(t,x)", X_xx_tx, "X_x0(t,x)", X_x0_tx,
    
             "X_00(t,x)", X_00_tx, "X_0x(t,x)", X_0x_tx, "u0(t,x)", u0_tx ]
    
    return vars_animate, var_names

    
# -------------------------------------