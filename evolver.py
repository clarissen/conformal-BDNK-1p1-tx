import numpy as np
import time

# home made modules
import parameters as params
import data_manager as data_m
import kurganovtadmor as kt
import bdnk as hydro
import post 
import animator
import config


class Evolve:

    

    #ideas
    # parallelize?

    # attributes
    # *****************************
    def __init__(self, it, q0):

        self.it = it
        # initial data
        self.q0 = q0
        self.f0 = hydro.get_flux(self.q0, self.it)
        # self.f0 = hydro.force_flux_init(self.f0) # MAKE SURE THIS IS OK
        self.s0 = hydro.get_src(self.q0, self.it)
        # self.s0 = hydro.force_src_init(self.s0) # MAKE SURE THIS IS OK


        # class wide parameters
        self.Km2, self.Km1, self.K, self.Kp1, self.Kp2 = params.indexing
        self.hx = params.hx
        self.ht = params.ht
        self.iterations = params.iterations
        self.t0 = params.t0
        self.theta_kt = params.theta_kt
        self.cs = params.cs
        self.cplus = params.cplus

        # pre-allocating memory to array objects that are evolved many times
        self.q = q0.copy()
        self.qa = np.zeros(q0.shape)
        self.qb = np.zeros(q0.shape)

        self.qpKp1 = np.zeros(q0[:,self.K].shape)
        self.qmKp1 = np.zeros(q0[:,self.K].shape)
        self.qpKm1 = np.zeros(q0[:,self.K].shape)
        self.qmKm1 = np.zeros(q0[:,self.K].shape)

        self.HL = np.zeros(q0[:,self.K].shape)
        self.HR = np.zeros(q0[:,self.K].shape)

        self.Zq = np.zeros(q0[:,self.K].shape)

        # any other things to pre-allocate?

    # *****************************

    # methods
    # *****************************

    # initializing the fluid state
    def start(self):

        # checking sim config before running
        data_m.sim_check(params.specs, params.sucomp)

        # get the simulation folder name
        sim_name = data_m.save_check("sim", params.unique, params.sucomp)

        print("-------CODE START-------")
        print("========================")

        return sim_name

        

    # saving sim data for post-processing
    def finish(self, sim_name, jsonfiles, npyfiles, txtfiles):

        data_m.save_sim(sim_name, jsonfiles, npyfiles, txtfiles)

        print("======================")

        for i in txtfiles[1]:
            print(i)
        for i in txtfiles[3]:
            print(i)

        print("======================")
        print("-------CODE END-------")

    def alarms(self, on, q, t, en_change0, mom_change0, en_change_time, mom_change_time):

        alarm_stop = False

        if on == True:

            en = q[0].sum()*self.hx - (self.q0[0].sum()*self.hx - ( self.q0[1][-1] - self.q0[1][0] )*t )
            mom = q[1].sum()*self.hx - (self.q0[1].sum()*self.hx - (self.f0[1][-1] - self.f0[1][0])* t )

            en_change = np.max(np.abs(en_change0 - en))
            mom_change = np.max(np.abs(mom_change0 - mom))
            if (en_change > 1e-4 ) or (mom_change > 1e-4):
                alarm_stop = True
                print("change in total energy or momentum exceeded 1e-4.")


            print('energy change: ', en_change)
            print('momentum change: ', mom_change )
            print('--------')

            en_change_time.append([en_change, t])
            mom_change_time.append([mom_change, t])

            return en_change_time, mom_change_time, alarm_stop
        
        else:
            return en_change_time, mom_change_time, alarm_stop

    def clock(self,i, t,start_time):
        # time keeping
        print('--------')
        print('time (min) so far = ' + str((time.time()- start_time)/60) )
        print('iteration = ' + str(i) + ', time (1/GeV) = ' + str(t) )
        print('--------')
        return i

    # numerical method
    # Kurganov-Tadmor (spatial) with 2nd order Runge-Kutta Total Variation Diminishing
    def kt_tvdrk2(self, alarms):

        start_time = time.time()

        # calling the allocated memory as new objects to make the code less verbose
        Km2, Km1, K, Kp1, Kp2 = self.Km2, self.Km1, self.K, self.Kp1, self.Kp2
        q, qa, qb = self.q, self.qa, self.qb
        f0, s0 = self.f0, self.s0

        # print("q(t=0)[0,240:260] = ", q[0,240:260])

        qpKp1, qmKp1, qpKm1, qmKm1 = self.qpKp1, self.qmKp1, self.qpKm1, self.qmKm1
        HL, HR = self.HL, self.HR
        Zq = self.Zq
        
        # energy and momentum monitoring
        en_change0 = q[0].sum()*self.hx - (self.q0[0].sum()*self.hx - ( self.q0[1][-1] - self.q0[1][0] )*self.t0 )
        mom_change0 = q[1].sum()*self.hx - (self.q0[1].sum()*self.hx - (f0[1][-1] - f0[1][0])*self.t0 )
        en_change_time = [ [en_change0, self.t0]]
        mom_change_time = [ [mom_change0, self.t0]]

        # lists that will be appended to for data storage 
        t_arr = [self.t0]
        qtx = [self.q0]
        ftx = [f0]
        stx = [s0]

        # setting up iterations and time keeping
        i = self.it
        t = self.t0 

        while i < self.iterations:

            # BEGIN TIME STEPPING 
            #--------------------------------------
            # RK stage 1
            # ===========
            # generating the spatial residual Zq (RHS of conservation law)
            qpKp1 = kt.qp(q[:, K]  , q[:, Kp1], q[:, Kp2], self.theta_kt, self.hx)     # q^{+}_{i+1/2}
            qmKp1 = kt.qm(q[:, Km1], q[:, K]  , q[:, Kp1], self.theta_kt, self.hx)     # q^{-}_{i+1/2}
            qpKm1 = kt.qp(q[:, Km1], q[:, K]  , q[:, Kp1], self.theta_kt, self.hx)     # q^{+}_{i-1/2}
            qmKm1 = kt.qm(q[:, Km2], q[:, Km1], q[:, K  ], self.theta_kt, self.hx)     # q^{-}_{i-1/2}           		
            HL = kt.Hflux(qmKm1, qpKm1, self.cplus, i) # H_{i-1/2} 
            HR = kt.Hflux(qmKp1, qpKp1, self.cplus, i) # H_{i+1/2}
            # Zq = - (HR - HL) / self.hx  + hydro.get_src(q[:,K], i)
            Zq = - (HR - HL) / self.hx  + hydro.get_src(q, i)[:,K]
            
            # update equation for RK stage 1 (LHS of conservation law)
            qa[:,K] = q[:,K] + self.ht * Zq
            # UPDATE ghost cells using Neumann boundary conditions
            qa[:,-1] = qa[:,-2] = qa[:,-3]
            qa[:,0] = qa[:,1] = qa[:,2]

            # advancing time
            # ta = t + self.ht

            # RK stage 2
            # ===========
            # generating the spatial residual, Zq, from qa
            qpKp1 = kt.qp(qa[:, K]  , qa[:, Kp1], qa[:, Kp2], self.theta_kt, self.hx)     # q^{+}_{i+1/2}
            qmKp1 = kt.qm(qa[:, Km1], qa[:, K]  , qa[:, Kp1], self.theta_kt, self.hx)     # q^{-}_{i+1/2}
            qpKm1 = kt.qp(qa[:, Km1], qa[:, K]  , qa[:, Kp1], self.theta_kt, self.hx)     # q^{+}_{i-1/2}
            qmKm1 = kt.qm(qa[:, Km2], qa[:, Km1], qa[:, K  ], self.theta_kt, self.hx)     # q^{-}_{i-1/2}           		
            HL = kt.Hflux(qmKm1, qpKm1, self.cplus, i) # H_{i-1/2} 
            HR = kt.Hflux(qmKp1, qpKp1, self.cplus, i) # H_{i+1/2}
            # Zq = - (HR - HL) / self.hx  + hydro.get_src(qa[:,K], i)
            Zq = - (HR - HL) / self.hx  + hydro.get_src(qa, i)[:,K]

            #update formula RK stage 2
            qb[:, K] = 0.5*q[:, K] + 0.5*qa[:, K] + 0.5*self.ht * Zq
            # UPDATE ghost cells using Neumann boundary conditions
            qb[:,-1] = qb[:,-2] = qb[:,-3]
            qb[:,0] = qb[:,1] = qb[:,2]

            # advancing time, iterations
            i += 1
            t += self.ht
            # update q with last stage RK qb
            q = qb

            # storing data
            if i % config.tmod == 0:
                t_arr.append(t) # storing time
                qtx.append(q.copy()) # storing q(t,x) !! used to be qb
                ftx.append(hydro.get_flux(q, i))
                stx.append(hydro.get_src(q, i))

            # ALARMS AND CHECKS
            if alarms == True:
                if i % 10 == 0:
                    j = self.clock(i,t,start_time)
                    en_change_time, mom_change_time, alarm_stop = self.alarms(alarms, q, t, en_change0, mom_change0, en_change_time, mom_change_time)
                    if alarm_stop == True:
                        break
            else:
                pass

            
            #--------------------------------------
            # END TIME STEPPING 

        end_time = time.time()

        # list of objects that will be saved as <file>.json
        jsonfiles = ["param_dict.json", params.param_dict, "post_dict.json", params.post_dict]
        # list of objects that will be saved as <file>.npy
        npyfiles = ["qtx.npy", np.array(qtx), "ftx.npy", np.array(ftx), "stx.npy", np.array(stx), "t_arr.npy", np.array(t_arr), "x.npy", params.x, "version.npy", params.version, "specs.npy", params.specs]
        # list of objects that will be saved as <file>.txt
        time_keep = [f'Simulation start time = {self.t0} 1/GeV', f'Simulation end time = {t} 1/GeV', f'achieving a maximum number of iterations =  {i}', \
                    f'code timer (s) = {time.time()- start_time}',f'code timer (min) = {(end_time- start_time)/60}', f'code time/iteration (s) = {(time.time()- start_time)/i}']
        txtfiles = ["sim_config.txt", params.sim_config, "time_keep.txt", time_keep, "en_change_time.txt", en_change_time, "mom_change_time.txt", mom_change_time]

        return jsonfiles, npyfiles, txtfiles
        

    # this runs everything necessary
    def execute(self, alarms):

        sim_name = self.start()

        jsonfiles, npyfiles, txtfiles = self.kt_tvdrk2(alarms)

        self.finish(sim_name, jsonfiles, npyfiles, txtfiles)

        post.calculate_vars(sim_name)

        animator.animate_all(sim_name)


    def memory(self):

        print("self.K memory: ", id(self.K))
        K = self.K
        print("K = self.K memory: ", id(K))

        print("self.q0 memory: ", id(self.q0))
        print("self.q memory: ", id(self.q))
        q = self.q
        print("q = self.q memory: ", id(q))



        print("self.qb memory: ", id(self.qb))
        qb = self.qb
        print("qb = self.qb memory: ", id(qb))
        q = qb
        print("q = qb memory: ", id(q))

        print("self.q memory: ", id(self.q))
        print("q memory: ", id(q))



    # *****************************