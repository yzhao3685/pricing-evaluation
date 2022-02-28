#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import set_up#our codes
import methods
from warnings import simplefilter
simplefilter(action='ignore')# ignore all warnings
smoothness, jump, exploration, randomness = 5, 0.25, 0.6, 0.6
setting = 'setting3'
policy_type,demand_type,logging_type,feature_dist = [setting,randomness],[setting,smoothness,jump],[setting,exploration],'uniform'

feature_dim,train_size,rndseed,sigma = 2,50,int(1000000*(time.time()%1)),0.2#bandwidth of kernels
numSeeds=10
t_start=time.time()
outfile = "n_{}_N_{}_seed_{}.txt".format(str(train_size),str(numSeeds),str(rndseed))
with open(outfile, "w") as f:
    f.write("simulated MSE: \n")
    f.write("LASSO,logistic,RF,NW,NW(sigma 0.2),Nathan,ours, ours(oracle base):\n")
#allocate memory
n_methods = 8
mse_arr = (np.zeros((numSeeds,n_methods,3)))
run_time_arr = (np.zeros((numSeeds,n_methods)))
time_arr = np.zeros(n_methods+1)
gamma=5
for i in range(numSeeds):
    print('\n')
    print('iterate :',i)
    with open(outfile, "a") as f:
        time_elapsed=int(round(time.time()-t_start))
        f.write(" \n")
        f.write("run %d time_elapsed(secs) %d \n"%(i,time_elapsed))
    rndseed=rndseed+i*10
    Train_data = set_up.prepare_data(rndseed,train_size,policy_type,demand_type,logging_type,feature_dist,feature_dim,sigma)
    #heuristics_lambda,heuristics_gamma = methods.nathans_heuristics(Train_data)
    #print('heuristics lambda, gamma: ',heuristics_lambda,heuristics_gamma)
    #predict then optimize methods
    time_arr[0]=time.time()
    mse_arr[i][0] = methods.simulate_reg_oracle_MSE(Train_data,'linear')
    time_arr[1]=time.time()
    mse_arr[i][1] = methods.simulate_reg_oracle_MSE(Train_data,'logistic')
    time_arr[2]=time.time()
    mse_arr[i][2] = methods.simulate_rf_MSE(Train_data,5,0)
    time_arr[3]=time.time()
    
    w_NW,optimal_sigma = methods.kernel_regression_oracle(Train_data)
    time_arr[4]=time.time()
    w_NW_vanilla = methods.kernel_regression(Train_data,sigma)
    time_arr[5]=time.time()
    w_nathan,optimal_lambda = methods.nathans_method_oracle(Train_data)
    time_arr[6]=time.time()
    
    hat_r_z, hat_r_y = methods.kernel_regression_r_z(Train_data,sigma)
    w_ours = methods.our_method_DR(Train_data,gamma,[],hat_r_z, hat_r_y)
    
    time_arr[7]=time.time()
    hat_r_z, hat_r_y = methods.kernel_regression_r_z(Train_data,optimal_sigma)#oracle base reg
    w_ours_oracle = methods.our_method_DR(Train_data,1,[],hat_r_z, hat_r_y)
    time_arr[8]=time.time()
    run_time_arr[i]=time_arr[1:]-time_arr[:n_methods]

    mse_arr[i][3] = methods.simulate_MSE(Train_data,w_NW)
    mse_arr[i][4] = methods.simulate_MSE(Train_data,w_NW_vanilla)
    mse_arr[i][5] = methods.simulate_MSE(Train_data,w_nathan)
    mse_arr[i][6] = methods.simulate_MSE(Train_data,w_ours)
    mse_arr[i][7] = methods.simulate_MSE(Train_data,w_ours_oracle)
    print('our method simulated mse: ',mse_arr[i][6])
    print('our method (oracle base reg) simulated mse: ',mse_arr[i][7])

    with open(outfile, "a") as f:
        [f.write('{:.2f} '.format(item)) for item in mse_arr[i,:,0]]
        f.write("lambda*,sigma* %.4f %.4f \n" % (
            optimal_lambda,optimal_sigma))

time_elapsed=int(round(time.time()-t_start))
with open(outfile, "a") as f:
    f.write("\n")
    f.write("LASSO,logistic,RF,NW,NW(sigma 0.2),Nathan,ours, ours(oracle base):\n")
    [f.write('{:.2f} '.format(item)) for item in np.average(mse_arr[:,:,0],axis=0)]
    f.write("avg simulated MSE \n")
    [f.write('{:.2f} '.format(item)) for item in np.average(mse_arr[:,:,1],axis=0)]
    f.write("avg simulated bias^2 \n")
    [f.write('{:.2f} '.format(item)) for item in np.average(mse_arr[:,:,2],axis=0)]
    f.write("avg simulated variance \n")

    f.write(" \n")
    f.write("avg runtime: \n")
    [f.write('{:.3f} '.format(item)) for item in np.average(run_time_arr,axis=0)]
    
    f.write("total runtime: \n")
    f.write("%d \n" % (time_elapsed))
    
