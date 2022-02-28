#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import set_up#our codes
import methods
from warnings import simplefilter
simplefilter(action='ignore')# ignore all warnings
smoothness, jump, exploration, randomness = 5, 0.25, 0.6, 0.6
setting = 'setting1'
policy_type,demand_type,logging_type,feature_dist = [setting,randomness],[setting,smoothness,jump],[setting,exploration],'uniform'
feature_dim,train_size,rndseed,sigma = 2,30,int(1000000*(time.time()%1)),0.2#bandwidth of kernels

t_start=time.time()
Gamma=1

Train_data = set_up.prepare_data(rndseed,train_size,policy_type,demand_type,logging_type,feature_dist,feature_dim,sigma)
_,_,_,hat_r_z_arr,hat_r_y_arr = methods.simulate_linear_MSE(Train_data,3,1)
hat_r_z,hat_r_y = methods.truncate_r(Train_data,hat_r_z_arr[0],hat_r_y_arr[0])

w2,obj2,time2,grad_arr,blackbox_obj_arr = methods.our_method_DR_keep_record(Train_data,Gamma,[],hat_r_z, hat_r_y)

w1,obj1,time1,iterates_quad_cut,surrogate_iterates_quad_cut = methods.our_method_DR_quadratic_cut(
    Train_data,Gamma,hat_r_z,hat_r_y,brute_force=0,returnDetails=2)

w3,obj3,time3,iterates_linear_cut,surrogate_iterates_linear_cut = methods.our_method_DR_linear_cut(
    Train_data,Gamma,hat_r_z,hat_r_y,brute_force=0,returnDetails=2)


cap = blackbox_obj_arr[-1]*2
from matplotlib import pyplot#obj iterates all together 
pyplot.plot(np.arange(len(iterates_quad_cut)), np.minimum(iterates_quad_cut,cap),label='quadratic cut')
pyplot.plot(np.arange(len(blackbox_obj_arr)), np.minimum(blackbox_obj_arr,cap),label='gradient method')
pyplot.plot(np.arange(len(iterates_linear_cut)), np.minimum(iterates_linear_cut,cap),label='linear cut')
title='wc mse of iterates'
pyplot.title(title)
pyplot.xlabel('num of iterates')
pyplot.ylabel('wc mse')
pyplot.legend(loc='best')
pyplot.show()


grad_norm = np.linalg.norm(grad_arr,axis=1)
from matplotlib import pyplot
pyplot.plot(np.arange(len(blackbox_obj_arr)), grad_norm,label='grad norm')
title='iterates of grad norm for gradient method'
pyplot.title(title)
pyplot.xlabel('num of iterates')
pyplot.ylabel('norm of grad')
pyplot.show()

from matplotlib import pyplot
perc_subopt_linear = 100*(iterates_linear_cut-iterates_linear_cut[-1])/iterates_linear_cut[-1]
perc_subopt_linear = np.minimum(100,perc_subopt_linear)
pyplot.plot(np.arange(len(iterates_linear_cut)), perc_subopt_linear,label='wc mse')
title='% suboptimality of iterates'
pyplot.title(title)
pyplot.xlabel('num of iterates')
pyplot.ylabel('objective value')
pyplot.show()

from matplotlib import pyplot#linear cut log scale
log_perc_subopt_linear = np.log((iterates_linear_cut-iterates_linear_cut[-1])/iterates_linear_cut[-1])
pyplot.plot(np.arange(len(iterates_linear_cut)), log_perc_subopt_linear,label='wc mse')
pyplot.axhline(y=-2.3, color='r', linestyle='-',label='10% suboptimal')
pyplot.axhline(y=-3.0, color='g', linestyle='-',label='5% suboptimal')
pyplot.axhline(y=-4.6, color='b', linestyle='-',label='1% suboptimal')
title='log % suboptimality of iterates'
pyplot.title(title)
pyplot.xlabel('num of iterates')
pyplot.ylabel('objective value')
pyplot.legend(loc='best')
pyplot.show()


##from matplotlib import pyplot
##pyplot.plot(np.arange(len(blackbox_obj_arr)), blackbox_obj_arr,label='wc mse')
##title='iterates of objective for gradient method'
##pyplot.title(title)
##pyplot.xlabel('num of iterates')
##pyplot.ylabel('objective value')
##pyplot.show()
##
##from matplotlib import pyplot
##pyplot.plot(np.arange(len(iterates_quad_cut)), iterates_quad_cut,label='wc mse')
##pyplot.plot(np.arange(len(iterates_quad_cut)), surrogate_iterates_quad_cut,label='wc mse over finite set r')
##title='iterates of objective for quadratic cut'
##pyplot.title(title)
##pyplot.xlabel('num of iterates')
##pyplot.ylabel('objective value')
##pyplot.show()


#plot subproblem grid search
##Train_data = set_up.prepare_data(rndseed,train_size,policy_type,demand_type,logging_type,feature_dist,feature_dim,sigma)
##_,_,_,hat_r_z_arr,hat_r_y_arr = methods.simulate_linear_MSE(Train_data,3,1)
##hat_r_z,hat_r_y = methods.truncate_r(Train_data,hat_r_z_arr[0],hat_r_y_arr[0])
##w1,obj,time1 = methods.our_method_DR_constraint_generation_scipy(Train_data,Gamma,hat_r_z,hat_r_y,brute_force=0,returnDetails=1)
##
##obj_arr = methods.subroutine_DR_brute_force(w1,Gamma,Train_data,hat_r_z, hat_r_y,4,solver='Gurobi')
##grid = np.arange(-Gamma*(2**0.5),Gamma*(2**0.5),0.1)

##from matplotlib import pyplot
##pyplot.plot(grid, obj_arr)
##title='subproblem objective in grid search'
##pyplot.title(title)
##pyplot.xlabel('variable associated with positive eval')
##pyplot.ylabel('subproblem objective')
##pyplot.show()
