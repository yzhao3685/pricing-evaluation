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
feature_dim,train_size,rndseed,sigma = 2,30,int(1000000*(time.time()%1)),0.2#bandwidth of kernels

t_start=time.time()

Train_data = set_up.prepare_data(rndseed,train_size,policy_type,demand_type,logging_type,feature_dist,feature_dim,sigma)
P_train,D_train = Train_data.P_train,Train_data.D_train
linear_mse,linear_bias,linear_var,hat_r_z_arr,hat_r_y_arr = methods.simulate_linear_MSE(Train_data,3,1)
hat_r_z,hat_r_y = methods.truncate_r(Train_data,hat_r_z_arr[0],hat_r_y_arr[0])

Gamma_list = np.arange(1,9)*5
w_ours_arr = np.zeros((len(Gamma_list),train_size)); our_est_arr = np.zeros(len(Gamma_list)); w_ours = np.zeros(train_size)
remove_simplex =0#whether we remove simplex constraint
for i in range(len(Gamma_list)):
    Gamma = Gamma_list[i]
    w_ours,record_arr = methods.optimize_bound_method(Train_data,Gamma,w_ours,hat_r_z,hat_r_y,epsilon=0.05,solver='scipy',remove_simplex=0)
    our_est = w_ours@(P_train*D_train - hat_r_z) + np.average(hat_r_y)
    if remove_simplex==0:
        if max(w_ours)>0.2:
            print('Gamma: ',Gamma,'large weight')
        elif max(w_ours)-min(w_ours)<0.003:
            print('Gamma: ',Gamma,'equal weights')
        else:
            print('Gamma: ',Gamma,'ok weights')
    else:
        if sum(w_ours)<0.05:
            print('weights very small')
        elif max(w_ours)-min(w_ours)<0.003:
            print('Gamma: ',Gamma,'equal weights')
        else:
            print('Gamma: ',Gamma,'ok weights')
    w_ours_arr[i] = w_ours; our_est_arr[i] = our_est
    

expected_val_test = Train_data.expected_val_test
w_NW_baseline = methods.kernel_regression(Train_data,0.2)
w_NW,opt_sigma = methods.kernel_regression_oracle(Train_data)
NW_pred_rev_arr = methods.simulate_MSE(Train_data,w_NW,returnDetails=1)

print('true rev: ',expected_val_test)
print('our estimate: ',our_est)
print('NW estimate: ',np.average(NW_pred_rev_arr))
print('LASSO estimate: ',np.average(np.average(hat_r_y_arr,axis=0)))

#plot recorded details
import matplotlib.pyplot as plt
n=record_arr.shape[0]
plt.plot(np.arange(n),record_arr[:,1] ,label='objective')
plt.plot(np.arange(n),record_arr[:,2] ,label='q(w) term')
plt.plot(np.arange(n),record_arr[:,3] ,label='bias term')
plt.plot(np.arange(n),record_arr[:,4] ,label='variance term')

plt.ylabel('terms in objective')
plt.xlabel('number of iterates')
plt.title('breakdown of objective')
plt.legend(loc='best')
plt.show()


##_,_,_,predict_rev_arr = methods.our_method_DR_simulate_MSE(Train_data,Gamma,hat_r_z_arr,hat_r_y_arr,method='opti bound',epsilon=0.05)
##
###plot histogram
##import matplotlib.pyplot as plt
##plt.hist(np.average(hat_r_y_arr,axis=0), density=True, bins=20, alpha=0.5,label='LASSO')  
##plt.hist(predict_rev_arr, density=True, bins=20, alpha=0.5,label='ours')
##plt.hist(NW_pred_rev_arr, density=True, bins=20, alpha=0.5,label='kernel reg')
##plt.axvline(x=expected_val_test,color='r',label='true rev')
##plt.ylabel('Probability')
##plt.xlabel('revenue')
##plt.title('dist of rev estimate')
##plt.legend(loc='best')
##plt.show()
##
###plot weights
##ind_very_low,ind_low,ind_med,ind_high,ind_very_high = set_up.price_levels(Train_data.X_train,Train_data.P_train,demand_type)
##ind_low.extend(ind_very_low)
##ind_high.extend(ind_very_high)
##import matplotlib.pyplot as plt
##plt.scatter(w_NW_baseline, w_ours*train_size,alpha=0.5,label='medium price')
##plt.scatter(w_NW_baseline[ind_low], w_ours[ind_low]*train_size,alpha=0.5,label='low price')
##plt.scatter(w_NW_baseline[ind_high], w_ours[ind_high]*train_size,alpha=0.5,label='high price')
##plt.ylabel('weights (unit is 1/n)')
##plt.xlabel('similarity score (kernel reg weights)')
##plt.title('our weights')
##plt.legend(loc='best')
##plt.show()


#unique inner problem solution
##for i in range(10):
##    hat_r_z,hat_r_y = methods.truncate_r(Train_data,hat_r_z_arr[i],hat_r_y_arr[i])
##    weights = methods.optimize_bound_method(Train_data,Gamma,[],hat_r_z,hat_r_y,epsilon=0.05)
##    alpha,alpha2 = methods.optimize_bound_subroutine_uniqueness(weights,0.05,Train_data,Gamma,hat_r_z,hat_r_y)
##    abs_diff = np.abs(alpha-alpha2)
##    print('alpha avg: ',round(np.average(np.abs(alpha)),10),
##        'avg diff: ',round(np.average(abs_diff),10),'max diff: ',round(max(abs_diff),10))


#weights are close, with or without change of variable
##weights1= methods.optimize_bound_method(Train_data,Gamma,[],hat_r_z,hat_r_y,epsilon=0.05,solver='Gurobi',change_of_var=1)
##weights2= methods.optimize_bound_method(Train_data,Gamma,[],hat_r_z,hat_r_y,epsilon=0.05,solver='Gurobi',change_of_var=0)



