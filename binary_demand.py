#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
from sklearn.linear_model import LogisticRegression
import set_up#our codes
import methods
policy_type,demand_type,logging_type,feature_dist = 1.2,'new','gaussian','uniform'
outfile = "demand_fun_{}.txt".format(demand_type+str(policy_type)+str(time.time()))
with open(outfile, "w") as f:
    f.write("")
t_1 = time.time()
feature_dim,train_size,rndseed,sigma = 1,30,927,0.2#bandwidth of kernel


# In[2]:


#test run
Train_data = set_up.prepare_data(rndseed,train_size,policy_type,demand_type,logging_type,feature_dist,feature_dim,sigma)
#DM,DR are so bad when there is no coverage. 
# reg_results = methods.direct_method_estimate(Train_data,2,2)
# DM_rev = reg_results.predict_rev
# DR_results = methods.doubly_robust_estimate(Train_data,reg_results)
# DR_rev = DR_results.predict_rev
# real_rev= Train_data.expected_val_test

w_nathan,nathan_est_val = methods.nathans_method(1,Train_data)
_,_,MSE_nathan = set_up.true_MSE(w_nathan,Train_data)

our_results = methods.FW(w_nathan,1,Train_data)
w_ours=our_results.w
_,_,MSE_ours=set_up.true_MSE(w_ours,Train_data)
print('sanity check passed')


# 

# In[ ]:


#now repeat above for different training datasets
numTrial=20
optimal_Gamma_arr=np.zeros(numTrial)
optimal_lambda_arr=np.zeros(numTrial)
MSE_ours_arr=np.zeros(numTrial)
MSE_nathan_arr=np.zeros(numTrial)

for j in range(0,numTrial):    
    Train_data = set_up.prepare_data(rndseed+9*j,train_size,policy_type,demand_type,logging_type,feature_dist,feature_dim,sigma)
    # nathan's method for oracle lambda
    lambda_list=np.append(np.arange(1,100)*0.0005,np.arange(1,201)*0.05)
    MSE_nathan_arr[j]=np.inf
    for l in range(0,len(lambda_list)):
        w_nathan,_ = methods.nathans_method(lambda_list[l],Train_data)
        _,_,MSE_nathan = set_up.true_MSE(w_nathan,Train_data)
        if MSE_nathan<MSE_nathan_arr[j]:
            MSE_nathan_arr[j]=MSE_nathan
            optimal_lambda_arr=lambda_list[l]
    # our method for oracle Gamma
    w_ours = methods.nathans_method(1,Train_data)#starting point
    Gamma_list=np.append(np.arange(1,11)*0.05,np.arange(1,11))
    MSE_ours_arr[j]=np.inf#set to infinity
    for l in range(0,len(Gamma_list)):
        our_results = methods.FW(w_nathan,Gamma_list[l],Train_data)
        w_ours=our_results.w
        _,_,MSE_ours=set_up.true_MSE(w_ours,Train_data)
        if MSE_ours<MSE_ours_arr[j]:
            MSE_ours_arr[j]=MSE_ours
            optimal_Gamma_arr=Gamma_list[l]
        print('gamma ',Gamma_list[l],' MSE ',MSE_ours)
    
    t_3 = time.time()
    print('random seed ',j, ' complete')
    with open(outfile, "a") as f:
        f.write("Seed {}:\n".format(j))
        f.write("%.4f %.4f \n" % (MSE_nathan_arr[j], MSE_ours_arr[j]))


# In[ ]:





# In[ ]:





# In[ ]:




