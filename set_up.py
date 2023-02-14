#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoLarsCV
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
import methods
from sklearn.model_selection import KFold
from scipy.stats import norm
from scipy.optimize import NonlinearConstraint
from sklearn.model_selection import train_test_split


def policy_to_evaluate(x_arr,rndseed, synthetic_new_policy):
    # synthetic_new_policy could take values likes 3, 4, 5
    p_arr = synthetic_new_policy + x_arr@np.array([1,-1]) * 0.5
    np.random.seed(rndseed+883)
    p_arr += np.random.normal(loc=0.0, scale=2, size=len(p_arr))
    return np.minimum(np.maximum(p_arr,1),10) * 2

def expected_demand(x_arr,p_arr):
##    temp = p_arr / 2 - 5 + np.arctan(np.divide(x_arr[:,0], x_arr[:,1])) # if p >= 7, then demand is 0.3, if p<=3, then demand is 0.7 (assume x is (0,0))
    temp = p_arr / 2 - 5 + x_arr@np.array([-1,1]) # if p >= 7, then demand is 0.3, if p<=3, then demand is 0.7 (assume x is (0,0))
    d_arr = 0.25 + 0.75 * sig(temp)      
    return d_arr

def historical_policy(x_arr,rndseed):
    p_arr = 7 + x_arr@np.array([1,-1]) * 0.5
    np.random.seed(rndseed+883)
    p_arr += np.random.normal(loc=0.0, scale=2, size=len(p_arr))
    return np.minimum(np.maximum(p_arr,1),10) * 2
      
def sig(x):#sigmoid function
    return 1/(1+np.exp(x))

def realized_demand(x_arr,p_arr,rndseed):
    np.random.seed(rndseed+52028)
    return np.random.binomial(1,expected_demand(x_arr,p_arr))       
    
def generate_feature(train_size,rndseed):
    np.random.seed(rndseed+883)
    return np.random.uniform(-1,1,(train_size,2))
    
class prepare_synthetic_data: 
    def __init__(self,rndseed,train_size,s_p,s_x, synthetic_new_policy):
        X_train = generate_feature(train_size,rndseed)
        P_train = historical_policy(X_train,rndseed)
        P_test = policy_to_evaluate(X_train,rndseed, synthetic_new_policy)
        self.D_train = realized_demand(X_train,P_train,rndseed)
        self.expected_R_train = np.multiply(P_train,expected_demand(X_train,P_train))
        self.expected_R_test = np.multiply(P_test,expected_demand(X_train,P_test))
        self.expected_val_test = sum(self.expected_R_test)/train_size
        self.expected_D_train = expected_demand(X_train,P_train)
        
        self.dim = 2
        self.G = compute_Gram_matrix(P_train,X_train,P_test,self.dim,s_p,s_x)
        self.numSimulation=100
        self.D_train_simul=np.zeros((self.numSimulation,train_size))
        for i in range(self.numSimulation):
            self.D_train_simul[i] = realized_demand(X_train,P_train,rndseed+i*10)
            
        self.P_test,self.P_train,self.X_train = P_test,P_train,X_train
        self.train_size = train_size
        self.s_p=s_p; self.s_x = s_x
        self.G_inv = np.linalg.pinv(self.G,rcond=1e-4,hermitian=True)
        self.Z_train = X_train
        self.dim = 2
        self.synthetic_new_policy = synthetic_new_policy
        
##class split_data: 
##    def __init__(self,Train_data,whichHalf,rndseed):#rndseed determines how we split data
##        num_fold=2#two fold 
##        assert Train_data.train_size%num_fold==0
##        X_arr,P_arr,D_arr = Train_data.X_train,Train_data.P_train,Train_data.D_train
##        kf = KFold(n_splits=num_fold,shuffle=True, random_state=rndseed)
##        kf.get_n_splits(X_arr)
##        k=0
##        for train_index, test_index in kf.split(X_arr):
##            k+=1
##            if k==whichHalf:
##                X_train, X_test = X_arr[train_index], X_arr[test_index]
##                P_train, P_test = P_arr[train_index], P_arr[test_index]
##                D_train, D_test = D_arr[train_index], D_arr[test_index]
##        train_size=len(P_train)
##        G,C = compute_Gram_matrix(P_train,X_train,P_test,Train_data.dim,Train_data.s_p,Train_data.s_x)
##        rowSumHalfG,D3,arr_m,_,_ = preparation(G,X_train,P_train,P_test,Train_data.demand_type,Train_data.dim)
##            
##        self.P_test,self.P_train,self.X_train,self.D_train = P_test,P_train,X_train,D_train
##        self.train_size,self.G,self.C = train_size,G,C
##        self.dim = Train_data.dim
##        self.realized_R_train, self.realized_val_test = np.multiply(P_train,D_train),P_test@D_test/train_size
        
def compute_Gram_matrix(P_train,X_train,P_test,dim, s_p,s_x):
    n= len(P_train)
    P_max = max(max(P_train),max(P_test)); P_min = min(min(P_train),min(P_test))
    assert P_max - P_min > 0.001 #avoid division by zero
    normalizing_constant = (P_max-P_min)*s_p
    P_train_norm=(P_train-P_min)/normalizing_constant
    P_test_norm=(P_test-P_min)/normalizing_constant
    x_min_arr = np.min(X_train,axis=0); x_max_arr = np.max(X_train,axis=0)
    assert min(x_max_arr-x_min_arr) > 0.001 #avoid division by zero
    normalizing_constant = (x_max_arr-x_min_arr)*s_x
    X_train_norm = np.divide((X_train-np.tile(x_min_arr,(n,1))),normalizing_constant)

    Z_and_Y=np.zeros((n*2,dim+1))
    Z_and_Y[0:n]=np.concatenate((X_train_norm,P_train_norm[:,None]),axis=1)
    Z_and_Y[n:]=np.concatenate((X_train_norm,P_test_norm[:,None]),axis=1)
    kernel = RBF(1)
    G=kernel(Z_and_Y)
    G=G+0.0001*np.identity(n*2)#make G positive semi-definite
    return G

###compute true MSE, use our formula for MSE
##def true_MSE(w,Train_data,isBernstein=0,hat_r_train=[],hat_r_test=[]):
##    estimate = w@Train_data.expected_R_train
##    if isBernstein==1:
##        estimate = w@(Train_data.expected_R_train-hat_r_train)+np.average(hat_r_test)
##    bias=estimate-Train_data.expected_val_test
##    temp=np.multiply(Train_data.expected_R_train,Train_data.P_train-Train_data.expected_R_train)
##    variance=temp@np.multiply(w,w)
##    return bias**2+variance, bias**2,variance,estimate
##
###approximate the RKHS norm of the revenue function, using fitted model for demand function
##def approx_RKHS_norm(Train_data,hat_r_train,s_p,s_x,returnDetails=0):
##    n=Train_data.train_size
##    delta_train = Train_data.P_train*Train_data.D_train - hat_r_train
##    G,_ = compute_Gram_matrix(delta_train, Train_data.X_train, delta_train, Train_data.dim, s_p,
##                              s_x)
##    G = G[:n, :n]
##    G_inv = np.linalg.pinv(G, hermitian=True)
##    naive_rkhs_norm = delta_train@G_inv@delta_train  #gives super large rkhs norm. likely overfitting r
##    #build optimization model
##    def rkhs_obj(x):
##        return x@x+(G @ x - delta_train) @ (G @ x - delta_train)
##    res = minimize(rkhs_obj, G_inv@delta_train, method='trust-constr', jac=False,tol=0.01,options={'verbose': 0})
##    x_star= res.x
##    approx_rkhs_norm = x_star@G@x_star
##
##    G_inv = np.linalg.pinv(G, rcond=0.01,hermitian=True)
##    pseudo_inv_rkhs_norm1 = delta_train@G_inv@delta_train
##    G_inv = np.linalg.pinv(G, rcond=0.02,hermitian=True)
##    pseudo_inv_rkhs_norm2 = delta_train@G_inv@delta_train
##    #import scipy
##    #evals = scipy.linalg.eigh(G, eigvals_only=True)
##    #print('evals ',evals)
##    print('rkhs norm naive',round(naive_rkhs_norm**0.5,3),'better ',round(approx_rkhs_norm**0.5,3))
##    if returnDetails==0:
##        return approx_rkhs_norm**0.5
##    return approx_rkhs_norm**0.5,naive_rkhs_norm**0.5,pseudo_inv_rkhs_norm1**0.5,pseudo_inv_rkhs_norm2**0.5

class prepare_Nomis_data: 
    def __init__(self,rndseed,s_p,s_x,train_size, new_policy_ratio):
        #load dataset
        df = pd.read_csv('NomisB_e-Car_Data.csv')
        sample_size = df.shape[0]
        Z_oracle= np.array(df[['FICO','Amount','Cost of Funds',
                              'Competition rate']])
        Z_arr = np.array(df[['FICO','Cost of Funds']])
        D_arr = np.array(df[['Outcome']]).flatten()
        P_arr = np.array(df[['Rate']]).flatten()
        # remove nan values in csv
        for i in range(len(P_arr)):
            if np.isnan(P_arr[i]):
                sample_size = i
                break
        Z_oracle = Z_oracle[:sample_size, :]
        Z_arr = Z_arr[:sample_size, :]
        D_arr = D_arr[:sample_size]
        P_arr = P_arr[:sample_size]

        def new_policy(arr):
            return new_policy_ratio*arr
        #impute counterfactual using xgboost
        pred_d_train_all, pred_d_test_all = methods.xgb_impute_counterfactual(Z_oracle, P_arr, new_policy(P_arr), D_arr, 2, rndseed)

        #generate training and testing data
        P_train, _, Z_train,_, expec_d_train,_,expec_d_test,_ = train_test_split(
            P_arr,Z_arr,pred_d_train_all, pred_d_test_all, train_size=train_size,random_state=rndseed+89485)
        train_size=Z_train.shape[0]
        P_test = new_policy(P_train)
        print('imputed value of old policy: ',np.average(expec_d_train*P_train))
        print('imputed value of new policy: ',np.average(expec_d_test*P_test),'\n')
        
        #generate binary demand realizations
        np.random.seed(rndseed+52028)
        D_train = np.random.binomial(1,expec_d_train)

        #preparation for our method and nathan's method
        dim = 2
        G = compute_Gram_matrix(P_train,Z_train,P_test,dim, s_p,s_x)
        R_train = D_train*P_train
        
        # simulate 100 demand realizations
        numSimulation=100
        D_train_simul=np.zeros((numSimulation,train_size))
        for i in range(numSimulation):# 100 simulations
            np.random.seed(rndseed+i*23)
            D_train_simul[i]=np.random.binomial(1,expec_d_train)

        self.P_train=P_train
        self.P_test = P_test
        self.D_train=D_train
        self.Z_train=Z_train  # for sppl
        self.X_train =Z_train # for our method and bope
        self.train_size=train_size

        self.D_train_simul,self.numSimulation = D_train_simul, numSimulation
        self.G = G; self.s_p=s_p; self.s_x = s_x
        self.G_inv = np.linalg.pinv(G,rcond=1e-4,hermitian=True)
        self.dim = 2
        self.expected_R_train = P_train*expec_d_train
        self.expected_R_test=P_test*expec_d_test
        self.expected_val_test = sum(self.expected_R_test)/train_size

