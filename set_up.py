#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoLarsCV
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
import methods
from sklearn.model_selection import KFold
from scipy.stats import norm

def policy_to_evaluate(x_arr,p_arr,policy_type,logging_type,rndseed):
    if policy_type[0]=='setting1':
        multiplier=3
        p_arr=7.0+x_arr@np.array([-1,1])
        if policy_type[1]!='deterministic':
            std=1/norm.ppf(0.5+0.5*policy_type[1])#std needed for "logging_type" percentage of prices to fall in middle level 
            np.random.seed(rndseed+1836)
            p_arr+=np.random.normal(loc=0.0,scale=std,size=len(p_arr))        
        return np.minimum(np.maximum(p_arr,0),10)*multiplier
    elif policy_type[0]=="setting2":
        multiplier=3
        theta=np.angle(x_arr[:,0]+x_arr[:,1]*1j)*1.5
        p_arr=3.0-theta
        halfn = int(round(len(p_arr)/2))
        p_arr[halfn:]+=4
        if policy_type[1]!='deterministic':
            std=1/norm.ppf(0.5+0.5*policy_type[1])#std needed for "logging_type" percentage of prices to fall in middle level 
            np.random.seed(rndseed+1836)
            p_arr+=np.random.normal(loc=0.0,scale=std,size=len(p_arr))        
        return np.minimum(np.maximum(p_arr,0),10)*multiplier
    else:
        multiplier=3
        p_arr=np.ones(len(p_arr))*5.0+x_arr@np.array([-1,1])
        if policy_type[1]!='deterministic':
            std=1/norm.ppf(0.5+0.5*policy_type[1])#std needed for "logging_type" percentage of prices to fall in middle level 
            np.random.seed(rndseed+1836)
            p_arr+=np.random.normal(loc=0.0,scale=std,size=len(p_arr))        
        return np.minimum(np.maximum(p_arr,0),10)*multiplier

def expected_demand(x_arr,p_arr,demand_type,dim):
    #assume demand type is a list, first entry is setting, second entry is smoothness parameter, third entry is jump size
    if demand_type[0]=='setting1':
        multiplier=3
        kappa=demand_type[1]#controls smoothness of demand
        height=demand_type[2]
        y=p_arr/multiplier-x_arr@np.array([-1,1])
        d_arr=height*sig(kappa*(y-2))+(0.5-height)*sig(kappa*(y-4))+(0.5-height)*sig(kappa*(y-6))+height*sig(kappa*(y-8))
    elif demand_type[0]=="setting2":
        multiplier=3
        kappa=demand_type[1]#controls smoothness of demand
        height=demand_type[2]
        theta=np.angle(x_arr[:,0]+x_arr[:,1]*1j)*1.5 #in [-pi,pi]
        y=p_arr/multiplier+theta
        d_arr=height*sig(kappa*(y-2))+(0.5-height)*sig(kappa*(y-4))+(0.5-height)*sig(kappa*(y-6))+height*sig(kappa*(y-8))
    else:
        multiplier=3
        kappa=demand_type[1]#controls smoothness of demand
        height=demand_type[2]
        y=p_arr/multiplier-x_arr@np.array([-1,1])
        d_arr=height*sig(kappa*(y-2))+(0.5-height)*sig(kappa*(y-4))+(0.5-height)*sig(kappa*(y-6))+height*sig(kappa*(y-8))
    return d_arr

def historical_policy(x_arr,logging_type,rndseed):
    #assume logging type is a list, first entry is settiing, second entry is % of prices falling withint the middle level 
    if logging_type[0]=='setting1':
        multiplier=3
        std=1/norm.ppf(0.5+0.5*logging_type[1])#std needed for "logging_type" percentage of prices to fall in middle level 
        p_arr=5+x_arr@np.array([-1,1])
        np.random.seed(rndseed+883)
        p_arr+=np.random.normal(loc=0.0,scale=std,size=len(p_arr))
        return np.minimum(np.maximum(p_arr,0),10)*multiplier
    elif logging_type[0]=="setting2":
        multiplier=3
        std=1/norm.ppf(0.5+0.5*logging_type[1])#std needed for "logging_type" percentage of prices to fall in middle level 
        theta=np.angle(x_arr[:,0]+x_arr[:,1]*1j)*1.5
        p_arr=5.0-theta
        np.random.seed(rndseed+883)
        p_arr+=np.random.normal(loc=0.0,scale=std,size=len(p_arr))
        return np.minimum(np.maximum(p_arr,0),10)*multiplier
    else:
        multiplier=3
        std=1/norm.ppf(0.5+0.5*logging_type[1])#std needed for "logging_type" percentage of prices to fall in middle level 
        p_arr=np.ones(x_arr.shape[0])*3+x_arr@np.array([-1,1])
        np.random.seed(rndseed+883)
        p_arr+=np.random.normal(loc=0.0,scale=std,size=len(p_arr))
        return np.minimum(np.maximum(p_arr,0),10)*multiplier
    
def price_levels(x_arr,p_arr,demand_type):
    ind_very_low,ind_low,ind_med,ind_high,ind_very_high = [],[],[],[],[]
    d_arr = expected_demand(x_arr,p_arr,demand_type,2)
    height=demand_type[2]
    #d_arr=height*sig(kappa*(y-2))+(0.5-height)*sig(kappa*(y-4))+(0.5-height)*sig(kappa*(y-6))+height*sig(kappa*(y-8))
    for i in range(len(p_arr)):
        if d_arr[i]<=height/2:
            ind_very_high.append(i)#highest prices have lowest demand
        elif height/2<d_arr[i]<=height+(0.5-height)/2:
            ind_high.append(i)
        elif height+(0.5-height)/2<d_arr[i]<=0.5+(0.5-height)/2:
            ind_med.append(i)
        elif 0.5+(0.5-height)/2<d_arr[i]<=1-height/2:
            ind_low.append(i)
        else:
            ind_very_low.append(i)
    return ind_very_low,ind_low,ind_med,ind_high,ind_very_high

    
def sig(x):#sigmoid function
    return 1/(1+np.exp(x))

def realized_demand(x_arr,p_arr,rndseed,demand_type,dim):
    np.random.seed(rndseed+52028)
    return np.random.binomial(1,expected_demand(x_arr,p_arr,demand_type,dim))       
    
def generate_feature(train_size,rndseed,feature_dist):
    np.random.seed(rndseed+883)
    return np.random.uniform(-1,1,(train_size,2))
    
class prepare_data: 
    def __init__(self,rndseed,train_size,policy_type,demand_type,logging_type,feature_dist,dim,sigma):
        X_train=generate_feature(train_size,rndseed,feature_dist)
        P_train=historical_policy(X_train,logging_type,rndseed)
        #P_train = np.maximum(P_train,5.0)#artificially lower  bound the price
        P_train[np.argmin(P_train)] = 0#test if we can handle this case
        P_test=policy_to_evaluate(X_train,P_train,policy_type,logging_type,rndseed)
        D_train=realized_demand(X_train,P_train,rndseed,demand_type,dim)
        
        G,C = compute_Gram_matrix(P_train,X_train,P_test,dim,sigma)
        rowSumHalfG,D3,arr_m,expected_R_train,expected_val_test = preparation(G,X_train,P_train,P_test,demand_type,dim)
        numSimulation=100
        D_train_simul=np.zeros((numSimulation,train_size))
        for i in range(numSimulation):# 100 simulations
            D_train_simul[i]=realized_demand(X_train,P_train,rndseed+i*10,demand_type,dim)
            
        self.P_test,self.P_train,self.X_train,self.D_train = P_test,P_train,X_train,D_train
        self.train_size,self.G,self.C = train_size,G,C
        self.rowSumHalfG,self.D3,self.arr_m,self.dim,self.sigma,self.demand_type = rowSumHalfG,D3,arr_m,dim,sigma,demand_type
        self.expected_R_train,self.expected_val_test = expected_R_train,expected_val_test
        self.D_train_simul,self.numSimulation = D_train_simul, numSimulation
        self.expected_D_train = expected_demand(X_train,P_train,demand_type,dim)
        self.expected_R_test=np.multiply(P_test,expected_demand(X_train,P_test,demand_type,dim))
        self.G_inv = np.linalg.pinv(G,hermitian=True)
        
class split_data: 
    def __init__(self,Train_data,whichHalf,rndseed):#rndseed determines how we split data
        num_fold=2#two fold 
        assert Train_data.train_size%num_fold==0
        X_arr,P_arr,D_arr = Train_data.X_train,Train_data.P_train,Train_data.D_train
        kf = KFold(n_splits=num_fold,shuffle=True, random_state=rndseed)
        kf.get_n_splits(X_arr)
        k=0
        for train_index, test_index in kf.split(X_arr):
            k+=1
            if k==whichHalf:
                X_train, X_test = X_arr[train_index], X_arr[test_index]
                P_train, P_test = P_arr[train_index], P_arr[test_index]
                D_train, D_test = D_arr[train_index], D_arr[test_index]
        train_size=len(P_train)
        G,C = compute_Gram_matrix(P_train,X_train,P_test,Train_data.dim,Train_data.sigma)
        rowSumHalfG,D3,arr_m,_,_ = preparation(G,X_train,P_train,P_test,Train_data.demand_type,Train_data.dim)
            
        self.P_test,self.P_train,self.X_train,self.D_train = P_test,P_train,X_train,D_train
        self.train_size,self.G,self.C = train_size,G,C
        self.rowSumHalfG,self.D3,self.arr_m,self.dim,self.sigma = rowSumHalfG,D3,arr_m,Train_data.dim,Train_data.sigma
        self.realized_R_train, self.realized_val_test = np.multiply(P_train,D_train),P_test@D_test/train_size
        
def compute_Gram_matrix(P_train,X_train,P_test,dim,sigma):
    train_size= len(P_train)
    P_train_norm=P_train/30
    P_test_norm=P_test/30
    price_multiplier=10#difference in price should be more important than difference in one feature dim
    P_train_norm=P_train_norm*price_multiplier
    P_test_norm=P_test_norm*price_multiplier
    Z_and_Y=np.zeros((train_size*2,dim+1))
    Z_and_Y[0:train_size]=np.concatenate((X_train/2,P_train_norm[:,None]),axis=1)
    Z_and_Y[train_size:]=np.concatenate((X_train/2,P_test_norm[:,None]),axis=1)
    kernel = RBF(sigma)
    G=kernel(Z_and_Y)
    G=G+0.0001*np.identity(train_size*2)#make G positive semi-definite
    C=np.linalg.cholesky(G*2)#C is lower triangular, CC^\top=G
    return G,C

def preparation(G,X_train,P_train,P_test,demand_type,dim):
    train_size = len(P_train)
    rowSumHalfG=np.transpose(np.sum(G[:,train_size:],axis=1).reshape(-1,1))#sum each row of right half of G
    D3=np.transpose(rowSumHalfG).dot(rowSumHalfG)/(train_size**2)#the third term in Dw
    arr_m=np.zeros((train_size,train_size*2,train_size*2))
    for i in range(0,train_size):
       temp2=G[:,i].reshape(-1,1)
       arr_m[i]=temp2.dot(np.transpose(temp2))
    expected_R_train=np.multiply(P_train,expected_demand(X_train,P_train,demand_type,dim))
    expected_R_test=np.multiply(P_test,expected_demand(X_train,P_test,demand_type,dim))
    expected_val_test=sum(expected_R_test)/train_size
    return rowSumHalfG,D3,arr_m,expected_R_train,expected_val_test

#compute true MSE, use our formula for MSE
def true_MSE(w,Train_data):
    bias=w@Train_data.expected_R_train-Train_data.expected_val_test
    temp=np.multiply(Train_data.expected_R_train,Train_data.P_train-Train_data.expected_R_train)
    variance=temp@np.multiply(w,w)
    return bias**2+variance, bias**2,variance,

