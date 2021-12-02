#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
from sklearn.gaussian_process.kernels import RBF

def policy_to_evaluate(x_arr,p_arr,policy_type,rndseed):
    np.random.seed(rndseed+913)
    return np.random.normal(loc=(5+2*x_arr*policy_type),scale=1.0)

def expected_demand(x_arr,p_arr,demand_type,dim):
    d_arr=np.zeros(len(p_arr))
    if demand_type=='Max':
        for i in range(0,len(p_arr)):
            temp=w@x_arr[i]-abs(x_arr[i]@np.ones(dim))*p_arr[i]
            d_arr[i]=1/(1+np.exp(-temp))
    elif demand_type=='old':
        d_arr=(5*x_arr+3*(10-p_arr)-10)/10#demand is very sensitive to price changes
        d_arr=np.minimum(d_arr,1)
        d_arr=np.maximum(d_arr,0)
    else:
        d_arr=(p_arr-5-2*x_arr+1)/2
        d_arr=np.maximum(np.minimum(d_arr,1),0)
    return d_arr

def realized_demand(x_arr,p_arr,rndseed,demand_type,dim):
    np.random.seed(rndseed+52028)
    return np.random.binomial(1,expected_demand(x_arr,p_arr,demand_type,dim))

def historical_policy(x_arr,logging_type,rndseed):
    P_high,P_low=10,5
    if logging_type=='discrete':#discrete logging policy, Max's set up
        P_ladder=np.arange(round(P_low),round(P_high)+1)
        output=np.zeros((len(x_arr),len(P_ladder)))
        softmax_multiplier=5.0
        for i in range(0,len(x_arr)):
            temp=demand_function(np.tile(x_arr[i],(len(P_ladder),1)),P_ladder)*softmax_multiplier
            output[i]=np.exp(temp)/sum(np.exp(temp))#softmax sampling   
        return output#output a matrix, each row is probability of prices for one feature vector    
    elif logging_type=='linear':
        return P_high-(P_high-P_low)*x_arr
    elif logging_type=='quadratic':
        return P_high*(1-x_arr**2)
    elif logging_type=='symmetric':
        return P_high*np.multiply(0.5-x_arr,0.5-x_arr)#price is high when deviate from 0.5
    elif logging_type=='gaussian':
        np.random.seed(rndseed+883)
        return np.random.normal(loc=(5+2*x_arr),scale=1.0)
    else:#uniform pricing policy
        np.random.seed(rndseed+883)
        return np.random.uniform(low=P_low,high=P_high,size=len(x_arr))
    
def generate_feature(train_size,rndseed,feature_dist):
    np.random.seed(rndseed+883)
    if feature_dist=='uniform':
        return np.random.uniform(0,1,train_size)
    elif feature_dist=='discrete':
        return np.random.choice(np.arange(10)/10,size=train_size)
    else:
        X_train=np.random.uniform(0,0.2,train_size)
        X_train[10:20]+=0.4
        X_train[20:30]+=0.8
        return X_train


# In[ ]:


class prepare_data: 
    def __init__(self,rndseed,train_size,policy_type,demand_type,logging_type,feature_dist,dim,sigma):
        X_train=generate_feature(train_size,rndseed,feature_dist)
        P_train=historical_policy(X_train,logging_type,rndseed)
        P_test=policy_to_evaluate(X_train,P_train,policy_type,rndseed)
        D_train=realized_demand(X_train,P_train,rndseed,demand_type,dim)
        G,C = compute_Gram_matrix(P_train,X_train,P_test,dim,sigma)
        rowSumHalfG,D3,arr_m,expected_R_train,expected_val_test = preparation(G,X_train,P_train,P_test,demand_type,dim)
        
        self.P_test,self.P_train,self.X_train,self.D_train = P_test,P_train,X_train,D_train
        self.train_size,self.G,self.C = train_size,G,C
        self.rowSumHalfG,self.D3,self.arr_m = rowSumHalfG,D3,arr_m
        self.expected_R_train,self.expected_val_test = expected_R_train,expected_val_test


# In[ ]:


def compute_Gram_matrix(P_train,X_train,P_test,dim,sigma):
   priceMultiplier=1#multiplier of price in kernel
   train_size=len(P_train)
   lb=min(P_train)
   ub=max(P_train)
   P_train_norm=priceMultiplier*(P_train-lb)/ (ub-lb)
   P_test_norm=priceMultiplier*(P_test-lb)/ (ub-lb)

   Z_train=np.zeros((train_size,dim+1))
   for i in range(0,train_size):
       Z_train[i]=np.append(X_train[i],P_train_norm[i])#historical policy
   Y_train=np.zeros((train_size,dim+1))
   for i in range(0,train_size):
       Y_train[i]=np.append(X_train[i],P_test_norm[i])#new policy

   #gram matrix
   kernel = RBF(sigma)
   Z_and_Y=np.zeros((train_size*2,dim+1))
   Z_and_Y[0:train_size,:]=Z_train
   Z_and_Y[train_size:2*train_size,:]=Y_train
   G=kernel(Z_and_Y)
   G=G+0.0001*np.identity(train_size*2)#make G positive semi-definite
   C=np.linalg.cholesky(G*2)#C is lower triangular, CC^\top=G
   return G,C

def preparation(G,X_train,P_train,P_test,demand_type,dim):
   train_size=len(P_train)
   rowSumHalfG=np.transpose(np.sum(G[:,train_size:2*train_size],axis=1).reshape(-1,1))#sum each row of right half of G
   D3=np.transpose(rowSumHalfG).dot(rowSumHalfG)/(train_size**2)#the third term in Dw
   arr_m=np.zeros((train_size,train_size*2,train_size*2))
   for i in range(0,train_size):
       temp2=G[:,i].reshape(-1,1)
       arr_m[i]=temp2.dot(np.transpose(temp2))
   expected_R_train=np.multiply(P_train,expected_demand(X_train,P_train,demand_type,dim))
   expected_R_test=np.multiply(P_test,expected_demand(X_train,P_test,demand_type,dim))
   expected_val_test=sum(expected_R_test)/train_size
   return rowSumHalfG,D3,arr_m,expected_R_train,expected_val_test


# In[ ]:


#compute true MSE, use our formula for MSE
def true_MSE(w,Train_data):
    bias=w@Train_data.expected_R_train-Train_data.expected_val_test
    temp=np.multiply(Train_data.expected_R_train,Train_data.P_train-Train_data.expected_R_train)
    variance=temp@np.multiply(w,w)
    return bias**2,variance,bias**2+variance

