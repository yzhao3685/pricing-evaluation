#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
from gurobipy import *
import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoLarsCV
import statistics

def nathans_method(lamda,Train_data):
    train_size=Train_data.train_size
    G_aug=np.copy(Train_data.G)
    for i in range(0,train_size):
        G_aug[i][i]=G_aug[i][i]+lamda**2#add variance term
    evals,evecs=scipy.linalg.eigh(G_aug, eigvals_only=False)#this method assumes real symmetric matrix, and is fast
    A=np.diag(evals)
    c=np.sum(evecs[0:train_size,0:2*train_size],axis=0)
    #build optimization model
    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    v=m.addMVar ( train_size*2, ub=100.0,lb=-100.0,name="v" )
    m.setObjective(v@A@v, GRB.MINIMIZE) 
    m.addConstr(v@c==1)#simplex constraint
    m.addConstr(evecs[train_size:2*train_size,0:2*train_size]@v==-np.ones(train_size)/train_size)
    m.update()
    m.optimize()
    #get weights
    v_star=np.zeros(2*train_size)
    for i in range(0,2*train_size):
        v_star[i]=v[i].x
    w=evecs[0:train_size,0:2*train_size]@v_star
    return w,Train_data.D_train@w


# In[ ]:


def subroutine_binary_search(weights,Gamma,Train_data):
    G,C,rowSumHalfG = Train_data.G,Train_data.C,Train_data.rowSumHalfG
    D3,arr_m,train_size = Train_data.D3,Train_data.arr_m,Train_data.train_size
    #compute D(w)
    wSumHalfG=(G[:,0:train_size].dot(weights)).reshape(-1,1)
    D2=2*wSumHalfG.dot(rowSumHalfG)/train_size#the second term in Dw
    D1=wSumHalfG.dot(np.transpose(wSumHalfG))#the first term in Dw
    for i in range(0,train_size):
        D1=D1-arr_m[i]*weights[i]**2
    Dw=-2*D1+D2+np.transpose(D2)-2*D3#Dw may not be PSD
    #compute S
    M=np.linalg.solve(C,Dw)#solve matrix M for CM=Dw
    B=np.linalg.solve(C,np.transpose(M))#solve matrix B for CB=M^\top
    evals,evecs=scipy.linalg.eigh(B, eigvals_only=False)#symmetric QR
    Q=np.copy(evecs)
    delta=np.copy(evals)
    S=np.linalg.solve(np.transpose(C),Q)
    #compute bw, epsilon
    temp2=np.multiply(np.multiply(weights,weights),Train_data.P_train)
    bw=-G[:,0:train_size]@temp2  
    epsilon=np.transpose(S).dot(bw)
    #binary search algorithm
    lambda_L=max(0,-min(delta))#ensure delta+lambda_L>=0
    lambda_U=100
    tol=0.000000001      
    while lambda_U-lambda_L>=tol:
        lambda_M=(lambda_U+lambda_L)/2
        x_star=np.divide(-epsilon,delta+lambda_M)
        if np.linalg.norm(x_star)>Gamma*(2**0.5):
            lambda_L=lambda_M
        else:
            lambda_U=lambda_M 
    optimal_dual=lambda_U
    x_star=np.divide(-epsilon,delta+optimal_dual) #recover opti primal variable from opti dual variable
    #get key values
    obj = np.multiply(delta,x_star)@x_star/2+epsilon@x_star
    obj_val = -obj #return phi(w) 
    q=S@x_star#worst case r()
    #compute gradient
    grad=np.zeros(train_size)
    for i in range(0,train_size):
        temp1=np.transpose(G[:,i]).reshape(-1,1)
        temp2=np.transpose(wSumHalfG)-weights[i]*np.transpose(temp1)-rowSumHalfG/train_size
        Hessian=2*temp1.dot(temp2)
        grad[i]=q.dot(Hessian.dot(np.transpose(q)))
        grad[i]=grad[i]+2*weights[i]*Train_data.P_train[i]*np.transpose(G[:,i]).dot(q)
    return obj_val, grad


# In[ ]:


class PGD: 
    def __init__(self,w,Gamma,Train_data):
        obj_val_old,grad=subroutine_binary_search(w,Gamma,Train_data) 
        k,L,maxIter = 1,max(10,Gamma*100),500    #L: a guess on lipschitz constant of phi(w) 
        MSE_arr,stepSize_arr=np.ones(maxIter+1)*obj_val_old,np.zeros(maxIter+1) 
        step_size,counter_small_steps=1,0
        eta,tau=0.1,0.8#parameters in Armijo rule
        while (k<=maxIter and counter_small_steps<20):
            gt,step_size=grad@grad,1/L
            w_projected=np.maximum(w-step_size*grad,np.zeros(Train_data.train_size))
            obj_val_new,grad=subroutine_binary_search(w_projected,Gamma,Train_data)
            #Backtracking. Use Armijo rule for step size
            counter_Armijo=0
            while obj_val_old-obj_val_new<step_size*gt*eta and counter_Armijo<20:
                counter_Armijo+=1
                step_size=step_size*tau
                w_projected=np.maximum(w-step_size*grad,np.zeros(Train_data.train_size))
                obj_val_new,grad=subroutine_binary_search(w_projected,Gamma,Train_data)
                if counter_Armijo==20:
                    counter_small_steps+=1
            obj_val_old,MSE_arr[k],stepSize_arr[k]=obj_val_new,obj_val_new,step_size
            w=w-step_size*grad
            k+=1
        #truncate dummy values
        total_iter=k
        MSE_arr,stepSize_arr = MSE_arr[0:k],stepSize_arr[0:k]

        self.w, self.MSE_arr, self.stepSize_arr = w, MSE_arr, stepSize_arr


# In[ ]:


class FW: 
    def __init__(self,w,Gamma,Train_data):
        obj_val_old,grad=subroutine_binary_search(w,Gamma,Train_data) 
        k,L,maxIter = 1, max(10,Gamma*100),500
        MSE_arr,stepSize_arr=np.ones(maxIter+1)*obj_val_old,np.zeros(maxIter+1)
        eta,tau,counter_small_steps=0.1,0.8,0
        while (k<=maxIter and counter_small_steps<20):
            ind=np.argmin(grad)
            s_t=np.zeros(Train_data.train_size)
            s_t[ind]=1
            descent_dir=s_t-w
            #compute step size
            gt=-grad@descent_dir
            step_size=min(gt/(L*descent_dir@descent_dir),1)
            #use Armijo rule for step size
            obj_val_new,grad=subroutine_binary_search(w+step_size*descent_dir,Gamma,Train_data)
            counter_Armijo=1
            while obj_val_old-obj_val_new<step_size*gt*eta and counter_Armijo<20:
                step_size=step_size*tau
                counter_Armijo+=1
                obj_val_new,grad=subroutine_binary_search(w+step_size*descent_dir,Gamma,Train_data)
                if counter_Armijo==20:
                    counter_small_steps+=1
            obj_val_old=obj_val_new
            MSE_arr[k],stepSize_arr[k] = obj_val_new,step_size
            w=w+step_size*descent_dir
            k+=1
        #truncate dummy values
        total_iter=k
        MSE_arr,stepSize_arr = MSE_arr[0:k],stepSize_arr[0:k]

        self.w, self.MSE_arr, self.stepSize_arr = w, MSE_arr, stepSize_arr


class direct_method_estimate: 
    def __init__(self,TrainData,deg1,deg2):
        X1=PolynomialFeatures(degree=deg1, include_bias=True).fit_transform(TrainData.X_train.reshape(-1,1))
        temp=PolynomialFeatures(degree=deg2, include_bias=True).fit_transform(TrainData.X_train.reshape(-1,1))
        X2=np.multiply(TrainData.P_train.reshape(-1,1),temp)
        numCol1=np.shape(X1)[1]
        numCol2=np.shape(X2)[1]
        X=np.array([[0.0 for i in range(numCol1+numCol2)] for j in range (TrainData.train_size)]) 
        X[:,0:numCol1]=X1
        X[:,numCol1:]=-X2
        model = LassoLarsCV(fit_intercept=False).fit(X, TrainData.D_train)
        HT,alpha=model.coef_,model.alpha_
        #evaluate on testing
        X2=np.multiply(TrainData.P_test.reshape(-1,1),temp)
        X[:,numCol1:]=-X2
        predict_demand = np.maximum(X@HT,np.zeros(TrainData.train_size))#truncate demand estimate
        predict_demand = np.minimum(predict_demand,np.ones(TrainData.train_size))
        predict_rev=predict_demand@TrainData.P_test/TrainData.train_size
        
        self.HT,self.numCol1,self.numCol2 = HT,numCol1,numCol2
        self.predict_rev,self.Xv,self.Xv2=predict_rev,X1,temp


# In[ ]:


#doubly robust estimators and IPS estimators 
class doubly_robust_estimate:
    def __init__(self,TrainData,reg_result): 
        n=TrainData.train_size
        Xv,Xv2=reg_result.Xv,reg_result.Xv2
        a_DM,b_DM,a_DR,b_DR,R_DR = np.ones(n),np.ones(n),np.ones(n),np.ones(n),np.ones(n)
        #estimate g and sigma
        model = LassoLarsCV(fit_intercept=False).fit(Xv, TrainData.P_train)
        g = model.coef_
        VarP = statistics.variance(np.subtract(TrainData.P_train,Xv@g))
        for i in range(0,n):
            a_DM[i]=reg_result.HT[0:reg_result.numCol1]@Xv[i]#direct method estimator
            b_DM[i]=reg_result.HT[reg_result.numCol1:]@Xv2[i]
            res=TrainData.D_train[i]-a_DM[i]+b_DM[i]*TrainData.P_train[i]
            a_DR[i]=a_DM[i]+(1+Xv[i]@g*(Xv[i]@g-TrainData.P_train[i])/VarP)*res                         
            b_DR[i]=b_DM[i]+((Xv[i]@g-TrainData.P_train[i])/VarP)*res
            D_DR=a_DR[i]-b_DR[i]*TrainData.P_test[i]
            D_DR=min(max(D_DR,0),1)#truncate demand estimate
            R_DR[i]=D_DR*TrainData.P_test[i]
            
        self.a_DM,self.b_DM,self.a_DR,self.b_DR=a_DM,b_DM,a_DR,b_DR
        self.predict_rev=np.average(R_DR)
