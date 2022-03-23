#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoLarsCV
import statistics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
import set_up
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from sklearn import ensemble
from gurobipy import *
from scipy.optimize import NonlinearConstraint
import cyipopt
import time

def nathan_method_helper(w,A):
    return w@A@w,2*(A@w)
def nathan_method_hess(w,A):
    return A*2

def nathans_method(lamda,Train_data):
    train_size=Train_data.train_size
    G_aug=np.copy(Train_data.G)
    for i in range(0,train_size):
        G_aug[i][i]=G_aug[i][i]+lamda**2
    evals,evecs=scipy.linalg.eigh(G_aug, eigvals_only=False)
    A=np.diag(evals)
    c=np.sum(evecs[0:train_size,0:2*train_size],axis=0)
    #write optimization problem
    cons_Matrix = np.concatenate((np.transpose(c.reshape(-1,1)),evecs[train_size:2*train_size,0:2*train_size]),axis=0)
    cons_RHS = -np.ones(train_size+1)/train_size
    cons_RHS[0]=1
    linear_constraint = LinearConstraint(cons_Matrix, cons_RHS, cons_RHS)
    w0=[1/(train_size*2)]*(train_size*2)
    res = minimize(nathan_method_helper, w0,args=(A), method='trust-constr', jac=True,hess=nathan_method_hess,tol=0.0001,
               constraints=[linear_constraint],options={'verbose': 0,'maxiter':100})#, bounds=bounds)#default maxIter=1000
    w=evecs[0:train_size,0:2*train_size]@res.x
    return w

def nathans_method_oracle(Train_data):
    lambda_list=np.append(np.arange(1,10)*0.01,np.arange(1,10)*0.1)
    lambda_list=np.append(lambda_list,np.arange(1,10))
    lambda_list=np.append(lambda_list,np.arange(1,11)*10)
    MSE_nathan_opt,w_nathan_opt,optimal_lambda=np.inf,np.zeros(Train_data.train_size),0.0
    MSE_arr = np.zeros(len(lambda_list))
    for l in range(0,len(lambda_list)):
        w_nathan = nathans_method(lambda_list[l],Train_data)
        MSE_nathan,_,_ = set_up.true_MSE(w_nathan,Train_data)
        MSE_arr[l]=MSE_nathan
        if MSE_nathan<MSE_nathan_opt:
            MSE_nathan_opt,w_nathan_opt=MSE_nathan,w_nathan
            optimal_lambda=lambda_list[l]
    print('nathans method true MSE: ',MSE_nathan_opt)
    return w_nathan_opt,optimal_lambda

def optimize_bound_method(Train_data,Gamma,w0,hat_r_z,hat_r_y,epsilon=0.05,solver='ipopt',remove_simplex=0):
    train_size = Train_data.train_size
    hat_r_z, hat_r_y = truncate_r(Train_data,hat_r_z,hat_r_y)
    if w0==[]:
        w0=[1/train_size]*train_size
    global record_arr
    res=[]; record_arr = np.zeros((1,6)); w_ours=w0
    if solver=='ipopt':
        if remove_simplex==0:
            w_ours = optimize_bound_ipopt(w0,epsilon,Train_data,Gamma,hat_r_z,hat_r_y,record_arr)
        else:
            w_ours = optimize_bound_ipopt_remove_simplex(w0,epsilon,Train_data,Gamma,hat_r_z,hat_r_y,record_arr)
    else:#use scipy trust region solver
        linear_constraint = LinearConstraint([1]*train_size, [1], [1])
        bounds = Bounds([0.0]*train_size, [5]*train_size)#weights must be positive
        if remove_simplex==0:
            res = minimize(optimize_bound_subroutine, w0,args=(epsilon,Train_data,Gamma,hat_r_z,hat_r_y,1), method='trust-constr',
                        jac=True,tol=0.0002,constraints=[linear_constraint],options={'verbose':0}, bounds=bounds)#default maxIter=1000
            w_ours = res.x
        else:
            res = minimize(optimize_bound_subroutine, w0,args=(epsilon,Train_data,Gamma,hat_r_z,hat_r_y,1), method='trust-constr',
                        jac=True,tol=0.0002,options={'verbose':0}, bounds=bounds)#default maxIter=1000
            w_ours = res.x            
    return w_ours, record_arr[1:,:]


def optimize_bound_ipopt(w0,epsilon,Train_data,Gamma,hat_r_z,hat_r_y,record_arr):
    class IPOPT_problem(object):
        def __init__(self):
            pass
        def objective(self, x):
            obj,grad=optimize_bound_subroutine(x,epsilon,Train_data,Gamma,hat_r_z,hat_r_y,1)
            return obj
        def gradient(self, x):
            obj,grad=optimize_bound_subroutine(x,epsilon,Train_data,Gamma,hat_r_z,hat_r_y)
            return grad
        def constraints(self, x):
            return sum(x)
        def jacobian(self, x):
            return np.ones(len(x))        
    lb, ub,cl,cu = [0.0]*Train_data.train_size, [np.inf]*Train_data.train_size, [1.0], [1.0]
    nlp = cyipopt.Problem(n=len(w0),m=len(cl),problem_obj=IPOPT_problem(),lb=lb,ub=ub,cl=cl,cu=cu)
    nlp.add_option('mu_strategy', 'adaptive')# barrier parameter adaptive
    nlp.add_option('tol', 2e-4)
    #nlp.add_option('max_iter',1000)#default 3,000 max_iter
    w_ours, info = nlp.solve(w0)
    return w_ours

def optimize_bound_ipopt_remove_simplex(w0,epsilon,Train_data,Gamma,hat_r_z,hat_r_y,record_arr):
    class IPOPT_problem(object):
        def __init__(self):
            pass
        def objective(self, x):
            obj,grad=optimize_bound_subroutine(x,epsilon,Train_data,Gamma,hat_r_z,hat_r_y,1)
            return obj
        def gradient(self, x):
            obj,grad=optimize_bound_subroutine(x,epsilon,Train_data,Gamma,hat_r_z,hat_r_y)
            return grad     
    lb, ub = [0.0]*Train_data.train_size, [np.inf]*Train_data.train_size
    nlp = cyipopt.Problem(n=len(w0),m=len(cl),problem_obj=IPOPT_problem(),lb=lb,ub=ub)
    nlp.add_option('mu_strategy', 'adaptive')# barrier parameter adaptive
    nlp.add_option('tol', 2e-4)
    #nlp.add_option('max_iter',1000)
    w_ours, info = nlp.solve(w0)
    return w_ours

def optimize_bound_subroutine(weights,epsilon,Train_data,Gamma,hat_r_z,hat_r_y,keep_record=0):
    train_size,P_train,P_test,G = Train_data.train_size,Train_data.P_train,Train_data.P_test,Train_data.G
    bw = np.append(weights, -np.ones(train_size)/train_size)
    Qw = np.diag(np.append(weights*weights,np.zeros(train_size)))
    vw = np.append(weights*weights*P_train,np.zeros(train_size))
    r0=np.append(hat_r_z,hat_r_y)
    #throw into solver
    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    alpha=m.addMVar (train_size*2, ub=np.inf,lb=-np.inf,name="alpha" )
    t=m.addMVar (1, ub=np.inf,lb=0,name="t" )
    m.setObjective((bw@G)@alpha+t*(2*math.log(1/epsilon))**0.5, GRB.MAXIMIZE)
    m.addConstr(t@t+alpha@(G@Qw@G+np.identity(train_size*2)*0.00000001)@alpha+(2*Qw@r0-vw)@alpha <= vw@r0-r0@Qw@r0)
    m.addConstr(alpha@G@alpha <= Gamma**2)
    m.addConstrs(G[i]@alpha <= (np.append(P_train,P_test)-r0)[i] for i in range(train_size*2))
    m.addConstrs(G[i]@alpha >= -r0[i] for i in range(train_size*2))
    m.params.method=0#0 is primal simplex, 1 is dual simplex, 2 is barrier
    m.update()
    m.optimize()
    assert m.status==2 or m.status==13#assert solver terminates successfully
    alpha_star=np.zeros(train_size*2)
    for i in range(0,train_size*2):
        alpha_star[i]=alpha[i].x
    t = t[0].x
    delta=G@alpha_star
    assert t>0 #otherwise will have division by 0
    qw = max(np.abs(weights)*P_train)
    obj_val = delta@bw+t*(2*math.log(1/epsilon))**0.5+qw*math.log(1/epsilon)/3
    r_z = hat_r_z+delta[0:train_size]
    grad_2nd_term= (2*math.log(1/epsilon))**0.5*0.5/t
    grad_2nd_term = grad_2nd_term*(2*weights*r_z*(P_train-r_z))
    grad_3rd_term = np.zeros(train_size)
    largest_index= np.argmax(np.abs(weights)*P_train)
    grad_3rd_term[largest_index]=P_train[largest_index]#replace by sign(weights[largest_index]), if weights are allowed to be negative
    grad_3rd_term = grad_3rd_term*math.log(1/epsilon)/3
    grad = delta[0:train_size]+grad_2nd_term+grad_3rd_term
    if keep_record==1:
        global record_arr
        new_row = [record_arr.shape[0],obj_val,qw*math.log(1/epsilon)/3,delta@bw,t*(2*math.log(1/epsilon))**0.5,max(weights)]
        record_arr = np.vstack((record_arr,new_row))
        if record_arr.shape[0]%100==0:
            print('iterate: ',record_arr.shape[0]-1,'objective: ',np.round(obj_val,4))
    return obj_val,grad


def truncate_r(Train_data,r_z,r_y):
    tol=0.01
    r_z1=np.minimum(Train_data.P_train-tol,r_z)
    r_y1 = np.minimum(Train_data.P_test-tol,r_y)
    r_z1, r_y1 = np.maximum(r_z1,tol),np.maximum(r_y1,tol)
    return r_z1,r_y1

def simulate_linear_MSE(TrainData,deg,returnEst):
    M=np.concatenate((TrainData.X_train,TrainData.P_train.reshape(-1,1)),axis=1)
    M=PolynomialFeatures(degree=deg, include_bias=True).fit_transform(M)
    M_test=np.concatenate((TrainData.X_train,TrainData.P_test.reshape(-1,1)),axis=1)
    M_test=PolynomialFeatures(degree=deg, include_bias=True).fit_transform(M_test)
    predict_rev_arr=np.zeros(TrainData.numSimulation)
    hat_r_z,hat_r_y=np.zeros((TrainData.numSimulation,TrainData.train_size)),np.zeros((TrainData.numSimulation,TrainData.train_size))
    for i in range(TrainData.numSimulation):
        model = LassoLarsCV(fit_intercept=False).fit(M, TrainData.D_train_simul[i])
        HT,alpha=model.coef_,model.alpha_
        predict_demand = np.minimum(np.maximum(M_test@HT,0),1)#truncate demand estimate
        predict_rev_arr[i]=predict_demand@TrainData.P_test/TrainData.train_size
        #for our method DR
        hat_r_y[i]=np.multiply(predict_demand,TrainData.P_test)#test data
        predict_demand_train = np.minimum(np.maximum(M@HT,0),1)
        hat_r_z[i]=np.multiply(predict_demand_train,TrainData.P_train)#train data
        
    simulated_MSE = (np.linalg.norm(predict_rev_arr-TrainData.expected_val_test)**2)/TrainData.numSimulation
    biasSq = (np.average(predict_rev_arr)-TrainData.expected_val_test)**2
    var = simulated_MSE - biasSq
    print('linear reg simulated MSE: ',simulated_MSE)
    if returnEst==0:
        return simulated_MSE,biasSq,var
    else:
        return simulated_MSE,biasSq,var,hat_r_z,hat_r_y

def simulate_reg_oracle_MSE(TrainData,methodName):
    deg_list = np.arange(1,5)
    mse_list,opt_biasSq,opt_var,biasSq,var  = np.ones(len(deg_list))*np.inf, 0, 0, 0, 0
    for i in range(len(deg_list)):
        if methodName=='linear':
            mse_list[i],biasSq,var = simulate_linear_MSE(TrainData,deg_list[i],0)
        else:
            mse_list[i],biasSq,var = simulate_logistic_MSE(TrainData,deg_list[i])
        if i==np.argmin(mse_list):
            opt_biasSq,opt_var = biasSq,var
    print(methodName,' optimal degree: ',deg_list[np.argmin(mse_list)])
    return min(mse_list), opt_biasSq,opt_var

def kernel_regression(Train_data,sigma):
    G,_ = set_up.compute_Gram_matrix(Train_data.P_train,Train_data.X_train,Train_data.P_test,Train_data.dim,sigma)
    train_size = Train_data.train_size
    w_NW=np.zeros(train_size)
    for i in range(0,train_size):
        w_NW+=G[0:train_size,train_size+i]/(sum(G[0:train_size,train_size+i])*train_size)
    return w_NW

def kernel_regression_oracle(Train_data):
    sigma_list=np.append(np.arange(1,10)*0.01,np.arange(1,10)*0.1)
    sigma_list=np.append(sigma_list,np.arange(1,10))
    sigma_list=np.append(sigma_list,np.arange(1,11)*10)    
    #sigma_list=np.append(sigma_list,np.arange(1,10)*10.0)
    MSE_NW_opt,w_NW_opt,optimal_sigma=np.inf,np.zeros(Train_data.train_size),0.0
    for l in range(0,len(sigma_list)):
        w_NW= kernel_regression(Train_data,sigma_list[l])
        MSE_NW,_,_ = set_up.true_MSE(w_NW,Train_data)
        if MSE_NW<MSE_NW_opt:
            MSE_NW_opt,w_NW_opt=MSE_NW,w_NW
            optimal_sigma=sigma_list[l]
    _,biasSq,variance, = set_up.true_MSE(w_NW_opt,Train_data)
    print('kernel reg true MSE: ',MSE_NW_opt,'bias^2: ',biasSq,'variance: ',variance)
    return w_NW_opt,optimal_sigma

def simulate_MSE(Train_data,weights,returnDetails=0):
    predict_rev_arr=np.zeros(Train_data.numSimulation)    
    for i in range(Train_data.numSimulation):
        predict_rev_arr[i] = weights@np.multiply(Train_data.P_train,Train_data.D_train_simul[i])
    simulated_MSE = (np.linalg.norm(predict_rev_arr-Train_data.expected_val_test)**2)/Train_data.numSimulation
    biasSq = (np.average(predict_rev_arr)-Train_data.expected_val_test)**2
    var = simulated_MSE - biasSq
    if returnDetails==0:
        return simulated_MSE,biasSq,var
    else:
        return predict_rev_arr






