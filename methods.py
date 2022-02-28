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
#import cyipopt
import time

def nathan_method_helper(w,A):
    return w@A@w,2*(A@w)
def nathan_method_hess(w,A):
    return A*2

def nathans_method(lamda,Train_data):
    train_size=Train_data.train_size
    G_aug=np.copy(Train_data.G)
    for i in range(0,train_size):
        G_aug[i][i]=G_aug[i][i]+lamda**2#add variance term
    evals,evecs=scipy.linalg.eigh(G_aug, eigvals_only=False)#this method assumes real symmetric matrix, and is fast
    A=np.diag(evals)
    c=np.sum(evecs[0:train_size,0:2*train_size],axis=0)
    #write optimization problem
    cons_Matrix = np.concatenate((np.transpose(c.reshape(-1,1)),evecs[train_size:2*train_size,0:2*train_size]),axis=0)
    cons_RHS = -np.ones(train_size+1)/train_size
    cons_RHS[0]=1
    linear_constraint = LinearConstraint(cons_Matrix, cons_RHS, cons_RHS)
    #bounds = Bounds([-100.0]*(2*train_size), [100.0]*(2*train_size))
    w0=[1/(train_size*2)]*(train_size*2)
    res = minimize(nathan_method_helper, w0,args=(A), method='trust-constr', jac=True,hess=nathan_method_hess,tol=0.0001,
               constraints=[linear_constraint],options={'verbose': 0,'maxiter':100})#, bounds=bounds)#default maxIter=1000
    w=evecs[0:train_size,0:2*train_size]@res.x
    return w

def nathans_method_oracle(Train_data):
    #nathans method with oracle param lambda
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

def nathans_heuristics(Train_data):
    #nathans heuristics to get some ok param lambda
    train_size = Train_data.train_size
    K = Train_data.G[0:train_size,0:train_size]
    evals=scipy.linalg.eigh(K, eigvals_only=True)
    lambda_list = np.append(np.arange(1,10),np.arange(1,11)*10)
    optimal_obj,optimal_lamda,optimal_gamma,optimal_alpha, optimal_p0 = np.inf,1,1,1,1
    for lamda in lambda_list:
        obj,_,gamma,alpha,p0 = neg_log_likelihood(lamda,Train_data,evals)
        if obj <=  optimal_obj:
            optimal_obj,optimal_lamda,optimal_gamma,optimal_alpha, optimal_p0 = obj,lamda,gamma,alpha,p0
    print('best alpha, p0: ',optimal_alpha, optimal_p0)
    return optimal_lamda,optimal_gamma

def simulate_MSE(Train_data,weights):
    #given any weights, we simulate the mse. Simulation uses oracle information (true demand function)
    predict_rev_arr=np.zeros(Train_data.numSimulation)    
    for i in range(Train_data.numSimulation):
        predict_rev_arr[i] = weights@np.multiply(Train_data.P_train,Train_data.D_train_simul[i])
    simulated_MSE = (np.linalg.norm(predict_rev_arr-Train_data.expected_val_test)**2)/Train_data.numSimulation
    biasSq = (np.average(predict_rev_arr)-Train_data.expected_val_test)**2
    var = simulated_MSE - biasSq
    return simulated_MSE,biasSq,var

def our_method(Train_data,Gamma,w0):
    #input w0 is some initial guess. output are final weights
    train_size = Train_data.train_size
    if w0==[]:
        w0=[1/train_size]*train_size
    linear_constraint = LinearConstraint([1]*train_size, [1], [1])#weights sum to 1
    bounds = Bounds([0.0]*train_size, [1.0]*train_size)
    res = minimize(subroutine, w0,args=(Gamma,Train_data,0), method='trust-constr', jac=True,tol=0.00001,
               constraints=[linear_constraint],options={'verbose':1}, bounds=bounds)#default maxIter=1000
    return res.x

def our_method_oracle(Train_data):
    #our method, choose oracle param gamma
    Gamma_list=np.append(np.arange(1,11)*0.2,np.arange(2,11)*2)
    #Gamma_list=max(rkhs_norm-5,0)+np.arange(11)
    train_size=Train_data.train_size
    w_ours = [1/train_size]*train_size
    MSE_ours_opt,w_ours_opt,optimal_gamma = np.inf,np.zeros(train_size),0.0
    MSE_arr = np.zeros(len(Gamma_list))
    for i in range(len(Gamma_list)):
        w_ours = our_method(Train_data,Gamma_list[i],w_ours)#warm start
        _,_,MSE_ours = set_up.true_MSE(w_ours,Train_data)
        print('Gamma: ',Gamma_list[i],'true MSE: ',MSE_ours)
        MSE_arr[i]=MSE_ours
        if MSE_ours<MSE_ours_opt:
            MSE_ours_opt,w_ours_opt,optimal_gamma=MSE_ours,w_ours,Gamma_list[i]
    print('ours true MSE: ',MSE_ours_opt,'optimal gamma: ',optimal_gamma)
    return MSE_ours_opt,w_ours_opt,optimal_gamma,MSE_arr,Gamma_list

def our_method_DR_simulate_MSE(Train_data,Gamma,hat_r_z,hat_r_y):
    #simulate the mse of our method. This is used if base regression is not kernel regression
    predict_rev_arr=np.zeros(Train_data.numSimulation)  
    w_ours = [1/Train_data.train_size]*Train_data.train_size  
    for i in range(Train_data.numSimulation):
        w_ours=our_method_DR(Train_data,Gamma,w_ours,hat_r_z[i], hat_r_y[i])#weights differ for different hat_r
        predict_rev_arr[i] = w_ours@np.multiply(Train_data.P_train,Train_data.D_train_simul[i])
        if i%10==0:
            print('run ',i,'finished')
    simulated_MSE = (np.linalg.norm(predict_rev_arr-Train_data.expected_val_test)**2)/Train_data.numSimulation
    biasSq = (np.average(predict_rev_arr)-Train_data.expected_val_test)**2
    var = simulated_MSE - biasSq
    print('ours DR simulated MSE: ',[simulated_MSE,biasSq,var])
    return simulated_MSE,biasSq,var

def our_method_DR(Train_data,Gamma,w0,hat_r_z, hat_r_y,method='nonconvex'):
    #input w0 is some initial guess. 1st output is weights. 
    time_start=time.time()
    hat_r_z,hat_r_y = truncate_r(Train_data,hat_r_z,hat_r_y)
    train_size = Train_data.train_size
    if w0==[]:
        w0=np.ones(train_size)/train_size
    linear_constraint = LinearConstraint([1]*train_size, [1], [1])#weights sum to 1
    bounds = Bounds([0.0]*train_size, [1.0]*train_size)#default maxIter=1000
    tolerance=0.01
    fun=subroutine_DR_nonconvex
    if method=='nonconvex':#use a nonconvex solver
        fun=subroutine_DR_nonconvex
    elif method=='binary search':
        #this does not impose bound constraints. without bound constraints, we can use binary search to solve the subproblem exactly
        fun=subroutine_DR_allow_negative
    else: #brute force search over a grid
        fun=subroutine_DR_brute_force
    res = minimize(fun, w0,args=(Gamma,Train_data,hat_r_z, hat_r_y,0), 
                   method='trust-constr', jac=True,tol=tolerance,constraints=[linear_constraint],options={'verbose': 2}, bounds=bounds)
    obj_final,_ = subroutine_DR_brute_force(res.x,Gamma,Train_data,hat_r_z, hat_r_y,0)
    weights=res.x
    time_elapsed = int(round(time.time()-time_start))
    print('trust region, method: ',method, ' obj: ',round(obj_final,2),'runtime: ',time_elapsed,'\n')
    return weights,obj_final,time_elapsed

def our_method_DR_keep_record(Train_data,Gamma,w0,hat_r_z, hat_r_y):
    #add two outputs: gradient and objective value at each iterate. This is only done for the brute force search. 
    time_start=time.time()
    hat_r_z,hat_r_y = truncate_r(Train_data,hat_r_z,hat_r_y)
    train_size = Train_data.train_size
    if w0==[]:
        w0=np.ones(train_size)/train_size
    linear_constraint = LinearConstraint([1]*train_size, [1], [1])#weights sum to 1
    bounds = Bounds([0.0]*train_size, [1.0]*train_size)#default maxIter=1000
    tolerance=0.01
    grad_arr,blackbox_obj_arr = [], []
    fun=subroutine_DR_brute_force_keep_record
    res = minimize(fun, w0,args=(Gamma,Train_data,hat_r_z, hat_r_y,grad_arr,blackbox_obj_arr), 
                   method='trust-constr', jac=True,tol=tolerance,constraints=[linear_constraint],options={'verbose': 2}, bounds=bounds)

    time_elapsed = int(round(time.time()-time_start))
    return res.x,res.fun,time_elapsed,grad_arr,blackbox_obj_arr


def our_method_DR_quadratic_cut(Train_data,Gamma,hat_r_z,hat_r_y,brute_force=0,returnDetails=0):
    #iterativel add quadratic cut
    time_start=time.time()
    hat_r_z,hat_r_y = truncate_r(Train_data,hat_r_z,hat_r_y)
    train_size = Train_data.train_size
    weights=np.ones(len(hat_r_z))/len(hat_r_z)
    t,sub_obj,i,certificate = 0,100,0,0
    linear_constraint = LinearConstraint([1]*train_size, [1], [1],keep_feasible=True)#weights sum to 1
    bounds = Bounds([0.0]*train_size, [1.0]*train_size)#default maxIter=1000
    grad, new_r_z, new_r_y = np.zeros(train_size),np.zeros(train_size),np.zeros(train_size)
    
    if brute_force==0:
        sub_obj,grad,new_r_z,new_r_y = subroutine_DR(weights,Gamma,Train_data,hat_r_z, hat_r_y,1)
    else:
        sub_obj,grad,new_r_z,new_r_y = subroutine_DR_brute_force(weights,Gamma,Train_data,hat_r_z, hat_r_y,1)
    r_z_arr,r_y_arr = np.vstack((hat_r_z,new_r_z)),np.vstack((hat_r_y,new_r_y))
    sub_obj_arr,t_arr = np.zeros(1000),np.zeros(1000)
    while certificate==0 and i<=1000:#we do use warm start
        res = minimize(mse_given_r, weights,args=(r_z_arr,r_y_arr,Train_data.P_train), 
                   method='trust-constr', jac=True,tol=0.01,constraints=[linear_constraint],options={'verbose':0}, bounds=bounds)
        weights,t = res.x,res.fun
        if brute_force==0:
            sub_obj,grad,new_r_z,new_r_y = subroutine_DR(weights,Gamma,Train_data,hat_r_z, hat_r_y,1)
        else:
            sub_obj,grad,new_r_z,new_r_y = subroutine_DR_brute_force(weights,Gamma,Train_data,hat_r_z, hat_r_y,1)
        sub_obj_arr[i],t_arr[i] = sub_obj,t
        i+=1
        r_z_arr,r_y_arr = np.vstack((r_z_arr,new_r_z)),np.vstack((r_y_arr,new_r_y))
        print('objective: ',t,' wc mse: ',sub_obj)
        if t >= sub_obj-0.01:
            certificate=1
            #print('grad: ',grad)#grad shows soln is not optimal

    if returnDetails==1:
        obj_final,_ = subroutine_DR_brute_force(weights,Gamma,Train_data,hat_r_z, hat_r_y,0)
        time_elapsed = int(round(time.time()-time_start))
        print('quadratic cut, brute force=',brute_force, ' obj: ',round(obj_final,2),'runtime: ',time_elapsed,'\n')
        return weights,obj_final,time_elapsed
    elif returnDetails==2:
        obj_final,_ = subroutine_DR_brute_force(weights,Gamma,Train_data,hat_r_z, hat_r_y,0)
        time_elapsed = int(round(time.time()-time_start))
        print('quadratic cut, brute force=',brute_force, ' obj: ',round(obj_final,2),'runtime: ',time_elapsed,'\n')
        return weights,obj_final,time_elapsed,sub_obj_arr[:i],t_arr[:i]
    else:
        return weights                          

def truncate_r(Train_data,r_z,r_y):
    #base regression may produce revenue <0 or revenue>price. We truncate it, otherwise the subproblem is infeasible. 
    tol=0.01
    r_z1=np.minimum(Train_data.P_train-tol,r_z)
    r_y1 = np.minimum(Train_data.P_test-tol,r_y)
    r_z1, r_y1 = np.maximum(r_z1,tol),np.maximum(r_y1,tol)
    return r_z1,r_y1

    
def our_method_DR_quadratic_cut_gurobi(Train_data,Gamma,hat_r_z,hat_r_y):
    #gurobi crashes if the number of quadratic constraints > 10
    hat_r_z, hat_r_y = truncate_r(Train_data,hat_r_z,hat_r_y)
    train_size,P_train,P_test = Train_data.train_size,Train_data.P_train,Train_data.P_test
    weights=np.ones(len(hat_r_z))/len(hat_r_z)
    t,subproblem,i,certificate = 0,100,0,0
    
    subproblem,_,new_r_z,new_r_y = subroutine_DR_brute_force(weights,Gamma,Train_data,hat_r_z, hat_r_y,1)
    new_r_z,new_r_y = truncate_r(Train_data,new_r_z,new_r_y)
    r_z_arr,r_y_arr = np.vstack((hat_r_z,new_r_z)),np.vstack((hat_r_y,new_r_y))
    while certificate==0 and i<=1000:
        num_cons=len(r_z_arr)
        r_y_avg_arr = np.average(r_y_arr,axis=1)
        H_arr = np.zeros((num_cons,train_size+1,train_size+1))
        b_arr = np.zeros((num_cons,train_size+1))
        for i in range(num_cons):
            tol=0.1
            assert min(min(r_z_arr[i]), min(P_train-r_z_arr[i]), min(r_y_arr[i]), min(P_test-r_y_arr[i])) >= 0
            var_arr = (Train_data.P_train-r_z_arr[i])*r_z_arr[i]
            H_arr[i][1:,1:] = np.outer(r_z_arr[i],r_z_arr[i])+np.diag(var_arr)
            b_arr[i][1:] = -2*r_y_avg_arr[i]*r_z_arr[i]
            b_arr[i][0] = -1
            H_arr[i][0,0]=0.0001
            evals = scipy.linalg.eigh(H_arr[i], eigvals_only=True)
            assert min(evals)>=0.0001
        m = Model()
        m.Params.LogToConsole = 0#suppress Gurobipy printing
        ub_arr = np.ones(train_size+1)
        ub_arr[0]=100
        tw=m.addMVar (train_size+1, ub=ub_arr,lb=np.zeros(train_size+1),name="tw" )#1st entry t, other entry weights    
        m.setObjective(tw[0], GRB.MINIMIZE)
        temp=np.ones(train_size+1)
        temp[0]=0
        m.addConstr(tw@temp==1)
        m.addConstrs(tw@H_arr[i]@tw+b_arr[i]@tw+r_y_avg_arr[i]**2 <= 0 for i in range(num_cons))
        m.params.method=1#0 is primal simplex, 1 is dual simplex, 2 is barrier
        m.params.NonConvex=0
        m.update()
        m.optimize()
        print('status: ',m.status)
        assert m.status==2 or m.status==13
        weights=np.zeros(train_size)
        for i in range(0,train_size):
            weights[i]=tw[i+1].x
        t = tw[0].x
        subproblem,_,new_r_z,new_r_y = subroutine_DR_brute_force(weights,Gamma,Train_data,hat_r_z, hat_r_y,1)
        new_r_z,new_r_y = truncate_r(Train_data,new_r_z,new_r_y)
        r_z_arr,r_y_arr = np.vstack((r_z_arr,new_r_z)),np.vstack((r_y_arr,new_r_y))
        if t >= subproblem-0.001:
            print('objective: ',t,' wc mse: ',subproblem)
            return weights,r_z_arr,r_y_arr
        print('objective: ',t,' wc mse: ',subproblem)
    return weights,r_z_arr,r_y_arr

def our_method_DR_linear_cut(Train_data,Gamma,hat_r_z,hat_r_y,brute_force=0,returnDetails=0):
    #iteratively add linear cut
    time_start=time.time()
    hat_r_z,hat_r_y = truncate_r(Train_data,hat_r_z,hat_r_y)
    train_size = Train_data.train_size
    weights=np.ones(len(hat_r_z))/len(hat_r_z)
    t,sub_obj,numIter,certificate = 0,100,0,0
    grad, new_r_z, new_r_y = np.zeros(train_size),np.zeros(train_size),np.zeros(train_size)
    if brute_force==0:
        sub_obj,grad,new_r_z,new_r_y = subroutine_DR(weights,Gamma,Train_data,hat_r_z, hat_r_y,1)
    else:
        sub_obj,grad,new_r_z,new_r_y = subroutine_DR_brute_force(weights,Gamma,Train_data,hat_r_z, hat_r_y,1)
    w_arr,grad_arr,obj_arr = weights.reshape((1,-1)), grad.reshape((1,-1)),[sub_obj]
    sub_obj_arr,t_arr = np.zeros(1000),np.zeros(1000)

    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    t_var=m.addMVar (1, ub=100.0,lb=-100.0,name="t" )#t_var is different from the constant t
    w=m.addMVar (train_size, ub=1.0,lb=0.0,name="w" )    
    m.setObjective(t_var*1, GRB.MINIMIZE) 
    m.addConstr(np.ones(train_size)@w==1.0)
    m.addConstr(t_var >= obj_arr[numIter]+w@grad_arr[numIter]-w_arr[numIter]@grad_arr[numIter])#add linear cut
    m.params.method=1#0 is primal simplex, 1 is dual simplex, 2 is barrier. Need dual simplex to do warm start
    
    while certificate==0 and numIter<=1000:
        numIter+=1
        if brute_force==0:
            sub_obj,grad,new_r_z,new_r_y = subroutine_DR(weights,Gamma,Train_data,hat_r_z, hat_r_y,1)
        else:
            sub_obj,grad,new_r_z,new_r_y = subroutine_DR_brute_force(weights,Gamma,Train_data,hat_r_z, hat_r_y,1)
        new_r_z,new_r_y = truncate_r(Train_data,new_r_z,new_r_y)
        w_arr,grad_arr,obj_arr = np.vstack((w_arr,weights)),np.vstack((grad_arr,grad)),np.append(obj_arr,sub_obj)

        m.addConstr(t_var >= obj_arr[numIter]+w@grad_arr[numIter]-w_arr[numIter]@grad_arr[numIter])
        #gurobi by default warm starts at previous solution, when dual simplex is used. 
        m.update()
        m.optimize()
        assert m.status==2
        weights=np.zeros(train_size)
        for i in range(train_size):
            weights[i]=w[i].x
        t=t_var[0].x

        sub_obj_arr[numIter-1],t_arr[numIter-1] = sub_obj,t
        if numIter%30==0:
            print('iterate: ',numIter,'objective: ',t,' wc mse: ',sub_obj)
        if t >= sub_obj-0.01:
            certificate=1
    if returnDetails==1:
        obj_final,_ = subroutine_DR_brute_force(weights,Gamma,Train_data,hat_r_z, hat_r_y,0)
        time_elapsed = int(round(time.time()-time_start))
        print('linear cut, brute force=',brute_force, ' obj: ',round(obj_final,2),'runtime: ',time_elapsed,'\n')
        return weights,obj_final,time_elapsed
    elif returnDetails==2:
        obj_final,_ = subroutine_DR_brute_force(weights,Gamma,Train_data,hat_r_z, hat_r_y,0)
        time_elapsed = int(round(time.time()-time_start))
        print('linear cut, brute force=',brute_force, ' obj: ',round(obj_final,2),'runtime: ',time_elapsed,'\n')
        return weights,obj_final,time_elapsed,sub_obj_arr[:numIter],t_arr[:numIter]
    else:
        return weights


def mse_given_r(weights,r_z_arr,r_y_arr,P_train):
    #input a finite set of revenue functions, output wc mse within the finite set of revenue functions
    worst_bias_arr = r_z_arr@weights-np.average(r_y_arr,axis=1)
    worst_var_arr = ((P_train-r_z_arr)*r_z_arr)@(weights*weights)
    mse_arr = worst_bias_arr*worst_bias_arr+worst_var_arr
    i = np.argmax(mse_arr)
    grad =2*worst_bias_arr[i]*r_z_arr[i] + 2*weights*(P_train-r_z_arr[i])*r_z_arr[i]  
    return max(mse_arr),grad

def subroutine_DR(weights,Gamma,Train_data,hat_r_z, hat_r_y,returnDetails,solver='Gurobi'):
    #solve the subproblem by throwing into a nonconvex solver.
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
    assert evals[0]<0 and evals[1]>-0.001#only one negative eval of Dw
    Q=np.copy(evecs)
    delta=np.copy(evals)
    S=np.linalg.solve(np.transpose(C),Q)
    #compute bw, epsilon
    temp2=np.multiply(np.multiply(weights,weights),Train_data.P_train)
    bw=-G[:,0:train_size]@temp2   
    #additional terms to bw that come from given estimate \hat r
    temp1=G[:,0:train_size]@weights-G[:,train_size:2*train_size]@np.ones(train_size)/train_size
    temp1=temp1*(weights@hat_r_z-hat_r_y@np.ones(train_size)/train_size)
    temp2=G[:,0:train_size]@(weights*weights*hat_r_z)
    bw=bw-2*temp1+2*temp2
    epsilon=np.transpose(S).dot(bw)
    #optimize
    lb = np.append(-hat_r_z,-hat_r_y)
    ub = np.append(Train_data.P_train-hat_r_z,Train_data.P_test-hat_r_y)
    consM,x_star = G@S,np.zeros(train_size*2)
    if solver=='Gurobi':
        m = Model()
        m.Params.LogToConsole = 0#suppress Gurobipy printing
        x=m.addMVar ( train_size*2, ub=100.0,lb=-100.0,name="x" )    
        m.setObjective(epsilon@x+x@np.diag(delta)@x*0.5, GRB.MINIMIZE) #epsilon@x+delta@y
        m.addConstr(x@x <=2*Gamma**2)
        m.addConstrs(consM[i]@x <= ub[i] for i in range(train_size*2))#cannot just add these ineq, because Ben-tal's reformulation no longer holds
        m.addConstrs(consM[i]@x >= lb[i] for i in range(train_size*2))
        m.params.NonConvex=2
        m.params.method=2#0 is primal simplex, 1 is dual simplex, 2 is barrier
        m.update()
        m.optimize()
        assert m.status==2
        for i in range(0,2*train_size):
            x_star[i]=x[i].x
    elif solver=='TR':#scipy tr solver not as good as Gurobi on this non-convex problem
        x0=np.zeros(train_size*2)
        def cons_f(x):
            return 0.5*x@x
        def cons_J(x):
            return x
        def cons_H(x,v):
            return np.identity(train_size*2)
        nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, Gamma**2,jac=cons_J,hess=cons_H,keep_feasible=True)#rkhs norm
        linear_constraint = LinearConstraint(consM, lb, ub,keep_feasible=True)#positive rev
        def obj_H(x,delta):
            return delta
        def obj_f_J(x,delta):
            return x@np.diag(delta)@x*0.5+epsilon@x, x@delta+epsilon       
        res = minimize(obj_f_J, x0,args=(delta), method='trust-constr', jac=True,hess=obj_H, tol=0.0001,
        constraints=[linear_constraint,nonlinear_constraint],options={'verbose':0})#default maxIter=1000
        x_star = res.x
    else: #scipy SLSQP solver currently not working. violates the rkhs cons
        x0=np.zeros(train_size*2)
        #ineq_cons is c(x) >= 0
        ineq_cons = {'type': 'ineq',
         'fun' : lambda x: np.concatenate((Gamma**2-x@x*0.5,ub-consM@x,consM@x-lb), axis=None),
         'jac' : lambda x: np.vstack((-x,-consM,consM))
                     }
        def obj_H(x,delta):
            return delta
        def obj_f_J(x,delta):
            return x@np.diag(delta)@x*0.5+epsilon@x, x@delta+epsilon       
        res = minimize(obj_f_J, x0,args=(delta), method='SLSQP', jac=True,hess=obj_H, tol=0.0001,
        constraints=[ineq_cons],options={'verbose':0})#default maxIter=1000
        x_star = res.x
        
    q=S@x_star
    if min(min(G@q-lb),min(ub-G@q)) < -0.0001:
        print('violates bound cons',' solver ',solver)
    if q@G@q> Gamma**2+0.0001:
        print('violates rkhs cons',' solver ',solver)
    #compute gradient
    delta_r_z=G[0:train_size,:]@q
    delta_r_y=G[train_size:2*train_size,:]@q
    r_z=delta_r_z+hat_r_z#worst case revenue=delta_r_z+hat_r_z
    r_y=delta_r_y+hat_r_y
    worst_bias=(weights@r_z-r_y@np.ones(train_size)/train_size)
    worst_var_arr = (Train_data.P_train-r_z)*r_z
    worst_var=weights@(weights*worst_var_arr)
    obj_val=worst_bias**2+worst_var
    grad=2*worst_bias*r_z+2*weights*worst_var_arr
    if returnDetails==0:
        return obj_val, grad
    elif returnDetails==1:
        return obj_val,grad,r_z,r_y
    elif returnDetails==2:
        return obj_val,worst_bias**2,worst_var,r_z,r_y
    elif returnDetails==3:
        return q@G@q,q
    else:
        return delta,epsilon,x_star,obj_val,grad

def subroutine_DR_brute_force(weights,Gamma,Train_data,hat_r_z, hat_r_y,returnDetails,solver='Gurobi'):
    #solve the subproblem by brute force search through a grid
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
    assert evals[0]<0 and evals[1]>-0.001#only one negative eval of Dw
    Q=np.copy(evecs)
    delta=np.copy(evals)
    S=np.linalg.solve(np.transpose(C),Q)
    #compute bw, epsilon
    temp2=np.multiply(np.multiply(weights,weights),Train_data.P_train)
    bw=-G[:,0:train_size]@temp2   
    #additional terms to bw that come from given estimate \hat r
    temp1=G[:,0:train_size]@weights-G[:,train_size:2*train_size]@np.ones(train_size)/train_size
    temp1=temp1*(weights@hat_r_z-hat_r_y@np.ones(train_size)/train_size)
    temp2=G[:,0:train_size]@(np.multiply(np.power(weights,2),hat_r_z))
    bw=bw-2*temp1+2*temp2   
    epsilon=np.transpose(S).dot(bw)
    #optimize    
    lb = np.append(-hat_r_z,-hat_r_y)
    ub = np.append(Train_data.P_train-hat_r_z,Train_data.P_test-hat_r_y)   
    grid = np.arange(-Gamma*(2**0.5),Gamma*(2**0.5),0.1)#constraint 0.5*sum_i x_i^2\le Gamma^2 forces bound on x[0]
    consM = G@S
    obj_gridSearch,x_star,feasible_count = np.inf,np.zeros(2*train_size),0
    obj_arr = -np.ones(len(grid))*np.inf
    obj_surrogate_arr = -np.ones(len(grid))*np.inf
    for i_x0 in grid:
        if solver=='Gurobi':
            m = Model()#x=0 should be feasible
            m.Params.LogToConsole = 0#suppress Gurobipy printing
            x=m.addMVar ( train_size*2-1, ub=100.0,lb=-100.0,name="x" )  #remove the 1st variable  
            m.setObjective(epsilon[1:]@x+x@np.diag(delta[1:])@x*0.5, GRB.MINIMIZE) #epsilon@x+delta@y
            m.addConstr(x@x+i_x0**2 <=2*Gamma**2)
            m.addConstrs(consM[i][1:]@x+consM[i][0]*i_x0 <= ub[i] for i in range(train_size*2))
            m.addConstrs(consM[i][1:]@x+consM[i][0]*i_x0  >= lb[i] for i in range(train_size*2))
            m.params.method=2#0 is primal simplex, 1 is dual simplex, 2 is barrier
            m.update()
            m.optimize()
            if m.status==2:#neither infeasible nor unbounded
                feasible_count+=1
                x_output=np.zeros(2*train_size)
                x_output[0]=i_x0
                for i in range(0,2*train_size-1):
                    x_output[i+1]=x[i].x
                obj_ = m.getObjective()
                obj = obj_.getValue()+epsilon[0]*i_x0+delta[0]*i_x0**2#add back removed terms in objective
                obj_arr[np.where(grid==i_x0)[0][0]] = obj
                obj_surrogate_arr[np.where(grid==i_x0)[0][0]] = epsilon[0]*i_x0+delta[0]*i_x0**2
                if obj < obj_gridSearch:
                    obj_gridSearch = obj
                    x_star = x_output
                    q=S@x_star
                    if min(min(G@q-lb),min(ub-G@q))<-0.0001:
                        print('violate ineq cons ')
                    if q@G@q> Gamma**2+0.0001:
                        print('violate rkhs cons')
        else: #scipy tr solver. currently not working, because it kills the entire program when subproblem is infeasible 
            def cons_f(x):
                return x@x*0.5+0.5*i_x0**2
            def cons_J(x):
                return x
            def cons_H(x,v):
                return np.identity(train_size*2-1)
            def obj_H(x,delta):
                return delta[1:]
            def obj_f_J(x,delta):
                return epsilon[1:]@x+x@np.diag(delta[1:])@x*0.5, x@delta[1:]+epsilon[1:]
            nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, Gamma**2,jac=cons_J,hess=cons_H,keep_feasible=True)#rkhs norm
            linear_constraint = LinearConstraint(consM[:,1:], lb, ub,keep_feasible=True)#positive rev
            bounds = Bounds([-np.inf]*(2*train_size-1), [np.inf]*(2*train_size-1))#fix x[0]
            x0=np.zeros(train_size*2-1)
            res = minimize(obj_f_J, x0,args=(delta), method='trust-constr', jac=True,hess=obj_H, tol=0.001,
                    constraints=[nonlinear_constraint,nonlinear_constraint],options={'verbose':0},bounds=bounds)#default maxIter=1000
            if res.success==True:#feasible
                q=S@res.x
                if min(min(G@q-lb),min(ub-G@q))<-0.0001:
                    print('violate ineq cons ')
                if q@G@q> Gamma**2+0.0001:
                    print('violate rkhs cons')
                    #print('x[0]=',round(res.x[0],2),' objective: ',round(res.fun,2))
                obj = res.fun+epsilon[0]*i_x0+delta[0]*i_x0**2#add back removed terms in objective
                if obj < obj_gridSearch:
                    obj_gridSearch = obj
                    x_star = res.x
                    
    if feasible_count==0:
        print('infeasible for all values on the grid')
    q=S@x_star
    #compute gradient
    delta_r_z=G[0:train_size,:]@q
    delta_r_y=G[train_size:2*train_size,:]@q
    r_z=delta_r_z+hat_r_z
    r_y=delta_r_y+hat_r_y
    worst_bias=(weights@r_z-r_y@np.ones(train_size)/train_size)
    worst_var_arr = (Train_data.P_train-r_z)*r_z
    worst_var=weights@(weights*worst_var_arr)
    obj_val=worst_bias**2+worst_var
    grad=2*worst_bias*r_z+2*weights*worst_var_arr
    if returnDetails==0:
        return obj_val, grad
    elif returnDetails==1:
        return obj_val,grad,r_z,r_y
    elif returnDetails==2:
        return obj_val,worst_bias**2,worst_var,r_z,r_y
    elif returnDetails==3:
        return q@G@q,q
    else:
        return obj_arr,obj_surrogate_arr,evals

def subroutine_DR_brute_force_keep_record(weights,Gamma,Train_data,hat_r_z, hat_r_y,grad_arr,blackbox_obj_arr):
    #only differs from"subroutine_DR_brute_force" in that we add two outputs: gradient and objective at each iteration
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
    assert evals[0]<0 and evals[1]>-0.001#only one negative eval of Dw
    Q=np.copy(evecs)
    delta=np.copy(evals)
    S=np.linalg.solve(np.transpose(C),Q)
    #compute bw, epsilon
    temp2=np.multiply(np.multiply(weights,weights),Train_data.P_train)
    bw=-G[:,0:train_size]@temp2   
    #additional terms to bw that come from given estimate \hat r
    temp1=G[:,0:train_size]@weights-G[:,train_size:2*train_size]@np.ones(train_size)/train_size
    temp1=temp1*(weights@hat_r_z-hat_r_y@np.ones(train_size)/train_size)
    temp2=G[:,0:train_size]@(np.multiply(np.power(weights,2),hat_r_z))
    bw=bw-2*temp1+2*temp2   
    epsilon=np.transpose(S).dot(bw)
    #optimize    
    lb = np.append(-hat_r_z,-hat_r_y)
    ub = np.append(Train_data.P_train-hat_r_z,Train_data.P_test-hat_r_y)   
    grid = np.arange(-Gamma*(2**0.5),Gamma*(2**0.5),0.1)#constraint 0.5*sum_i x_i^2\le Gamma^2 forces bound on x[0]
    consM = G@S
    obj_gridSearch,x_star,feasible_count = np.inf,np.zeros(2*train_size),0
    obj_arr = -np.ones(len(grid))*np.inf
    for i_x0 in grid:
        m = Model()#x=0 should be feasible
        m.Params.LogToConsole = 0#suppress Gurobipy printing
        x=m.addMVar ( train_size*2-1, ub=100.0,lb=-100.0,name="x" )  #remove the 1st variable  
        m.setObjective(epsilon[1:]@x+x@np.diag(delta[1:])@x*0.5, GRB.MINIMIZE) #epsilon@x+delta@y
        m.addConstr(x@x+i_x0**2 <=2*Gamma**2)
        m.addConstrs(consM[i][1:]@x+consM[i][0]*i_x0 <= ub[i] for i in range(train_size*2))
        m.addConstrs(consM[i][1:]@x+consM[i][0]*i_x0  >= lb[i] for i in range(train_size*2))
        m.params.method=2#0 is primal simplex, 1 is dual simplex, 2 is barrier
        m.update()
        m.optimize()
        if m.status==2:#neither infeasible nor unbounded
            feasible_count+=1
            x_output=np.zeros(2*train_size)
            x_output[0]=i_x0
            for i in range(0,2*train_size-1):
                x_output[i+1]=x[i].x
            obj_ = m.getObjective()
            obj = obj_.getValue()+epsilon[0]*i_x0+delta[0]*i_x0**2#add back removed terms in objective
            obj_arr[np.where(grid==i_x0)[0][0]] = obj
            if obj < obj_gridSearch:
                obj_gridSearch = obj
                x_star = x_output
                q=S@x_star
                if min(min(G@q-lb),min(ub-G@q))<-0.0001:
                    print('violate ineq cons ')
                if q@G@q> Gamma**2+0.0001:
                    print('violate rkhs cons')
    if feasible_count==0:
        print('infeasible for all values on the grid')
    q=S@x_star
    #compute gradient
    delta_r_z=G[0:train_size,:]@q
    delta_r_y=G[train_size:2*train_size,:]@q
    r_z=delta_r_z+hat_r_z
    r_y=delta_r_y+hat_r_y
    worst_bias=(weights@r_z-r_y@np.ones(train_size)/train_size)
    worst_var_arr = (Train_data.P_train-r_z)*r_z
    worst_var=weights@(weights*worst_var_arr)
    obj_val=worst_bias**2+worst_var
    grad=2*worst_bias*r_z+2*weights*worst_var_arr
    #update records
    grad_arr.append(grad)
    blackbox_obj_arr.append(obj_val)
    #print('obj arr:',blackbox_obj_arr)
    return obj_val, grad

def subroutine_DR_nonconvex(weights,Gamma,Train_data,hat_r_z, hat_r_y,returnDetails):#without diagonalization
    G,C,rowSumHalfG = Train_data.G,Train_data.C,Train_data.rowSumHalfG
    D3,arr_m,train_size = Train_data.D3,Train_data.arr_m,Train_data.train_size
    #compute D(w)
    wSumHalfG=(G[:,0:train_size].dot(weights)).reshape(-1,1)
    D2=2*wSumHalfG.dot(rowSumHalfG)/train_size#the second term in Dw
    D1=wSumHalfG.dot(np.transpose(wSumHalfG))#the first term in Dw
    for i in range(0,train_size):
        D1=D1-arr_m[i]*weights[i]**2
    Dw=-2*D1+D2+np.transpose(D2)-2*D3#Dw may not be PSD
    #compute bw, epsilon
    temp2=np.multiply(np.multiply(weights,weights),Train_data.P_train)
    bw=-G[:,0:train_size]@temp2   
    #additional terms to bw that come from given estimate \hat r
    temp1=G[:,0:train_size]@weights-G[:,train_size:2*train_size]@np.ones(train_size)/train_size
    temp1=temp1*(weights@hat_r_z-hat_r_y@np.ones(train_size)/train_size)
    temp2=G[:,0:train_size]@(np.multiply(np.power(weights,2),hat_r_z))
    bw=bw-2*temp1+2*temp2
    #truncate hat_r to make opti problem feasible
    hat_r_z = np.minimum(Train_data.P_train,hat_r_z)
    hat_r_y = np.minimum(Train_data.P_test,hat_r_y)
    #optimize    
    def cons_f(x):
        return x@G@x
    def cons_J(x):
        return 2*G@x
    def cons_H(x,v):
        return 2*G
    nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, Gamma**2,jac=cons_J,hess=cons_H,keep_feasible=True)#rkhs norm
    lb = np.append(-hat_r_z,-hat_r_y)
    ub = np.append(Train_data.P_train-hat_r_z,Train_data.P_test-hat_r_y)+0.0001
    linear_constraint = LinearConstraint(G, lb, ub,keep_feasible=False)#positive rev
    def obj_H(q,Dw,bw):
        return Dw
    def obj_f_J(q,Dw,bw):
        return 0.5*q@Dw@q+bw@q, Dw@q+bw
    q0=np.zeros(train_size*2)
    res = minimize(obj_f_J, q0,args=(Dw,bw), method='trust-constr', jac=True,hess=obj_H, tol=0.01,
    constraints=[linear_constraint,nonlinear_constraint],options={'verbose':0})#default maxIter=1000
    q = res.x
    if min(min(G@q-lb),min(ub-G@q))<-0.0001:
        print('violates bound constraint')
    #else:
    #    print('no constraint violation')
    #compute gradient
    delta_r_z=G[0:train_size,:]@q
    delta_r_y=G[train_size:2*train_size,:]@q
    r_z=delta_r_z+hat_r_z#worst case revenue=delta_r_z+hat_r_z
    r_y=delta_r_y+hat_r_y
    worst_bias=(weights@r_z-r_y@np.ones(train_size)/train_size)
    worst_var_arr = (Train_data.P_train-r_z)*r_z
    worst_var=weights@(weights*worst_var_arr)
    obj_val=worst_bias**2+worst_var
    grad=2*worst_bias*r_z+2*weights*worst_var_arr
    if returnDetails==0:
        return obj_val, grad
    elif returnDetails==1:
        return obj_val,worst_bias**2,worst_var
    elif returnDetails==2:
        return obj_val,worst_bias**2,worst_var,r_z,r_y
    else:
        return q@G@q,q


def subroutine_DR_allow_negative(weights,Gamma,Train_data,hat_r_z, hat_r_y,returnDetails):
    #assume we allow negative revenue. can use binary search over the dual to solve the subproblem. very fast
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
    #additional terms to bw that come from given estimate \hat r
    temp1=G[:,0:train_size]@weights-G[:,train_size:2*train_size]@np.ones(train_size)/train_size
    temp1=temp1*(weights@hat_r_z-hat_r_y@np.ones(train_size)/train_size)
    temp2=G[:,0:train_size]@(np.multiply(np.power(weights,2),hat_r_z))
    bw=bw-2*temp1+2*temp2   
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
    #obj = np.multiply(delta,x_star)@x_star/2+epsilon@x_star #can use as a sanity check. Note no var clipping here
    #obj_val = -obj #return phi(w) 
    q=S@x_star#worst case r()
    #compute gradient
    delta_r_z=G[0:train_size,:]@q
    delta_r_y=G[train_size:2*train_size,:]@q
    r_z=delta_r_z+hat_r_z#worst case revenue=delta_r_z+hat_r_z
    r_y=delta_r_y+hat_r_y
    worst_bias=(weights@r_z-r_y@np.ones(train_size)/train_size)
    worst_var_arr = (Train_data.P_train-r_z)*r_z
    #worst_var_arr = np.maximum(0.1,worst_var_arr)#variance clipping
    worst_var=weights@(weights*worst_var_arr)
    obj_val=worst_bias**2+worst_var
    grad=2*worst_bias*r_z+2*weights*worst_var_arr
    if returnDetails==0:
        return obj_val, grad
    elif returnDetails==1:
        return obj_val,worst_bias**2,worst_var
    elif returnDetails==2:
        return obj_val,worst_bias**2,worst_var,r_z,r_y
    else:
        return delta,epsilon,x_star,obj_val,grad

def simulate_linear_MSE(TrainData,deg,returnEst):
    #simulate MSE for LASSO regression estimate
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
        
        
def simulate_logistic_MSE(TrainData,deg):
    #simulate MSE for logistic regression estimate
    M=np.concatenate((TrainData.X_train,TrainData.P_train.reshape(-1,1)),axis=1)
    M=PolynomialFeatures(degree=deg, include_bias=True).fit_transform(M)
    M_test=np.concatenate((TrainData.X_train,TrainData.P_test.reshape(-1,1)),axis=1)
    M_test=PolynomialFeatures(degree=deg, include_bias=True).fit_transform(M_test)
    predict_rev_arr=np.zeros(TrainData.numSimulation)
    for i in range(TrainData.numSimulation):
        #smaller C suggest stronger regularization
        model = LogisticRegression(C=0.5, penalty='l1', solver='liblinear',max_iter=500).fit(M,TrainData.D_train_simul[i])
        #model = LogisticRegressionCV(penalty='l1',cv=2,max_iter=500).fit(M,TrainData.D_train_simul[i])#cv blows up mse
        predict_demand = model.predict(M_test)
        predict_rev_arr[i]=predict_demand@TrainData.P_test/TrainData.train_size

        #predict_probs = model.predict_proba(M_test)
        #predict_prob=predict_probs[:,0]#Prob[D=1]
        #predict_rev_arr[i]=predict_prob@TrainData.P_test/TrainData.train_size#continuous outcome blows up mse
    simulated_MSE = (np.linalg.norm(predict_rev_arr-TrainData.expected_val_test)**2)/TrainData.numSimulation
    biasSq = (np.average(predict_rev_arr)-TrainData.expected_val_test)**2
    var = simulated_MSE - biasSq
    print('logistic reg simulated MSE: ',simulated_MSE)
    return simulated_MSE,biasSq,var

def simulate_reg_oracle_MSE(TrainData,methodName):
    #simulate MSE for either linear regression or logistic regression, with oracle params (degree of polynomial)
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
    #kernel regression with oracle bandwidth sigma
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

def simulate_rf_MSE(Train_data,depth,useClassifier):
    #simulate MSE of random forest 
    regressor = []
    if useClassifier == 1:
        regressor = ensemble.RandomForestClassifier(max_depth=depth)
    else:
        regressor = ensemble.RandomForestRegressor(max_depth=depth)
    Z_arr = np.concatenate((Train_data.X_train,Train_data.P_train[:, None]),axis=1)
    Y_arr = np.concatenate((Train_data.X_train,Train_data.P_test[:, None]),axis=1)
    predict_rev_arr = np.zeros(Train_data.numSimulation)
    for i in range(Train_data.numSimulation):
        regressor.fit(Z_arr, Train_data.D_train_simul[i])
        D_pred = regressor.predict(Y_arr)
        D_pred = np.maximum(np.zeros(Train_data.train_size),np.minimum(D_pred,np.ones(Train_data.train_size)))#truncate demand predictions
        predict_rev_arr[i] = D_pred@Train_data.P_test/Train_data.train_size
    simulated_MSE = (np.linalg.norm(predict_rev_arr-Train_data.expected_val_test)**2)/Train_data.numSimulation
    biasSq = (np.average(predict_rev_arr)-Train_data.expected_val_test)**2
    var = simulated_MSE - biasSq
    print('random forest simulated MSE: ',simulated_MSE)
    return simulated_MSE,biasSq,var






