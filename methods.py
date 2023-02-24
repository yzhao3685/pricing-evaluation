import numpy as np
import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoLarsCV
from sklearn.gaussian_process.kernels import RBF
import set_up
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from gurobipy import *
from scipy.optimize import NonlinearConstraint
import time
from sklearn.ensemble import GradientBoostingClassifier
import math
from sklearn import ensemble
import pdb


#helper function for SPPE method          
class direct_method_estimate: 
    def __init__(self,TrainData,idx=-1):
        D_arr = TrainData.D_train
        if idx!=-1:
            D_arr = TrainData.D_train_simul[idx]
        # assume semiparametric model, demand = a_fcn(z)-b_fcn(z)*p
        Deg1=1#degree of polynomial we fit a(z)
        Deg2=1#degree of polynomial we fit b(z)
        X1=PolynomialFeatures(degree=Deg1, include_bias=True).fit_transform(TrainData.Z_train)
        temp=PolynomialFeatures(degree=Deg2, include_bias=True).fit_transform(TrainData.Z_train)
        X2=np.multiply(TrainData.P_train.reshape(-1,1),temp)
        numCol1=np.shape(X1)[1]
        numCol2=np.shape(X2)[1]
        X=np.array([[0.0 for i in range(numCol1+numCol2)] for j in range (TrainData.train_size)]) 
        X[:,0:numCol1]=X1
        X[:,numCol1:]=-X2
        model = LassoLarsCV(fit_intercept=False).fit(X, D_arr)
        HT=model.coef_
        alpha=model.alpha_
        model = LassoLarsCV(fit_intercept=False).fit(X1, TrainData.P_train)
        g=model.coef_
        model2=LassoLarsCV(fit_intercept=False).fit(X1, np.multiply(
            TrainData.P_train,TrainData.P_train))
        g2=model2.coef_
    
        self.HT=HT
        self.numCol1=numCol1
        self.numCol2=numCol2
        self.g=g
        self.g2=g2

class doubly_robust_estimate:
    def __init__(self,TrainData,reg_result,idx=-1):
        D_arr = TrainData.D_train
        if idx!=-1:
            D_arr = TrainData.D_train_simul[idx]
        train_size=TrainData.train_size
        Xv=PolynomialFeatures(degree=1, include_bias=True).fit_transform(TrainData.Z_train)
        Xv2=PolynomialFeatures(degree=1, include_bias=True).fit_transform(TrainData.Z_train)
        a_DM=np.array([0.0]*train_size)
        b_DM=np.array([0.0]*train_size)
        a_DR=np.array([0.0]*train_size)
        b_DR=np.array([0.0]*train_size)
        res=np.array([0.0]*train_size)
        #estimate g and sigma
        g=reg_result.g
        g2=reg_result.g2
        g_z=Xv.dot(g)
        g2_z=Xv.dot(g2)
        varP=g2_z-np.multiply(g_z,g_z)
        
        for i in range(0,train_size):
            p=TrainData.P_train[i]
            A=[[1,-g_z[i]],[-g_z[i],g2_z[i]]]
            Ainv=np.linalg.pinv(A, rcond=1e-4)
            f=Ainv@[1,-p]
            
            a_DM[i]=reg_result.HT[0:reg_result.numCol1]@Xv[i]
            b_DM[i]=reg_result.HT[reg_result.numCol1:]@Xv2[i]
            res[i]=D_arr[i]-a_DM[i]+b_DM[i]*TrainData.P_train[i]

            a_DR[i]=a_DM[i]+f[0]*res[i]
            b_DR[i]=b_DM[i]+f[1]*res[i]
        counter=0
        for i in range(0,train_size):
            if b_DM[i]>0:
                counter=counter+1
              
        self.a_DM=a_DM
        self.b_DM=b_DM
        self.a_DR=a_DR
        self.b_DR=b_DR

class evaluate_policy:
    def __init__(self,TrainData,a_b_est):
        train_size=TrainData.train_size
        R_DM=np.array([0.0]*train_size)
        R_DR=np.array([0.0]*train_size)
        D_DM=np.array([0.0]*train_size)
        D_DR=np.array([0.0]*train_size)
        for i in range(0,train_size):
            p=TrainData.P_test[i]
            D_DM[i]=a_b_est.a_DM[i]-a_b_est.b_DM[i]*p
            D_DR[i]=a_b_est.a_DR[i]-a_b_est.b_DR[i]*p
            R_DM[i]=p*D_DM[i]
            R_DR[i]=p*D_DR[i]
        self.V_DM=np.average(R_DM)
        self.V_DR=np.average(R_DR)
        self.D_DR=D_DR
        self.D_DM=D_DM
        self.R_DM=R_DM
        self.R_DR=R_DR

def BOPE_heuristics(Train_data,hat_r_train,s_p_arr,s_x_arr,fix_gamma=0,fix_sigma=0):
    #compute heuristic choice of sigma, gamma, and kernel bandwidth s
    P_train =Train_data.P_train
    n=Train_data.train_size
    opt_log_likelihood=-np.inf
    opt_gamma=1;opt_sigma=1;opt_s_p=1;opt_s_x=1
    delta_r = Train_data.D_train*Train_data.P_train - hat_r_train
    log_likelihood_arr = np.zeros((len(s_p_arr),len(s_x_arr)))
    for j in range(len(s_p_arr)):
        for k in range(len(s_x_arr)):
            G = set_up.compute_Gram_matrix(P_train,Train_data.X_train,P_train,Train_data.dim,s_p_arr[j],s_x_arr[k])
            G = G[:n,:n] #only take the training part. ignore the testing part
            evals,evecs = scipy.linalg.eigh(G, eigvals_only=False)
            v = evecs@delta_r

            bounds = Bounds([0.01]*2, [50.0]*2, keep_feasible=False)
            res = minimize(BOPEE_log_likelihood, [5,5], args=(evals,v), method='trust-constr', jac=False,
                           tol=1e-4, options={'verbose': 0}, bounds=bounds)
            params = res.x;
            gamma=params[0];sigma=params[1]
            if fix_gamma>0:
                gamma= fix_gamma
            if fix_sigma>0:
                sigma = fix_sigma
            x = [gamma,sigma]
            log_likelihood = -BOPE_log_likelihood(x, evals, v)
            if log_likelihood > opt_log_likelihood:
                opt_log_likelihood=log_likelihood
                opt_gamma=gamma;opt_sigma=sigma;opt_s_p=s_p_arr[j];opt_s_x=s_x_arr[k]
            log_likelihood_arr[j,k]=log_likelihood
        print('x_p ',s_p_arr[j],'best likelihood ',opt_log_likelihood)
    return opt_gamma, opt_sigma,opt_s_p,opt_s_x, log_likelihood_arr


def BOPE_log_likelihood(x,evals,v):
    gamma= x[0];sigma=x[1];n=len(evals)
    log_likelihood = 0.5 * sum(np.log(evals * gamma ** 2 + sigma ** 2))
    log_likelihood += 0.5 * sum(np.divide(v*v,evals*gamma**2+sigma**2))
    return log_likelihood

def Bernstein_heuristics(Train_data,hat_r_train,s_p_arr,s_x_arr,gamma_arr=[]):
    #compute heuristic choice of kernel parameters s_p,s_x. s_p is the bandwidth for price. s_x is the bandwidth for feature
    if gamma_arr==[]:
        gamma_arr = np.ones(1)
    n=Train_data.train_size
    log_likelihood_arr = np.zeros((len(s_p_arr),len(s_x_arr)))
    best_gamma_arr = np.zeros((len(s_p_arr),len(s_x_arr)))
    P_train = Train_data.P_train
    for j in range(len(s_p_arr)):
        r_star0 = P_train / 2
        for k in range(len(s_x_arr)):
            G = set_up.compute_Gram_matrix(P_train,Train_data.X_train,P_train,Train_data.dim,s_p_arr[j],s_x_arr[k])
            G = G[:n,:n] #only take the training part. ignore the testing part
            G_inv = np.linalg.pinv(G, hermitian=True)
            likelihood_vs_gamma = np.zeros(len(gamma_arr))
            for l in range(len(gamma_arr)):
                gamma = gamma_arr[l]
                r_star,hess = log_posterior_density_opt(G_inv,Train_data,hat_r_train,r_star0,gamma)
                log_likelihood = -0.5*n*np.log(2*np.pi)
                log_likelihood += -0.5*(r_star-hat_r_train)@G_inv@(r_star-hat_r_train)/(gamma**2)
                determinant = np.linalg.det(hess@G*gamma**2+np.identity(n))
                log_likelihood += -0.5*np.log(determinant)

                r_star0=r_star #warm start
                likelihood_vs_gamma[l] = log_likelihood
            best_gamma_ind = np.argmax(likelihood_vs_gamma)
            best_gamma_arr[j,k] = gamma_arr[best_gamma_ind]
            log_likelihood_arr[j,k] = likelihood_vs_gamma[best_gamma_ind]
            print('s_p ',s_p_arr[j],'s_x',s_x_arr[k],'best gamma ',gamma_arr[best_gamma_ind],'det ',determinant)
    opt_s_p,opt_s_x,opt_s_p_ind,opt_s_x_ind = bernstein_find_opt_params_from_likelihood(log_likelihood_arr,s_p_arr,s_x_arr)
    opt_gamma = best_gamma_arr[opt_s_p_ind,opt_s_x_ind]
    return opt_s_p,opt_s_x,opt_gamma

def bernstein_find_opt_params_from_likelihood(log_likelihood_arr,s_p_arr,s_x_arr):
    #choose the smallest bandwidth such that log likelihood is no more than 0.5% away from optimal log likelihood
    opt_log_likelihood = np.max(log_likelihood_arr); counter=0;opt_s_p_ind=0; opt_s_p=0; opt_s_x=0; opt_s_x_ind=0
    for i in range(len(s_p_arr)):
        if max(log_likelihood_arr[i])> 1.001*opt_log_likelihood and counter==0:
            opt_s_p = s_p_arr[i]; counter+=1; opt_s_p_ind = i
    assert opt_s_p > 0
    counter=0
    for i in range(len(s_x_arr)):
        if log_likelihood_arr[opt_s_p_ind,i]> 1.001*opt_log_likelihood and counter==0:
            opt_s_x = s_x_arr[i]; counter+=1
    assert opt_s_x > 0
    return opt_s_p,opt_s_x,opt_s_p_ind,opt_s_x_ind

def log_posterior_density_opt(G_inv,Train_data,hat_r_train,r_star0,gamma):
    #compute r* that maximizes log posterior density
    R_train=Train_data.D_train*Train_data.P_train
    P_train = Train_data.P_train
    if r_star0==[]:
        r_star0=P_train/2
    #bound constraints are needed to avoid division by zero in hessian (results in crazy large hessian or numerical issues in solver)
    bounds = Bounds([0.1]*len(P_train), P_train-0.1,keep_feasible=True)
    #scipy only has 'minimize'. So we negate the objective function to do maximization
    res = minimize(log_posterior_density_obj_grad, r_star0,args=(R_train,P_train,G_inv,gamma,hat_r_train)
                   , method='trust-constr',jac=True,hess=log_posterior_density_hess,
                    tol=1e-4,options={'verbose':0}, bounds=bounds)
    r_star = res.x; opt_obj = res.fun
    diagonals =np.divide(R_train,P_train*r_star*r_star)\
               +np.divide(P_train-R_train,P_train*(P_train-r_star)*(P_train-r_star))
    if min(diagonals)<0: #sanity check
        print('H has negative values ',diagonals)
    assert min(diagonals) > 0
    hess = np.diag(diagonals)#Hessian of the marginal likelihood, not Hessian of the log posterior
    return r_star,hess

def log_posterior_density_obj_grad(r_star,R_train,P_train,G_inv,gamma,hat_r_train):
    #helper function for BOPE-Bern method heuristics bandwidth
    R_train_ratio = np.divide(R_train,P_train)
    r_star_ratio = np.divide(r_star,P_train)
    obj = R_train_ratio@np.log(r_star_ratio)+(1-R_train_ratio)@np.log(1-r_star_ratio)\
            - 0.5*(r_star-hat_r_train)@G_inv@(r_star-hat_r_train)/(gamma**2)
    grad = np.divide(R_train,P_train*r_star)-np.divide(1-R_train_ratio,P_train-r_star)\
           -G_inv@(r_star-hat_r_train)/(gamma**2)
    # scipy only has 'minimize'. So we negate the objective function to do maximization
    return -obj,-grad

def log_posterior_density_hess(r_star,R_train,P_train,G_inv,gamma,hat_r_train):
    #helper function for BOPE-Bern method heuristics bandwidth
    diagonals =np.divide(R_train,P_train*r_star*r_star)\
               +np.divide(P_train-R_train,P_train*(P_train-r_star)*(P_train-r_star))
    hess = -np.diag(diagonals)-G_inv/(gamma**2)
    # scipy only has 'minimize'. So we negate the objective function to do maximization
    return -hess

def BOPE_Bern_method(Train_data,Gamma,w0,hat_r_train,hat_r_test,epsilon=0.05,remove_simplex=1,verbose=0):
    train_size = Train_data.train_size
    hat_r_train, hat_r_test = truncate_r(Train_data,hat_r_train,hat_r_test)
    if w0==[]:
        w0=[1/train_size]*train_size
    global record_arr
    record_arr = np.zeros((1,6))
    sub_data = prepare_subproblem_data(epsilon,Gamma,hat_r_train,hat_r_test,Train_data)
    linear_constraint = LinearConstraint([1]*train_size, [1], [1])
    bounds = Bounds([0.0]*train_size, [1.0]*train_size)
    global last_delta_r
    last_delta_r=np.zeros(train_size*2)
    if remove_simplex==0:
        res = minimize(BOPE_Bern_subroutine, w0,args=(sub_data,Train_data,1), method='trust-constr',
                    jac=True,tol=0.000002,constraints=[linear_constraint],options={'verbose':verbose}, bounds=bounds)#default maxIter=1000
        w_ours = res.x
    else:
        res = minimize(BOPE_Bern_subroutine, w0,args=(sub_data,Train_data,1), method='trust-constr',
                    jac=True,tol=0.000002,options={'verbose':verbose})#default maxIter=1000
        w_ours = res.x
    rkhs_norm, _ = BOPE_Bern_subroutine(w_ours, sub_data, Train_data, 0, 2)
    print('BOPE-Bern method, rkhs norm of wc delta r ',rkhs_norm)
    return w_ours 

class prepare_subproblem_data: 
    def __init__(self,epsilon,Gamma,hat_r_train,hat_r_test,Train_data):
        n = Train_data.train_size
        hat_r_train, hat_r_test = truncate_r(Train_data, hat_r_train, hat_r_test)
        self.epsilon=epsilon;self.Gamma=Gamma;self.hat_r_train=hat_r_train;self.hat_r_test=hat_r_test
        self.r0=np.append(hat_r_train,hat_r_test)

def BOPE_Bern_subroutine(weights,sub_data,Train_data,keep_record=0,returnDetails=0):
    train_size,P_train,P_test,G_inv = Train_data.train_size,Train_data.P_train,Train_data.P_test,Train_data.G_inv
    epsilon=sub_data.epsilon;Gamma=sub_data.Gamma;r0=sub_data.r0
    bw = np.append(weights, -np.ones(train_size)/train_size)
    Qw = np.diag(np.append(weights*weights,np.zeros(train_size)))
    vw = np.append(weights*weights*P_train,np.zeros(train_size))

    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    delta=m.addMVar (train_size*2, ub=np.append(Train_data.P_train,Train_data.P_test)-r0,lb=-r0,name="delta" )
    t=m.addMVar (1, ub=100,lb=0,name="t" )
    m.setObjective(bw@delta+t*(2*math.log(1/epsilon))**0.5, GRB.MAXIMIZE)
    m.addConstr(t@t+delta@Qw@delta+(2*r0@Qw-vw)@delta <= vw@r0-r0@Qw@r0)
    m.addConstr(delta@G_inv@delta <= Gamma**2)
    m.update()
    #m.printStats()
    m.optimize()
    if m.status!=2 and m.status!=13:
        print('gurobi status code: ',m.status)
        print('quad cons RHS: ',vw @ r0 - r0 @ Qw @ r0,'vw@r0: ',vw@r0,'r0@Qw@r0: ',r0@Qw@r0)
        print('ub: ',ub)
    assert m.status==2 or m.status==13#assert solver terminates successfully
    delta_star=np.zeros(train_size*2)
    for i in range(0,train_size*2):
        delta_star[i]=delta[i].x
    t = t[0].x
    delta=delta_star
    assert t>=0 #otherwise will have division by 0
    global last_delta_r
    last_delta_r = delta
    qw = max(np.abs(weights)*P_train)
    obj_val = delta@bw+t*(2*math.log(1/epsilon))**0.5+qw*math.log(1/epsilon)/3
    r_train = sub_data.hat_r_train+delta[0:train_size]
    grad_2nd_term= (2*math.log(1/epsilon))**0.5*0.5/t
    grad_2nd_term = grad_2nd_term*(2*weights*r_train*(P_train-r_train))
    grad_3rd_term = np.zeros(train_size)
    largest_index= np.argmax(np.abs(weights)*P_train)
    grad_3rd_term[largest_index]=P_train[largest_index]*np.sign(weights[largest_index])
    grad_3rd_term = grad_3rd_term*math.log(1/epsilon)/3
    grad = delta[0:train_size]+grad_2nd_term+grad_3rd_term
    if returnDetails == 1:
        r_test = sub_data.hat_r_test+delta[train_size:]
        return np.append(r_train,r_test)
    elif returnDetails==2:#return rkhs norm of delta
        return delta@G_inv@delta,delta
    elif returnDetails==3:
        return delta
    if keep_record==1:
        global record_arr
        new_row = [record_arr.shape[0],obj_val,qw*math.log(1/epsilon)/3,delta@bw,t*(2*math.log(1/epsilon))**0.5,max(weights)]
        record_arr = np.vstack((record_arr,new_row))
    return obj_val,grad

def truncate_r(Train_data,r_train,r_test):
    tol=0.01
    r_train1=np.minimum(Train_data.P_train-tol,r_train)
    r_test1 = np.minimum(Train_data.P_test-tol,r_test)
    r_train1, r_test1 = np.maximum(r_train1,tol),np.maximum(r_test1,tol)
    return r_train1,r_test1

def BOPE_B_method_subroutine(weights,gamma,Train_data,hat_r_train, hat_r_test,returnDetails):
    train_size,P_train,P_test = Train_data.train_size,Train_data.P_train,Train_data.P_test
    r0=np.append(hat_r_train,hat_r_test)
    bw = np.append(weights, -np.ones(train_size) / train_size)
    Qw = np.diag(np.append(weights*weights,np.zeros(train_size)))
    vw = np.append(weights*weights*P_train,np.zeros(train_size))
    obj_quad_term = np.outer(bw, bw)
    G_inv = Train_data.G_inv
    def cons_f(x):
        return x@G_inv@x
    def cons_J(x):
        return 2*G_inv@x
    def cons_H(x,v):
        return 2*G_inv
    nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, gamma**2,jac=cons_J,hess=cons_H,keep_feasible=True)#rkhs norm
    ub = np.append(Train_data.P_train,Train_data.P_test)-r0+0.0001
    def obj_H(x):
        return -(2*obj_quad_term+vw)
    def obj_f_J(x): #negate the mse because scipy is doing minimization, and inner problem is doing maximization
        return -(x@obj_quad_term@x+vw@x-x@Qw@x), \
               -(2*obj_quad_term@x+vw-2*Qw@x)
    global last_delta_r
    x0 = np.zeros(train_size*2)
    bounds = Bounds(-r0-0.0001, ub)#default maxIter=1000
    res = minimize(obj_f_J, x0,method='trust-constr', jac=True,hess=obj_H, tol=0.01,
            constraints=[nonlinear_constraint],options={'verbose':0},bounds=bounds)#default maxIter=1000
    delta_r = res.x
    last_delta_r = delta_r
    #compute gradient
    delta_r_train=delta_r[0:train_size]
    delta_r_test=delta_r[train_size:2*train_size]
    r_train=delta_r_train+hat_r_train#worst case revenue=delta_r_train+hat_r_train
    r_test=delta_r_test+hat_r_test
    worst_bias=(weights@r_train-r_test@np.ones(train_size)/train_size)
    worst_var_arr = (Train_data.P_train-r_train)*r_train
    worst_var=weights@(weights*worst_var_arr)
    obj_val=worst_bias**2+worst_var
    grad=2*worst_bias*r_train+2*weights*worst_var_arr
    assert np.isinf(obj_val) == False
    assert np.isnan(obj_val) == False
    assert sum(np.isinf(grad)) == 0
    assert sum(np.isnan(grad)) == 0
    if returnDetails==0:
        return obj_val, grad
    elif returnDetails==1:
        return obj_val,worst_bias**2,worst_var


def BOPE_B_method(Train_data,Gamma,w0,hat_r_train, hat_r_test,remove_simplex=1):
    hat_r_train,hat_r_test = truncate_r(Train_data,hat_r_train,hat_r_test)
    train_size = Train_data.train_size
    if w0==[]:
        w0=np.ones(train_size)/train_size
    linear_constraint = LinearConstraint([1]*train_size, [1], [1])#weights sum to 1
    bounds = Bounds([0.0]*train_size, [100.0]*train_size)#default maxIter=1000
    tolerance=0.01
    fun=BOPE_B_method_subroutine
    global last_delta_r
    last_delta_r=np.zeros(train_size*2)
    if remove_simplex==0:
        res = minimize(fun, w0,args=(Gamma,Train_data,hat_r_train, hat_r_test,0),
                       method='trust-constr', jac=True,tol=tolerance,constraints=[linear_constraint],options={'verbose': 0}, bounds=bounds)
    else:
        res = minimize(fun, w0,args=(Gamma,Train_data,hat_r_train, hat_r_test,0),
                       method='trust-constr', jac=True,tol=tolerance,options={'verbose': 0})
    weights=res.x; wc_mse = res.fun
    print('BOPE-B method, rkhs norm of wc delta r ',last_delta_r@Train_data.G_inv@last_delta_r)
    return weights


def BOPE_method_subroutine(weights,Gamma,sigma,Train_data,r0):
    train_size = Train_data.train_size; G_inv = Train_data.G_inv
    bw = np.append(weights, -np.ones(train_size)/train_size)
    obj_quad_term = np.outer(bw,bw)
    def cons_f(x):
        return x @ G_inv @ x
    def cons_J(x):
        return 2 * G_inv @ x
    def cons_H(x, v):
        return 2 * G_inv
    nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, Gamma ** 2, jac=cons_J, hess=cons_H,
                                               keep_feasible=True)
    ub = np.append(Train_data.P_train , Train_data.P_test)-r0
    bounds = Bounds(-r0, ub)
    def obj_H(x):
        return -(2*obj_quad_term)
    def obj_f_J(x):
        return -(x @ obj_quad_term @ x), \
               -(2*obj_quad_term@x)
    x0 = np.zeros(train_size * 2)
    res = minimize(obj_f_J, x0, method='trust-constr', jac=True, hess=obj_H, tol=0.01,
                   constraints=[nonlinear_constraint],options={'verbose': 0},bounds=bounds)  # default maxIter=1000
    delta = res.x
    global last_delta_r; last_delta_r = delta
    if min(min(delta + r0), min(ub - delta)) < -0.1:
        print('violates bound constraint, lb violation: ',max(0-delta - r0),'ub violation: ',max(delta-ub))
    #compute gradient
    obj_val = delta@obj_quad_term@delta+(weights@weights)*sigma**2
    grad = (2*bw@delta)*delta[:train_size]+2*weights*sigma**2
    return obj_val, grad

def BOPE_method(Train_data,Gamma,sigma,w0,hat_r_train, hat_r_test,remove_simplex=1):
    hat_r_train,hat_r_test = truncate_r(Train_data,hat_r_train,hat_r_test)
    r0 = np.append(hat_r_train,hat_r_test)
    train_size = Train_data.train_size
    assert min(r0)>0
    if w0==[]:
        w0=np.ones(train_size)/train_size
    tolerance=0.01; fun=BOPE_method_DR_subroutine
    global last_delta_r; last_delta_r=np.zeros(2*train_size)
    if remove_simplex==1:
        res = minimize(fun, w0,args=(Gamma,sigma,Train_data,r0),
                       method='trust-constr', jac=True,tol=tolerance,options={'verbose': 0})
    else:
        linear_constraint = LinearConstraint([1] * train_size, [1], [1])  # weights sum to 1
        bounds = Bounds([0.0] * train_size, [100.0] * train_size)  # default maxIter=1000
        res = minimize(fun, w0,args=(Gamma,sigma,Train_data,r0),
                       method='trust-constr', jac=True,tol=tolerance,constraints=[linear_constraint],options={'verbose': 0}, bounds=bounds)
    weights=res.x; wc_mse = res.fun
    print('BOPE method, rkhs norm of wc delta r ',last_delta_r@Train_data.G_inv@last_delta_r)
    return weights


def simulate_linear_MSE(TrainData, deg, returnEst):
    M = np.concatenate((TrainData.X_train, TrainData.P_train.reshape(-1, 1)), axis=1)
    M = PolynomialFeatures(degree=deg, include_bias=True).fit_transform(M)
    M_test = np.concatenate((TrainData.X_train, TrainData.P_test.reshape(-1, 1)), axis=1)
    M_test = PolynomialFeatures(degree=deg, include_bias=True).fit_transform(M_test)
    predict_rev_arr = np.zeros(TrainData.numSimulation)
    hat_r_train, hat_r_test = np.zeros((TrainData.numSimulation, TrainData.train_size)), np.zeros(
        (TrainData.numSimulation, TrainData.train_size))
    for i in range(TrainData.numSimulation):
        model = LassoLarsCV(fit_intercept=False).fit(M, TrainData.D_train_simul[i])
        HT, alpha = model.coef_, model.alpha_
        predict_demand = np.minimum(np.maximum(M_test @ HT, 0), 1)  # truncate demand estimate
        predict_rev_arr[i] = predict_demand @ TrainData.P_test / TrainData.train_size
        # for BOPE-B method 
        hat_r_test[i] = np.multiply(predict_demand, TrainData.P_test)  # test data
        predict_demand_train = np.minimum(np.maximum(M @ HT, 0), 1)
        hat_r_train[i] = np.multiply(predict_demand_train, TrainData.P_train)  # train data

    simulated_MSE = (np.linalg.norm(predict_rev_arr - TrainData.expected_val_test) ** 2) / TrainData.numSimulation
    biasSq = (np.average(predict_rev_arr) - TrainData.expected_val_test) ** 2
    var = simulated_MSE - biasSq
    print('linear reg simulated MSE: ', simulated_MSE)
    if returnEst == 0:
        return simulated_MSE, biasSq, var
    else:
        return simulated_MSE, biasSq, var, hat_r_train, hat_r_test

def xgb_impute_counterfactual(X_train, P_train, P_test, D_train, deg, rndseed):
    # impute counterfactual for nomis pricing dataset
    M_train = np.concatenate((X_train, P_train.reshape(-1, 1)), axis=1)
    M_train = PolynomialFeatures(degree=deg, include_bias=True).fit_transform(M_train)
    M_test = np.concatenate((X_train, P_test.reshape(-1, 1)), axis=1)
    M_test = PolynomialFeatures(degree=deg, include_bias=True).fit_transform(M_test)
    
    model = ensemble.GradientBoostingRegressor(random_state=rndseed+8238).fit(M_train, D_train)
    predict_d_train = model.predict(M_train)
    predict_d_train = np.minimum(np.maximum(predict_d_train, 0), 1)  # truncate demand estimate
    predict_d_test = model.predict(M_test)
    predict_d_test = np.minimum(np.maximum(predict_d_test, 0), 1)  # truncate demand estimate

    train_mse = np.average(np.square(predict_d_train-D_train))
    print('xgb training mse: ',train_mse)
    
    return  predict_d_train, predict_d_test
