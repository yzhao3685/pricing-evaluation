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


#helper function for Victor's method          
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

def nathan_DR_heuristics(Train_data,hat_r_train,s_p_arr,s_x_arr,fix_gamma=0,fix_sigma=0):
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
            res = minimize(nathan_log_likelihood, [5,5], args=(evals,v), method='trust-constr', jac=False,
                           tol=1e-4, options={'verbose': 0}, bounds=bounds)
            params = res.x;
            gamma=params[0];sigma=params[1]
            if fix_gamma>0:
                gamma= fix_gamma
            if fix_sigma>0:
                sigma = fix_sigma
            x = [gamma,sigma]
            log_likelihood = -nathan_log_likelihood(x, evals, v)
            #log_likelihood = -res.fun #objective is negative log likelihood
            if log_likelihood > opt_log_likelihood:
                opt_log_likelihood=log_likelihood
                opt_gamma=gamma;opt_sigma=sigma;opt_s_p=s_p_arr[j];opt_s_x=s_x_arr[k]
            log_likelihood_arr[j,k]=log_likelihood
        print('x_p ',s_p_arr[j],'best likelihood ',opt_log_likelihood)
    return opt_gamma, opt_sigma,opt_s_p,opt_s_x, log_likelihood_arr


def nathan_log_likelihood(x,evals,v):
    gamma= x[0];sigma=x[1];n=len(evals)
    log_likelihood = 0.5 * sum(np.log(evals * gamma ** 2 + sigma ** 2))
    log_likelihood += 0.5 * sum(np.divide(v*v,evals*gamma**2+sigma**2))
    #log_likelihood += 0.5 * delta_r @ np.linalg.solve(G * gamma ** 2 + np.identity(n) * sigma ** 2, delta_r)
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

##def Bernstein_heuristics_debugging(Train_data,hat_r_train,s_p_arr,s_x_arr,gamma_arr=[]):
##    #compute heuristic choice of kernel parameters s_p,s_x. s_p is the bandwidth for price. s_x is the bandwidth for feature
##    if gamma_arr==[]:
##        gamma_arr = np.ones(1)
##    n=Train_data.train_size
##    log_likelihood_arr = np.zeros((len(s_p_arr),len(s_x_arr)))
##    best_gamma_arr = np.zeros((len(s_p_arr),len(s_x_arr)))
##    log_likelihood_2nd_term = np.zeros((len(s_p_arr), len(s_x_arr)))
##    log_likelihood_3rd_term = np.zeros((len(s_p_arr), len(s_x_arr)))
##    P_train = Train_data.P_train
##    for j in range(len(s_p_arr)):
##        r_star0 = P_train / 2
##        for k in range(len(s_x_arr)):
##            G,_ = set_up.compute_Gram_matrix(P_train,Train_data.X_train,P_train,Train_data.dim,s_p_arr[j],s_x_arr[k])
##            G = G[:n,:n] #only take the training part. ignore the testing part
##            G_inv = np.linalg.pinv(G, hermitian=True)
##            likelihood_vs_gamma = np.zeros(len(gamma_arr));second_term_vs_gamma = np.zeros(len(gamma_arr));
##            third_term_vs_gamma = np.zeros(len(gamma_arr));rkhs_norm_vs_gamma = np.zeros(len(gamma_arr))
##            for l in range(len(gamma_arr)):
##                gamma = gamma_arr[l]
##                r_star,hess = log_posterior_density_opt(G_inv,Train_data,hat_r_train,r_star0,gamma)
##                log_likelihood = -0.5*n*np.log(2*np.pi)
##                #log_likelihood = 0 #for debugging
##                log_likelihood += -0.5*(r_star-hat_r_train)@G_inv@(r_star-hat_r_train)/(gamma**2)
##                second_term_vs_gamma[l] = -0.5 * (r_star - hat_r_train) @ G_inv @ (r_star - hat_r_train) / (gamma ** 2)
##                determinant = np.linalg.det(hess@G*gamma**2+np.identity(n))
##                third_term_vs_gamma[l] =  -0.5 * np.log(determinant)
##                log_likelihood += -0.5*np.log(determinant)
##                rkhs_norm_vs_gamma[l] = (r_star-hat_r_train)@G_inv@(r_star-hat_r_train)
##
##                r_star0=r_star #warm start
##                likelihood_vs_gamma[l] = log_likelihood
##            best_gamma_ind = np.argmax(likelihood_vs_gamma)
##            best_gamma_arr[j,k] = gamma_arr[best_gamma_ind]
##            log_likelihood_arr[j,k] = likelihood_vs_gamma[best_gamma_ind]
##            log_likelihood_2nd_term[j, k] = second_term_vs_gamma[best_gamma_ind]
##            log_likelihood_3rd_term[j, k] = third_term_vs_gamma[best_gamma_ind]
##            print('s_p ',s_p_arr[j],'s_x',s_x_arr[k],'best gamma ',gamma_arr[best_gamma_ind],'det ',determinant,
##                  'rkhs norm ',rkhs_norm_vs_gamma[best_gamma_ind])
##
##    opt_s_p,opt_s_x,opt_s_p_ind,opt_s_x_ind = bernstein_find_opt_params_from_likelihood(log_likelihood_arr,s_p_arr,s_x_arr)
##    opt_gamma = best_gamma_arr[opt_s_p_ind,opt_s_x_ind]
##    return opt_s_p,opt_s_x,opt_gamma,log_likelihood_arr, log_likelihood_2nd_term,log_likelihood_3rd_term


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
    #scipy only has 'minimize'. So I negate the objective function to do maximization
    res = minimize(log_posterior_density_obj_grad, r_star0,args=(R_train,P_train,G_inv,gamma,hat_r_train)
                   , method='trust-constr',jac=True,hess=log_posterior_density_hess,
                    tol=1e-4,options={'verbose':0}, bounds=bounds)
    r_star = res.x; opt_obj = res.fun
    diagonals =np.divide(R_train,P_train*r_star*r_star)\
               +np.divide(P_train-R_train,P_train*(P_train-r_star)*(P_train-r_star))
    if min(diagonals)<0: #sanity check
        print('H has negative values ',diagonals)
    assert min(diagonals) > 0
    hess = np.diag(diagonals)#hessian of the marginal likelihood, not hessian of the log posterior
    return r_star,hess

def log_posterior_density_obj_grad(r_star,R_train,P_train,G_inv,gamma,hat_r_train):
    #helper function for bernstein method heuristics bandwidth
    R_train_ratio = np.divide(R_train,P_train)
    r_star_ratio = np.divide(r_star,P_train)
    obj = R_train_ratio@np.log(r_star_ratio)+(1-R_train_ratio)@np.log(1-r_star_ratio)\
            - 0.5*(r_star-hat_r_train)@G_inv@(r_star-hat_r_train)/(gamma**2)
    grad = np.divide(R_train,P_train*r_star)-np.divide(1-R_train_ratio,P_train-r_star)\
           -G_inv@(r_star-hat_r_train)/(gamma**2)
    # scipy only has 'minimize'. So I negate the objective function to do maximization
    return -obj,-grad

def log_posterior_density_hess(r_star,R_train,P_train,G_inv,gamma,hat_r_train):
    #helper function for bernstein method heuristics bandwidth
    diagonals =np.divide(R_train,P_train*r_star*r_star)\
               +np.divide(P_train-R_train,P_train*(P_train-r_star)*(P_train-r_star))
    hess = -np.diag(diagonals)-G_inv/(gamma**2)
    # scipy only has 'minimize'. So I negate the objective function to do maximization
    return -hess

def bernstein_method(Train_data,Gamma,w0,hat_r_train,hat_r_test,epsilon=0.05,remove_simplex=1,verbose=0):
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
        res = minimize(bernstein_subroutine, w0,args=(sub_data,Train_data,1), method='trust-constr',
                    jac=True,tol=0.000002,constraints=[linear_constraint],options={'verbose':verbose}, bounds=bounds)#default maxIter=1000
        w_ours = res.x
    else:
        res = minimize(bernstein_subroutine, w0,args=(sub_data,Train_data,1), method='trust-constr',
                    jac=True,tol=0.000002,options={'verbose':verbose})#default maxIter=1000
        w_ours = res.x
    rkhs_norm, _ = bernstein_subroutine(w_ours, sub_data, Train_data, 0, 2)
    print('bernstein method, rkhs norm of wc delta r ',rkhs_norm)
    return w_ours #record_arr[1:,:]

def bernstein_method_abs(Train_data,Gamma,w0,hat_r_train,hat_r_test,epsilon=0.05,remove_simplex=1,verbose=0):
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
        res = minimize(bernstein_abs_subroutine, w0,args=(sub_data,Train_data,1), method='trust-constr',
                    jac=True,tol=0.000002,constraints=[linear_constraint],options={'verbose':verbose}, bounds=bounds)#default maxIter=1000
        w_ours = res.x
    else:
        res = minimize(bernstein_abs_subroutine, w0,args=(sub_data,Train_data,1), method='trust-constr',
                    jac=True,tol=0.000002,options={'verbose':verbose})#default maxIter=1000
        w_ours = res.x
    rkhs_norm, _ = bernstein_abs_subroutine(w_ours, sub_data, Train_data, 0, 2)
    print('bernstein method, rkhs norm of wc delta r ',rkhs_norm)
    return w_ours #record_arr[1:,:]


class prepare_subproblem_data: 
    def __init__(self,epsilon,Gamma,hat_r_train,hat_r_test,Train_data):
        n = Train_data.train_size
        hat_r_train, hat_r_test = truncate_r(Train_data, hat_r_train, hat_r_test)
        self.epsilon=epsilon;self.Gamma=Gamma;self.hat_r_train=hat_r_train;self.hat_r_test=hat_r_test
        self.r0=np.append(hat_r_train,hat_r_test)

def bernstein_subroutine(weights,sub_data,Train_data,keep_record=0,returnDetails=0):
    train_size,P_train,P_test,G_inv = Train_data.train_size,Train_data.P_train,Train_data.P_test,Train_data.G_inv
    epsilon=sub_data.epsilon;Gamma=sub_data.Gamma;r0=sub_data.r0
    bw = np.append(weights, -np.ones(train_size)/train_size)
    Qw = np.diag(np.append(weights*weights,np.zeros(train_size)))
    vw = np.append(weights*weights*P_train,np.zeros(train_size))

    # #cons: t @ t + delta @ Qw @ delta + (2 * r0 @ Qw - vw) @ delta <= vw @ r0 - r0 @ Qw @ r0
    # cons_matrix = np.zeros((2*train_size+1,2*train_size+1))
    # cons_matrix[0,0] = 1; cons_matrix[1:,1:] = Qw; cons_linear = np.append(0,2 * r0 @ Qw - vw)
    # def cons_f(x):
    #     return x@cons_matrix@x+cons_linear@x
    # def cons_J(x):
    #     return 2*cons_matrix@x+cons_linear
    # def cons_H(x,v):
    #     return 2*cons_matrix
    # nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, vw @ r0 - r0 @ Qw @ r0,jac=cons_J,hess=cons_H,
    #                                            keep_feasible=True)
    # ub = np.append(Train_data.P_train,Train_data.P_test)-r0+0.0001
    # ub = np.append(np.inf,ub); lb = np.append(0,-r0 - 0.0001) # enforce t>0 to avoid division by 0 or overflow
    # #obj: bw@delta+t*(2*math.log(1/epsilon))**0.5-delta@G_inv@delta*(1/(2*Gamma**2)), GRB.MAXIMIZE
    # obj_matrix = np.zeros((2*train_size+1,2*train_size+1))
    # obj_matrix[1:,1:]=-G_inv/(2*Gamma**2); obj_linear = np.append(2*math.log(1/epsilon),bw)
    # def obj_H(x):
    #     return -(2*obj_matrix)
    # def obj_f_J(x): #negate the mse because scipy is doing minimization, and inner problem is doing maximization
    #     return -(x@obj_matrix@x+obj_linear@x), -(2*obj_matrix@x+obj_linear)
    # x0 = np.zeros(1+train_size*2)
    # bounds = Bounds(lb, ub)
    # res = minimize(obj_f_J, x0,method='trust-constr', jac=True,hess=obj_H, tol=0.01,options={'verbose':0},
    #     constraints=[nonlinear_constraint],bounds=bounds)#default maxIter=1000
    # delta = res.x[1:]; t= res.x[0]

    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    delta=m.addMVar (train_size*2, ub=np.append(Train_data.P_train,Train_data.P_test)-r0,lb=-r0,name="delta" )
    t=m.addMVar (1, ub=100,lb=0,name="t" )
    #m.setObjective(bw@delta+t*(2*math.log(1/epsilon))**0.5-delta@G_inv@delta*(1/(2*Gamma**2)), GRB.MAXIMIZE)#move cons to obj
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
        #if record_arr.shape[0]%100==0:
        #    print('iterate: ',record_arr.shape[0]-1,'objective: ',np.round(obj_val,4))
    return obj_val,grad


def bernstein_abs_subroutine(weights,sub_data,Train_data,keep_record=0,returnDetails=0):
    train_size,P_train,P_test,G_inv = Train_data.train_size,Train_data.P_train,Train_data.P_test,Train_data.G_inv
    epsilon=sub_data.epsilon;Gamma=sub_data.Gamma;r0=sub_data.r0
    bw = np.append(weights, -np.ones(train_size)/train_size)
    Qw = np.diag(np.append(weights*weights,np.zeros(train_size)))
    vw = np.append(weights*weights*P_train,np.zeros(train_size))
    # solve for the +bw@delta
    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    delta=m.addMVar (train_size*2, ub=np.append(Train_data.P_train,Train_data.P_test)-r0,lb=-r0,name="delta" )
    t=m.addMVar (1, ub=100,lb=0,name="t" )
    #m.setObjective(bw@delta+t*(2*math.log(1/epsilon))**0.5-delta@G_inv@delta*(1/(2*Gamma**2)), GRB.MAXIMIZE)#move cons to obj
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
    t_1 = t[0].x
    delta_star_1 = delta_star
    assert t_1 >= 0 #otherwise will have division by 0
    obj_positive_ = bw@delta_star_1 + t_1 * (2*math.log(1/epsilon))**0.5
    
    # solve for the -bw@delta
    m1 = Model()
    m1.Params.LogToConsole = 0#suppress Gurobipy printing
    delta1 = m1.addMVar (train_size*2, ub=np.append(Train_data.P_train,Train_data.P_test)-r0,lb=-r0,name="delta" )
    t1 = m1.addMVar (1, ub=100,lb=0,name="t" )
    #m.setObjective(bw@delta+t*(2*math.log(1/epsilon))**0.5-delta@G_inv@delta*(1/(2*Gamma**2)), GRB.MAXIMIZE)#move cons to obj
    m1.setObjective(-bw@delta1 + t1 * (2*math.log(1/epsilon))**0.5, GRB.MAXIMIZE)
    m1.addConstr(t1 @ t1 + delta1 @ Qw @ delta1 + (2*r0@Qw-vw) @ delta1 <= vw@r0-r0@Qw@r0)
    m1.addConstr(delta1 @ G_inv @ delta1 <= Gamma**2)
    m1.update()
    #m.printStats()
    m1.optimize()
    if m1.status!=2 and m1.status!=13:
        print('gurobi status code: ',m1.status)
        print('quad cons RHS: ',vw @ r0 - r0 @ Qw @ r0,'vw@r0: ',vw@r0,'r0@Qw@r0: ',r0@Qw@r0)
        print('ub: ', ub)
    assert m1.status==2 or m1.status==13#assert solver terminates successfully
    delta_star=np.zeros(train_size*2)
    for i in range(0,train_size*2):
        delta_star[i]=delta1[i].x
    t_2 = t1[0].x
    delta_star_2 = delta_star
    assert t_2 >= 0 #otherwise will have division by 0
    obj_negative_ = - bw @ delta_star_2 + t_2 * (2 * math.log(1/epsilon))**0.5

    # choose the best delta
    if obj_negative_ <= obj_positive_:
        delta = delta_star_1; t = t_1; sign = 1
    else:
        delta = delta_star_2; t = t_2; sign = -1
        
    global last_delta_r
    last_delta_r = delta
    qw = max(np.abs(weights)*P_train)
    obj_val = sign * delta@bw+t*(2*math.log(1/epsilon))**0.5+qw*math.log(1/epsilon)/3
    r_train = sub_data.hat_r_train+delta[0:train_size]
    grad_2nd_term= (2*math.log(1/epsilon))**0.5*0.5/t
    grad_2nd_term = grad_2nd_term*(2*weights*r_train*(P_train-r_train))
    grad_3rd_term = np.zeros(train_size)
    largest_index= np.argmax(np.abs(weights)*P_train)
    grad_3rd_term[largest_index]=P_train[largest_index]*np.sign(weights[largest_index])
    grad_3rd_term = grad_3rd_term*math.log(1/epsilon)/3
    grad = sign * delta[0:train_size]+grad_2nd_term+grad_3rd_term
    if returnDetails == 1:
        r_test = sub_data.hat_r_test+delta[train_size:]
        return np.append(r_train,r_test)
    elif returnDetails==2:#return rkhs norm of delta
        return delta@G_inv@delta,delta
    elif returnDetails==3:
        return delta, sign
    if keep_record==1:
        global record_arr
        new_row = [record_arr.shape[0],obj_val,qw*math.log(1/epsilon)/3,delta@bw,t*(2*math.log(1/epsilon))**0.5,max(weights)]
        record_arr = np.vstack((record_arr,new_row))
        #if record_arr.shape[0]%100==0:
        #    print('iterate: ',record_arr.shape[0]-1,'objective: ',np.round(obj_val,4))
    return obj_val,grad


def hoeffding_method(Train_data,Gamma,w0,hat_r_train,hat_r_test,epsilon=0.05,remove_simplex=1,verbose=0):
    train_size = Train_data.train_size
    hat_r_train, hat_r_test = truncate_r(Train_data,hat_r_train,hat_r_test)
    if w0==[]:
        w0=[1/train_size]*train_size
    global record_arr
    record_arr = np.zeros((1,6)); w_ours=w0
    sub_data = prepare_subproblem_data(epsilon,Gamma,hat_r_train,hat_r_test,Train_data)
    linear_constraint = LinearConstraint([1]*train_size, [1], [1])
    bounds = Bounds([0.0]*train_size, [1.0]*train_size)
    global last_delta_r
    last_delta_r=np.zeros(train_size*2)
    if remove_simplex==0:
        res = minimize(hoeffding_subroutine, w_ours,args=(sub_data,Train_data,1), method='trust-constr',
                    jac=True,tol=0.000002,constraints=[linear_constraint],options={'verbose':verbose}, bounds=bounds)#default maxIter=1000
        w_ours = res.x
    else:
        res = minimize(hoeffding_subroutine, w_ours,args=(sub_data,Train_data,1), method='trust-constr',
                    jac=True,tol=0.000002,options={'verbose':verbose})#default maxIter=1000
        w_ours = res.x
    rkhs_norm, _ = hoeffding_subroutine(w_ours, sub_data, Train_data, 0, 2)
    print('hoeffding method, rkhs norm of wc delta r ',rkhs_norm)
    return w_ours #record_arr[1:,:]

def hoeffding_subroutine(weights,sub_data,Train_data,keep_record=0,returnDetails=0):
    train_size,P_train,P_test,G_inv = Train_data.train_size,Train_data.P_train,Train_data.P_test,Train_data.G_inv
    epsilon=sub_data.epsilon;Gamma=sub_data.Gamma;r0=sub_data.r0
    bw = np.append(weights, -np.ones(train_size)/train_size)
    
    m = Model()
    m.Params.LogToConsole = 0#suppress Gurobipy printing
    delta=m.addMVar (train_size*2, ub=np.append(Train_data.P_train,Train_data.P_test)-r0,lb=-r0,name="delta" )
    m.setObjective(bw@delta, GRB.MAXIMIZE)
    m.addConstr(delta@G_inv@delta <= Gamma**2)
    m.update()
    m.optimize()
    if m.status!=2 and m.status!=13:
        print('gurobi status code: ',m.status)
        print('ub: ',ub)
    assert m.status==2 or m.status==13#assert solver terminates successfully
    delta_star=np.zeros(train_size*2)
    for i in range(0,train_size*2):
        delta_star[i]=delta[i].x
    delta=delta_star
    global last_delta_r
    last_delta_r = delta
    t = (weights * weights * P_train @ P_train)**0.5
    obj_val = delta@bw + t*(2*math.log(1/epsilon))**0.5
    grad_2nd_term = (2*math.log(1/epsilon))**0.5 * 0.5/t
    grad_2nd_term = grad_2nd_term  * (2 * weights * P_train * P_train)
    grad = delta[0:train_size] + grad_2nd_term
    if returnDetails == 1:
        r_test = sub_data.hat_r_test+delta[train_size:]
        return np.append(r_train,r_test)
    elif returnDetails==2:#return rkhs norm of delta
        return delta@G_inv@delta,delta
    elif returnDetails==3:
        return delta
    return obj_val,grad


def truncate_r(Train_data,r_train,r_test):
    tol=0.01
    r_train1=np.minimum(Train_data.P_train-tol,r_train)
    r_test1 = np.minimum(Train_data.P_test-tol,r_test)
    r_train1, r_test1 = np.maximum(r_train1,tol),np.maximum(r_test1,tol)
    return r_train1,r_test1


def min_mse_method_subroutine(weights,gamma,Train_data,hat_r_train, hat_r_test,returnDetails):
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
        # -(2*obj_quad_term+vw-2*Qw-G_inv/(gamma**2))
    def obj_f_J(x): #negate the mse because scipy is doing minimization, and inner problem is doing maximization
        return -(x@obj_quad_term@x+vw@x-x@Qw@x), \
               -(2*obj_quad_term@x+vw-2*Qw@x)
        #-(x @ obj_quad_term @ x + vw @ x - x @ Qw @ x - x @ G_inv @ x / (2 * gamma ** 2)), \
        #-(2 * obj_quad_term @ x + vw - 2 * Qw @ x - G_inv @ x / (gamma ** 2))
    global last_delta_r
    # x0=last_delta_r   #warm start from previous iterate's solution
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


def min_mse_method(Train_data,Gamma,w0,hat_r_train, hat_r_test,remove_simplex=1):
    hat_r_train,hat_r_test = truncate_r(Train_data,hat_r_train,hat_r_test)
    train_size = Train_data.train_size
    if w0==[]:
        w0=np.ones(train_size)/train_size
    linear_constraint = LinearConstraint([1]*train_size, [1], [1])#weights sum to 1
    bounds = Bounds([0.0]*train_size, [100.0]*train_size)#default maxIter=1000
    tolerance=0.01
    fun=min_mse_method_subroutine
    global last_delta_r
    last_delta_r=np.zeros(train_size*2)
    if remove_simplex==0:
        res = minimize(fun, w0,args=(Gamma,Train_data,hat_r_train, hat_r_test,0),
                       method='trust-constr', jac=True,tol=tolerance,constraints=[linear_constraint],options={'verbose': 0}, bounds=bounds)
    else:
        res = minimize(fun, w0,args=(Gamma,Train_data,hat_r_train, hat_r_test,0),
                       method='trust-constr', jac=True,tol=tolerance,options={'verbose': 0})
    weights=res.x; wc_mse = res.fun
    #print('sum weights: ',sum(weights))
    print('wc mse method, rkhs norm of wc delta r ',last_delta_r@Train_data.G_inv@last_delta_r)
    return weights


##def min_mse_method_subroutine_oracle(weights,Train_data,returnDetails=0):
##    #use true revenue, instead of w.c. revenue
##    train_size = Train_data.train_size
##    r_train=Train_data.expected_R_train
##    r_test=Train_data.expected_R_test
##    bias=(weights@r_train-r_test@np.ones(train_size)/train_size)
##    var_arr = (Train_data.P_train-r_train)*r_train
##    var=weights@(weights*var_arr)
##    obj_val=bias**2+var
##    grad=2*bias*r_train+2*weights*var_arr
##    #if returnDetails==0:
##    return obj_val, grad
##
##def min_mse_method_oracle(Train_data,w0,remove_simplex=1):
##    train_size = Train_data.train_size
##    if w0==[]:
##        w0=np.ones(train_size)/train_size
##    linear_constraint = LinearConstraint([1]*train_size, [1], [1])#weights sum to 1
##    bounds = Bounds([0.0]*train_size, [100.0]*train_size)#default maxIter=1000
##    tolerance=0.01
##    fun=min_mse_method_subroutine_oracle
##    res=[]
##    if remove_simplex==0:
##        res = minimize(fun, w0,args=(Train_data,0),
##                       method='trust-constr', jac=True,tol=tolerance,constraints=[linear_constraint],options={'verbose': 0}, bounds=bounds)
##    else:
##        res = minimize(fun, w0,args=(Train_data,0),
##                       method='trust-constr', jac=True,tol=tolerance,options={'verbose': 0})
##    weights=res.x; true_mse = res.fun
##    #print('sum weights: ',sum(weights))
##    return weights


def nathan_method_DR_subroutine(weights,Gamma,sigma,Train_data,r0):
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
        #-(2*obj_quad_term - G_inv/(Gamma**2))
    def obj_f_J(x):
        return -(x @ obj_quad_term @ x), \
               -(2*obj_quad_term@x)
        # return -(x @ obj_quad_term @ x - x@G_inv@x/(2*Gamma**2)), \
        #        -(2*(bw@x)*bw - G_inv@x/(Gamma**2)) #inner problem is maximizing wc mse
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

def nathans_method_DR(Train_data,Gamma,sigma,w0,hat_r_train, hat_r_test,remove_simplex=1):
    hat_r_train,hat_r_test = truncate_r(Train_data,hat_r_train,hat_r_test)
    r0 = np.append(hat_r_train,hat_r_test)
    train_size = Train_data.train_size
    assert min(r0)>0
    if w0==[]:
        w0=np.ones(train_size)/train_size
    tolerance=0.01; fun=nathan_method_DR_subroutine
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
    print('nathan method, rkhs norm of wc delta r ',last_delta_r@Train_data.G_inv@last_delta_r)
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
        # for our method DR
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

##def simulate_XGB_MSE(TrainData, deg, returnEst):
##    M = np.concatenate((TrainData.X_train, TrainData.P_train.reshape(-1, 1)), axis=1)
##    M = PolynomialFeatures(degree=deg, include_bias=True).fit_transform(M)
##    M_test = np.concatenate((TrainData.X_train, TrainData.P_test.reshape(-1, 1)), axis=1)
##    M_test = PolynomialFeatures(degree=deg, include_bias=True).fit_transform(M_test)
##    predict_rev_arr = np.zeros(TrainData.numSimulation)
##    hat_r_train, hat_r_test = np.zeros((TrainData.numSimulation, TrainData.train_size)), np.zeros(
##        (TrainData.numSimulation, TrainData.train_size))
##    for i in range(TrainData.numSimulation):
##        model = GradientBoostingClassifier().fit(M, TrainData.D_train_simul[i])
##
##        predict_d_test = model.predict(M_test)
##        predict_d_test = np.minimum(np.maximum(predict_d_test, 0), 1)  # truncate demand estimate
##        predict_rev_arr[i] = predict_d_test @ TrainData.P_test / TrainData.train_size
##        # for our method DR
##        hat_r_test[i] = np.multiply(predict_d_test, TrainData.P_test)  # test data
##        predict_d_train = model.predict(M)
##        predict_d_train = np.minimum(np.maximum(predict_d_train, 0), 1)
##        hat_r_train[i] = np.multiply(predict_d_train, TrainData.P_train)  # train data
##
##    simulated_MSE = (np.linalg.norm(predict_rev_arr - TrainData.expected_val_test) ** 2) / TrainData.numSimulation
##    biasSq = (np.average(predict_rev_arr) - TrainData.expected_val_test) ** 2
##    var = simulated_MSE - biasSq
##    print('xgboost simulated MSE: ', simulated_MSE)
##    if returnEst == 0:
##        return simulated_MSE, biasSq, var
##    else:
##        return simulated_MSE, biasSq, var, hat_r_train, hat_r_test

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


# loss functions method
def loss_functions_method(Train_data, simul_num=0, deg=2):
    P_ladder = np.arange(1,10)*2 + 1
    def old_policy_dist(x_arr): 
        # assume input x is only one feature. 1d array.
        output=np.zeros(len(P_ladder))
        loc1 = 7 + x_arr@np.array([1,-1]) * 0.5
        scale1 = 2
        for i in range(len(P_ladder)):
            a = P_ladder[i] - 1
            b = P_ladder[i] + 1
            prob1 = scipy.stats.norm.cdf(a, loc=loc1, scale = scale1)
            prob2 = scipy.stats.norm.cdf(b, loc=loc1, scale = scale1)
            output[i] = prob2 - prob1        
        return output#output a matrix, each row is probability of prices for one feature vector

    def new_policy_dist(x_arr): 
        # assume input x is only one feature. 1d array.
        output=np.zeros(len(P_ladder))
        loc1 = Train_data.synthetic_new_policy + x_arr@np.array([1,-1]) * 0.5
        scale1 = 2
        for i in range(len(P_ladder)):
            a = P_ladder[i] - 1
            b = P_ladder[i] + 1
            prob1 = scipy.stats.norm.cdf(a, loc=loc1, scale = scale1)
            prob2 = scipy.stats.norm.cdf(b, loc=loc1, scale = scale1)
            output[i] = prob2 - prob1        
        return output#output a matrix, each row is probability of prices for one feature vector
    def loss_function(valuation,x):#Input one single feature x
        # loss function is used to evaluate the new policy
        dist=new_policy_dist(x)
        output=0
        for i in range(0, len(P_ladder)):
            if valuation>=P_ladder[i]:
                output-=P_ladder[i]*dist[i]
        return output
    def compute_T(x):
        # should use old policy to compute T
        T=np.zeros((2*len(P_ladder), len(P_ladder) + 1))
        dist=old_policy_dist(x)
        for i in range(0, 2 * len(P_ladder)):
            for j in range(0, len(P_ladder) + 1):
                if i <= len(P_ladder) -1 and j > i:#note i<=nP-1 because first index is 0
                    T[i,j]=dist[i]
                elif i >= len(P_ladder) and j <= i - len(P_ladder):#note i>=len(P_ladder) because first index is 0
                    T[i,j]=dist[i - len(P_ladder)]
        return T

    def R_mv(T,f_y):
        #f_y is prob dist, and must be all positive under coverage assumption.Else inverse calculation will have problems
        assert len(f_y)==2 * len(P_ladder) and min(f_y) > 0
        matrix1=np.transpose(T)@np.diag(1/f_y)@T
        matrix2=np.transpose(T)@np.diag(1/f_y)
        return  np.linalg.solve(matrix1,matrix2)#return matrix1^{-1}@matrix2

    #linear regression demand prediction
    M = np.concatenate((Train_data.X_train, Train_data.P_train.reshape(-1, 1)), axis=1)
    M = PolynomialFeatures(degree=deg, include_bias=True).fit_transform(M)
    model2=LassoLarsCV(cv=5, normalize=False).fit(M, Train_data.D_train_simul[simul_num]) # use D_simul
    
    R_arr=np.zeros(Train_data.train_size)
    f_v_matrix=np.zeros((Train_data.train_size, len(P_ladder) + 1))
    for i in range(0, Train_data.train_size):
        x_arr=np.tile(Train_data.X_train[i],(len(P_ladder) , 1))
        z_arr=np.concatenate((x_arr,P_ladder.reshape(-1,1)),axis=1)
        M_counterfactual = PolynomialFeatures(degree=deg, include_bias=True).fit_transform(z_arr)
        conditional_prob_arr = model2.predict(M_counterfactual)#prob will purchase conditional on price and feature
        conditional_prob_arr = np.maximum(conditional_prob_arr, np.ones(len(P_ladder)) / 100) # avoid negative prob
        conditional_prob_arr = np.minimum(conditional_prob_arr, np.ones(len(P_ladder)) * 99 / 100) # avoid negative prob
        dist = old_policy_dist(Train_data.X_train[i])
        f_y=np.append(dist * conditional_prob_arr, dist * (1 - conditional_prob_arr)) 
        T=compute_T(Train_data.X_train[i]) 
        R=R_mv(T,f_y)
        f_v=R@f_y
        f_v_matrix[i]=f_v
        # up til here, we use the old policy to get customer valuation. Then we use learned valuation to evaluate new policy
        
        for j in range(0, len(P_ladder)):
            R_arr[i]-=f_v[j]*loss_function(P_ladder[j], Train_data.X_train[i]) #should use target policy in loss_function
    loss_functions_method_predict_rev = np.average(R_arr)
    return loss_functions_method_predict_rev
