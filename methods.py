import numpy as np
import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoLarsCV
import statistics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process.kernels import RBF
import set_up
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from sklearn import ensemble
from gurobipy import *
from scipy.optimize import NonlinearConstraint
import time
from sklearn.ensemble import GradientBoostingClassifier

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
        MSE_nathan,_,_,_ = set_up.true_MSE(w_nathan,Train_data)
        MSE_arr[l]=MSE_nathan
        if MSE_nathan<MSE_nathan_opt:
            MSE_nathan_opt,w_nathan_opt=MSE_nathan,w_nathan
            optimal_lambda=lambda_list[l]
    print('nathans method true MSE: ',MSE_nathan_opt)
    return w_nathan_opt,optimal_lambda

def min_mse_gamma(Train_data,hat_r_train,hat_r_test,gamma_list,remove_simplex=1):
    #choose gamma that minimizes true mse
    n_gamma=len(gamma_list);mse_arr = np.zeros(n_gamma)
    all_weights_arr = np.zeros((n_gamma,Train_data.train_size))
    opt_gamma=0; opt_weights = 0; opt_mse=np.inf
    train_size = Train_data.train_size
    w_ours = [1 / train_size] * train_size
    hat_r_train, hat_r_test = truncate_r(Train_data,hat_r_train,hat_r_test) #else will cause negative variance issues
    for i in range(n_gamma):
        w,wc_mse = min_mse_method(Train_data,gamma_list[i],w_ours,hat_r_train, hat_r_test,remove_simplex)
        mse_arr[i],biasSq,var,estimate = set_up.true_MSE(w,Train_data)
        print('gamma: ',gamma_list[i],' true mse: ',mse_arr[i],'weights: ',w)
        if mse_arr[i] < opt_mse:
            opt_mse=mse_arr[i];opt_gamma=gamma_list[i];opt_weights=w
    return opt_weights,opt_gamma,all_weights_arr


def nathan_DR_heuristics_old(Train_data,hat_r_train,bandwidth_arr):
    #compute heuristic choice of sigma, gamma, and kernel bandwidth s
    R_train=Train_data.D_train*Train_data.P_train
    n=Train_data.train_size
    opt_log_likelihood=-np.inf
    opt_gamma=1;opt_sigma=1;opt_h=1
    normalized_train = (set_up.compute_Gram_matrix(Train_data.P_train,Train_data.X_train,Train_data.P_test,
        Train_data.dim,1,1))[0:n]
    delta_r = R_train - hat_r_train
    log_likelihood_arr = np.zeros(len(bandwidth_arr))
    for k in range(len(bandwidth_arr)):
        h = bandwidth_arr[k]
        kernel = RBF(h)
        G = kernel(normalized_train)
        G = G + 0.0001 * np.identity(n)  # make G positive semi-definite
        evals,evecs = scipy.linalg.eigh(G, eigvals_only=False)

        bounds = Bounds([0.01]*2, [50.0]*2, keep_feasible=True)
        res = minimize(nathan_log_likelihood, [1,1], args=(evals,delta_r,G), method='trust-constr', jac=False,
                       tol=1e-8, options={'verbose': 0}, bounds=bounds)
        params = res.x;
        gamma=params[0];sigma=params[1]
        log_likelihood = -res.fun #objective is negative log likelihood
        if log_likelihood > opt_log_likelihood:
            opt_log_likelihood=log_likelihood
            opt_gamma=gamma;opt_sigma=sigma;opt_h=h
        log_likelihood_arr[k]=log_likelihood
    return opt_gamma, opt_sigma,opt_h,log_likelihood_arr

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
            G,_ = set_up.compute_Gram_matrix(P_train,Train_data.X_train,P_train,Train_data.dim,s_p_arr[j],s_x_arr[k])
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

def bernstein_heuristics_gamma(Train_data,gamma_list,hat_r_train,hat_r_test,epsilon,remove_simplex,onlyGamma=0):
    #heuristics choice of gamma described in section 5.3 in Vishal's write up.
    #The "mse of estimator" approach of choosing gamma. This does not really work.
    # #Most of the time MSE(r0) does not even reach 5% of revenue estimate. and this ratio changes a lot across datasets/bandwidth
    counter =0; bern_gamma=gamma_list[-1]; n=Train_data.train_size
    r0_mse_arr = np.zeros(len(gamma_list)); all_weights_arr = np.zeros((len(gamma_list),Train_data.train_size))
    w_bernstein = np.ones(n)/n
    for i in range(len(gamma_list)):
        gamma = gamma_list[i]
        w_bernstein = optimize_bound_method(Train_data, gamma, w_bernstein, hat_r_train, hat_r_test, epsilon,
                                                    remove_simplex)#warm start
        r0_mse_arr[i] = w_bernstein*w_bernstein*(Train_data.P_train-hat_r_train)@hat_r_train
        all_weights_arr[i] = w_bernstein
        percentage = r0_mse_arr[i] / np.average(hat_r_train)
        print('gamma',gamma,'ro mse',r0_mse_arr[i],'% of base est',percentage)
        if percentage>0.05 and counter==0:
            bern_gamma = gamma;counter=1
            if onlyGamma==1:
                print('selecte gamma',bern_gamma)
                return bern_gamma
    print('selected gamma',bern_gamma)
    if onlyGamma == 1:
        return bern_gamma
    return bern_gamma,r0_mse_arr,all_weights_arr


def Bernstein_heuristics(Train_data,hat_r_train,s_p_arr,s_x_arr,gamma_arr=[]):
    #compute heuristic choice of kernel parameters s_p,s_x. s_p is the bandwidth for price. s_x is the bandwidth for feature
    if gamma_arr==[]:
        gamma_arr = np.ones(1)
    n=Train_data.train_size
    log_likelihood_arr = np.zeros((len(s_p_arr),len(s_x_arr)))
    log_likelihood_2nd_term = np.zeros((len(s_p_arr), len(s_x_arr)))
    log_likelihood_3rd_term = np.zeros((len(s_p_arr), len(s_x_arr)))
    P_train = Train_data.P_train
    for j in range(len(s_p_arr)):
        r_star0 = P_train / 2
        for k in range(len(s_x_arr)):
            G,_ = set_up.compute_Gram_matrix(P_train,Train_data.X_train,P_train,Train_data.dim,s_p_arr[j],s_x_arr[k])
            G = G[:n,:n] #only take the training part. ignore the testing part
            G_inv = np.linalg.pinv(G, hermitian=True)
            likelihood_vs_gamma = np.zeros(len(gamma_arr));second_term_vs_gamma = np.zeros(len(gamma_arr));
            third_term_vs_gamma = np.zeros(len(gamma_arr));rkhs_norm_vs_gamma = np.zeros(len(gamma_arr))
            for l in range(len(gamma_arr)):
                gamma = gamma_arr[l]
                r_star,hess = log_posterior_density_opt(G_inv,Train_data,hat_r_train,r_star0,gamma)
                log_likelihood = -0.5*n*np.log(2*np.pi)
                #log_likelihood = 0 #for debugging
                log_likelihood += -0.5*(r_star-hat_r_train)@G_inv@(r_star-hat_r_train)/(gamma**2)
                second_term_vs_gamma[l] = -0.5 * (r_star - hat_r_train) @ G_inv @ (r_star - hat_r_train) / (gamma ** 2)
                determinant = np.linalg.det(hess@G+np.identity(n))
                third_term_vs_gamma[l] =  -0.5 * np.log(determinant)
                log_likelihood += -0.5*np.log(determinant)
                rkhs_norm_vs_gamma[l] = (r_star-hat_r_train)@G_inv@(r_star-hat_r_train)

                r_star0=r_star #warm start
                likelihood_vs_gamma[l] = log_likelihood
            best_gamma = np.argmax(likelihood_vs_gamma)
            log_likelihood_arr[j,k] = likelihood_vs_gamma[best_gamma]
            log_likelihood_2nd_term[j, k] = second_term_vs_gamma[best_gamma]
            log_likelihood_3rd_term[j, k] = third_term_vs_gamma[best_gamma]
            print('s_p ',s_p_arr[j],'s_x',s_x_arr[k],'best gamma ',gamma_arr[best_gamma],'det ',determinant,
                  'rkhs norm ',rkhs_norm_vs_gamma[best_gamma])

    opt_s_p,opt_s_x = bernstein_find_opt_params_from_likelihood(log_likelihood_arr,s_p_arr,s_x_arr)
    return opt_s_p,opt_s_x,log_likelihood_arr, log_likelihood_2nd_term,log_likelihood_3rd_term

def bernstein_find_opt_params_from_likelihood(log_likelihood_arr,s_p_arr,s_x_arr):
    #choose the smallest bandwidth such that log likelihood is no more than 0.5% away from optimal log likelihood
    opt_log_likelihood = np.max(log_likelihood_arr); counter=0;opt_s_p_ind=0; opt_s_p=0; opt_s_x=0
    for i in range(len(s_p_arr)):
        if max(log_likelihood_arr[i])> 1.001*opt_log_likelihood and counter==0:
            opt_s_p = s_p_arr[i]; counter+=1; opt_s_p_ind = i
    assert opt_s_p > 0
    counter=0
    for i in range(len(s_x_arr)):
        if log_likelihood_arr[opt_s_p_ind,i]> 1.001*opt_log_likelihood and counter==0:
            opt_s_x = s_x_arr[i]; counter+=1
    assert opt_s_x > 0
    return opt_s_p,opt_s_x

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
    res=[]; record_arr = np.zeros((1,6)); w_ours=w0
    sub_data = prepare_subproblem_data(epsilon,Gamma,hat_r_train,hat_r_test,Train_data)
    linear_constraint = LinearConstraint([1]*train_size, [1], [1])
    bounds = Bounds([0.0]*train_size, [100.0]*train_size)
    if remove_simplex==0:
        res = minimize(bernstein_subroutine, w_ours,args=(sub_data,Train_data,1), method='trust-constr',
                    jac=True,tol=0.000002,constraints=[linear_constraint],options={'verbose':verbose}, bounds=bounds)#default maxIter=1000
        w_ours = res.x
    else:
        res = minimize(bernstein_subroutine, w_ours,args=(sub_data,Train_data,1), method='trust-constr',
                    jac=True,tol=0.000002,options={'verbose':verbose})#default maxIter=1000
        w_ours = res.x
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
    assert t>0 #otherwise will have division by 0
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
    if keep_record==1:
        global record_arr
        new_row = [record_arr.shape[0],obj_val,qw*math.log(1/epsilon)/3,delta@bw,t*(2*math.log(1/epsilon))**0.5,max(weights)]
        record_arr = np.vstack((record_arr,new_row))
        #if record_arr.shape[0]%100==0:
        #    print('iterate: ',record_arr.shape[0]-1,'objective: ',np.round(obj_val,4))
    return obj_val,grad

def oracle_bernstein_method(Train_data,w0,epsilon=0.05,remove_simplex=1,verbose=0):
    train_size = Train_data.train_size
    hat_r_train, hat_r_test = truncate_r(Train_data,Train_data.expected_R_train,Train_data.expected_R_test)
    if w0==[]:
        w0=[1/train_size]*train_size
    global record_arr
    res=[]; record_arr = np.zeros((1,6)); w_ours=w0; gamma=1#dummy gamma
    sub_data = prepare_subproblem_data(epsilon,gamma,hat_r_train,hat_r_test,Train_data)
    linear_constraint = LinearConstraint([1]*train_size, [1], [1])
    bounds = Bounds([0.0]*train_size, [100.0]*train_size)
    if remove_simplex==0:
        res = minimize(oracle_bernstein_method_subroutine, w_ours,args=(sub_data,Train_data,1), method='trust-constr',
                    jac=True,tol=0.0002,constraints=[linear_constraint],options={'verbose':verbose}, bounds=bounds)#default maxIter=1000
        w_ours = res.x
    else:
        res = minimize(oracle_bernstein_method_subroutine, w_ours,args=(sub_data,Train_data,1), method='trust-constr',
                    jac=True,tol=0.0002,options={'verbose':verbose})#default maxIter=1000
        w_ours = res.x
    return w_ours #record_arr[1:,:]

def oracle_bernstein_method_subroutine(weights,sub_data,Train_data,keep_record=0,returnDetails=0):
    train_size,P_train,P_test = Train_data.train_size,Train_data.P_train,Train_data.P_test
    r=np.append(Train_data.expected_R_train,Train_data.expected_R_test)
    bw = np.append(weights, -np.ones(train_size) / train_size)
    r0 = sub_data.r0;epsilon=sub_data.epsilon
    Qw = np.diag(np.append(weights*weights,np.zeros(train_size)))
    vw = np.append(weights*weights*P_train,np.zeros(train_size))
    var = vw@r0-r0@Qw@r0#terms associated with delta are all zeros
    t = var**0.5
    assert t>0 #otherwise will have division by 0
    qw = max(np.abs(weights)*P_train)
    obj_val = bw@(r-r0) + t*(2*math.log(1/epsilon))**0.5+qw*math.log(1/epsilon)/3
    r_train =Train_data.expected_R_train
    grad_2nd_term= (2*math.log(1/epsilon))**0.5*0.5/t
    grad_2nd_term = grad_2nd_term*(2*weights*r_train*(P_train-r_train))
    grad_3rd_term = np.zeros(train_size)
    largest_index= np.argmax(np.abs(weights)*P_train)
    grad_3rd_term[largest_index]=P_train[largest_index]*np.sign(weights[largest_index])
    grad_3rd_term = grad_3rd_term*math.log(1/epsilon)/3
    grad = (r_train-sub_data.hat_r_train)+ grad_2nd_term+grad_3rd_term
    #print('bias term in oracle bound: ',bw@(r-r0))
    if keep_record==1:
        global record_arr
        new_row = [record_arr.shape[0],obj_val,qw*math.log(1/epsilon)/3,0,t*(2*math.log(1/epsilon))**0.5,max(weights)]
        record_arr = np.vstack((record_arr,new_row))
        #if record_arr.shape[0]%100==0:
        #    print('iterate: ',record_arr.shape[0]-1,'objective: ',np.round(obj_val,4))
    return obj_val,grad

def truncate_r(Train_data,r_train,r_test):
    tol=0.01
    r_train1=np.minimum(Train_data.P_train-tol,r_train)
    r_test1 = np.minimum(Train_data.P_test-tol,r_test)
    r_train1, r_test1 = np.maximum(r_train1,tol),np.maximum(r_test1,tol)
    return r_train1,r_test1


def min_mse_method_subroutine(weights,Gamma,Train_data,hat_r_train, hat_r_test,returnDetails):#without diagonalization
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
    temp1=temp1*(weights@hat_r_train-hat_r_test@np.ones(train_size)/train_size)
    temp2=G[:,0:train_size]@(np.multiply(np.power(weights,2),hat_r_train))
    bw=bw-2*temp1+2*temp2
    #truncate hat_r to make opti problem feasible
    hat_r_train = np.minimum(Train_data.P_train,hat_r_train)
    hat_r_test = np.minimum(Train_data.P_test,hat_r_test)
    #optimize
    def cons_f(x):
        return x@G@x
    def cons_J(x):
        return 2*G@x
    def cons_H(x,v):
        return 2*G
    nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, Gamma**2,jac=cons_J,hess=cons_H,keep_feasible=True)#rkhs norm
    lb = np.append(-hat_r_train,-hat_r_test)
    ub = np.append(Train_data.P_train-hat_r_train,Train_data.P_test-hat_r_test)+0.0001
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
    #compute gradient
    delta_r_train=G[0:train_size,:]@q
    delta_r_test=G[train_size:2*train_size,:]@q
    r_train=delta_r_train+hat_r_train#worst case revenue=delta_r_train+hat_r_train
    r_test=delta_r_test+hat_r_test
    worst_bias=(weights@r_train-r_test@np.ones(train_size)/train_size)
    worst_var_arr = (Train_data.P_train-r_train)*r_train
    worst_var=weights@(weights*worst_var_arr)
    obj_val=worst_bias**2+worst_var
    grad=2*worst_bias*r_train+2*weights*worst_var_arr
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
    res=[]
    if remove_simplex==0:
        res = minimize(fun, w0,args=(Gamma,Train_data,hat_r_train, hat_r_test,0),
                       method='trust-constr', jac=True,tol=tolerance,constraints=[linear_constraint],options={'verbose': 0}, bounds=bounds)
    else:
        res = minimize(fun, w0,args=(Gamma,Train_data,hat_r_train, hat_r_test,0),
                       method='trust-constr', jac=True,tol=tolerance,options={'verbose': 0})
    weights=res.x; wc_mse = res.fun
    #print('sum weights: ',sum(weights))
    return weights,wc_mse

def min_mse_method_subroutine_oracle(weights,Train_data,returnDetails=0):
    #use true revenue, instead of w.c. revenue
    train_size = Train_data.train_size
    r_train=Train_data.expected_R_train
    r_test=Train_data.expected_R_test
    bias=(weights@r_train-r_test@np.ones(train_size)/train_size)
    var_arr = (Train_data.P_train-r_train)*r_train
    var=weights@(weights*var_arr)
    obj_val=bias**2+var
    grad=2*bias*r_train+2*weights*var_arr
    #if returnDetails==0:
    return obj_val, grad

def min_mse_method_oracle(Train_data,w0,remove_simplex=1):
    train_size = Train_data.train_size
    if w0==[]:
        w0=np.ones(train_size)/train_size
    linear_constraint = LinearConstraint([1]*train_size, [1], [1])#weights sum to 1
    bounds = Bounds([0.0]*train_size, [100.0]*train_size)#default maxIter=1000
    tolerance=0.01
    fun=min_mse_method_subroutine_oracle
    res=[]
    if remove_simplex==0:
        res = minimize(fun, w0,args=(Train_data,0),
                       method='trust-constr', jac=True,tol=tolerance,constraints=[linear_constraint],options={'verbose': 0}, bounds=bounds)
    else:
        res = minimize(fun, w0,args=(Train_data,0),
                       method='trust-constr', jac=True,tol=tolerance,options={'verbose': 0})
    weights=res.x; true_mse = res.fun
    #print('sum weights: ',sum(weights))
    return weights


def nathan_method_DR_subroutine(weights,Gamma,sigma,Train_data,r0):
    train_size = Train_data.train_size; G_inv = Train_data.G_inv
    bw = np.append(weights, -np.ones(train_size)/train_size)
    obj_quad_term = np.outer(bw,bw)

    # m = Model()
    # m.Params.LogToConsole = 0#suppress Gurobipy printing
    # delta=m.addMVar (train_size*2, ub=np.append(Train_data.P_train,Train_data.P_test)-r0,lb=-r0,name="delta" )
    # m.setObjective(delta @ obj_quad_term @ delta, GRB.MAXIMIZE) #maximize a convex function
    # m.addConstr(delta @ G_inv @ delta <= Gamma**2)
    # m.params.NonConvex = 2
    # m.update()
    # #m.printStats()
    # m.optimize()
    # if m.status!=2 and m.status!=13:
    #     print('gurobi status code: ',m.status)
    #     print('quad cons RHS: ',vw @ r0 - r0 @ Qw @ r0,'vw@r0: ',vw@r0,'r0@Qw@r0: ',r0@Qw@r0)
    #     print('ub: ',ub)
    # assert m.status==2 or m.status==13#assert solver terminates successfully
    # delta_star=np.zeros(train_size*2)
    # for i in range(0,train_size*2):
    #     delta_star[i]=delta[i].x
    # delta = delta_star

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
        return -2*obj_quad_term
    def obj_f_J(x):
        return -x @ obj_quad_term @ x, -2*(bw@x)*bw #inner problem is maximizing wc mse
    x0 = np.zeros(train_size * 2)
    res = minimize(obj_f_J, x0, method='trust-constr', jac=True, hess=obj_H, tol=0.01,
                   constraints=[nonlinear_constraint],options={'verbose': 0},bounds=bounds)  # default maxIter=1000
    delta = res.x
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
    tolerance=0.01; fun=nathan_method_DR_subroutine ; res=[]
    if remove_simplex==1:
        res = minimize(fun, w0,args=(Gamma,sigma,Train_data,r0),
                       method='trust-constr', jac=True,tol=tolerance,options={'verbose': 0})
    else:
        linear_constraint = LinearConstraint([1] * train_size, [1], [1])  # weights sum to 1
        bounds = Bounds([0.0] * train_size, [100.0] * train_size)  # default maxIter=1000
        res = minimize(fun, w0,args=(Gamma,sigma,Train_data,r0),
                       method='trust-constr', jac=True,tol=tolerance,constraints=[linear_constraint],options={'verbose': 0}, bounds=bounds)
    weights=res.x; wc_mse = res.fun
    return weights

def nathans_method_DR_oracle(Train_data,Gamma,hat_r_train, hat_r_test,remove_simplex):
    lambda_list=np.append(np.arange(1,10)*0.1,np.arange(1,10))
    MSE_nathan_opt,w_nathan_opt,optimal_lambda=np.inf,np.zeros(Train_data.train_size),0.0
    MSE_arr = np.zeros(len(lambda_list))
    w_nathan = np.zeros(Train_data.train_size)
    for i in range(0,len(lambda_list)):
        w_nathan = nathans_method_DR(Train_data,Gamma,lambda_list[i],w_nathan,hat_r_train, hat_r_test,remove_simplex)
        MSE_nathan,_,_,_ = set_up.true_MSE(w_nathan,Train_data)
        MSE_arr[i]=MSE_nathan
        if MSE_nathan<MSE_nathan_opt:
            MSE_nathan_opt,w_nathan_opt=MSE_nathan,w_nathan
            optimal_lambda=lambda_list[i]
    return w_nathan_opt,optimal_lambda


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

def simulate_XGB_MSE(TrainData, deg, returnEst):
    M = np.concatenate((TrainData.X_train, TrainData.P_train.reshape(-1, 1)), axis=1)
    M = PolynomialFeatures(degree=deg, include_bias=True).fit_transform(M)
    M_test = np.concatenate((TrainData.X_train, TrainData.P_test.reshape(-1, 1)), axis=1)
    M_test = PolynomialFeatures(degree=deg, include_bias=True).fit_transform(M_test)
    predict_rev_arr = np.zeros(TrainData.numSimulation)
    hat_r_train, hat_r_test = np.zeros((TrainData.numSimulation, TrainData.train_size)), np.zeros(
        (TrainData.numSimulation, TrainData.train_size))
    for i in range(TrainData.numSimulation):
        model = GradientBoostingClassifier().fit(M, TrainData.D_train_simul[i])

        predict_d_test = model.predict(M_test)
        predict_d_test = np.minimum(np.maximum(predict_d_test, 0), 1)  # truncate demand estimate
        predict_rev_arr[i] = predict_d_test @ TrainData.P_test / TrainData.train_size
        # for our method DR
        hat_r_test[i] = np.multiply(predict_d_test, TrainData.P_test)  # test data
        predict_d_train = model.predict(M)
        predict_d_train = np.minimum(np.maximum(predict_d_train, 0), 1)
        hat_r_train[i] = np.multiply(predict_d_train, TrainData.P_train)  # train data

    simulated_MSE = (np.linalg.norm(predict_rev_arr - TrainData.expected_val_test) ** 2) / TrainData.numSimulation
    biasSq = (np.average(predict_rev_arr) - TrainData.expected_val_test) ** 2
    var = simulated_MSE - biasSq
    print('xgboost simulated MSE: ', simulated_MSE)
    if returnEst == 0:
        return simulated_MSE, biasSq, var
    else:
        return simulated_MSE, biasSq, var, hat_r_train, hat_r_test