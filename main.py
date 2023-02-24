import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoLarsCV
import statistics
import time
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from sklearn.gaussian_process.kernels import RBF
import methods
from warnings import simplefilter
import set_up
import pdb

def parse_arguments():
    parser = argparse.ArgumentParser('convbarycenter', add_help=False)
    parser.add_argument(
        '--Nomis',
        dest='Nomis',
        type=bool,
        default=False,
        help="""If True, we use the Nomis dataset. If False, we use the synthetic dataset"""
        )
    parser.add_argument(
        '--new_policy',
        dest='new_policy',
        type=float,
        default=1.1,
        help="""If Nomis is True and new_policy=1.1, then new policy is 10% price increase"""
        )
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        default=int(1000000*(time.time()%1)),
        help="""random seed"""
        )
    parser.add_argument(
        '--train_size',
        dest='train_size',
        type=int,
        default=50,
        help="""training dataset size"""
        )
    parser.add_argument(
        '--epsilon',
        dest='epsilon',
        type=float,
        default=0.1,
        help="""If epsilon = 0.1, then we compute one-sided 90% confidence bound"""
        )
    parser.add_argument(
        '--remove_simplex',
        dest='remove_simplex',
        type=bool,
        default=True,
        help="""If True, we do not impose simplex constraints on weights"""
        )    
    return parser.parse_args()


def main():
    simplefilter(action='ignore')
    args = parse_arguments()
    rndseed = args.seed
    new_policy_ratio = args.new_policy
    Nomis = args.Nomis
    train_size = args.train_size
    remove_simplex = args.remove_simplex
    epsilon = args.epsilon
    s_p,s_x = 1, 1 # temporary values, will use heuristics to calculate hyper-parameters 
    if Nomis:
        Train_data = set_up.prepare_Nomis_data(rndseed,s_p,s_x,train_size, new_policy_ratio)
    else:
        Train_data = set_up.prepare_synthetic_data(rndseed, train_size, s_p, s_x)
    _, _, _, hat_r_train, hat_r_test = methods.simulate_linear_MSE(Train_data, 1, 1)
    print('linear regression est(avg over simulations): ',np.average(np.average(hat_r_test,axis=1)))

    # ##################### BOPE heuristics
    Train_data = Train_data
    train_size = Train_data.train_size
    s_p_arr = np.array([1,2.5, 5, 10]);
    s_x_arr = s_p_arr
    BOPE_gamma, BOPE_sigma,BOPE_s_p,BOPE_s_x,BOPE_log_likelihood_arr = methods.BOPE_heuristics(
        Train_data,hat_r_train[0],s_p_arr,s_x_arr)
    print('BOPE gamma',BOPE_gamma, 'sigma',BOPE_sigma, 's_p',BOPE_s_p, 's_x',BOPE_s_x)
    if Nomis:
        Train_data_BOPE = set_up.prepare_Nomis_data(rndseed,BOPE_s_p,BOPE_s_x,train_size, new_policy_ratio)
    else:
        Train_data_BOPE = set_up.prepare_synthetic_data(rndseed, train_size, BOPE_s_p, BOPE_s_x)
    _, _, _, hat_r_train, hat_r_test = methods.simulate_linear_MSE(Train_data, 1, 1)
    
    ############### BOPE-B heuristics
    s_p_arr = np.array([1, 2.5, 5, 10]);
    s_x_arr = s_p_arr
    gamma_arr = np.array([1,2.5,5])
    bern_s_p, bern_s_x, our_gamma = methods.Bernstein_heuristics(Train_data,
            hat_r_train[0], s_p_arr, s_x_arr,gamma_arr)
    print('our gamma',our_gamma,'s_p',bern_s_p,'s_x',bern_s_x)
    if Nomis:
        Train_data_ours = set_up.prepare_Nomis_data(rndseed,bern_s_p,bern_s_x,train_size, new_policy_ratio)
    else:
        Train_data_ours = set_up.prepare_synthetic_data(rndseed, train_size, bern_s_p, bern_s_x)

    #distribution of estimates/bounds for different demand realizations
    t_start= time.time()
    n_sim = Train_data.numSimulation; d_sim = Train_data.D_train_simul
    w_BOPE_B_arr, w_BOPE_arr, w_bernstein_arr, w_hoeffding_arr =(np.zeros((n_sim,train_size)) for _ in range(4))
    delta_arr = np.zeros((n_sim,train_size * 2))
    est_BOPE_B_arr, est_BOPE_Bern_arr, est_LASSO_arr, est_BOPE_arr, est_SPPL_arr, est_SPPL_dm_arr, \
                    est_hoeffding_arr  = (np.zeros(n_sim) for _ in range(7))
    hoef_bound_arr, BOPE_Bern_bound_arr, BOPE_B_bound_arr = (np.zeros(n_sim) for _ in range(3))
    w_BOPE, w_bernstein, w_hoeffding = (np.ones(train_size)/train_size for _ in range(3))
    P_train = Train_data.P_train

    for i in range(n_sim):
        print('iterate ',i)
        #LASSO 
        est_LASSO_arr[i] = np.average(hat_r_test[i])

        #SPPL
        reg_result = methods.direct_method_estimate(Train_data,i)
        a_b_est = methods.doubly_robust_estimate(Train_data,reg_result,i)
        est_val = methods.evaluate_policy(Train_data,a_b_est)
        est_SPPL_arr[i] = est_val.V_DR
        est_SPPL_dm_arr[i] = est_val.V_DM
        print('SPPL\'s est: ',est_val.V_DR)

        #BOPE_B
        w_BOPE_B = methods.BOPE_B_method(Train_data_ours, our_gamma, [], hat_r_train[i], hat_r_test[i], remove_simplex)
        w_BOPE_B_arr[i] = w_BOPE_B
        est_BOPE_B_arr[i] = w_BOPE_B @ (d_sim[i]*P_train- hat_r_train[i]) + np.average(hat_r_test[i])

        #BOPE-Bern
        w_bernstein = methods.BOPE_Bern_method(Train_data_ours, our_gamma, w_bernstein, hat_r_train[i], hat_r_test[i], epsilon,
                                                    remove_simplex) 
        est_BOPE_Bern_arr[i] = w_bernstein @ (d_sim[i]*P_train- hat_r_train[i]) + np.average(hat_r_test[i])
        w_bernstein_arr[i] = w_bernstein

        #BOPE
        w_BOPE = methods.BOPE_method(Train_data_BOPE, BOPE_gamma, BOPE_sigma, [], 
                                             hat_r_train[i], hat_r_test[i], remove_simplex) #no warm start
        est_BOPE_arr[i] = w_BOPE @ (d_sim[i]*P_train- hat_r_train[i]) + np.average(hat_r_test[i])
        w_BOPE_arr[i] = w_BOPE
        
        #WC Bernstein bound for BOPE_B weights
        sub_data = methods.prepare_subproblem_data(epsilon, our_gamma, hat_r_train[i], hat_r_test[i], Train_data_ours)
        rhs_mse, _ = methods.BOPE_Bern_subroutine(w_BOPE_B, sub_data, Train_data_ours)
        BOPE_B_bound_arr[i] = est_BOPE_B_arr[i] - rhs_mse
        print('Bernstein bound BOPE_B weights: ', BOPE_B_bound_arr[i])

        #WC Bernstein bound for BOPE_Bern weights 
        rhs_mse, _ = methods.BOPE_Bern_subroutine(w_bernstein, sub_data, Train_data_ours)
        BOPE_Bern_bound_arr[i] = est_BOPE_B_arr[i] - rhs_mse
        print('Bernstein bound BOPE_Bern weights: ', BOPE_Bern_bound_arr[i])
        print('iterate ',i, 'LASSO ',round(est_LASSO_arr[i],3), 'est ',round(est_BOPE_B_arr[i],3),
              'avg est ',round(np.average(est_BOPE_B_arr[:i]),3), 'sum w ',sum(w_BOPE_B))
    time_elapsed = round(time.time() - t_start)
    print('total runtime: ', time_elapsed)

    # print average and std of bounds and estimates
    outfile = "bounds_n_{}_seed_{}.txt".format(str(train_size),str(rndseed))
    candidate_methods = [est_BOPE_Bern_arr, BOPE_Bern_bound_arr, est_BOPE_B_arr, BOPE_B_bound_arr
        ,est_BOPE_arr, est_SPPL_arr, est_LASSO_arr]
    num_methods = len(candidate_methods)
    avg_result_arr = np.zeros(num_methods); std_result_arr = np.zeros(num_methods)
    V_pi1 = Train_data.expected_val_test
    for i in range(num_methods):
        avg_result_arr[i] = np.average(candidate_methods[i])
        std_result_arr[i] = np.std(candidate_methods[i])
    with open(outfile, "w") as f:
        f.write("BOPE-Bern est, BOPE-Bern bound, BOPE-B est, BOPE-B bound, BOPE est, SPPL est, LASSO est:\n")
        f.write("average: \n")
        [f.write('{:.2f} '.format(item)) for item in avg_result_arr]
        f.write("std: \n")
        [f.write('{:.2f} '.format(item)) for item in std_result_arr]
        f.write("ground truth: \n")
        f.write('{:.4f} '.format(V_pi1))

    print('simulated mse: ')
    V_pi1 = Train_data.expected_val_test
    candidate_methods = [est_BOPE_B_arr,est_BOPE_arr,est_LASSO_arr,est_SPPL_arr]
    num_methods = len(candidate_methods)
    sim_mse_arr = np.zeros(num_methods);sim_biasSq_arr = np.zeros(num_methods);sim_var_arr = np.zeros(num_methods)
    for i in range(num_methods):
        est_mse_arr = candidate_methods[i]
        sim_mse_arr[i] = (est_mse_arr - V_pi1) @ (est_mse_arr - V_pi1) / train_size
        sim_biasSq_arr[i] = (np.average(est_mse_arr)-V_pi1)**2
        sim_var_arr[i] =sim_mse_arr[i] - sim_biasSq_arr[i]

    outfile = "n_{}_seed_{}.txt".format(str(train_size),str(rndseed))
    with open(outfile, "w") as f:
        f.write("BOPE-B, BOPE, LASSO,SPPL:\n")
        f.write("simulated MSE: \n")
        [f.write('{:.4f} '.format(item)) for item in sim_mse_arr]
        f.write("simulated bias^2: \n")
        [f.write('{:.4f} '.format(item)) for item in sim_biasSq_arr]
        f.write("simulated variance: \n")
        [f.write('{:.4f} '.format(item)) for item in sim_var_arr]


if __name__ == '__main__':
    main()

