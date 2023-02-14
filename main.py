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
        '--Nomis_new_policy',
        dest='Nomis_new_policy',
        type=float,
        default=1.1,
        help="""If Nomis is True and new_policy=1.1, then new policy is 10% price increase"""
        )
    parser.add_argument(
        '--synthetic_new_policy',
        dest='synthetic_new_policy',
        type=float,
        default=3,
        help="""If Nomis is False and new_policy=3, then new policy has prices centered around 3"""
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
    simplefilter(action='ignore')# ignore all warnings
    args = parse_arguments()
    rndseed = args.seed
    new_policy_ratio = args.Nomis_new_policy
    synthetic_new_policy = args.synthetic_new_policy
    Nomis = args.Nomis
    train_size = args.train_size
    remove_simplex = args.remove_simplex
    epsilon = args.epsilon
    s_p,s_x = 1, 1 # temporary values, will use heuristics to calculate hyper-parameters 
    if Nomis:
        Train_data = set_up.prepare_Nomis_data(rndseed,s_p,s_x,train_size, new_policy_ratio)
    else:
        Train_data = set_up.prepare_synthetic_data(rndseed, train_size, s_p, s_x, synthetic_new_policy)
    _, _, _, hat_r_train, hat_r_test = methods.simulate_linear_MSE(Train_data, 1, 1)
    print('linear regression est(avg over simulations): ',np.average(np.average(hat_r_test,axis=1)))

    # ##################### nathan heuristics
    Train_data = Train_data
    train_size = Train_data.train_size
    s_p_arr = np.array([1,2.5, 5, 10]);
    s_x_arr = s_p_arr
    nathan_gamma, nathan_sigma,nathan_s_p,nathan_s_x,nathan_log_likelihood_arr = methods.nathan_DR_heuristics(
        Train_data,hat_r_train[0],s_p_arr,s_x_arr)
    print('nathan gamma',nathan_gamma, 'sigma',nathan_sigma, 's_p',nathan_s_p, 's_x',nathan_s_x)
    if Nomis:
        Train_data_nathan = set_up.prepare_Nomis_data(rndseed,nathan_s_p,nathan_s_x,train_size, new_policy_ratio)
    else:
        Train_data_nathan = set_up.prepare_synthetic_data(rndseed, train_size, nathan_s_p, nathan_s_x, synthetic_new_policy)
    _, _, _, hat_r_train, hat_r_test = methods.simulate_linear_MSE(Train_data, 1, 1)
    
    ############### our heuristics
    s_p_arr = np.array([1, 2.5, 5, 10]);
    s_x_arr = s_p_arr
    gamma_arr = np.array([1,2.5,5])
    bern_s_p, bern_s_x, our_gamma = methods.Bernstein_heuristics(Train_data,
            hat_r_train[0], s_p_arr, s_x_arr,gamma_arr)
    print('our gamma',our_gamma,'s_p',bern_s_p,'s_x',bern_s_x)
    if Nomis:
        Train_data_ours = set_up.prepare_Nomis_data(rndseed,bern_s_p,bern_s_x,train_size, new_policy_ratio)
    else:
        Train_data_ours = set_up.prepare_synthetic_data(rndseed, train_size, bern_s_p, bern_s_x, synthetic_new_policy)

    #distribution of estimates/bounds for different demand realizations
    t_start= time.time()
    n_sim = Train_data.numSimulation; d_sim = Train_data.D_train_simul
    w_mse_arr, w_nathan_arr, w_bernstein_arr, w_hoeffding_arr =(np.zeros((n_sim,train_size)) for _ in range(4))
    delta_arr = np.zeros((n_sim,train_size * 2))
    est_wc_mse_arr, est_bernstein_arr, est_base_arr, est_nathan_arr, est_sppl_arr, est_sppl_dm_arr, \
                    est_hoeffding_arr  = (np.zeros(n_sim) for _ in range(7))
    hoef_bound_arr, bern_bound_arr, bern_bound_w_mse_arr = (np.zeros(n_sim) for _ in range(3))
    w_nathan, w_bernstein, w_hoeffding = (np.ones(train_size)/train_size for _ in range(3))
    P_train = Train_data.P_train

    for i in range(n_sim):
        print('iterate ',i)       
        #sppl
        reg_result = methods.direct_method_estimate(Train_data,i)
        a_b_est = methods.doubly_robust_estimate(Train_data,reg_result,i)
        est_val = methods.evaluate_policy(Train_data,a_b_est)
        est_sppl_arr[i] = est_val.V_DR
        est_sppl_dm_arr[i] = est_val.V_DM
        print('SPPL\'s est: ',est_val.V_DR)

        #wc mse
        w_mse = methods.min_mse_method(Train_data_ours, our_gamma, [], hat_r_train[i], hat_r_test[i], remove_simplex)
        w_mse_arr[i] = w_mse
        est_wc_mse_arr[i] = w_mse @ (d_sim[i]*P_train- hat_r_train[i]) + np.average(hat_r_test[i])

        #wc hoeffding
        w_hoeffding = methods.hoeffding_method(Train_data_ours, our_gamma, w_hoeffding, hat_r_train[i], hat_r_test[i], epsilon,
                                                    remove_simplex) #warm start
        est_hoeffding_arr[i] = w_hoeffding @ (d_sim[i]*P_train- hat_r_train[i]) + np.average(hat_r_test[i])
        w_hoeffding_arr[i] = w_hoeffding

        #wc bernstein
        w_bernstein = methods.bernstein_method_abs(Train_data_ours, our_gamma, w_bernstein, hat_r_train[i], hat_r_test[i], epsilon,
                                                    remove_simplex) #warm start
        est_bernstein_arr[i] = w_bernstein @ (d_sim[i]*P_train- hat_r_train[i]) + np.average(hat_r_test[i])
        w_bernstein_arr[i] = w_bernstein
        
        sub_data = methods.prepare_subproblem_data(epsilon, our_gamma, hat_r_train[i], hat_r_test[i], Train_data_ours)
        # plug mse weights into the bernstein bound
        rhs_mse, _ = methods.bernstein_subroutine(w_mse, sub_data, Train_data_ours)
        bern_bound_w_mse_arr[i] = est_wc_mse_arr[i] - rhs_mse
        print('bernstein bound mse weights: ', bern_bound_w_mse_arr[i])

        # plug hoeffding weights into the bernstein bound (no abs value)
        rhs_mse, _ = methods.bernstein_subroutine(w_hoeffding, sub_data, Train_data_ours)
        hoef_bound_arr[i] = est_wc_mse_arr[i] - rhs_mse
        print('bernstein bound hoeffding weights: ', hoef_bound_arr[i])

        # plug bernstein weights (with abs value) into the bernstein bound (no abs value)
        rhs_mse, _ = methods.bernstein_subroutine(w_bernstein, sub_data, Train_data_ours)
        bern_bound_arr[i] = est_wc_mse_arr[i] - rhs_mse
        print('bernstein bound bernstein weights: ', bern_bound_arr[i])
        
        # check if the wc bias is positive
        delta, sign = methods.bernstein_abs_subroutine(w_bernstein,sub_data,Train_data_ours,keep_record=0,returnDetails=3)
        bw = np.append(w_bernstein, -np.ones(train_size)/train_size)
        wc_bias = sign * delta@bw
        delta_arr[i] = delta
        print('wc bias: ', wc_bias)
        print('true bias: ',est_bernstein_arr[i] - Train_data_ours.expected_val_test)
        assert wc_bias >= 0
        
        #bope
        w_nathan = methods.nathans_method_DR(Train_data_nathan, nathan_gamma, nathan_sigma, [], 
                                             hat_r_train[i], hat_r_test[i], remove_simplex) #no warm start
        est_nathan_arr[i] = w_nathan @ (d_sim[i]*P_train- hat_r_train[i]) + np.average(hat_r_test[i])
        w_nathan_arr[i] = w_nathan
        
        #base estimate
        est_base_arr[i] = np.average(hat_r_test[i])
        print('iterate ',i, 'base ',round(est_base_arr[i],3), 'est ',round(est_wc_mse_arr[i],3),
              'avg est ',round(np.average(est_wc_mse_arr[:i]),3), 'sum w ',sum(w_mse))
    time_elapsed = round(time.time() - t_start)
    print('total runtime: ', time_elapsed)


    #process results
    candidate_methods = [est_hoeffding_arr, hoef_bound_arr, est_bernstein_arr, bern_bound_arr, est_wc_mse_arr, bern_bound_w_mse_arr
        ,est_nathan_arr, est_sppl_arr, est_base_arr]
    candidate_methods = [np.maximum(result_arr, np.zeros(result_arr.shape)) for result_arr in candidate_methods]
    
    #plot histogram of bounds
    import matplotlib.pyplot as plt
    hist_weights = np.ones_like(bern_bound_arr) / n_sim
    plt.hist(bern_bound_arr, weights=hist_weights, bins=20, alpha=0.5, label='bernstein bounds')
    hist_weights = np.ones_like(hoef_bound_arr) / n_sim
    plt.hist(hoef_bound_arr, weights=hist_weights, bins=20, alpha=0.5, label='hoeffding bounds')
    hist_weights = np.ones_like(bern_bound_w_mse_arr) / n_sim
    plt.hist(bern_bound_w_mse_arr, weights=hist_weights, bins=20, alpha=0.5, label='bernstein bounds with mse weights')
    plt.axvline(x=Train_data.expected_val_test,color='r',label='true rev')
    plt.ylabel('Probability')
    plt.xlabel('revenue')
    plt.title('revenue bounds, different demand realizations')
    plt.legend(loc='best')
    plt.savefig('bounds_n_'+str(train_size)+'_'+str(rndseed)+'_.png')

    # print average and std of bounds and estimates
    outfile = "bounds_n_{}_seed_{}.txt".format(str(train_size),str(rndseed))
    num_methods = len(candidate_methods)
    avg_result_arr = np.zeros(num_methods); std_result_arr = np.zeros(num_methods)
    V_pi1 = Train_data.expected_val_test
    for i in range(num_methods):
        avg_result_arr[i] = np.average(candidate_methods[i])
        std_result_arr[i] = np.std(candidate_methods[i])
    with open(outfile, "w") as f:
        f.write("hoef est, hoef bound, bern est, bern bound, mse est, mse bound, nathan est, sppl, base:\n")
        f.write("average: \n")
        [f.write('{:.2f} '.format(item)) for item in avg_result_arr]
        f.write("std: \n")
        [f.write('{:.2f} '.format(item)) for item in std_result_arr]
        f.write("ground truth: \n")
        f.write('{:.4f} '.format(V_pi1))

    print('simulated mse: ')
    V_pi1 = Train_data.expected_val_test
    candidate_methods = [est_wc_mse_arr,est_nathan_arr,est_base_arr,est_sppl_arr]
    num_methods = len(candidate_methods)
    sim_mse_arr = np.zeros(num_methods);sim_biasSq_arr = np.zeros(num_methods);sim_var_arr = np.zeros(num_methods)
    for i in range(num_methods):
        est_mse_arr = candidate_methods[i]
        sim_mse_arr[i] = (est_mse_arr - V_pi1) @ (est_mse_arr - V_pi1) / train_size
        sim_biasSq_arr[i] = (np.average(est_mse_arr)-V_pi1)**2
        sim_var_arr[i] =sim_mse_arr[i] - sim_biasSq_arr[i]

    outfile = "n_{}_seed_{}.txt".format(str(train_size),str(rndseed))
    with open(outfile, "w") as f:
        f.write("wc mse,nathan, linear reg,sppl:\n")
        f.write("simulated MSE: \n")
        [f.write('{:.4f} '.format(item)) for item in sim_mse_arr]
        f.write("simulated bias^2: \n")
        [f.write('{:.4f} '.format(item)) for item in sim_biasSq_arr]
        f.write("simulated variance: \n")
        [f.write('{:.4f} '.format(item)) for item in sim_var_arr]

    #plot histogram of estimates
    import matplotlib.pyplot as plt
    hist_weights = np.ones_like(est_bernstein_arr) / n_sim
    plt.hist(est_bernstein_arr, weights=hist_weights, bins=20, alpha=0.5, label='bernstein estimates')
    hist_weights = np.ones_like(est_hoeffding_arr) / n_sim
    plt.hist(est_hoeffding_arr, weights=hist_weights, bins=20, alpha=0.5, label='hoeffding estimates')
    hist_weights = np.ones_like(est_wc_mse_arr) / n_sim
    plt.hist(est_wc_mse_arr, weights=hist_weights, bins=20, alpha=0.5, label='wcmse estimates')
    plt.axvline(x=Train_data.expected_val_test,color='r',label='true rev')
    plt.ylabel('Probability')
    plt.xlabel('revenue')
    plt.title('revenue estimates, different demand realizations')
    plt.legend(loc='best')
    plt.savefig('est_n_'+str(train_size)+'_'+str(rndseed)+'_.png')


if __name__ == '__main__':
    main()

