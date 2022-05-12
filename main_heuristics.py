#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time,csv
import set_up,methods
from warnings import simplefilter
from matplotlib import pyplot

simplefilter(action='ignore')# ignore all warnings

if __name__ == "__main__":
    smoothness, jump, exploration, randomness = 5, 0.25, 0.6, 0.6
    setting = 'setting4'
    policy_type,demand_type,logging_type,feature_dist = [setting,randomness],[setting,smoothness,jump],[setting,exploration],'uniform'
    feature_dim=2;train_size=30; rndseed = int(1000000*(time.time()%1)); s_p=1; s_x=1 #bandwidth of kernels
    rndseed=  491709 #104162
    t_start=time.time()

    #hyper parameters
    epsilon = 0.10;
    remove_simplex = 1;

    Train_data = set_up.prepare_data(rndseed, train_size, policy_type, demand_type, logging_type, feature_dist, feature_dim,
                                     s_p,s_x)
    P_train, D_train = Train_data.P_train, Train_data.D_train; P_test = Train_data.P_test
    expected_R_train = Train_data.expected_R_train;
    expected_R_test = Train_data.expected_R_test
    hat_r_train = np.ones(train_size) * np.average(P_train * D_train);
    hat_r_test = np.copy(hat_r_train)  # constant baseline

    # hat_r_train = Train_data.expected_R_train
    # hat_r_test = Train_data.expected_R_test#oracle info

    hat_r_train, hat_r_test = methods.truncate_r(Train_data,hat_r_train,hat_r_test) #else will cause negative variance issues

    # _,_,all_weights_arr, r0_mse_arr = methods.oracle_gamma(Train_data,hat_r_train,hat_r_test,kernel_multiplier_list,
    #                                                        'Bernstein',epsilon,remove_simplex)
    #
    # #####################nathan heuristics
    # s_p_arr = np.array([1,2.5,5,10,25,50]); s_x_arr = s_p_arr
    # nathan_gamma, nathan_sigma,nathan_s_p,nathan_s_x,nathan_log_likelihood_arr = methods.nathan_DR_heuristics(
    #     Train_data,hat_r_train,s_p_arr,s_x_arr,5,1)
    # print('Nathans heuristics gamma, sigma, x_p,s_x: ',nathan_gamma, nathan_sigma, nathan_s_p,nathan_s_x)
    # pyplot.cla()
    # for i in range(len(s_p_arr)):
    #     pyplot.plot(s_x_arr, nathan_log_likelihood_arr[i])
    #     pyplot.scatter(s_x_arr, nathan_log_likelihood_arr[i], alpha=0.5, s=25,label = 'bandwidth for price='+str(s_p_arr[i]))
    # title = '(Nathan heuristics) log likelihood'
    # pyplot.title(title)
    # pyplot.xlabel('bandwidth for feature X')
    # pyplot.legend(loc='best')
    # pyplot.savefig(setting+'_n_'+str(train_size)+'_'+str(rndseed)+'_heuristics_nathan.png')
    # pyplot.switch_backend('agg')

    # pyplot.cla()
    # pyplot.plot(nathan_bandwidth_arr, nathan_log_likelihood_arr)
    # pyplot.scatter(nathan_bandwidth_arr, nathan_log_likelihood_arr, alpha=0.5, s=25)
    # title = '(Nathan heuristics) log marginal likelihood'
    # pyplot.title(title)
    # pyplot.xlabel('bandwidth')
    # pyplot.grid
    # pyplot.savefig(setting+'_n_'+str(train_size)+'_'+str(rndseed)+'_heuristics_nathan.png')
    # pyplot.switch_backend('agg')




    ##############bernstein heuristics kernel bandwidth
    s_p_arr = np.array([1,2.5,5,10,25,50]); s_x_arr = s_p_arr
    gamma_arr = np.array([0.1,0.25,0.5,1,2.5,5,10])
    bern_s_p,bern_s_x,bern_log_likelihood_arr, log_likelihood_2nd_term,log_likelihood_3rd_term\
        = methods.Bernstein_heuristics(Train_data, hat_r_train,s_p_arr,s_x_arr,gamma_arr)
    print('bernstein heuristics s_p: ',bern_s_p, 's_x ',bern_s_x)

    pyplot.cla()
    for i in range(len(s_p_arr)):
        pyplot.plot(s_x_arr, bern_log_likelihood_arr[i])
        #pyplot.plot(s_x_arr, log_likelihood_2nd_term[i],label='2nd term')
        #pyplot.plot(s_x_arr, log_likelihood_3rd_term[i],label='3rd term')
        pyplot.scatter(s_x_arr, bern_log_likelihood_arr[i], alpha=0.5, s=25,label = 'bandwidth for price='+str(s_p_arr[i]))
    title = '(Bernstein heuristics) log likelihood'
    pyplot.title(title)
    pyplot.xlabel('bandwidth for feature X')
    pyplot.legend(loc='best')
    pyplot.savefig(setting+'_n_'+str(train_size)+'_'+str(rndseed)+'_heuristics_bernstein.png')
    pyplot.switch_backend('agg')




    #################  mse(w,r) heuristics for gamma
    # s_p_arr = np.array([1,2.5,5,10,25,50]); s_x_arr = s_p_arr
    # bern_s_p,bern_s_x,bern_log_likelihood_arr= methods.Bernstein_heuristics(Train_data, hat_r_train,s_p_arr,s_x_arr)
    # Train_data_bern = set_up.prepare_data(rndseed, train_size, policy_type, demand_type, logging_type, feature_dist, feature_dim,
    #                                  bern_s_p,bern_s_x)
    # gamma_list = np.arange(1, 20) * 2
    # bern_gamma,r0_mse_arr,bern_weights_arr = methods.bernstein_heuristics_gamma(Train_data_bern,
    #         gamma_list, hat_r_train, hat_r_test,epsilon, remove_simplex,0)
    #




    #################  mse(w,r) heuristics for gamma. multiple seeds. often cannot reach 5%.
    # sample_size_arr = [30,50]; heuristics_gamma_arr = np.zeros(len(sample_size_arr)); n_seeds=2
    # gamma_list = np.arange(1, 20) * 2
    # s_p_arr = np.array([1, 2.5, 5, 10]);
    # s_x_arr = s_p_arr
    # for i in range(len(sample_size_arr)):
    #     train_size = sample_size_arr[i]
    #     Train_data_bern = set_up.prepare_data(rndseed, train_size, policy_type, demand_type, logging_type,
    #                                           feature_dist, feature_dim,1, 1)
    #     hat_r_train = np.ones(train_size) * np.average(Train_data_bern.P_train * Train_data_bern.D_train)
    #     hat_r_train, _ = methods.truncate_r(Train_data_bern, hat_r_train, hat_r_train)
    #     bern_s_p, bern_s_x, bern_log_likelihood_arr = methods.Bernstein_heuristics(Train_data_bern, hat_r_train,
    #                                                                                s_p_arr, s_x_arr)
    #     print('bernstein heuristics s_p: ', bern_s_p, 's_x ', bern_s_x)
    #     for j in range(n_seeds):
    #         Train_data_bern = set_up.prepare_data(rndseed+5*j, train_size, policy_type, demand_type, logging_type, feature_dist, feature_dim,
    #                                          bern_s_p,bern_s_x)
    #         hat_r_train = np.ones(train_size) * np.average(Train_data_bern.P_train * Train_data_bern.D_train)
    #         hat_r_test = np.copy(hat_r_train)
    #         hat_r_train, hat_r_test = methods.truncate_r(Train_data_bern, hat_r_train,hat_r_test)
    #         heuristics_gamma = methods.bernstein_heuristics_gamma(Train_data_bern,
    #                                            gamma_list, hat_r_train, hat_r_test, epsilon, remove_simplex, 1)
    #         heuristics_gamma_arr[i] += float(heuristics_gamma)/n_seeds
    #     print('sample size ',train_size , 'heuristics gamma: ',heuristics_gamma_arr[i])
    # pyplot.cla()
    # pyplot.plot(sample_size_arr, heuristics_gamma_arr)
    # pyplot.scatter(sample_size_arr, heuristics_gamma_arr, alpha=0.5, s=25)
    # title = 'heuristics gamma (avg over 5 seeds)'
    # pyplot.title(title)
    # pyplot.xlabel('sample size')
    # pyplot.savefig(setting+'_n_'+str(train_size)+'_'+str(rndseed)+'_bern_gamma.png')
    # pyplot.switch_backend('agg')
    #




    # ########################  approx rkhs norm
    # sample_size_arr = [30,50,100,200,300,400]; rkhs_norm_arr = np.zeros(len(sample_size_arr)); n_seeds=10
    # s_p_arr = np.array([1, 2.5, 5, 10]);
    # s_x_arr = s_p_arr
    # naive_rkhs_norm_arr = np.zeros(len(sample_size_arr));pseudo1_rkhs_norm_arr = np.zeros(len(sample_size_arr))
    # pseudo2_rkhs_norm_arr = np.zeros(len(sample_size_arr))
    # for i in range(len(sample_size_arr)):
    #     train_size = sample_size_arr[i]
    #     Train_data_bern = set_up.prepare_data(rndseed, train_size, policy_type, demand_type, logging_type,
    #                                           feature_dist, feature_dim,1, 1)
    #     hat_r_train = np.ones(train_size) * np.average(Train_data_bern.P_train * Train_data_bern.D_train)
    #     #hat_r_train = Train_data_bern.expected_R_train  # use oracle info
    #     #bern_s_p,bern_s_x,bern_log_likelihood_arr= methods.Bernstein_heuristics(Train_data_bern, hat_r_train,s_p_arr,s_x_arr)
    #     bern_s_p, bern_s_x = 2.5,5
    #     #print('bernstein heuristics s_p: ',bern_s_p, 's_x ',bern_s_x)
    #     for j in range(n_seeds):
    #         Train_data_bern = set_up.prepare_data(rndseed+5*j, train_size, policy_type, demand_type, logging_type, feature_dist, feature_dim,
    #                                          bern_s_p,bern_s_x)
    #         hat_r_train = np.ones(train_size) * np.average(Train_data_bern.P_train * Train_data_bern.D_train)
    #         #hat_r_train = Train_data_bern.expected_R_train #use oracle info
    #         hat_r_train, _ = methods.truncate_r(Train_data_bern, hat_r_train,hat_r_train)
    #         approx_rkhs_norm, naive_rkhs_norm, pseudo_inv_rkhs_norm1, pseudo_inv_rkhs_norm2 = \
    #             set_up.approx_RKHS_norm(Train_data_bern, hat_r_train, bern_s_p, bern_s_x,1)
    #         #approx_rkhs_norm = set_up.approx_RKHS_norm(Train_data_bern,hat_r_train,bern_s_p,bern_s_x)
    #         rkhs_norm_arr[i] += approx_rkhs_norm/n_seeds
    #         naive_rkhs_norm_arr[i] += naive_rkhs_norm/n_seeds
    #         pseudo1_rkhs_norm_arr[i] += pseudo_inv_rkhs_norm1/n_seeds
    #         pseudo2_rkhs_norm_arr[i] += pseudo_inv_rkhs_norm2/n_seeds
    #     print('sample size ',train_size , 'avg approx rkhs norm: ',rkhs_norm_arr[i])
    # pyplot.cla()
    # pyplot.plot(sample_size_arr, rkhs_norm_arr,label='Yunfan write up')
    # pyplot.scatter(sample_size_arr, rkhs_norm_arr, alpha=0.5, s=25)
    #
    # pyplot.plot(sample_size_arr, naive_rkhs_norm_arr,label = 'naive inverse')
    # pyplot.scatter(sample_size_arr, naive_rkhs_norm_arr, alpha=0.5, s=25)
    # pyplot.plot(sample_size_arr, pseudo1_rkhs_norm_arr,label='pseudo inv, rcond=0.01')
    # pyplot.scatter(sample_size_arr, pseudo1_rkhs_norm_arr, alpha=0.5, s=25)
    # pyplot.plot(sample_size_arr, pseudo2_rkhs_norm_arr,label='pseudo inv, rcond=0.02')
    # pyplot.scatter(sample_size_arr, pseudo2_rkhs_norm_arr, alpha=0.5, s=25)
    # title = 'approx rkhs norm (avg over 10 seeds)'
    # pyplot.title(title)
    # pyplot.xlabel('sample size')
    # pyplot.legend(loc='best')
    # pyplot.savefig(setting+'_n_'+str(train_size)+'_'+str(rndseed)+'_rkhs_norm.png')
    # pyplot.switch_backend('agg')





