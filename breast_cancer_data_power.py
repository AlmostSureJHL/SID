# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 16:13:32 2018

@author: David
"""
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from CPH_test import CPH_test
from wild_bootstrap_LR import wild_bootstrap_test_logrank_covariates
from opthsic import opt_hsic
from SDCOV_T import SDCov_gauss, SDCov_gauss_wild_bootstrap
from multiprocessing import Pool

os.environ['R_HOME'] = 'D:\R4.1\R-4.2.1'
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

d_1 = {'package.dependencies': 'package_dot_dependencies',
     'package_dependencies': 'package_uscore_dependencies'}
dcortools = importr('dcortools', robject_translations=d_1)
tools = importr('tools', robject_translations=d_1)
graphics = importr('graphics')

path = 'E:/ECNU/Reaserch/SurvEnergy/python_simulation_result/'
data = 'gbsg.xlsx'  # 'biofeedback.xlsx'
file = path + data
dataset = pd.read_excel(file, header=0)  # header=0表示第一行是表头
dataset = dataset.values

def rejection_rate(args):
    a = args[0]
    method = args[1]
    kernel_x = args[2]
    kernel_z = args[3]
    alpha = args[4]
    n = 50*a[1]
    num_observed = 0
    seed = a[0]
    num_rejections = 0
    local_state = np.random.RandomState(seed)
    seeds_for_test = local_state.choice(1000000, size=num_repetitions, replace=False)
    for repetition in range(num_repetitions):
        if repetition % 5 == 0:
            print('n is', n, 'repetition', repetition, 'method', method, 'kernel_x', kernel_x)
        if scenario == 1:
            lambda1 = 10
            samp = local_state.choice(range(685),n,False)
            x = dataset[samp, 2].reshape(-1,1)
            z = dataset[samp, 9]
            d = dataset[samp, 10]
            bandwidth = lambda1*sum(d) * round((4 / (1 + 2)) ** (1 / (1 + 4)) * n ** (-1 / (1 + 4)), 3) / n
            print('cens. prop.', round(1 - sum(d) / n, 3), 'bandwidth', bandwidth)
        else:
            x = 0
            x = x[:, np.newaxis]
            z = 0
            n = 0
            print('ERROR')


        # do the test
        if method == 'cph':
            num_rejections += 1 if CPH_test(x=x, z=z, d=d) <= 0.1 else 0
        elif method == 'lr':
            num_rejections += 1 if wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d,
                                                                          seed=seeds_for_test[repetition],
                                                                          kernel_x=kernel_x,
                                                                          kernel_z=kernel_z) <= 0.1 else 0
        elif method == 'Dcov_IPCW':
            str_list1 = ','.join([str(i) for i in np.log(z)])
            str_list2 = ','.join([str(i) for i in d])
            str_list3 = ','.join([str(i) for i in x[:, 0]])
            if x.shape[1] > 1:
                for j in range(1, x.shape[1]):
                    str_list3 = str_list3 + ',' + ','.join([str(i) for i in x[:, j]])
            else:
                str_list3 = str_list3
            str_array = 'ipcw.dcov.test(cbind(time=c(' + str_list1 + '),status=c(' + str_list2 + '))' + \
                        ',' + 'X=matrix(c(' + str_list3 + '),' + str(x.shape[0]) + ',' + str(x.shape[1]) + \
                        '),B=1999)$pval'
            num_rejections += 1 if robjects.r(str_array)[0] <= 0.1 else 0
        elif method == 'SDCOV_gauss':
            num_rejections += 1 if SDCov_gauss(x, z, d, bandwidth, kernel_x) <= 0.1 else 0
        elif method == 'SDCov_gauss_wild_bootstrap':
            num_rejections += 1 if SDCov_gauss_wild_bootstrap(x, z, d, bandwidth, kernel_x, alpha) <= 0.1 else 0
        else:
            num_rejections += 1 if opt_hsic(x=x, z=z, d=d, seed=seeds_for_test[repetition]) <= 0.1 else 0

    print('percentage observed', num_observed / (n * num_repetitions))
    return (num_rejections / num_repetitions)


scenario = 1
num_repetitions = 200
dimensions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
seeds = [i for i in range(len(dimensions))]
inputs = [[seeds[i], dimensions[i]] for i in range(len(dimensions))]
print(inputs)

power = pd.DataFrame()
power1 = np.random.uniform(0, 1, len(dimensions))
dimensions_1 = [i * 50 for i in dimensions]
power['n_sample'] = dimensions_1
processes_count = 3

if __name__ == '__main__':
    time_start = time.time()
    processes_pool = Pool(processes_count)
    #### cox logrank #####  cox logrank
    #### cox logrank #####  cox logrank
    method = 'cph'
    kernel_x = ''
    kernel_z = ''
    alpha = []
    args = []
    for i in range(len(dimensions)):
        args.append([inputs[i], method, kernel_x, kernel_z, alpha])

    power1 = processes_pool.map(rejection_rate, args)
    power['CPH'] = pd.DataFrame(power1)

    #####JASA  kenerl  ##### JASA  kenerl
    #####JASA  kenerl  ##### JASA  kenerl
    method = 'lr'
    kernel_x = 'gau'
    kernel_z = 'gau'
    alpha = []
    args = []
    for i in range(len(dimensions)):
        args.append([inputs[i], method, kernel_x, kernel_z, alpha])

    power1 = processes_pool.map(rejection_rate, args)
    power['lr'] = pd.DataFrame(power1)

    #####Biometrics  kenerl  ##### Biometrics  kenerl
    #####Biometrics  kenerl  ##### Biometrics  kenerl
    # method = 'Dcov_IPCW'
    # kernel_x = ''
    # kernel_z = ''
    # alpha = []
    # args = []
    # for i in range(len(dimensions)):
    #     args.append([inputs[i], method, kernel_x, kernel_z, alpha])
    # power1 = processes_pool.map(rejection_rate, args)
    # power['IPCW'] = pd.DataFrame(power1)

    #####Our methods Energy Distance
    method = 'SDCov_gauss_wild_bootstrap'
    kernel_x = 'SDCOV'
    kernel_z = ''
    alpha = 1
    args = []
    for i in range(len(dimensions)):
        args.append([inputs[i], method, kernel_x, kernel_z, alpha])

    power1 = processes_pool.map(rejection_rate, args)
    power['Energy_1'] = pd.DataFrame(power1)

    # #####Our methods Energy Distance
    # method = 'SDCov_gauss_wild_bootstrap'
    # kernel_x = 'SDCOV'
    # kernel_z = ''
    # alpha = 0.5
    # args = []
    # for i in range(len(dimensions)):
    #     args.append([inputs[i], method, kernel_x, kernel_z, alpha])
    #
    # power1 = processes_pool.map(rejection_rate, args)
    # power['Energy_0.5'] = pd.DataFrame(power1)
    #
    # #####Our methods kernel-based  Guass kernel
    # method = 'SDCov_gauss_wild_bootstrap'
    # kernel_x = 'Guass'
    # kernel_z = ''
    # alpha = []
    # args = []
    # for i in range(len(dimensions)):
    #     args.append([inputs[i], method, kernel_x, kernel_z, alpha])
    #
    # power1 = processes_pool.map(rejection_rate, args)
    # power['SDCOV_gau'] = pd.DataFrame(power1)
    #
    # #####Our methods kernel-based  laplace kernel
    # method = 'SDCov_gauss_wild_bootstrap'
    # kernel_x = 'laplace'
    # kernel_z = ''
    # alpha = []
    # args = []
    # for i in range(len(dimensions)):
    #     args.append([inputs[i], method, kernel_x, kernel_z, alpha])
    #
    # power1 = processes_pool.map(rejection_rate, args)
    # power['SDCOV_lap'] = pd.DataFrame(power1)

    ##### ##### ##### ##### ##### ##### ##### #####
    time_end = time.time()
    print('time_cost:', time_end - time_start)
    processes_pool.close()
    processes_pool.join()
    print(power)

    ##### empirical power ##########
    plt.figure(figsize=(8, 6))  # size of picture
    plt.xlabel("n_sample")
    plt.ylabel("empirical power")
    plt.title("power of breast cancer data")
    plt.plot(power['n_sample'], power['CPH'], label="CPH", color='k', linestyle='-.', marker='+')
    plt.plot(power['n_sample'], power['lr'], label="lr(gau,gau)", color='g', linestyle='--', marker='>')
    plt.plot(power['n_sample'], power['Energy_1'], label="Energy_1", color='y', linestyle='solid', marker='+')
    #plt.plot(power['n_sample'], power['Energy_0.5'], label="Energy_0.5", color='y', linestyle='solid', marker='*')
    #plt.plot(power['n_sample'], power['IPCW'], label="Ipcw", color='r', linestyle='dashed', marker='h')
    #plt.plot(power['n_sample'], power['SDCOV_gau'], label="CCF_gau", color='r', linestyle=':', marker='*')
    #plt.plot(power['n_sample'], power['SDCOV_lap'], label="CCF_lap", color='b', linestyle='-', marker='s')
    plt.legend(loc='best')
    plt.show()