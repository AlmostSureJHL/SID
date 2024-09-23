import os
import pandas as pd
import sys
from sklearn.preprocessing import normalize, StandardScaler
import numpy as np

os.environ['R_HOME'] = 'D:\R4.1\R-4.2.1'
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

d = {'package.dependencies': 'package_dot_dependencies',
     'package_dependencies': 'package_uscore_dependencies'}
dcortools = importr('dcortools', robject_translations=d)
tools = importr('tools', robject_translations=d)

sys.path.append(r'E:\ECNU\Reaserch\SurvEnergy\python_simulation_result')
from CPH_test import CPH_test
from wild_bootstrap_LR import wild_bootstrap_test_logrank_covariates
from SDCOV_T import SDCov_gauss_wild_bootstrap

path = 'E:/ECNU/Reaserch/SurvEnergy/python_simulation_result/'
data = 'costdata.xlsx'  # 'biofeedback.xlsx'
file = path + data
dataset = pd.read_excel(file, header=0)  # header=0表示第一行是表头
dataset = dataset.values

d = dataset[:, 3]
z = dataset[:, 2]
minVals = z.min(0)
maxVals = z.max(0)
z = (z -minVals)/(maxVals-minVals)
# x = np.hstack((dataset[:, 4].reshape(-1,1), dataset[:, 6].reshape(-1,1)))
x = dataset[:, 6].reshape(-1,1)
x = normalize(x, axis=0, norm='l2')
# x = scale(x)
n = len(dataset)

dim = 1

# test by lr statistic
method = 'lr'
kernel_x = 'gau'
kernel_z = 'gau'
P_lr = wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d,
                                              seed=1,
                                              kernel_x=kernel_x,
                                              kernel_z=kernel_z)

# test by CPH statistic
method = 'cph'
kernel_x = ''
kernel_z = ''
P_CPH = CPH_test(x=x, z=z, d=d)

lambda1 = 0.5 # cure model 0.5
#####Our methods Energy Distance
method = 'SDCOV_gauss'
kernel_x = 'SDCOV'
kernel_z = ''
bandwidth = 0.6 * sum(d) * round((4 / (1 + 2)) ** (1 / (1 + 4)) * n ** (-1 / (1 + 4)), 3) / n
P_ENERGY_5 = SDCov_gauss_wild_bootstrap(x, z, d, bandwidth, kernel_x, 0.5)

#####Our methods Energy Distance
method = 'SDCOV_gauss'
kernel_x = 'SDCOV'
kernel_z = ''
bandwidth = 0.7 * sum(d) * round((4 / (1 + 2)) ** (1 / (1 + 4)) * n ** (-1 / (1 + 4)), 3) / n
P_ENERGY_1 = SDCov_gauss_wild_bootstrap(x, z, d, bandwidth, kernel_x, 1)

#####Our methods kernel-based  Guass kernel
method = 'SDCOV_gauss'
kernel_x = 'Guass'
kernel_z = ''
bandwidth = 0.7 * sum(d) * round((4 / (1 + 2)) ** (1 / (1 + 4)) * n ** (-1 / (1 + 4)), 3) / n
P_GAUSS = SDCov_gauss_wild_bootstrap(x, z, d, bandwidth, kernel_x, [])

#####Our methods kernel-based  laplace kernel
method = 'SDCOV_gauss'
kernel_x = 'laplace'
kernel_z = ''
bandwidth = 0.5 * sum(d) * round((4 / (1 + 2)) ** (1 / (1 + 4)) * n ** (-1 / (1 + 4)), 3) / n
P_LAPLACE = SDCov_gauss_wild_bootstrap(x, z, d, bandwidth, kernel_x, [])

print(P_ENERGY_5, P_ENERGY_1, P_GAUSS, P_LAPLACE)

#####Biometrics  kenerl  ##### Biometrics  kenerl
str_list1 = ','.join([str(i) for i in z])
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

P_IPCW = robjects.r(str_array)[0]
