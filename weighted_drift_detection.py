#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:02:35 2022

@author: DanBickelhaupt
"""

# %% Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from copy import deepcopy

# %% Data Processing Functions
def min_max_transform(data, feature_list, a=0, b=1):
    data_trans = data.copy()    
    for feat in feature_list:
        data_trans[feat] = (data_trans[feat] - data_trans[feat].min()) / (data_trans[feat].max() - data_trans[feat].min()) * (b - a) - a
    return data_trans

# %% T^2 Control Chart Functions
def hotelling_t2(data, alpha, plot_flag=False): # data is a np.array of size n x p; alpha is the confidence interval (float)
    # Make a copy of input data and save its dimensions    
    x = deepcopy(data)
    n = x.shape[0]
    p = x.shape[1]
    # Calculate Upper Control Limit
    b = beta.ppf(alpha,p/2,0.5*(n-p-1)) # b = Beta inverse cumulative distribution function, see https://www.mathworks.com/help/stats/betainv.html
    ucl = b*(n-1)**2/n # ucl equation from Bell's thesis, section 1.2
    # Calcuate covariance matrix - from Bell's thesis, section 1.2
    xbar = np.mean(x,axis=0).reshape((p,1))
    cov_mat = np.zeros((p,p))
    for i in range(n):
        xi = x[i,:].reshape((p,1))
        cov_mat += np.matmul( (xi - xbar), (xi - xbar).T )
    cov_mat *= (1/(n-1))
    invS = np.linalg.inv(cov_mat)
    # Calculate T^2 distribution - from Bell's thesis, section 1.2
    t2 = np.zeros((n,1))
    for i in range(n):
        xi = x[i,:].reshape((p,1))
        t2[i] = np.matmul( np.matmul( (xi - xbar).T, invS ), (xi - xbar) )
    # Plot Control Chart
    if plot_flag:
        plt.figure()
        plt.plot(list(range(n)), t2,'-k')
        plt.plot(list(range(n)), np.ones(n)*ucl, '--r')
    return t2, ucl
def wt_hotelling_t2(data, wt, alpha, plot_flag=False): # data is a np.array of size n x p; alpha is the confidence interval (float)
    # Make a copy of input data and save its dimensions        
    x = deepcopy(data)
    x_wt = wt*x
    n = x_wt.shape[0]
    p = x_wt.shape[1]
    # Calculate Upper Control Limit
    b = beta.ppf(alpha,p/2,0.5*(n-p-1)) # b = Beta inverse cumulative distribution function, see https://www.mathworks.com/help/stats/betainv.html
    ucl = b*(n-1)**2/n # ucl equation from Bell's thesis, section 1.2
    # Calcuate covariance matrix - from Bell's thesis, section 1.2
    xbar = np.mean(x_wt,axis=0).reshape((p,1))
    cov_mat = np.zeros((p,p))
    for i in range(n):
        xi = x[i,:].reshape((p,1))
        cov_mat += np.matmul( (xi - xbar), (xi - xbar).T )
    cov_mat *= (1/(n-1))
    invS = np.linalg.inv(cov_mat)
    # Calculate T^2 distribution - from Bell's thesis, section 1.2
    t2 = np.zeros((n,1))
    for i in range(n):
        xi = x[i,:].reshape((p,1))
        t2[i] = np.matmul( np.matmul( (xi - xbar).T, invS ), (xi - xbar) )
    # Plot Control Chart
    if plot_flag:
        plt.figure()
        plt.plot(list(range(n)), t2,'-k')
        plt.plot(list(range(n)), np.ones(n)*ucl, '--r')
    return t2, ucl
def timeseriesplot(t_data, y_data, title, xlabel, ylabel, linestyle, legend=[]):
    n = t_data.shape[1]
    plt.figure()
    for i in range(n):
        plt.plot(t_data[:,i], y_data[:,i],linestyle[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend != []: plt.legend(legend,loc='best')
    plt.grid(True)

# %% Workspace

# Read in data and determine training dataset columns
data = pd.read_csv('Annual_FinStatement_Data.csv')
input_features = data.columns
training_features = list(input_features[4:])
training_features.remove('EV')
training_features.remove('MarketCapDate')
training_features.remove('OpCashFlow') # results in NaN data so remove

data['StatementDate'] = pd.to_datetime(data['StatementDate'])
year_list = data['StatementDate'].dt.year.unique()
year_list.sort()

drift_stats = np.zeros((len(training_features),len(year_list),2))
data_tr = data[training_features]

for j, year in enumerate(year_list):
    # Extract only data from current year
    year_data = data_tr.loc[data['StatementDate'].dt.year == year]
    # Min-max normalize the current year's data
    year_data = min_max_transform(year_data, year_data.columns)
    # Compute mean & stddev for each feature of current year's data
    for i, feat in enumerate(training_features):
        drift_stats[i,j,0] = year_data[feat].mean()
        drift_stats[i,j,1] = year_data[feat].std()

plot_flag_1 = False
if plot_flag_1:
    for i in range(len(training_features)):
        plt.figure()
        plt.plot(year_list, drift_stats[i,:,0],'-or')
        plt.plot(year_list, drift_stats[i,:,1],'-xb')
        plt.title(training_features[i])
        plt.legend(['mu', 'sd'])
        plt.figure()
        plt.plot(year_list[1:], np.abs(drift_stats[i,1:,0] - drift_stats[i,:-1,0]),'-or')
        plt.plot(year_list[1:], np.abs(drift_stats[i,1:,1] - drift_stats[i,:-1,1]),'-xb')
        plt.title(training_features[i]+' - Delta')
        plt.legend(['mu', 'sd'])

# Weights from model_weight_generation.py - population of 300, 20 generations
ga_weights =  [0.7578815951977947, 0.37365528035000084, 0.5307331275143887,
              0.07161700578018437, 0.8726429354739297, 0.9838827072731333,
              0.9340805180375689, 0.3513082770864525, 0.5705181570980394,
              0.33602251168298025, 0.07197985089160741, 0.16080548872287015,
              0.24450418620304615, 0.3081539742892327, 0.4274079723174876,
              0.522171204467514, 0.4597273812391205, 0.009155958271628616,
              0.29519005797888176, 0.8386134747901682, 0.46670715404992524]
ga_weights = np.array(ga_weights)
ga_weights = np.delete(ga_weights, 12) # OpCashFlow feature (Feature 12) has NaN data, so remove it.

# %% Use Hotelling's T^2 Distribution to Generate a Control Chart

# reformat data
mu = drift_stats[:-1,:,0].T # ignore last year x feat matrix bc it's marketcap data
sd = drift_stats[:-1,:,1].T

# run t^2 distr on mu & sd values
alpha = 0.95 # 95% confidence interval
mu_t2, ucl = hotelling_t2(mu, alpha)
sd_t2, ucl = hotelling_t2(sd, alpha)
mu_t2_wt, ucl_wt = wt_hotelling_t2(mu, ga_weights, alpha)
sd_t2_wt, ucl_wt = wt_hotelling_t2(sd, ga_weights, alpha)

mu_avg = np.mean(mu, axis=1).reshape(-1,1)
sd_avg = np.mean(sd, axis=1).reshape(-1,1)

linestyle = ['-ok','-ob','--r']
leg = ['t^2, unweighted','t^2, weighted','UCL']
years = np.array(year_list).reshape(-1,1)
t = np.hstack((years, years, years))
y = np.hstack((mu_t2, mu_t2_wt, np.ones((len(mu_t2_wt),1))*ucl))
timeseriesplot(t, y, 'MU', 'Year', '', linestyle, leg)

y = np.hstack((sd_t2, sd_t2_wt, np.ones((len(sd_t2_wt),1))*ucl))
timeseriesplot(t, y, 'SD', 'Year', '', linestyle, leg)

comb_t2, ucl = hotelling_t2(mu+sd, alpha)
comb_t2_wt, ucl_wt = wt_hotelling_t2(mu+sd, ga_weights, alpha)
y = np.hstack((comb_t2, comb_t2_wt, np.ones((len(comb_t2),1))*ucl))
timeseriesplot(t, y, 'MU+SD', 'Year', '', linestyle, leg)
