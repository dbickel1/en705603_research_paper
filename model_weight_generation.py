#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 13:48:38 2022

@author: DanBickelhaupt
"""

# %% Packages
import random
from random import gauss
import numpy as np
import pandas as pd
from pprint import pprint
import time

# %% Data Processing Functions
def min_max_transform(data, feature_list, a=0, b=1):
    data_trans = data.copy()    
    for feat in feature_list:
        data_trans[feat] = (data_trans[feat] - data_trans[feat].min()) / (data_trans[feat].max() - data_trans[feat].min()) * (b - a) - a
    return data_trans

# %% kNN Functions
# w = np.array(N,1) , dataset = np.array(M,N) , querypt = np.array(N,1)
def w_knn(w, dataset, query, k, verbose=False):
    if type(dataset) != np.ndarray:
        dataset = np.array(dataset)
    if type(query) != np.ndarray:
        query = np.array(query)
    if type(w) != np.ndarray:
        w = np.array(w)
    output = np.zeros((len(query)))
    for i, querypt in enumerate(query):
        dist = np.sum(w.ravel() * ((dataset[:,:-1] - querypt[:-1])**2), axis=1)
        dist = np.delete(dist,0) # remove edge-case where querypt is in dataset
        nearest_idx = np.argsort(dist)
        output[i] = np.sum(dataset[nearest_idx[0:5],-1]) / len(dataset[nearest_idx[0:5]])
        if verbose and (i%round(len(query)/100) == 0):
            print('\t.....{}% complete.....'.format(i/round(len(query)/100)))
    return output

def mse(y_pred, y_actl):
    return  np.sum((y_actl - y_pred)**2) / len(y_actl)

# %% Genetic Algorithm Functions
def crossover(Pc, rand_num, parent1, parent2):
    if rand_num < Pc:
        random.seed(a=None)
        crossover_position = random.randint(0,len(parent1))
        child1 = parent1[0:crossover_position] + parent2[crossover_position:]
        child2 = parent2[0:crossover_position] + parent1[crossover_position:]
        return child1, child2
    return parent1, parent2

def tournament_select(population, fitness_vals, sample_num):
    random.seed(a=None)
    rand_selection_idx = random.sample(list(range(len(population))),k=sample_num)
    fit_select = []
    for i in rand_selection_idx: fit_select.append(fitness_vals[i])
    parent_idx = fitness_vals.index(max(fit_select))
    selected_parent = population[parent_idx]
    return selected_parent

def pick_parents(population, fitness, tournament_samp_size):
    parent1 = tournament_select(population, fitness, tournament_samp_size)
    parent2 = tournament_select(population, fitness, tournament_samp_size)
    return parent1, parent2

def real_generate_random_population(N, parameters):
    random.seed(a=None)
    rand_pop = []
    for i in range(N):
        rand_indv = []
        for j in range(parameters['num_dimensions']):
            rand_num = random.random()*(parameters['map_high'] - parameters['map_low']) + parameters['map_low']
            rand_indv.append(rand_num)
        rand_pop.append(rand_indv)
    return rand_pop

def real_eval_fitness_function(population, parameters):
    fitness = []
    f = []
    f_fxn = parameters['f_fxn']
    fit_fxn = parameters['fitness_fxn']
    for i in range(len(population)):
        cur_indv = population[i]
        cur_f = f_fxn(cur_indv)
        cur_indv_fit_fx = fit_fxn(cur_f)
        fitness.append(cur_indv_fit_fx)
        f.append(cur_f)
    return fitness, f

def real_compile_result(population, fitness, f):    
    result_idx = fitness.index(max(fitness))
    result_fitness = fitness[result_idx]
    result_f = f[result_idx]
    result_solution = population[result_idx]
    result = {
                'solution': result_solution,
                'fitness': result_fitness,
                'f': result_f
            }
    return result

def real_mutate_child(Pm, rand_num, child, gauss_stdev):
    if rand_num < Pm:
        random.seed(a=None)
        mutation_position = random.randint(0,len(child)-1)
        mutated_child = child[:]
        mutation_value = gauss(mutated_child[mutation_position], gauss_stdev)
        if mutation_value >= 0:
            mutated_child[mutation_position] = mutation_value            
        return mutated_child
    return child

def real_reproduce(parent1, parent2, parameters):
    Pc = parameters['Pc']
    Pm = parameters['Pm']
    mutation_stdev = parameters['mutation_stdev']
    random.seed(a=None)
    rand_num1 = random.random()
    rand_num2 = random.random()
    rand_num3 = random.random()
    child1, child2 = crossover(Pc, rand_num1, parent1, parent2)
    child1 = real_mutate_child(Pm, rand_num2, child1, mutation_stdev)
    child2 = real_mutate_child(Pm, rand_num3, child2, mutation_stdev)
    return child1, child2

def real_ga( parameters, debug=False):    
    pop_size = parameters['population_size']
    population = real_generate_random_population(pop_size, parameters)
    generations = 0
    while generations < parameters['generation_limit']:
        fitness, f = real_eval_fitness_function(population, parameters)
        next_population = []
        for n in range(int(pop_size/2)):
            parent1, parent2 = pick_parents(population, fitness, parameters['tournament_sample'])
            child1, child2 = real_reproduce(parent1, parent2, parameters)
            next_population.append(child1)
            next_population.append(child2)
        if (debug == True): #and (generations%round(parameters['generation_limit']/20) == 0):
            cur_result = real_compile_result(population, fitness, f)
            print('Generation #: {}, Best Individual in Current Generation ='.format(generations))
            pprint(cur_result, compact=True)
        generations += 1
        population = next_population
    result = real_compile_result(population, fitness, f)
    return result

# %% Organize data
data = pd.read_csv('Annual_FinStatement_Data.csv')
input_features = data.columns
training_features = list(input_features[4:])
training_features.remove('EV')
training_features.remove('MarketCapDate')

data['StatementDate'] = pd.to_datetime(data['StatementDate'])
year_list = [2016]
downselect_data = pd.DataFrame()
for year in year_list:
    downselect_data = pd.concat([downselect_data, data.loc[data['StatementDate'].dt.year == year]], axis=0, ignore_index=True)

data_good = downselect_data[training_features]
data_good = min_max_transform(data_good, data_good.columns)

# %% Calculate weights
real_ga_parameters = {'Pc': 0.9, 'Pm': 0.05, 'map_low': 0, 'map_high': 1, \
                      'num_dimensions': data_good.shape[1]-1, \
                      'population_size': 300, 'generation_limit': 20, 'tournament_sample': 7,\
                      'f_fxn': lambda w: mse(w_knn(w,real_ga_parameters['train'],real_ga_parameters['test'],real_ga_parameters['k'])\
                                              ,real_ga_parameters['test'].iloc[:,-1]), \
                      'fitness_fxn': lambda xs: 1/(1+xs),\
                      'mutation_stdev': 0.1, 'train': data_good, 'test': data_good, 'k': 5}

    
# Timing notes: 7.46s per sample of population per generation
print('..... Start weight optimization .....')
start_time = time.time()
results = real_ga(real_ga_parameters, debug=True)
print("--- %s seconds ---" % (time.time() - start_time))

ga_weights = results['solution']
mse_tr = results['f']
