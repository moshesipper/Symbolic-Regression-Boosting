# Symbolic-Regression Boosting (SyRBo)
# copyright 2021 moshe sipper  
# www.moshesipper.com 

USAGE = '  python syrbo.py resdir dsname n_replicates stages'

from string import ascii_lowercase
from random import choices
from sys import argv, stdin
from os import makedirs
from os.path import exists
from pandas import read_csv
from statistics import median
from mlxtend.evaluate import permutation_test
from pathlib import Path
from operator import itemgetter
import numpy as np
from decimal import Decimal
from time import process_time
# from copy import deepcopy

from sklearn.datasets import fetch_openml, make_regression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.datasets import load_boston, load_diabetes
from sklearn.preprocessing import normalize

from pmlb import fetch_data, regression_dataset_names
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

def _if3(x1, x2, x3): 
    return np.where(np.greater_equal(x1, np.zeros(x1.shape)), x2, x3)

def _if4(x1, x2, x3, x4): 
    return np.where(np.greater_equal(x1, x2), x3, x4)

if3 = make_function(function=_if3, name='if3', arity=3)
if4 = make_function(function=_if4, name='if4', arity=4)

class SyRBo:
    def __init__(self, stages=-1, population_size=-1, generations=-1): 
        self.stages = stages
        self.population_size = population_size
        self.generations = generations
        self.boosters = []
   
    def fit(self, X, y):
        for stage in range(self.stages):
            gp = SymbolicRegressor(population_size=self.population_size, generations=self.generations,\
                   function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', if3, if4))
            gp.fit(X, y)
            self.boosters.append(gp)
            p = np.nan_to_num(gp.predict(X))
            y -= p
        
    def predict(self, X): 
        pred = np.zeros(X.shape[0])
        for i in range(self.stages): 
            pred += np.nan_to_num(self.boosters[i].predict(X))
        return np.nan_to_num(pred)
    
    def score(self, X, y, sample_weight=None):
        return mean_absolute_error(y, self.predict(X))
    
    def get_params(self, deep=True):
        return { 'stages': self.stages, 'population_size': self.population_size, 'generations': self.gens } 

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
# end class 

Algorithms = [SyRBo, SymbolicRegressor]
AlgParams ={
SyRBo: { 'stages': -1, 'population_size': 200, 'generations': 200 }, 
SymbolicRegressor: {   'population_size': 200, 'generations': 200 }
}

def rand_str(n): return ''.join(choices(ascii_lowercase, k=n))

def fprint(fname, s):
    if stdin.isatty(): print(s) # running interactively 
    with open(Path(fname),'a') as f: f.write(s)

def get_args():        
    if len(argv) == 5: 
        resdir, dsname, n_replicates, stages =\
            argv[1]+'/', argv[2], int(argv[3]), int(argv[4])
    else: # wrong number of args
        exit('-'*80                       + '\n' +\
             'Incorrect usage:'           + '\n' +\
             '  python ' + ' '.join(argv) + '\n' +\
             'Please use:'                + '\n' +\
             USAGE                        + '\n' +\
             '-'*80)
                    
    if not exists(resdir): makedirs(resdir)
    fname = resdir + dsname + '_' + rand_str(6) + '.txt'
    return fname, resdir, dsname, n_replicates, stages

def get_dataset(dsname):
    if dsname ==  'regtest':
        X, y = make_regression(n_samples=10, n_features=2, n_informative=1)
    elif dsname == 'boston':
        X, y = load_boston(return_X_y=True)
    elif dsname == 'diabetes':
        X, y = load_diabetes(return_X_y=True)
    elif dsname in regression_dataset_names: # PMLB datasets
        X, y = fetch_data(dsname, return_X_y=True, local_cache_dir='pmlb') #../datasets/pmlbreg/
    else:
        try: # dataset from openml?
            X, y = fetch_openml(dsname, return_X_y=True, as_frame=False, cache=False)
        except:
            try: # a csv file in datasets folder?
                data = read_csv('../datasets/' + dsname + '.csv', sep=',')
                array = data.values
                X, y = array[:,0:-1], array[:,-1] # target is last col
                # X, y = array[:,1:], array[:,0] # target is 1st col
            except Exception as e: 
                print('looks like there is no such dataset')
                exit(e)
                
    X = normalize(X, norm='l2')
    # scaler = RobustScaler()
    # X = scaler.fit_transform(X)
                               
    n_samples, n_features = X.shape
    return X, y, n_samples, n_features

def print_params(fname, dsname, n_replicates, n_samples, n_features, stages):
    fprint(fname,\
        'dsname: ' + dsname + '\n' +\
        'n_samples: ' + str(n_samples) + '\n' +\
        'n_features: ' + str(n_features) + '\n' +\
        'n_replicates: ' + str(n_replicates) + '\n' +
        'stages: ' + str(stages) + '\n')

# main 
def main():
    start_time = process_time()
    fname, resdir, dsname, n_replicates, stages = get_args()
    X, y, n_samples, n_features = get_dataset(dsname)
    print_params(fname, dsname, n_replicates, n_samples, n_features, stages)
    global AlgParams
    AlgParams[SyRBo]['stages'] = stages

    fprint(fname, '\n')
       
    allreps = dict.fromkeys([alg for alg in Algorithms]) # for recording scores across all replicates
    for k in allreps: 
        allreps[k] = []    
    alltimes = dict.fromkeys([alg for alg in Algorithms]) # for recording runtimes across all replicates
    for k in alltimes: 
        alltimes[k] = []    
    for rep in range(1, n_replicates+1):
        onerep = dict.fromkeys([alg for alg in Algorithms]) # for recording scores across one replicate
        for k in onerep: 
            onerep[k] = [] 
        kf = KFold(n_splits=5, shuffle=True) # 5-fold cross validation        
        fold = 1
        for train_index, test_index in kf.split(X):
            onefold = dict.fromkeys([alg for alg in Algorithms], 0) # for recording scores across one fold
            for alg in Algorithms:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]            
                modeler = alg()
                modeler.set_params(**AlgParams[alg])
                alg_start_time = process_time()
                modeler.fit(X_train,y_train)
                alltimes[alg].append(process_time() - alg_start_time)
                score = mean_absolute_error(y_test, modeler.predict(X_test))
                onefold[alg] = score
                onerep[alg].append(score)
                allreps[alg].append(score)
           
            # stats for one fold
            s = 'replicate ' + str(rep) + ', fold ' + str(fold)
            for alg in Algorithms: s += ', ' + alg.__name__ + ' ' + str(round(onefold[alg],5))            
            fprint(fname, s + '\n')
            
            fold += 1
        
        # stats for one replicate
        s = 'replicate ' + str(rep)
        for alg in Algorithms: 
            s += ', ' + alg.__name__ + ' ' + str(round(median(onerep[alg]),5))
        fprint(fname, s + '\n')
    
    # done all replicate runs, compute and report final summary of experiment's stats    
    rankings = dict.fromkeys([alg for alg in Algorithms])
    medians = []
    for alg in Algorithms: 
        medians.append([alg, median(allreps[alg]),2])
    medians = sorted(medians, key=itemgetter(1))
    s_all = '*all, '
    for i in range(len(medians)):
        s = '#' + str(i+1) + ': ' + medians[i][0].__name__ + ' ' + str(medians[i][1]) + ', ' 
        s_all += s
        if medians[i][0] == SyRBo:
           s_progboost= '*SyRBo, ' + s
        rankings[medians[i][0]] = i+1
    rounds=10000 # number of permutation-test rounds
    if medians[0][0] == SyRBo: # ranked first
        pval  = permutation_test(allreps[SyRBo], allreps[medians[1][0]],  method='approximate', num_rounds=rounds,\
                                 func=lambda x, y: np.abs(np.median(x) - np.median(y)))
        pp = ' !' if pval<0.05 else ''
        s_progboost += 'pval: ' + '%.1E' % Decimal(pval) + pp + ', '
    if medians[1][0] == SyRBo: # ranked second
        pval  = permutation_test(allreps[SyRBo], allreps[medians[0][0]],  method='approximate', num_rounds=rounds,\
                                 func=lambda x, y: np.abs(np.median(x) - np.median(y)))
        pp = ' =' if pval>=0.05 else ''
        s_progboost += 'pval: ' + '%.1E' % Decimal(pval) + pp + ', '
    fprint(fname, '\n\n')
    fprint(fname, "*Summary of experiment's results over " + str(n_replicates) + ' replicates: \n')
    fprint(fname, s_all[:-2] + '\n')
    fprint(fname, s_progboost[:-2] + '\n')
    s1, s2 = '*algs, ', '*rankings, '
    for r in rankings:
        s1 += r.__name__ + ', '
        s2 += str(rankings[r]) + ', '
    fprint(fname, s1[:-2] + '\n')
    fprint(fname, s2[:-2] + '\n')
    fprint(fname, '*Time (total), ' + str(process_time() - start_time) + '\n')
    s1, s2 = '*Times (algs), ', '*Times (all runs), '
    for alg in Algorithms: 
        s1 += alg.__name__ + ', '
        s2 += str(median(alltimes[alg])) + ', '
    fprint(fname, s1[:-2] + '\n')
    fprint(fname, s2[:-2] + '\n')
    fprint(fname, '\n')

##############        
if __name__== "__main__":
  main()

