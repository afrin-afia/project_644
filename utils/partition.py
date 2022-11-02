import configparser
import numpy as np
import math

def zipf(k, N, s):
    H = sum(1/i**s for i in range(1, N+1))
    return 1/k**s / H

def bu(k, N, s):
    a = N - (N-1)*math.exp(-s)
    if k == 1:
        return a / N
    else:
        return (1 - a/N) / (N-1)
    
def coef_var(y):
    return np.std(y)/np.mean(y)

def maxmin_r(y):
    return np.max(y)/np.min(y)

def find_root(f,N,p,param, debug=False):
    # Newton's method
    ep = 0.0001
    tol = 0.000001
    s = 1
    x = list(range(N))
    
    for i in range(100):
        obja = get_obj(f,N,s+ep,param)
        objb = get_obj(f,N,s-ep,param)
        d = (obja - objb)/(2*ep)
        z = get_obj(f,N,s,param) - p
        s =  max(0,s - z/d)
        
        z_off = get_obj(f,N,s,param)
        err = abs(z_off - p)

        if debug:
            print(f'i: {i}, s: {s:.3f}, Obj: {z_off:.3f}, err: {err:.3f}, d: {d:.3f}')
            
        if err < tol:
            break
    return s

def get_obj(f,N,s,param):
    if param == 'coef_var':
        x = list(range(N))
        y = [f(i+1,N,s) for i in x]
        obj = coef_var(y)
        
    elif param == 'maxmin_r':
        x = list(range(N))
        y = [f(i+1,N,s) for i in x]
        obj = maxmin_r(y)
    elif param == 'dom_prop':
        obj = f(1, N, s)
        
    return obj

def unbal_param(N=10, p=0.5, param='dom_prop', distr='zipf', debug=False):
    '''
    Finds the appropiate parameter 's' such that the distribution behaves as expected
    - N: Number of classes in distribution
    - p: Parameter to control

    param (default 's'):
    - param='s' : the skweness parameter of the distribution. It is not very meaningful by itself
    - param='coef_var': the coefficient of variation of the resulting distribution. coef_var = std(y) / mean(y)
    - param='maxmin_r': the ratio of the dominant class to the minoryty class. maxmin_r = max(y) / min(y)
    - param='dom_prop': the proportion of the dominant class. dom_prop = y[0]

    distr (default='zipf')
    - distr='zipf': Uses Zipf's law as the distribution, a discrete finite power law.
    - distr='bu': Uses a Bernoulli-Uniform distribution. There is a dominant class and all the others are equal.
    '''

    if distr == 'zipf':
        f = zipf
    elif distr == 'bu':
        f = bu
    else:
        raise Exception("distr not in ['zipf', 'bu']. Choose a valid distribution.")

    if param == 's':
        s=p
    elif param in ['coef_var', 'maxmin_r', 'dom_prop']:
        if param in ['dom_prop']:
            if not (1/N <= p <= 1):
                raise Exception("1/N <= p < 1 if param=='dom_prop'")
        elif param in ['coef_var', 'maxmin_r']:
            if not (p > 0):
                raise Exception("p > 0 if param in ['coef_var', 'maxmin_r']")
                
        s = find_root(f,N,p,param, debug)
    else:
        raise Exception("param not in ['s', 'coef_var', 'maxmin_r', 'dom_prop']. Choose a valid parameter")
    if s > 25:
        print("Distr warning: s > 25 may cause numerical instability!!!")
    return s

def get_distr(N, s, distr='zipf'):
    if distr == 'zipf':
        f = zipf
    elif distr == 'bu':
        f = bu
    x = np.array(list(range(N)))
    p = np.array([f(i+1, N, s) for i in x])
    return p

def unbal_split(D, N, p, param="dom_prop", distr="zipf", shuffle=False, np_generator=np.random.default_rng()):
    '''
    Returns the dataset D into N unbalanced partitions such that the parameter param at each partition has the value of p.
    - D: Dataset
    - N: Number of partitions. Has to equal the number of classes in the dataset

    param (default 'dom_prop'):
    - param='s' : the skweness parameter of the distribution. It is not very meaningful by itself
    - param='coef_var': the coefficient of variation of the resulting distribution. coef_var = std(y) / mean(y)
    - param='maxmin_r': the ratio of the dominant class to the minoryty class. maxmin_r = max(y) / min(y)
    - param='dom_prop': the proportion of the dominant class. dom_prop = y[0]

    distr (default='zipf')
    - distr='zipf': Uses Zipf's law as the distribution, a discrete finite power law.
    - distr='bu': Uses a Bernoulli-Uniform distribution. There is a dominant class and all the others are equal.

    - shuffle: Shuffle the entries of the dataset
    - np_generator: Random number generator for numpy 
    '''

    M = D.shape[0]
    
    if shuffle:
        D = np_generator.permutation(D)
    # Don't separate x and y
    s = unbal_param(N, p, param=param, distr=distr)
    p = get_distr(N, s, distr=distr)
    
    # Shift distribution for every client
    P = np.array([np.roll(p, i) for i in range(N)])
    
    # Cummulatively get the indices
    Q = (P* M/N).round(0).cumsum(axis = 0) .astype(int)
    
    #D = train_d
    Dc = []

    for i in range(N):
        Di = np.array([])
        for j in range(N):
            start = 0 if i == 0 else Q[i-1, j]
            end = Q[i,j]
            # Assuming that the label is the last column
            idx = np.where(D[:,-1] == j)[0][start:end]
            Di = np.vstack((Di, D[idx])) if Di.size else D[idx]
        Dc.append(Di)
    # Return dataset
    return Dc

if __name__ == "__main__":
    print("Hello world")
    config = configparser.ConfigParser()
    config.read('../config.ini')
    print(config['PATHS']['FASHION'])

