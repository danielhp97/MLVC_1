import numpy as np
import pandas as pd
from numpy import linalg as LA
import random

from old import Y_data_train as Y
from old import Y_data_test as Yt
from old import X_data_train as X
from old import X_data_test as Xt
# perceptron algorithm


# slides pag. 91

#condition for solution on training dataset
# sgn(w^T x_n) = t_n

# learning rule:
# w = w + x_n t_n 

############ Auxiliar functions
def gradiente(w,x,I):
    sum = 0
    for i in range(0,I):
        sum += w*pow(x,i)#np.dot() # must have 2 np arrays.
    return sum


def batch_gradient(w,x,y):
    sum_gradiente = 0
    n = len(x) - 1 if x else None;
    if n == None : raise Exception("Empty Array")
    for i in range(1,n):
        sum_pow = 0
        for j in range(0,I):
            sum_pow += pow(x[i],j)
        sum_gradiente += (gradiente(w,x[i],I) - y[i]) * sum_pow
    sum_gradiente = sum_gradiente / n
    return sum_gradiente


# calculate stochastic gradient


def Cost_function(w,X,Y):
    val = 0
    I= len(w)-1
    nt=len(X)
    for i in range(1,nt):
        val += pow((gradiente(w,X[i],I) -Y[i]),2)
    val= val/(2*nt)
    return(val)


def step(w,Cost_k,grad_full,s_k,X,Y,k,nt):
    eta_k = backtrackingArmijo(w,Cost_k,grad_full,s_k,X,Y)

def step_simple():
    eta_k = # calculate eta_k
    # COST(w + nk sk) <= COST(w)    

def backtrackingArmijo(w,Cost_k,grad_full,s_k,X,Y):
    delta=0.1
    eta_k=1
    d=0
    while(d==0):
        if( Cost_function(w+eta_k*s_k,X,Y ) >= (Cost_k+ delta*eta_k * np.dot(grad_full.T,s_k)) ):
            d=0
            eta_k=eta_k/2
            if eta_k*LA.norm(s_k)<=1e-8:
                eta_k=1
        else:
            d=1
    return eta_k


def sign(x):
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    else:
        return 0.0


############ Main functions

def perc(w,X):
    # w is the weight vector
    # X is the input matrix made by columns with input vectors
    vector = X.loc[:,0]
    label = X.loc[:,1]   
    linear_sum = np.dot(w, vector)
    prediction = sign(linear_sum)
    error = label - prediction
    return prediction, error
    # output: binary vector composed by class labels 1 & -1
    pass

def percTrain(X,t,maxIts,online):
    # X is the input matrix
    # t is the input vector target values
    # maxIts is iteration limit
    # online: true if online version of optimization, false if batch version
    t = Y
    wk = 0 # initialized optimal weight vector 
    Cost_k = 0 # cost function result
    i = 0 # iteration count
    if online: #online version
        stop = 0 # out of iterartion criteria
        lr = 1 # learning rate
        indice = random.randint(0,len(X))

        lr = 1 # choose step size
        while stop==0:
            # stop rule
            if wk.T @ (X[:,j] * t[j]) <= tol and i< maxIts:
                stop = 1
            else:
                #calculate new weight vector
                st_grad = (gradiente(w,x[i],I) - y[i]) #calculate stochastic gradient
                s_k = -st_grad # calculate search direction
                step_k = step_simple() #calculate step
                wk = wk - step_k * st_grad
                wk = wk - step_k * s_k # Calculate new point
    else: # batch version
        stop = 0 # out of iterartion criteria
        while stop==0:
            if(LA.norm(batch_gradient(w1,X,Y)) <= tol and i< maxIts):
                stop = 1
            else: 
                print("Iteration n:", i)
                grad_k = batch_gradient(w1,X,Y) # calculate gradient on wk
                s_k = -grad_k # calcular search direction
                Cost_k = Cost_function(w1,X,Yt) # calculate OF on point wk
                grad_full = batch_gradient(w1,X,Y) # Calculate complete gradient on point wk
                eta_k = step(w1, Cost_k, grad_full, s_k, X, Y, k, nt) # Calculate next step size
                wk = wk + eta_k*s_k # Calculate new point
                i += 1
    print (f'{i} epochs needed. w = {wk}')
    return wk



if __name__=="main":
    random.seed(123)
    # define variables
    I = 7 # polinomial grade
    k = 1 # initial point
    w1 = np.full((1,I),0) # weight vector initialization
    tol = 1e-4 # tolerance value
    nt = len (X) # number or train values
    maxIts = 10*nt # max number of iterations
    print("Training:") 
    w = percTrain(X,Y,maxIts,online=False) # training weight vector
    print("Perceptron working...")
    results, errors = perc(w,Xt) #

    # perc(w,X)