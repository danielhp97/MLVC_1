import numpy as np
import pandas as pd
from numpy import linalg as LA

from old import Y_data_train as Y
from old import X_data_train as X
# perceptron algorithm


# slides pag. 91

#condition for solution on training dataset
# sgn(w^T x_n) = t_n

# learning rule:
# w = w + x_n t_n 

# define variables
I = 7 # polinomial grade
k = 1 # initial point
w1 = np.full((1,I),0) # weight vector initialization
tol = 1e-4 # tolerance value
nt = len (train_values) # number or train values
maxit = 10*n # max number of iterations


def perc(w,X):
    # w is the weight vector
    # X is the input matrix made by columns with input vectors

    # output: binary vector composed by class labels 1 & -1
    pass

def percTrain(X,t,maxIts,online):
    # X is the input matrix
    # t is the input vector target values
    # maxIts is iteration limit
    # online: true if online version of optimization, false if batch version

    if online:
        #online version
    else:
        #batch version


    # return: w vector
    pass

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


if __name__="main":
    wk = 0
    Custo_k = 0
    f = 1
    d = 0
    while(d==0):
        if(LA.norm(batch_gradient(w1,X,Y)) <= tol and f< maxit):
            d = 1
        else:# stopagge criteria
            grad_k = batch_gradient(w1,X,Y) # calculate gradient on wk
            print("grad_k: ", grad_k)
            s_k = -grad_k # calcular  search direction
            Custo_k = fun_Custo(w1,X,Y_data_test) # calculate OF on point wk
            grad_full = batch_gradient(w1,X,Y) # Calculate complete gradient on point wk
            print("grad_k: ", grad_full)
            eta_k = passo(w1,Custo_k,grad_full,s_k,X,Y,k,nt) # Calculate next step size
            wk = wk + eta_k*s_k # Calculate new point
    print("Optimal Solution:", wk)