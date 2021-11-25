import pandas as pd
import numpy as np
#lms regresssion

# minimize F(w)

w= [0.1,0.2,0.3,0.4,0.5] # weight vector
x=[0.52,0.53,0.54,0.44,0.34]
t=[1,1,-1,1,-1] # points (x,y)
# f(w,n)
def f(w,x):
    first_sum = 0
    for i in len(w):
        first_sum += pow(w[i]*x^[i],2)
# result is
result = f(w,x)

#w1g1(xn) + w2g2(xn) + · · · + wIgI(xn)


# we need to replicate online algorith (same rule for weight) but apply this rule to calculate weight vector

def percTrain_lms(X,t,maxIts):
    stop=0
    i=0
    tol = 1e-4 # tolerance value
    ## homogenize feature vectors
    ## needed to be transposed due to upcoming matrix multiplication
    x_hom = np.vstack((np.ones(len(X[0]),dtype=int), X))
    w = np.zeros(len(x_hom))
    while stop==0:
        for epoch in range(1,maxIts):
            i = epoch
            # we can add random  initialization:
            # random.randint(0,len(X))
            for j in range(len(t)):
                if w.T @ (x_hom[:,j] * t[j]) <= 0:
                    w += pow(w[j]*x^[j],2)
                    stop=1
                    if stop==1:
                        break
    return w


result_weight_vector = percTrain_lms(x,t,10)




##### results:

# weight vector

# closed form: linear lms: calculate w*

#