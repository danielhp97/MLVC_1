from packages import *

def perc(w, X):
    # w is the weight vector
    # X is the input matrix made by columns with input vectors
    vector = X.loc[:,0]
    label = X.loc[:,1]   
    linear_sum = np.dot(w, vector)
    prediction = sign(linear_sum)
    error = label - prediction
    return prediction, error
    # output: binary vector composed by class labels 1 & -1

def percTrain(X, t, maxIts, online):

    gamma = 1

    ## homogenize feature vectors
    ## needed to be transposed due to upcoming matrix multiplication
    x_hom = np.vstack((np.ones(len(X[0]),dtype=int), X)).T
    w = np.zeros(len(x_hom))
   
    
    for epoch in range(1,maxIts):
        running = False
        for j in range(len(t)):
            if w.T @ (x_hom[:,j] * t[j]) <= 0:
                w = w.T + gamma * x_hom[:,j] * t[j]
                running = True
        if running == False:
            break

    print (f'{epoch} epochs needed. w = {w}')
    return w