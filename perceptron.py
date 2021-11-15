from packages import *

def perc(w, X):

    x_hom = np.vstack((np.ones(len(X[0]),dtype=int), X))

    g = w @ x_hom
    prediction = np.sign(g)

    print (f'Prediction: {prediction}')
    return prediction

    # output: binary vector composed by class labels 1 & -1

def percTrain(X, t, maxIts, online):

    gamma = 1

    ## homogenize feature vectors
    ## needed to be transposed due to upcoming matrix multiplication
    x_hom = np.vstack((np.ones(len(X[0]),dtype=int), X))
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
    # output: trained weight vector