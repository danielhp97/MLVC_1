from packages import *

def perc(w, X):

    x_hom = np.vstack((np.ones(len(X[0]),dtype=int), X))

    g = w @ x_hom
    prediction = np.sign(g)

    print (f'Prediction: {prediction}')
    return prediction

    # output: binary vector composed by class labels 1 & -1

def percTrain(X, t, maxIts, online):
    # X is the input matrix
    # t is the input vector target values
    # maxIts is iteration limit
    # online: true if online version of optimization, false if batch version
    I = 2 # polinomial grade
    gamma = 1
    stop=0
    i=0
    tol = 1e-4 # tolerance value
    ## homogenize feature vectors
    ## needed to be transposed due to upcoming matrix multiplication
    x_hom = np.vstack((np.ones(len(X[0]),dtype=int), X))
    w = np.zeros(len(x_hom))
   
    if online:
        while stop==0:
            for epoch in range(1,maxIts):
                # we can add random  initialization:
                # random.randint(0,len(X))
                for j in range(len(t)):
                    if w.T @ (x_hom[:,j] * t[j]) <= 0:
                        w = w.T + gamma * x_hom[:,j] * t[j]
                        stop=1
                        if stop==1:
                            break
    else:
        while stop==0:
            if(LA.norm(batch_gradient(w,X,t)) <= tol and i< maxIts):
                stop = 1
            else:
                grad_k = batch_gradient(w,X,t) # calculate gradient on wk
                s_k = - grad_k # calculate search direction
                Cost_k = Cost_function(w,X,t) # calculate OF on point wk
                grad_full = batch_gradient(w,X,t) # Calculate complete gradient on point wk
                eta_k = step(w, Cost_k, grad_full, s_k, X, t, k=1, nt=len(X)) # Calculate next step size
                wk = wk + eta_k*s_k # Calculate new point
                i += 1

    plotDataAndDecitionBoundary(X,t,w)
    print (f'{epoch} epochs needed. w = {w}')
    return w
    # output: trained weight vector

def plotDataAndDecitionBoundary (X, t, w):

    # Plot the decision boundary using the resulting weight vector w. 
    # If you implemented everything correctly, 
    # it should appear as a straight line separating the red and the blue dots.
    # Please make sure to write your code in a way, that it also works for other w vectors.
    # [1 Points]

    plt.scatter (X [0], X[1], c= ['r' if c == 1 else 'b' for c in t])

    # YOUR CODE HERE
    axes = plt.gca()
    [left, right] = axes.get_xlim()
    db_x = np.linspace(left, right, 100)

    wn = [-w[2], w[1]] # Hyperebene bzw. Normalvektor auf w
    kwn = wn[1]/wn[0]  # Steigung Normalvektor/Hyperebene
    d = -w[0]/w[2]     # Offset Normalvektor/Hyperebene

    # y = kx + d
    db_y = kwn*db_x + d

    # plot deciion boundary
    plt.plot (db_x, db_y, 'g-')
