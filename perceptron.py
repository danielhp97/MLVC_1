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