from packages import *

# auxiliar functions

def gradiente(w,x,I):
    sum = 0
    for i in range(0,I):
        sum += w*pow(x,i)#np.dot() # must have 2 np arrays.
    return sum


def batch_gradient(w,x,y):
    I = 1
    sum_gradiente = 0
    print("x")
    print(x)
    print("len x")
    print(len(x))
    n = len(x) - 1 #if x else None
    #if n == None : raise Exception("Empty Array")
    for i in range(1,n):
        sum_pow = 0
        for j in range(0,I):
            sum_pow += pow(x[i],j)
        sum_gradiente += (gradiente(w,x[i],I) - y[i]) * sum_pow
    sum_gradiente = sum_gradiente / n
    return sum_gradiente


def Cost_function(w,X,Y):
    val = 0
    I= len(w)-1
    nt=len(X)
    for i in range(1,nt):
        val += pow((gradiente(w,X[i],I) -Y[i]),2)
    val= val/(2*nt)
    return(val)


def step(w,x,y,gamma):
    eta_k = batch_gradient(w,x,y) * gamma * 0.1
    return eta_k


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






def percTrain_not_working(X, t, maxIts, online):
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
    wk = np.zeros(len(x_hom))
    if online:
        while stop==0:
            for epoch in range(1,maxIts):
                i = epoch
                # we can add random  initialization:
                # random.randint(0,len(X))
                for j in range(len(t)):
                    if w.T @ (x_hom[:,j] * t[j]) <= 0:
                        w = w.T + gamma * x_hom[:,j] * t[j]
                        stop=1
                        if stop==1:
                            break
    elif online=="test":
        while stop==0:
            for epoch in range(1,maxIts):
                i = epoch
                # we can add random  initialization:
                # random.randint(0,len(X))
                for j in range(len(t)):
                    if w.T @ (x_hom[:,j] * t[j]) <= tol:
                        w = w.T + gamma * x_hom[:,j] * t[j]
                        stop=1
                    else:
                        eta_k = w.T @ (x_hom[:,j] * t[j])
                        s_k = -eta_k
                        w = w + eta_k* s_k
                        if stop==1:
                            break
    else:
        while stop==0:
            X = X[0]
            print("back to beggining")
            print(i)
            if(LA.norm(batch_gradient(w,X,t)) <= tol and i< maxIts):
                stop = 1
            else:
                grad_k = batch_gradient(w,X,t) # calculate gradient on wk
                s_k = - grad_k # calculate search direction
                Cost_k = Cost_function(w,X,t) # calculate OF on point wk
                grad_full = batch_gradient(w,X,t) # Calculate complete gradient on point wk
                eta_k = step(w, X, t, gamma) # Calculate next step size
                wk = wk + eta_k*s_k # Calculate new point
                i += 1

    plotDataAndDecitionBoundary(X,t,w)
    print (f'{i} epochs needed. w = {w}')
    return w
    # output: trained weight vector


def gradiente(w,x,I):
    sum = 0
    for i in range(0,I):
        sum += w*pow(x,i)#np.dot() # must have 2 np arrays.
    return sum


def batch_gradient(w,x,y):
    I = 2
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


def Cost_function(w,X,Y):
    val = 0
    I= len(w)-1
    nt=len(X)
    for i in range(1,nt):
        val += pow((gradiente(w,X[i],I) -Y[i]),2)
    val= val/(2*nt)
    return(val)


def step(w,x,y,gamma):
    eta_k = batch_gradient(w,x,y) * gamma * 0.1


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

def gradiente(w,x,I):
    sum = 0
    for i in range(0,I):
        sum += w*pow(x,i)#np.dot() # must have 2 np arrays.
    return sum

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
