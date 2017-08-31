import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def extractTrainingSet(G):
    Phi = np.zeros((len(G), len(G[0][0])))
    Y = np.zeros((len(G),len(G[0][1])))
    for i in range(len(G)):
        Phi[i] = G[i][0]
        Y[i] = G[i][1]
    return Phi, Y

def bls(Phi, Y): #batchLeastSquares
    """performs batch least squares on data set G"""
    PTP = np.dot(Phi.T,Phi)
    PTY = np.dot(Phi.T,Y)
    return np.dot(np.linalg.inv(PTP),PTY)

def wbls(Phi, Y, W): #weightedBatchLeastSquares
    """Weighted Batch Least Squares"""
    #TODO: get dim  M and throw error if W is not MxM
    PTWP = np.dot(Phi.T, np.dot(W, Phi))
    PTWY = np.dot(Phi.T, np.dot(W, Y))
    return np.dot(np.lingalg.inv(PTWP, PTWY))

def rls(Phi, Y, alpha=2000, NRLS=20):
    """Recursive Least Squares"""
    M = Phi.shape[0]
    N = Phi.shape[1]
    
    # initialize thetahat and P
    thetahat = np.array([0.,0.]).T
    P = alpha * np.identity(N)
    for i in range(M * NRLS):
        x = Phi[i%M, :]
        y = Y[i%M]
        c = 1 + np.dot(x.T, np.dot(P, x))
        Px = np.dot(P, x)
        P = np.dot(np.identity(2) - (np.outer(Px, x.T) / c), P)
        diff = y - np.dot(x.T, thetahat)
        #FIXME: this line works only because y is a scalar
        thetahat = thetahat + np.dot(P, x) * diff
    return thetahat

def wrls(Phi, Y, alpha=2000, forget_factor=1, NRLS=20):
    """Weighted Recursive Least Squares"""
    Phi, Y = extractTrainingSet(G)
    M = Phi.shape[0]
    N = Phi.shape[1]
    
    # initialize thetahat and P
    thetahat = np.array([0.,0.]).T
    P = alpha * np.identity(N)
    for i in range(M * NRLS):
        x = Phi[i%M, :]
        y = Y[i%M]
        c = forget_factor * 1 + np.dot(x.T, np.dot(P, x))
        Px = np.dot(P, x)
        P = np.dot(np.identity(2) - (np.outer(Px, x.T) / c), P)
        P = P / forget_factor
        diff = y - np.dot(x.T, thetahat)
        #FIXME: this line works only because y is a scalar
        thetahat = thetahat + np.dot(P, x) * diff
    return thetahat

def xsiFuzzyGauss(x, centers, spreads):
    R = centers.shape[0]
    n = centers.shape[1]
    xsi = np.zeros(x.shape)
    den = 0
    for i in range(R):
        prod = 1
        for j in range(n):
            prod *= np.exp(-0.5 * ((x[j] - centers[i][j]) / spreads[i][j]) ** 2)
        den += prod
    for i in range(R):
        num = 1
        for j in range(n):
            num *= np.exp(-0.5 * ((x[j] - centers[i][j]) / spreads[i][j]) ** 2)
        xsi[i] = num / den
    return xsi

def fuzzyGaussBLS(X, C, S, Y):
    Phi = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Phi[i,:] = xsiFuzzyGauss(X[i,:], C, S)
    return bls(Phi, Y)
        
def calcUcrisp(x, b, C, S):
    num = 0
    den = 0
    for i in range(C.shape[0]):
        prod = 1
        for j in range(C.shape[1]):
            prod *= np.exp(-0.5 * ((x[j] - C[i][j]) / S[i][j]) ** 2)
        num += b[i] * prod
        den += prod
    return num / den

def fuzzyGaussRLS(X, C, S, Y):
    Phi = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Phi[i,:] = xsiFuzzyGauss(X[i,:], C, S)
    return rls(Phi, Y)
            
            
if __name__ == '__main__':
    G = [[[1.,1.],[1.]],[[2.,1.],[1.]],[[3.,1.],[3.]]]
    Phi, Y = extractTrainingSet(G)
    
    # Test bls and rls
    thetaHat = bls(Phi, Y)
    print(thetaHat)
    thetaHat2 = rls(Phi, Y)
    thetaHat3 = wrls(Phi, Y, alpha=100, forget_factor=0.9)
    print(thetaHat2)

    # Test fuzzyGaussBLS
    X = np.array([[0.,2.],[2.,4.],[3.,6.]])
    #C = X[:2, :]
    C = np.array([[1.5,3.],[3.,5.]])
    #print(C)
    S = 2 * np.ones((2,2))
    Y = [1.,5.,6.]
    theta = fuzzyGaussBLS(X, C, S, Y)
    #print(theta)
    for i in range(X.shape[0]):
        print(calcUcrisp(X[i,:], theta, C, S))

    X2 = np.array([[1,2],[2.5,5],[4,7]])
    for i in range(X2.shape[0]):
        print(calcUcrisp(X2[i,:], theta, C, S))

    # Test fuzzyGaussRLS
    theta2 = fuzzyGaussBLS(X, C, S, Y)
    print(theta2)
    for i in range(X.shape[0]):
        print(calcUcrisp(X[i,:], theta2, C, S))

    for i in range(X2.shape[0]):
        print(calcUcrisp(X2[i,:], theta2, C, S))







