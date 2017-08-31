import numpy as np
import scipy as sp
import matplotlib as plt

class trainedStandardFuzzySystem:
    def __init__(self, X, Y, C0, S0, b0):
        self.X = X
        self.Y = Y
        self.C = C0
        self.S = S0
        self.b = b0
        self.R = C0.shape[0]
        self.n = C0.shape[1]
       
    def calcGauss(self, xij, cij, sij):
        x = self.X[xij[0]][xij[1]]
        c = self.C[cij[0]][cij[1]]
        s = self.S[sij[0]][sij[1]]
        return np.exp(-0.5 * ((x - c) / s) ** 2)

    def fxm(self, m):
        num = 0
        den = 0
        for i in range(self.R):
            prod = 1
            for j in range(self.n):
                prod *= self.calcGauss([m,j], [i,j], [i,j])
            num += self.b[i] * prod
            den += prod
        #print("fxm: ", num / den)
        return num / den
    
    def diff_m(self, m):
        #print("diff_m: ", self.fxm(m) - Y[m])
        return self.fxm(m) - Y[m]

    def error_m(self, m):
        return 0.5 * np.inner(self.diff(m), self.diff(m))

    def mu_i_m(self, i, m):
        prod = 1
        for j in range(self.n):
            prod *= self.calcGauss([m,j], [i,j], [i,j])
        #print("mu_i_m: ",i,prod)
        return prod

    def centerUpdate(self, step, i, m):
        den = 0
        for i in range(self.R):
            den += self.mu_i_m(i, m)
        dem_dbi = self.diff_m(m) * self.mu_i_m(i, m) / den
        self.b[i] -= step * dem_dbi
        print(-1 * step * dem_dbi)
        
        
def gradientDescent(X, Y, C0, S0, b0, tol = 1e-5, max_k=1000):
    # Initialize centers, widths, and b_i's
    C = C0
    S = S0
    b = b0
    k = 0
    # while error > tol or k < max_k
    # TODO: do grad descent on C, S, and B 
    
if __name__ == '__main__':
    C0 = np.array([[0,2],[2,4]])
    S0 = np.ones((2,2))
    b0 = np.array([1.,5.])
    X = np.array([[0.,2.],[2.,4.],[3.,6.]])
    Y = np.array([1.,5.,6.])
    tsfs = trainedStandardFuzzySystem(X, Y, C0, S0, b0) 
    tsfs.centerUpdate(1,0,2)   
    tsfs.centerUpdate(1,1,2)   
    print(tsfs.b[0])
    print(tsfs.b[1])
    






