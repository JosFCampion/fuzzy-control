from numpy import sin, cos
import numpy as np
# import scipy as sp
import matplotlib.pyplot as plt
import fuzzy_controller as fc

class InvPend:
    def __init__(self, init_cond, step_size):
        self.x = init_cond
        self.h = step_size
    
    def eom(self, x, u):
        dx1 = x[1]
        alpha = (-x[2] - 0.25 * x[1] ** 2 * sin(x[0])) / 1.5
        beta = 0.5 * ((4 / 3) - (1 / 3) * cos(x[0]) ** 2)
        dx2 = (9.8 * sin(x[0]) + cos(x[0]) * alpha) / beta
        dx3 = -100 * x[2] + 100 * u
        return np.array([dx1, dx2, dx3]) 

    def euler_step(self, x, u):
        self.x += self.h * self.eom(x, u)

    def rk4_step(self, x, u):
        k1 = self.h * self.eom(x, u)
        k2 = self.h * self.eom(self.x + k1 / 2, u)
        k3 = self.h * self.eom(self.x + k2 / 2, u)
        k4 = self.h * self.eom(self.x + k3 / 2, u)
        self.x += (k1 + 2 * k2 + 2 * k3 + k4) / 6

if __name__ == '__main__':
    fuz_ctrl = fc.init_fuz_ctrl() 
    
    x0 = np.array([0.1, 0, 0])
    h = 0.001
    inv_pend = InvPend(x0, h)
    
    nsteps = 3000
    y = []
    u = []
    for i in range(nsteps):
        g0 = 2.
        g1 = 0.1
        g2 = 5.
        e = -inv_pend.x[0]
        de = -inv_pend.x[1]
        c_input = g2 * np.array([g0 * e, g1 * de])
        ui = fuz_ctrl.calc_ucrisp(c_input)
        inv_pend.rk4_step(inv_pend.x, ui)
        u.append(ui)
        y.append(inv_pend.x[0])
    
    t = np.arange(0., nsteps * h, h)
    plt.plot(t, y)
    plt.show()
    plt.plot(t, u)
    plt.show()
    
