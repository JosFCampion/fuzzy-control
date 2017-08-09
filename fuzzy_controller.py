import numpy as np
# import scipy as sp
import matplotlib.pyplot as plt

class TriMemFun:
    """triangular membership function """
    def __init__(self, width_of_tri, center_of_mf, l_r_or_c='c'):
        self.w = width_of_tri
        self.c = center_of_mf
        if not (l_r_or_c == 'l' or l_r_or_c == 'r' or l_r_or_c == 'c'):
            raise ValueError(
            "Invalid type (choose 'l' for left, 'r' for right, or 'c' for center).")
        self.t = l_r_or_c # type = left, right, or center
    
    def calc_output(self, x):
        """calculates membership value for given input variable"""
        if self.t == 'l':
            if x < self.c:
                return 1.
            return max(0, 1 + (self.c - x) / (0.5 * self.w))
        elif self.t == 'r':
            if x < self.c:
                return max(0, 1. + (x - self.c) / (0.5 * self.w))
            return 1.
        else: # type == 'c'
            if x < self.c:
                return max(0, 1 + (x - self.c) / (0.5 * self.w))
            return max(0, 1 + (self.c - x) / (0.5 * self.w))

    def chopped_area(self, h):
        if self.t != 'c':
            raise ValueError("Area under curve only applies to type 'c' mfs.")
        return self.w * (h - h ** 2 / 2)

    def plot(self, show=True):
        X = np.array([self.c - self.w, self.c - 0.5 * self.w, self.c,
                      self.c + 0.5 * self.w, self.c + self.w])
        Y = np.array([0,0,1,0,0])
        if self.t == 'r':
            Y = np.array([0,0,1,1,1])
        if self.t == 'l':
            Y = np.array([1,1,1,0,0])
        plt.plot(X,Y)
        if show: plt.show()

class TriUnivDisc:
    """triangular universe of discourse; made of up several
       triangular membership functions """
    def __init__(self, tri_mem_funs):
        self.mfs = tri_mem_funs

    def plot(self):
        for i in range(len(self.mfs)):
            self.mfs[i].plot(False)
        plt.show()
            
class FuzzySystem:
    """MISO fuzzy system with two inputs, one outputs, and rule base"""
    def __init__(self, univ_disc_in1, univ_disc_in2, 
                 univ_disc_out, rules_base):
        self.ud_in1 = univ_disc_in1
        self.ud_in2 = univ_disc_in2
        self.ud_out = univ_disc_out
        self.rules = rules_base

    def calc_ucrisp(self, x):
        """performs fuzzy control process on input x"""
        n1 = len(self.ud_in1.mfs)
        n2 = len(self.ud_in2.mfs) 

        # Find the values of all membership functions given the values for x1 and x2
        mf1_vals = []
        for i in range(n1):
            mf1_vals.append(self.ud_in1.mfs[i].calc_output(x[0]))
        mf2_vals = []
        for j in range(n2):
            mf2_vals.append(self.ud_in2.mfs[j].calc_output(x[1]))

        # Find the values for the premise membership functions for a given x1 and x2
        # using the minimum operation
        prem_mat = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                prem_mat[i,j] = min(mf1_vals[i], mf2_vals[j])

        # Find the areas under the membership functions for all possible implied
        # fuzzy sets
        imps = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                imps[i,j] = self.ud_out.mfs[self.rules[i,j]].chopped_area(prem_mat[i,j])

        # Defuzzification: calculate ucrisp, the output of the fuzzy controller
        num = 0
        den = 0
        for i in range(n1):
            for j in range(n2):
                num += imps[i,j] * self.ud_out.mfs[self.rules[i,j]].c
                den += imps[i,j]
        return num / den

def init_fuz_ctrl():
    ws = [np.pi/2, np.pi/4, 20]
    c1s = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
    c2s = [-np.pi/4, -np.pi/8, 0, np.pi/8, np.pi/4]
    c3s = [-20, -10, 0, 10, 20]
    centers = [c1s, c2s, c3s]

    # Input 1: e(t)
    input1_mfs = []
    input1_mfs.append(TriMemFun(ws[0], c1s[0], 'l'))
    for i in range(1,4):
        input1_mfs.append(TriMemFun(ws[0], c1s[i], 'c'))
    input1_mfs.append(TriMemFun(ws[0], c1s[-1], 'r'))
    ud_in1 = TriUnivDisc(input1_mfs)
    
    # Input 2: de(t)/dt
    input2_mfs = []
    input2_mfs.append(TriMemFun(ws[1], c2s[0], 'l'))
    for i in range(1,4):
        input2_mfs.append(TriMemFun(ws[1], c2s[i], 'c'))
    input2_mfs.append(TriMemFun(ws[1], c2s[-1], 'r'))
    ud_in2 = TriUnivDisc(input2_mfs)
   
    # Output: u(t)
    output_mfs = []
    for i in range(5):
        output_mfs.append(TriMemFun(ws[2], c3s[i], 'c'))
    ud_out = TriUnivDisc(output_mfs)
    
    rule_base = np.array([[4,4,4,3,2],[4,4,3,2,1],[4,3,2,1,0],
                         [3,2,1,0,0],[2,1,0,0,0]])
    
    return FuzzySystem(ud_in1, ud_in2, ud_out, rule_base)
        
if __name__ == '__main__':
    x = np.array([0., np.pi/8 - np.pi/32]) 
    fuz_sys = init_fuz_ctrl()
    print("ucrisp: ", fuz_sys.calc_ucrisp(x))
    
    
