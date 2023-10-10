import numpy as np
import tensorflow as tf
import gpflow as gpf
from gpflow.config import default_float
gpf.config.set_default_float(tf.float64)

# benchmarks, relevant references can be found in our paper
# dim: input dimension
# p: number of constraints
# name: problem's name with dim and p
# opt: global minimal value
# x: global minimum
# lb, ub: input domain
# all query points will first be re-scaled into [0, 1]^dim 
# please do not directly query the given global minimum as they are not re-scaled

class Gardner():
    def __init__(self, dim = 2, p = 2):
        self.dim = dim
        self.p = p
        self.name = f'Gardner(d=2,p={self.p})'
        self.opt = np.arcsin(0.95) - 1
        self.x = np.array([[4.71238898, 1.2532359]])
        self.lb = np.array([[0, 0]])
        self.ub = np.array([[2*np.pi, 2*np.pi]])
        
    def __call__(self, x0, index):
        x = x0 * (self.ub - self.lb) + self.lb
        if index == 0:
            ans = np.sin(x[:,0]) + x[:,1]
        else: 
            ans = np.sin(x[:,0]) * np.sin(x[:,1]) + 0.95
        return ans

class Gramacy():
    def __init__(self, dim = 2, p = 2):
        self.dim = dim
        self.p = p
        self.name = f'Gramacy(d={self.dim},p={self.p})'
        self.opt = 0.5998
        self.x = np.array([[0.1954, 0.4044]])
        self.lb = np.array([[0, 0]])
        self.ub = np.array([[1, 1]])
        
    def __call__(self, x, index):
        
        if index == 0:
            ans = x[:,0] + x[:,1]
        elif index == 1: 
            ans = 1.5 - x[:, 0] - 2. * x[:,1] - 0.5 * np.sin(2. * np.pi * (x[:,0] ** 2 - 2. * x[:,1]))
        elif index == 2:
            ans = x[:,0] ** 2 + x[:,1] ** 2 - 1.5
        return ans 

class Sasena():
    def __init__(self, dim = 2, p = 3):
        self.dim = dim
        self.p = p
        self.name = f'Sasena(d={self.dim},p={self.p})'
        self.opt = - 0.7483
        self.x = np.array([[0.2017, 0.8332]])
        self.lb = np.array([[0., 0.]])
        self.ub = np.array([[1., 1.]])
        
    def __call__(self, x0, index):
        x = x0 * (self.ub - self.lb) + self.lb
        
        if index == 0:
            ans = - np.square(x[:,0] - 1.) - np.square(x[:,1] - 0.5)
        elif index == 1:
            ans = (np.square(x[:,0] - 3.) + np.square(x[:,1] + 2.)) * np.exp(- np.power(x[:,1], 7)) - 12.
        elif index == 2:
            ans = 10. * x[:,0] + x[:,1] - 7.
        elif index == 3:
            ans = np.square(x[:,0] - 0.5) + np.square(x[:,1] - 0.5) - 0.2
        return ans

class G4():
    def __init__(self, dim = 5, p = 6):
        self.dim = dim
        self.p = p
        self.name = f'G4(d={self.dim},p={self.p})'
        self.opt = - 30665.539
        self.x = np.array([[78., 33., 29.995256025682, 45., 36.775812905788]])
        self.lb = np.array([[78., 33., 27., 27., 27.]])
        self.ub = np.array([[102., 45., 45., 45., 45.]])
        
    def __call__(self, x0, index):
        x = x0 * (self.ub - self.lb) + self.lb
        if index == 0:
            ans = 5.3578547 * np.square(x[:,2]) + 0.8356891 * x[:,0] * x[:,4] + 37.293239 * x[:,0] - 40792.141
        elif index in [1, 2]:
            ans = 85.334407 + 0.0056858 * x[:,1] * x[:,4] + 0.0006262 * x[:,0] * x[:,3] - 0.0022053 * x[:,2] * x[:,4]
            if index == 1:
                return ans - 92.
            else:
                return - ans
        elif index in [3, 4]:
            ans = 80.51249 + 0.0071317 * x[:,1] * x[:,4] + 0.0029955 * x[:,0] * x[:,1] + 0.0021813 * np.square(x[:,2])
            if index == 3:
                return ans - 110.
            else:
                return 90. - ans
        else:
            ans = 9.300961 + 0.0047026 * x[:,2] * x[:,4] + 0.0012547 * x[:,0] * x[:,2] + 0.0019085 * x[:,2] * x[:,3]
            if index == 5:
                return ans - 25.
            else:
                return 20. - ans
        return ans

class G6():
    def __init__(self, dim = 2, p = 2):
        self.dim = dim
        self.p = p
        self.name = f'G6(d={self.dim},p={self.p})'
        self.x = np.array([[14.095, 0.84296]])
        self.opt = - 6961.81388
        self.lb = np.array([[13.5, 0.5]])
        self.ub = np.array([[14.5, 1.5]])
        
    def __call__(self, x0, index):
        x = x0 * (self.ub - self.lb) + self.lb
        
        if index == 0:
            ans = np.power(x[:,0] - 10., 3) + np.power(x[:,1] - 20., 3)
        elif index == 1:
            ans = - np.square(x[:,0] - 5.) - np.square(x[:,1] - 5.) + 100.
        elif index == 2:
            ans = np.square(x[:,0] - 6.) + np.square(x[:,1] - 5.) - 82.81
        return ans
    
class G7():
    def __init__(self, dim = 10, p = 8):
        self.dim = dim
        self.p = p
        self.name = f'G7(d={self.dim},p={self.p})'
        self.opt = 24.3062091
        self.x = np.array([[2.171996, 2.363683, 8.773926, 5.095984, 0.9906548, 1.430574, 1.321644, 9.828726, 8.280092, 8.375927]])
        self.lb = -10. * np.ones([1, self.dim])
        self.ub = 10. * np.ones([1, self.dim])
        
    def __call__(self, x0, index):
        x = x0 * (self.ub - self.lb) + self.lb
        if index == 0:
            ans = np.square(x[:,0]) + np.square(x[:,1]) + x[:,0] * x[:,1] - 14. * x[:,0] - 16. * x[:,1] + np.square(x[:,2] - 10.) + 4. * np.square(x[:,3] - 5.) + np.square(x[:,4] - 3.) + 2. * np.square(x[:,5] - 1.) + 5. * np.square(x[:,6]) + 7. * np.square(x[:,7] - 11.) + 2. * np.square(x[:,8] - 10.) + np.square(x[:,9] - 7.) + 45.
        elif index == 1:
            ans = (4. * x[:,0] + 5. * x[:,1] - 3. * x[:,6] + 9. * x[:,7] - 105.) / 10.
        elif index == 2:
            ans = (10. * x[:,0] - 8. * x[:,1] - 17. * x[:,6] + 2. * x[:,7]) / 10.
        elif index == 3:
            ans = (- 8. * x[:,0] + 2. * x[:,1] + 5. * x[:,8] - 2. * x[:,9] - 12.) / 10.
        elif index == 4:
            ans = (3. * np.square(x[:,0] - 2.) + 4. * np.square(x[:,1] - 3) + 2. * np.square(x[:,2]) - 7. * x[:,3] - 120.) / 100.
        elif index == 5:
            ans = (5. * np.square(x[:,0]) + 8. * x[:,1] + np.square(x[:,2] - 6.) - 2. * x[:,3] - 40.) / 100.
        elif index == 6:
            ans = (0.5 * np.square(x[:,0] - 8.) + 2. * np.square(x[:,1] - 4.) + 3. * np.square(x[:,4]) - x[:,5] - 30.) / 100.
        elif index == 7:
            ans = (np.square(x[:,0]) + 2. * np.square(x[:,1] - 2.) - 2. * x[:,0] * x[:,1] + 14. * x[:,4] - 6. * x[:,5]) / 100.
        elif index == 8:
            ans = (- 3. * x[:,0] + 6. * x[:,1] + 12. * np.square(x[:,8] - 8.) - 7. * x[:, 9]) / 100.
        return ans
    
class G8():
    def __init__(self, dim = 2, p = 2):
        self.dim = dim
        self.p = p
        self.name = f'G8(d={self.dim},p={self.p})'
        self.opt = - 0.095825
        self.x = np.array([[1.2279713, 4.2453733]])
        self.lb = np.array([[0.5, 0.5]])
        self.ub = np.array([[10., 10.]])
        
    def __call__(self, x0, index):
        x = x0 * (self.ub - self.lb) + self.lb
        
        if index == 0:
            ans = np.power(np.sin(2. * np.pi * x[:,0]), 3) * np.sin(2. * np.pi * x[:,1]) / (np.power(x[:,0], 3) * (x[:,0] + x[:,1]))
            ans = - ans
        elif index == 1:
            ans = np.square(x[:,0]) - x[:,1] + 1.
        elif index == 2:
            ans = 1. - x[:,0] + np.square(x[:,1] - 4)
        return ans

class G9():
    def __init__(self, dim = 7, p = 4):
        self.dim = dim
        self.p = p
        self.name = f'G9(d={self.dim},p={self.p})'
        self.opt = 680.6300573
        self.x = np.array([[2.330499, 1.951372, -0.4775414, 4.365726, -0.6244870, 1.038131, 1.594227]])
        self.lb = - 10. * np.ones([1, self.dim])
        self.ub = 10. * np.ones([1, self.dim])
        
    def __call__(self, x0, index):
        x = x0 * (self.ub - self.lb) + self.lb

        if index == 0:
            ans = np.square(x[:,0] - 10.) + 5. * np.square(x[:,1] - 12.) + np.power(x[:,2], 4) + 3. * np.square(x[:,3] - 11.) + 10. * np.power(x[:,4], 6) + 7. * np.square(x[:,5]) + np.power(x[:,6], 4) - 4. * x[:,5] * x[:,6] - 10. * x[:,5] - 8. * x[:,6]
        elif index == 1:
            ans = ( 2. * np.square(x[:,0]) + 3. * np.power(x[:,1], 4) + x[:,2] + 4. * np.square(x[:,3]) + 5. * x[:,4] - 127. ) / 10000.
        elif index == 2:
            ans = (7. * x[:,0] + 3. * x[:,1] + 10. * np.square(x[:,2]) + x[:,3] - x[:,4] - 282.)/100.
        elif index == 3:
            ans = (23. * x[:,0] + np.square(x[:,1]) + 6. * np.square(x[:,5]) - 8.* x[:,6] - 196.)/100.
        elif index == 4:
            ans = (4. * np.square(x[:,0]) + np.square(x[:,1]) - 3. * x[:,0] * x[:,1] + 2. * np.square(x[:,2]) + 5. * x[:,5] - 11. * x[:,6]) / 100.

        return ans

class G10():
    def __init__(self, dim = 8, p = 6):
        self.dim = dim
        self.p = p
        self.name = f'G10(d={self.dim},p={self.p})'
        self.opt = 7049.3307
        self.x = np.array([[579.3167, 1359.943, 5110.071, 182.0174, 295.5985, 217.9799, 286.4162, 395.5979]])
        self.lb = np.array([[100., 1000., 1000., 10., 10., 10., 10., 10.]])
        self.ub = np.array([[10000., 10000., 10000., 1000., 1000., 1000., 1000., 1000.]])
        
    def __call__(self, x0, index):
        x = x0 * (self.ub - self.lb) + self.lb
        if index == 0:
            ans = x[:,0] + x[:,1] + x[:,2]
        elif index == 1:
            ans = - 1. + 0.0025 * (x[:,3] + x[:,5])
        elif index == 2:
            ans = - 1. + 0.0025 * (- x[:,3] + x[:,4] + x[:,6])
        elif index == 3:
            ans = - 1. + 0.01 * (- x[:,4] + x[:,7])
        elif index == 4:
            ans = (100. * x[:,0] - x[:,0] * x[:,5] + 833.33252 * x[:,3] - 83333.333) / 1e6
        elif index == 5:
            ans = (x[:,1] * x[:,3] - x[:,1] * x[:,6] - 1250. * x[:,3] + 1250. * x[:,4]) / 1e6
        elif index == 6:
            ans = (x[:,2] * x[:,4] - x[:,2] * x[:,7] - 2500. * x[:,4] + 1250000.) / 1e6

        return ans

class Tension_Compression():
    def __init__(self, dim = 3, p = 4):
        self.dim = dim
        self.p = p
        self.name = f'Tension_Compression(d={self.dim},p={self.p})'
        self.opt = 0.012666
        self.x = np.array([[11.21390736278739, 0.35800478345599, 0.05174250340926]])
        self.lb = np.array([[2., 0.25, 0.05]])
        self.ub = np.array([[15., 1.3, 2.]])
        
    def __call__(self, x0, index):
        x = x0 * (self.ub - self.lb) + self.lb
        if index == 0:
            ans = np.square(x[:,2]) * x[:,1] * (x[:,0] + 2.)
        elif index == 1:
            ans = 1. - np.power(x[:,1], 3) * x[:,0] / (71785. * np.power(x[:,2], 4))
        elif index == 2:
            ans = (4. * np.square(x[:,1]) - x[:,2] * x[:,1]) / (12566. * np.power(x[:,2], 3) * (x[:,1] - x[:,2])) + 1. / (5108. * np.square(x[:,2])) - 1.
        elif index == 3:
            ans = 1. - 140.45 * x[:,2] / (x[:,0] * np.square(x[:,1]))
        elif index == 4:
            ans = (x[:,1] + x[:,2]) / 1.5 - 1.
        return ans
    
class Pressure_Vessel():
    def __init__(self, dim = 4, p = 3):
        self.dim = dim
        self.p = p
        self.name = f'Pressure_Vessel(d={self.dim},p={self.p})'
        self.x = np.array([[0.8125, 0.4375, 42.0984, 176.6368]])
        self.opt = 6059.715
        self.lb = np.array([[0., 0., 10., 150.]])
        self.ub = np.array([[10., 10., 50., 200.]])
        
    def __call__(self, x0, index):
        x = x0 * (self.ub - self.lb) + self.lb
        x[:,0] = np.round(x[:,0] / 0.0625) * 0.0625
        x[:,1] = np.round(x[:,1] / 0.0625) * 0.0625
        if index == 0:
            ans = 0.6224 * x[:,0] * x[:,2] * x[:,3] + 1.7781 * x[:,1] * np.square(x[:,2]) + 3.1661 * np.square(x[:,0]) * x[:,3] + 19.84 * np.square(x[:,0]) * x[:,2]
        elif index == 1:
            ans = - x[:,0] + 0.0193 * x[:,2]
        elif index == 2:
            ans = - x[:,1] + 0.00954 * x[:,2]
        elif index == 3:
            ans = (- np.pi * np.square(x[:,2]) * x[:,3] - 4. * np.pi / 3. * np.power(x[:,2], 3) + 1296000) / 1000000.
        return ans
    
class Welded_Beam():
    def __init__(self, dim = 4, p = 5):
        self.dim = dim
        self.p = p
        self.name = f'Welded_Beam(d={self.dim},p={self.p})'
        self.opt = 2.381065
        self.x = np.array([[0.24435257, 6.2157922, 8.2939046, 0.24435257]])
        self.lb = np.array([[0.125, 0.1, 0.1, 0.1]])
        self.ub = np.array([[10., 10., 10., 10.]])
        
    def __call__(self, x0, index):
        x = x0 * (self.ub - self.lb) + self.lb
        
        if index == 0:
            ans = 1.10471 * np.square(x[:,0]) * x[:,1] + 0.04811 * x[:,2] * x[:,3] * (14. + x[:,1])
        elif index == 1:
            tau1 = 6000. / (np.sqrt(2.) * x[:,0] * x[:,1])
            tau21 = 6000. * (14. + 0.5 * x[:,1]) * np.sqrt(0.25 * (np.square(x[:,1]) + np.square(x[:,0] + x[:,2])))
            tau22 = 2. * (0.707 * x[:,0] * x[:,1] * (np.square(x[:,1]) / 12. + 0.25 * np.square(x[:,0] + x[:,2])))
            tau2 = tau21 / tau22
            tau = np.square(tau1) + np.square(tau2) + x[:,1] * tau1 * tau2 / np.sqrt(0.25 * (np.square(x[:,1]) + np.square(x[:,0] + x[:,2])))
            ans = np.sqrt(tau) - 13000
        elif index == 2:
            ans = 504000. / (np.square(x[:,2]) * x[:,3]) - 30000.
        elif index == 3:
            ans = x[:,0] - x[:,3]
        elif index == 4:
            ans = 6000. - 64746.022 * (1. - 0.0282346 * x[:,2]) * x[:,2] * np.power(x[:,3], 3)
        elif index == 5:
            ans = 2.1952 / (np.power(x[:,2], 3) * x[:,3]) - 0.25

        return ans
    
class Speed_Reducer():
    def __init__(self, dim = 7, p = 11):
        self.dim = dim
        self.p = p
        self.name = f'Speed_Reducer(d={self.dim},p={self.p})'
        self.opt = 2996.3482
        self.x = np.array([[3.5, 0.7, 17., 7.3, 7.8, 3.350215, 5.286683]])
        self.lb = np.array([[2.6, 0.7, 17., 7.3, 7.8, 2.9, 4.9]])
        self.ub = np.array([[3.6, 0.8, 28., 8.3, 8.3, 3.9, 5.9]])
        
    def __call__(self, x0, index):
        x = x0 * (self.ub - self.lb) + self.lb
        if index == 0:
            ans = 0.7854 * x[:,0] * np.square(x[:,1]) * (3.3333 * np.square(x[:,2]) + 14.9334 * x[:,2] - 43.0934) - 1.508 * x[:,0] * (np.square(x[:,5]) + np.square(x[:,6])) + 7.4777 * (np.power(x[:,5], 3) + np.power(x[:,6], 3)) + 0.7854 * (x[:,3] * np.square(x[:,5]) + x[:,4] * np.square(x[:,6]))
        elif index == 1:
            ans = 27. / (x[:,0] * np.square(x[:,1]) * x[:,2]) - 1.
        elif index == 2:
            ans = 397.5 / (x[:,0] * np.square(x[:,1]) * np.square(x[:,2])) - 1
        elif index == 3:
            ans = 1.93 * np.power(x[:,3], 3) / (x[:,1] * x[:,2] * np.power(x[:,5], 4)) - 1.
        elif index == 4:
            ans = 1.93 * np.power(x[:,4], 3) / (x[:,1] * x[:,2] * np.power(x[:,6], 4)) - 1.
        elif index == 5:
            ans = np.sqrt(np.square(745. * x[:,3] / (x[:,1] * x[:,2])) + 16900000.) / np.power(x[:,5], 3) - 110.
        elif index == 6:
            ans = np.sqrt(np.square(745. * x[:,4] / (x[:,1] * x[:,2])) + 157500000.) / np.power(x[:,6], 3) - 85.
        elif index == 7:
            ans = x[:,1] * x[:,2] - 40.
        elif index == 8:
            ans = - x[:,0] / x[:,1] + 5.
        elif index == 9:
            ans = x[:,0] / x[:,1] - 12.
        elif index == 10:
            ans = (1.5 * x[:,5] + 1.9) / x[:,3] - 1.
        elif index == 11:
            ans = (1.1 * x[:,6] + 1.9) / x[:,4] - 1.
        return ans