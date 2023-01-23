import numpy as np
import tensorflow as tf

from numpy.random import random
from scipy.integrate import solve_ivp

class DataLoader():

    # settings read from config (set as class attributes)
    args = ['seed', 'g', 'l', 'T',
            'theta0', 'omega0', 'N_col']
    
       
    def __init__(self, config):

        # load and set class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])
        # set seed for data sampling
        np.random.seed(self.seed)
        
        # convert degrees to radians
        self.theta0 = np.radians(self.theta0)
        self.omega0 = np.radians(self.omega0)
        
        
    def array2tensor(self, array, exp_dim=True):
        '''
        Auxiliary function: converts numpy-array to tf-tensor, suitable for training
        '''         
        if exp_dim:
            array = np.expand_dims(array, axis=1)
        
        return tf.convert_to_tensor(array, dtype=tf.float32)
    
                
    def t_line(self, t_delta=0.01, tensor=True): 
        '''
        Returns an equally-spaced data array for postprocessing
        and visualization of final predictions
        '''    
        t_line = np.arange(0, self.T, t_delta)  
        
        if tensor == True:
            return self.array2tensor(t_line)
        else:
            return t_line
        
        
    def collocation(self, N=None):
        '''
        Returns an uniformly sampled collocation data set
        '''      
        # take default data settings if N is not provided
        N = self.N_col if N == None else N       
        t_col = self.T * random(N)  
        return self.array2tensor(t_col)
 
    
    def diff_equations(self, t, y):
        '''
        Auxiliary function: Used in Runge-Kutta Integration
        '''  
        theta, omega = y[0], y[1]
        return np.array([omega, -self.g / self.l * np.sin(theta)]) 


    def reference(self, N_eval=None):
        '''
        Determines reference solution by using Runge-Kutta Integration
        '''          
        if N_eval is None:
            t_line = self.t_line(tensor=False)
        else:
            t_line = np.linspace(0, self.T, N_eval) 
            
        # initial conditions
        init_y = [self.theta0, self.omega0]

        # solve ODE
        results = solve_ivp(self.diff_equations, (0, max(t_line)), 
                            init_y, method='RK45', t_eval=t_line, 
                            rtol=1e-8)
        
        t_line = self.array2tensor(results.t)
        theta = self.array2tensor(results.y[0])
        omega = self.array2tensor(results.y[1])
        return t_line, theta, omega