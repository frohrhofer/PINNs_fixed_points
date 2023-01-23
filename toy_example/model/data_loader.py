import numpy as np
import tensorflow as tf

from numpy.random import random

class DataLoader():
 
    # settings read from config (set as class attributes)
    args = ['seed', 'T', 'y0', 'N_col']
    
    
    def __init__(self, config):
        
        # load and set class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])
        # set seed for data sampling
        np.random.seed(self.seed)
        
        
    def array2tensor(self, array, exp_dim=True):
        '''
        Auxiliary function: converts numpy-array to tf-tensor
        '''         
        if exp_dim:
            array = np.expand_dims(array, axis=1)       
        return tf.convert_to_tensor(array, dtype=tf.float32)
    
                
    def t_line(self, t_delta=0.01): 
        '''
        Returns an equally-spaced data array for postprocessing
        and visualization of final predictions
        '''    
        t_line = np.arange(0, self.T, t_delta)        
        return self.array2tensor(t_line)
    
    
    def collocation(self, N=None):
        '''
        Returns an uniformly sampled collocation data set
        '''      
        # take default data settings if N is not provided
        N = self.N_col if N == None else N       
        t_col = self.T * random(N)  
        return self.array2tensor(t_col)
    
    
    def reference(self, t):
        '''
        Determines analytical solution for toy example equation
        '''       
        eps = abs(self.y0)
        try:
            c = np.sqrt(1/eps**2-1)
        except ZeroDivisionError:
            c = 0
        sign = np.sign(self.y0)
        
        y_true = sign*(1+c**2*np.exp(-2*t))**(-1/2)       
        return self.array2tensor(y_true, exp_dim=False)