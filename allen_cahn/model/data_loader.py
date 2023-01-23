import numpy as np
import pandas as pd
import tensorflow as tf

from pyDOE import lhs
from numpy.random import seed, random


class DataLoader():
    
    # settings read from config (set as class attributes)
    args = ['seed', 'x_domain', 't_domain', 
            'N_col', 'N_IC', 'N_BC']
    # holds reference DataFrame
    df_ref = None
    
    def __init__(self, config):
        
        # load and set class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])

        # seed for data sampling
        seed(self.seed)

        # computational domain settings
        self.xmin, self.xmax = min(self.x_domain), max(self.x_domain)
        self.tmin, self.tmax = min(self.t_domain), max(self.t_domain)       
        # needed for latin-hypercube sampling
        self.Xmin = np.array([self.xmin, self.tmin])
        self.Xmax = np.array([self.xmax, self.tmax])  
        
        # load reference data
        self.df_ref = pd.read_csv('data/allen_cahn.csv') 
        print("*** DataLoader initialized ***")
                    
        
    def array2tensor(self, array, exp_dim=False):
        '''
        Auxiliary function: converts numpy-array to tf-tensor, expands dimensions
        '''         
        if exp_dim:
            array = np.expand_dims(array, axis=1)      
        return tf.convert_to_tensor(array, dtype=tf.float32)
    
    
    def collocation(self, N=None):  
        '''
        Samples collocation points using latin-hypercube sampling
        '''
        # take default data settings if N is not provided
        N = self.N_col if N == None else N
        # latin-hypercube sampling
        X_col = self.Xmin + (self.Xmax - self.Xmin) * lhs(2, N)
        return self.array2tensor(X_col)
           
    
    def initial_condition(self, N=None):
        '''
        Samples data from the initial condition
        ''' 
        # take default data settings if N is not provided
        N = self.N_IC if N == None else N
        # equally spaced points along IC axis
        #x_IC = np.linspace(self.xmin, self.xmax, N)
        x_IC = (self.xmax-self.xmin) * random(N) + self.xmin
        # add zero time stamps      
        X_IC = np.stack([x_IC, np.zeros(N)], axis=1)
        # initial condition
        u_IC = x_IC**2 * np.cos(np.pi*x_IC)
        # provide data through tf.tensors
        X_IC = self.array2tensor(X_IC)
        u_IC = self.array2tensor(u_IC, exp_dim=True)        
        return X_IC, u_IC
    
    
    def boundary_condition(self, N=None):
        
        # take default data settings if N is not provided
        N = self.N_BC if N == None else N
        # equally spaced time points along x-axis
        #t_BC = np.linspace(self.tmin, self.tmax, N) 
        t_BC = (self.tmax-self.tmin) * random(N) + self.tmin
        # top and bottom boundary points
        x_BC_top = np.ones(N) * self.xmax
        x_BC_bottom = np.ones(N) * self.xmin
        # stack to 2d array
        X_BC_top = np.stack([x_BC_top, t_BC], axis=1)
        X_BC_bottom = np.stack([x_BC_bottom, t_BC], axis=1)
        # provide data through tf.tensors
        X_BC_top = self.array2tensor(X_BC_top)
        X_BC_bottom = self.array2tensor(X_BC_bottom)       
        return X_BC_top, X_BC_bottom
    
    
    def reference_mesh(self):
        '''
        Provides mesh data from reference solution
        '''
        X_ref = tf.convert_to_tensor(self.df_ref[['x', 't']], dtype=tf.float32)
        u_ref = tf.convert_to_tensor(self.df_ref[['u']], dtype=tf.float32)   
        
        return X_ref, u_ref
    
    
    def reference_xcut(self, time=0):  
        '''
        Provides data for xcut at a selected time
        from reference data
        '''
        # apply time domain masked
        df_t = self.df_ref[self.df_ref['t'] == time]    
        # extract features and labels
        X_cut = tf.convert_to_tensor(df_t[['x', 't']], dtype=tf.float32)
        u_cut = tf.convert_to_tensor(df_t[['u']], dtype=tf.float32)
        
        return X_cut, u_cut
        
        
        
        
        