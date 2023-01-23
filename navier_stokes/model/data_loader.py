import sys
import tensorflow as tf
import pandas as pd
import numpy as np

from pathlib import Path
from pyDOE import lhs
from numpy.random import seed, random

class DataLoader:
    
    # settings read from config (set as class attributes)
    args = ['seed',
            'x_domain', 'y_domain', 't_domain', 't_initial',
            'N_initial', 'N_batch', 'N_collocation', 
            'N_cylinder', 'N_inlet', 'N_outlet', 'N_wall']
    
    
    def __init__(self, config):
        
        # load and set class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])
        
        # seed for data sampling
        seed(self.seed)
        
        # spatial domain settings      
        self.xmin, self.xmax = min(self.x_domain), max(self.x_domain)
        self.ymin, self.ymax = min(self.y_domain), max(self.y_domain) 
        
        # temporal domain settings
        # for initial sequence training data
        self.tmin_init, self.tmax_init = min(self.t_initial), max(self.t_initial)    
        # and collocation (and BC) training data
        self.tmin, self.tmax = min(self.t_domain), max(self.t_domain) 
        print("*** DataLoader initialized ***")
              
    
    def load_csv(self, masked=True): 
        '''
        Loads CFD data found as .csv file in 'data' folder
        and returns DataFrame
        '''
        csv_file = Path('data/navier_stokes.csv')
        # check if data has been already downloaded
        if not csv_file.is_file():
            print("ERROR: Could not find data file! (run notebook in 'data' folder)")
            sys.exit(1)

        data = pd.read_csv(csv_file)   
        # apply domain mask (reduce computational domain)
        if masked == True:
            mask_x = (data['x']>=self.xmin) & (data['x']<=self.xmax)
            mask_y = (data['y']>=self.ymin) & (data['y']<=self.ymax)
            data = data[mask_x & mask_y]
        print("*** CFD data loaded ***")
        return data
          
    
    def features_and_labels(self, data):
        '''
        Splits DataFrame into features and labels DataFrame
        '''
        X = tf.convert_to_tensor(data[['x', 'y', 't']], dtype=tf.float32)
        U = tf.convert_to_tensor(data[['u', 'v', 'p']], dtype=tf.float32)
        return X, U
    
    
    def get_CFD_dataset(self):
        '''
        Loads CFD data, extracts features and labels,
        and provides data through shuffled Dataset
        '''        
        CFD_data = self.load_csv(masked=True)
        
        # truncate time domain
        mask_t = (CFD_data['t']>=self.tmin_init) & (CFD_data['t']<=self.tmax_init)
        CFD_data = CFD_data[mask_t]
        
        # reduce total number of training points
        if len(CFD_data) > self.N_initial:
            CFD_data = CFD_data.sample(self.N_initial, random_state=self.seed)
        
        # extract features and labels
        X_CFD, U_CFD = self.features_and_labels(CFD_data)
               
        # shuffling and batching
        CFD_dataset = tf.data.Dataset.from_tensor_slices((X_CFD, U_CFD))
        CFD_dataset = CFD_dataset.shuffle(buffer_size=len(CFD_dataset)).batch(self.N_batch)
        
        print("Trainset size: ", X_CFD.shape[0])
        return CFD_dataset
  

    def sample_collocation(self, N=None):
        '''
        Samples collocation points inside compuational domain
        (excludes data points inside cylinder)
        '''
        # take default data settings if N is not provided
        N = self.N_collocation if N == None else N
        
        # sample collocation data until N reached
        X_col = []
        while len(X_col) < N:          
            x_sample = (self.xmax - self.xmin) * random() + self.xmin
            y_sample = (self.ymax - self.ymin) * random() + self.ymin           
            # exclude coordinates that fall inside cylinder
            if (x_sample**2 + y_sample**2) < 0.5**2:
                continue
            else:
                t_sample = (self.tmax - self.tmin) * random() + self.tmin
                X_col.append([x_sample, y_sample, t_sample])
                
        return tf.convert_to_tensor(X_col, dtype=tf.float32)
    
    
    def sample_cylinder(self, N=None):
        '''
        Samples BC data at cylinder shell
        '''
        # take default data settings if N is not provided
        N = self.N_cylinder if N == None else N
        
        # sample random angles
        alpha = 2 * np.pi * random(N)
        # calculating cartesian coordinates
        x_cylinder = 0.5 * np.cos(alpha)
        y_cylinder = 0.5 * np.sin(alpha)     
        # add random time samples and concatenating arrays
        t_cylinder = (self.tmax - self.tmin) * random(N) + self.tmin
        X_cylinder = np.stack([x_cylinder, y_cylinder, t_cylinder], axis=1)
        
        return tf.convert_to_tensor(X_cylinder, dtype=tf.float32)
    
    
    def sample_inlet(self, N=None):
        '''
        Samples BC data at inlet (at xmin)
        '''
        # take default data settings if N is not provided
        N = self.N_inlet if N == None else N
        
        # lhs sampling over y and t domain
        X_min = [self.ymin, self.tmin]
        X_range = [self.ymax-self.ymin, self.tmax-self.tmin]
        yt_ticks =  X_min + X_range * lhs(2, N)
        # x_ticks at xmin
        x_ticks = np.expand_dims(np.ones(N) * self.xmin, axis=0).T
        # concatenating arrays
        X_inlet = np.concatenate([x_ticks, yt_ticks], axis=1)
        
        return tf.convert_to_tensor(X_inlet, dtype=tf.float32)
    
    
    def sample_outlet(self, N=None):
        '''
        Samples BC data at outlet (at xmax)
        '''
        # take default data settings if N is not provided
        N = self.N_outlet if N == None else N
        
        # lhs sampling over y and t domain
        X_min = [self.ymin, self.tmin]
        X_range = [self.ymax-self.ymin, self.tmax-self.tmin]
        yt_ticks =  X_min + X_range * lhs(2, N)
        # x_ticks at xmax
        x_ticks = np.expand_dims(np.ones(N) * self.xmax, axis=0).T
        # concatenating arrays to 3d
        X_outlet = np.concatenate([x_ticks, yt_ticks], axis=1)

        return tf.convert_to_tensor(X_outlet, dtype=tf.float32)
        
        
    def sample_wall(self, N=None):
        '''
        Samples BC data at top and bottom wall (ymin, ymax)
        '''
        # take default data settings if N is not provided
        N = self.N_wall if N == None else N

        # lhs sampling over x and t domain
        X_min = [self.xmin, self.tmin]
        X_range = [self.xmax-self.xmin, self.tmax-self.tmin]  
        X_wall = X_min + X_range * lhs(2, N) 
        x_ticks, t_ticks = X_wall[:,0:1], X_wall[:,1:2]
        # random choice for top or bottom wall (ymin, ymax)
        y_ticks = np.expand_dims(np.random.choice([self.ymin, self.ymax], N), axis=0).T
        # concatenating arrays
        X_wall = np.concatenate([x_ticks, y_ticks, t_ticks], axis=1)

        return tf.convert_to_tensor(X_wall, dtype=tf.float32)
        

    
    
   
    