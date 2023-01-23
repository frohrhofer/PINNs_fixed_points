import numpy as np
import tensorflow as tf

class Loss():
    '''
    This class provides the physics loss function 
    '''       
     # settings read from config (set as class attributes)
    args = ['g', 'l', 'theta0', 'omega0']
    
    
    def __init__(self, model, config):
        
        # load and set class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])
            
        # convert degrees to radians
        self.theta0 = np.radians(self.theta0)
        self.omega0 = np.radians(self.omega0)
        
        
        # save neural network (weights are updated during training)
        self.model = model
        
        
    def initial_condition(self):
        '''
        Determines IC loss for angle and velocity
        '''        
        t0 = tf.constant([0.])    
        with tf.GradientTape() as tape:
            tape.watch(t0)
            theta0 = self.model(t0)
        omega0 = tape.gradient(theta0, t0)
        
        # IC loss for angle
        loss_IC1 = tf.reduce_mean(tf.square(theta0 - self.theta0))
        # and velocity
        loss_IC2 = tf.reduce_mean(tf.square(omega0 - self.omega0))
        return loss_IC1 + loss_IC2
               
        
    def pendulum(self, t_col):
        '''
        Determines physics loss of the pendulum's differential equation
        '''
        # the tf-GradientTape function is used to retreive network derivatives
        with tf.GradientTape() as t:
            t.watch(t_col)
            with tf.GradientTape() as tt:
                tt.watch(t_col)    
                theta = self.model(t_col)
            omega = tt.gradient(theta, t_col) 
        omega_t = t.gradient(omega, t_col)
        
        res = omega_t + self.g/self.l * tf.math.sin(theta)
        loss = tf.reduce_mean(tf.square(res))
        return loss

    
    
    
