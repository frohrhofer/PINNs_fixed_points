import numpy as np
import tensorflow as tf

class Loss():
    '''
    This class provides the data, physics and BC loss functions 
    '''   
    def __init__(self, model):      
        # store model (weights are updated during training)
        self.model = model
        
        
    def InitialCondition(self, X, U_true):    
        '''
        Simple Mean-Squared Error of training data
        for initial sequence
        '''
        U_pred = self.model(X)  
        return tf.reduce_mean(tf.square(U_pred - U_true))
 

    def NavierStokes(self, X_collocation):
        '''
        Physics Loss Function (Navier-Stokes Equations)
        '''  
        # tape forward propergation to retrieve gradients
        with tf.GradientTape() as t:
            t.watch(X_collocation)
            with tf.GradientTape() as tt:
                tt.watch(X_collocation)
                U = self.model(X_collocation)
            U_d = tt.batch_jacobian(U, X_collocation)        
        U_dd = t.batch_jacobian(U_d, X_collocation)
        
        # U shape: (x_col, f_i)
        u, v, p = U[:, 0], U[:, 1], U[:, 2]     
        # U_d shape: (x_col, f_i, dx_i)
        u_x, u_y, u_t = U_d[:, 0, 0], U_d[:, 0, 1], U_d[:, 0, 2]
        v_x, v_y, v_t = U_d[:, 1, 0], U_d[:, 1, 1], U_d[:, 1, 2] 
        p_x, p_y = U_d[:, 2, 0], U_d[:, 2, 1]       
        # U_dd shape: (x_col, f_i, dx_i, dx_j)
        u_xx, u_yy = U_dd[:, 0, 0, 0], U_dd[:, 0, 1, 1]
        v_xx, v_yy = U_dd[:, 1, 0, 0], U_dd[:, 1, 1, 1]       
        # Navier-Stokes (with Reynolds Number = 100)
        res_x = u_t + u*u_x + v*u_y + p_x - (u_xx + u_yy) / 100
        res_y = v_t + u*v_x + v*v_y + p_y - (v_xx + v_yy) / 100
  
        loss_Fx = tf.reduce_mean(tf.square(res_x))
        loss_Fy = tf.reduce_mean(tf.square(res_y))       
        return loss_Fx + loss_Fy
    
    
    def Inlet(self, X_inlet):
        '''
        Inlet Boundary Condition
        ''' 
        # GradientTape for determining derivatives      
        with tf.GradientTape() as t:
            t.watch(X_inlet)
            U_pred = self.model(X_inlet)
        U_d = t.batch_jacobian(U_pred, X_inlet) 
        # derivatives in x direction
        p_x = U_d[:, 2, 0]
        
        loss_u = tf.reduce_mean(tf.square(U_pred[:,0] - 1))  # u = 1
        loss_v = tf.reduce_mean(tf.square(U_pred[:,1]))      # v = 0
        loss_p = tf.reduce_mean(tf.square(p_x))              # zero Gradient
        return loss_u + loss_v + loss_p
 

    def Outlet(self, X_outlet):
        '''
        Outlet Boundary Condition:
        Zero pressure condition
        ''' 
        # GradientTape for determining derivatives 
        with tf.GradientTape() as t:
            t.watch(X_outlet)
            U_pred = self.model(X_outlet)
        U_d = t.batch_jacobian(U_pred, X_outlet)        
        # derivatives in x direction
        u_x, v_x, p_x = U_d[:, 0, 0], U_d[:, 1, 0], U_d[:, 2, 0]
       
        loss_u = tf.reduce_mean(tf.square(u_x)) # zeroGradient
        loss_v = tf.reduce_mean(tf.square(v_x)) # zeroGradient
        loss_p = tf.reduce_mean(tf.square(p_x)) # zeroGradient
        return loss_u + loss_v + loss_p
    
    
    def Wall(self, X_wall):
        '''
        Top and Bottom Wall Boundary Condition: 
        Moving wall with no-slip condition
        ''' 
        # GradientTape for determining derivatives 
        with tf.GradientTape() as t:
            t.watch(X_wall)
            U_pred = self.model(X_wall)
        U_d = t.batch_jacobian(U_pred, X_wall)        
        # derivatives in y direction
        u_y, p_y = U_d[:, 0, 1], U_d[:, 2, 1]
        
        loss_u = tf.reduce_mean(tf.square(U_pred[:,0] - 1)) # u = 1
        loss_v = tf.reduce_mean(tf.square(U_pred[:,1]))     # v = 0
        loss_p = tf.reduce_mean(tf.square(p_y))             # zero Gradient
        return loss_u + loss_v + loss_p
 

    def Cylinder(self, X_cyl):
        '''
        Cylinder (shell) Boundary Condition
        No-slip and no-penetration condition
        '''         
        # GradientTape for determining derivatives      
        with tf.GradientTape() as t:
            t.watch(X_cyl)
            U_pred = self.model(X_cyl)
        U_d = t.batch_jacobian(U_pred, X_cyl) 
        # derivatives in x and y direction
        p_x, p_y = U_d[:, 2, 0], U_d[:, 2, 1]
        # derivative normal to cylinder surface, cos(phi)=x/r, sin(phi)=y/r
        x, y = X_cyl[:, 0], X_cyl[:, 1]
        p_r = (2*x) * p_x + (2*y) * p_y
        
        loss_u = tf.reduce_mean(tf.square(U_pred[:,0])) # u = 0
        loss_v = tf.reduce_mean(tf.square(U_pred[:,1])) # v = 0
        loss_p = tf.reduce_mean(tf.square(p_r))         # zero (normal) gradient
        return loss_u + loss_v + loss_p
    