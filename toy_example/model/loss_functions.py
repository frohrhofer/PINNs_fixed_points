import tensorflow as tf


class Loss():
    '''
    This class provides the physics loss function to the network training
    '''   
    def __init__(self, model):
        
        # save neural network (weights are updated during training)
        self.model = model

        
    def toy_example(self, t_col):
        '''
        Determines physics loss residuals of the differential equation
        '''
        # the tf-GradientTape function is used to retreive network derivatives
        with tf.GradientTape() as tape:
            tape.watch(t_col)
            y = self.model(t_col)
        y_t = tape.gradient(y, t_col) 
        
        res = y_t - (y - y**3)
        loss = tf.reduce_mean(tf.square(res))
        return loss
        