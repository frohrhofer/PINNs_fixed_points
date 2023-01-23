import tensorflow as tf

from pathlib import Path

from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

from model.data_loader import DataLoader
from model.loss_functions import Loss
from model.callback import CustomCallback


class PhysicsInformedNN(Sequential):
    '''
    This class provides the Physics-Informed Neural Network
    '''       
    # settings read from config (set as class attributes)
    args = ['version', 'seed',
            'N_hidden', 'N_neurons', 'activation',
            'N_epochs', 'learning_rate', 'decay_rate']
    # default log Path
    log_path = Path('logs')
    
    def __init__(self, config, verbose=False): 
        
        # call parent constructor & build NN
        super().__init__(name='PhysicsInformedNN')    
        # load and set class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])
        
        self.build_layers(verbose) 
        # data loader for sampling data at each training epoch
        self.data = DataLoader(config) 
        # loss functions for IC and physics
        self.loss = Loss(self)
        # callback for log recording and saving
        self.callback = CustomCallback(config) 
        # create model path to save logs
        self.path = self.log_path.joinpath(self.version)
        self.path.mkdir(parents=True, exist_ok=True)
        print('*** PINN build & initialized ***')            
        

    def build_layers(self, verbose):
        '''
        Builds the network layers
        '''         
        # set seed for weights initialization
        tf.random.set_seed(self.seed)         
        # build input layer
        self.add(InputLayer(input_shape=(2,)))
        # build hidden layers
        for i in range(self.N_hidden):
            self.add(Dense(units=self.N_neurons, 
                           activation=self.activation))
        # build linear output layer
        self.add(Dense(units=1, 
                       activation=None))
        if verbose:
            self.summary()             
 
        
    def train(self):
        '''
        Training loop with batch gradiend-descent optimization 
        Samples training data (collocation, IC, BC) at each batch iteration
        '''                 
        # learning rate schedule
        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=self.decay_rate)          
        # Adam optimizer with default settings for momentum
        self.optimizer = Adam(learning_rate=lr_schedule)    
            
        print("Training started...")
        for epoch in range(self.N_epochs):
                                
            X_col = self.data.collocation() 
            X_IC, u_IC = self.data.initial_condition()
            X_BC_top, X_BC_bottom = self.data.boundary_condition()
                      
            # perform one train step
            train_logs = self.train_step(X_IC, u_IC, 
                                         X_BC_top, X_BC_bottom, 
                                         X_col)
            # provide logs to callback 
            self.callback.write_logs(train_logs, epoch)
           
        # save log
        self.callback.save_logs(self.path)
        print("Training finished!")
        return self.callback.log
    
    
    @tf.function
    def train_step(self, X_IC, u_IC, X_BC_top, X_BC_bottom, X_col):
        '''
        Performs a single SGD training step by minimizing the 
        IC, BC and physics loss residuals using MSE
        '''      
        # open a GradientTape to record forward/loss pass                   
        with tf.GradientTape() as tape: 

            # inital condition loss
            loss_IC = self.loss.initial_condition(X_IC, u_IC)
            # boundary condition loss
            loss_BC = self.loss.boundary_condition(X_BC_top, X_BC_bottom)
            # physics loss
            loss_AC = self.loss.allen_cahn(X_col)
            # final training loss (with greater weight for IC loss)
            loss_train = 100 * loss_IC + loss_BC + loss_AC
                       
        # retrieve gradients
        grads = tape.gradient(loss_train, self.weights)        
        # perform single GD step 
        self.optimizer.apply_gradients(zip(grads, self.weights))       
        
        # save logs for recording
        train_logs = {'loss_train': loss_train, 'loss_IC': loss_IC, 
                      'loss_BC': loss_BC, 'loss_AC': loss_AC}       
        return train_logs
    
    
    
