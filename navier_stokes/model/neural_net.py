import tensorflow as tf

from pathlib import Path

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

from model.data_loader import DataLoader
from model.loss_functions import Loss
from model.callback import CustomCallback


class PhysicsInformedNN(Model):
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
        # loss functions for data, physics and BC
        self.loss = Loss(self)
        # callback for log recording and saving
        self.callback = CustomCallback(config)  
        # create model path to save logs
        self.path = self.log_path.joinpath(self.version)
        self.path.mkdir(parents=True, exist_ok=True)
        print('*** PINN build & initialized ***')
        
        
    def build_layers(self, verbose):
        '''
        Builds nested neural network (tf.Sequential) and its layers
        The nested neural network is needed for the hard constraints)
        '''
        # set seed for weights initialization
        tf.random.set_seed(self.seed) 
        # nested neural network (for hard constraints)
        self.neural_net = Sequential(name='nested_PINN')  
        # build input layer (x,y,t)
        self.neural_net.add(InputLayer(input_shape=(3,)))
        # build hidden layers
        for i in range(self.N_hidden):
            self.neural_net.add(Dense(units=self.N_neurons, 
                                activation=self.activation))
        # build 2d linear output layer (Psi, p)
        self.neural_net.add(Dense(units=2, activation=None))
        # provide weights to outer class
        self._weights = self.neural_net.weights
        # print network summary
        if verbose:
            self.neural_net.summary() 
            
            
    def call(self, X, training=False):  
        '''
        Overwrites default call function for
        implementing hard constraints (continuity equation)
        '''        
        with tf.GradientTape() as tape:
            tape.watch(X)  
            U = self.neural_net(X)
        U_d = tape.batch_jacobian(U, X)
        # get u and v from (latent) stream function
        u, v = U_d[:, 0, 1], -U_d[:, 0, 0]
        # get pressure from regular prediction
        p = U[:, 1]
        # stack to 3d output
        return tf.stack([u, v, p], axis=1) 
    
    
    def train(self):
        '''
        Training loop with batch gradiend-descent optimization 
        Samples training data (collocation, BC) at each batch iteration
        '''                                          
        # learning rate schedule
        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=self.decay_rate)                                    
        # Adam optimizer with default settings for momentum
        self.optimizer = Adam(learning_rate=lr_schedule) 
        
        # CFD dataset for imposing initial sequence
        CFD_dataset = self.data.get_CFD_dataset()
                        
        print("*** Training started... ***")        
        for epoch in range(self.N_epochs):
          
            # take batches from CFD dataset
            for (X_CFD, U_CFD) in CFD_dataset:
                
                # sample new training data at each batch iteration
                X_collocation = self.data.sample_collocation()                                      
                X_inlet = self.data.sample_inlet()
                X_outlet = self.data.sample_outlet()
                X_wall = self.data.sample_wall()
                X_cylinder = self.data.sample_cylinder()
                
                # single optimization step
                logs = self.train_step(X_CFD, U_CFD, X_collocation, 
                                       X_inlet, X_outlet, X_wall, X_cylinder)   
                
            # write logs to callback
            self.callback.write_logs(logs, epoch)  
                   
        # save log
        self.callback.save_logs(self.path)
        print("### Training finished ###")
        return self.callback.log
     
    
    @tf.function
    def train_step(self, X_CFD, U_CFD, X_collocation, X_inlet, X_outlet, X_wall, X_cylinder):
        '''
        Performs a single gradient-descent optimization step
        '''
        
        # open a GradientTape to record forward/loss pass                   
        with tf.GradientTape() as tape:    
            
            # Initial Sequence Data Loss
            loss_IC = self.loss.InitialCondition(X_CFD, U_CFD)
            # Physics Loss (Navier-Stokes Equations)
            loss_NS = self.loss.NavierStokes(X_collocation)
            # BC Losses
            loss_inlet = self.loss.Inlet(X_inlet)
            loss_outlet = self.loss.Outlet(X_outlet)
            loss_wall = self.loss.Wall(X_wall)
            loss_cylinder = self.loss.Cylinder(X_cylinder)
            loss_BC = loss_inlet + loss_outlet + loss_wall + loss_cylinder
            
            # total train loss
            loss_train = loss_IC + loss_NS + loss_BC
            
        # retrieve gradients
        grads = tape.gradient(loss_train, self.weights)                    
        # perform single GD step 
        self.optimizer.apply_gradients(zip(grads, self.weights)) 
        
        # save logs for recording
        logs = {'loss_train': loss_train, 'loss_IC': loss_IC, 
                'loss_NS': loss_NS, 'loss_BC': loss_BC}         
        return logs