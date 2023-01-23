import json
import numpy as np

from pathlib import Path

class CustomCallback():
    '''
    This class provides a custom callback that stores, prints and saves
    training logs (suitable for postproessing in json-format)
    '''
    # settings read from config (set as class attributes)
    args = ['N_epochs', 'freq_log', 'freq_print', 'keys_print']
    
    def __init__(self, config):    
        
        # load and set class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])
        
        # determines digits for 'fancy' log printing
        self.digits = int(np.log10(self.N_epochs)+1)      
        # create log from config file (is saved with training logs)
        self.log = config.copy()
  
        
    def write_logs(self, logs, epoch, force_print=False):  
        '''
        This function is called during network training and
        stores/prints training logs
        '''      
        # store training logs
        if (epoch % self.freq_log) == 0:
            # exceptions errors are used to catch the different data formats provided
            for key, item in logs.items():
                # append if list already exists
                try:
                    self.log[key].append(item.numpy().astype(np.float64))
                # create list otherwise
                except KeyError:
                    try:
                        self.log[key] = [item.numpy().astype(np.float64)]
                    # if list is given 
                    except AttributeError:
                        self.log[key] = item
                         
        # print training logs
        if (epoch % self.freq_print) == 0:
            
            s = f"{epoch:{self.digits}}/{self.N_epochs}"
            for key in self.keys_print:
                try:
                    s += f" | {key}: {logs[key]:2.2e}"
                except:
                    pass
            print(s) 
            
            
    def save_logs(self, path):
        '''
        Saves recorded training logs in json-format
        '''        
        log_file = path.joinpath(f'log.json')       
        with open(log_file, "w") as f:
            json.dump(self.log, f, indent=2)
        print("*** logs saved ***")