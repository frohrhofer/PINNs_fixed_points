import yaml

def load_config(file, config_update=None, verbose=True): 
    '''
    This function can be used to load and update (different) config files
    ''' 
    with open(file) as f:
        config = yaml.full_load(f)
    
    if config_update:
        config.update(config_update)
        
    if verbose:
        for key, item in config.items():
            print(key,":", item)
        
    return config