import torch

def load_weights(model, checkpoint_path, device='cpu'):
    r""" Loads the weights stored in a checkpoint file.
    
    Arguments:
        model: an instance of the model whose weights should be initialised
        checkpoint_path (str): full path pointing to the .pth file where the
            checkpoint is stored
        device (str): set to 'cpu' to load the weights on the CPU, otherwise
            pass a device ordinal to it ('0')
            
    Returns:
        an instance of model initialised with the weights stored in checkpoint_path"""
    
    if device == 'cpu':
        state_dict = torch.load(checkpoint_path, map_location=torch.device(device))
    else:
        state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(int(device)))
    
    model.load_state_dict(state_dict)

    return model, state_dict

def load_weights_from_dict(model, state_dict):
    r""" Loads a model with weights stored in a state dictionary
    
    Arguments:
        model: an instance of the model whose weights should be initialised
        state_dict (collections.OrderedDict): a dictionary containing the state 
            in which the model should be initialised

    Returns:
        an instance of an initialised model with the state in state_dict
    """ 

    model.load_state_dict(state_dict)

    return model

