import torch
from copy import deepcopy
from models.mobile_net import MobileNetV2
from utils.data_utils import classify_img, load_img, preprocess_img
from utils.model_utils import load_weights

# Constants
EPS = 0.001 # For numerical stability when dividing by sqrt(variance) in batch norm

def get_batch_norm_tensors_names(layer):
    r"""Returns a dict containing the names of the batch norm tensor in a layer built with
    nn.Sequential
    
    Arguments:
        layer(str): the name of the layer for which batch norm tensors are to be retrieved"""

    batch_norm_state = '.'.join(layer.split(".")[:-1]) + '.' + str(int(layer.split(".")[-1]) + 1)
    return  {'gamma': batch_norm_state + '.weight', 
             'beta' : batch_norm_state + '.bias',
             'running_mean': batch_norm_state + '.running_mean',
             'running_var': batch_norm_state + '.running_var',
             }

def get_batch_norm_state(tensor_names, state_dict):
    r"""Retrieves the tensors of the batch norm from the model state dictionary """
    
    return  {'gamma': state_dict[tensor_names['gamma']],
             'beta': state_dict[tensor_names['beta']],
             'running_mean': state_dict[tensor_names['running_mean']],
             'running_var': state_dict[tensor_names['running_var']],
             }
    
def compute_weight_scalers_and_biases(batch_norm_state):
    r""" Computes a scaling of the weights and biases in a conv layer s.t. the batch
    normalisation does not need to be explicitly applied after forward pass"""

    weight_scalers = batch_norm_state['gamma']/torch.sqrt(batch_norm_state['running_var'] + EPS)
    biases = batch_norm_state['beta'] - weight_scalers * batch_norm_state['running_mean']

    return weight_scalers, biases

def scale_weights(layer, state_dict):
    r""" Scales the weights of a layer so that batch norm if performed in the forward pass during inference"""

    batch_norm_names = get_batch_norm_tensors_names(layer)
    batch_norm_state = get_batch_norm_state(batch_norm_names, state_dict)
    w_scalers, biases = compute_weight_scalers_and_biases(batch_norm_state)
    weights = state_dict[layer + '.weight']
    scaled_weights = w_scalers.view(weights.shape[0], 1, 1, 1) * weights

    return {'scaled_weights': scaled_weights,
            'biases': biases}

def create_new_state_dict(layer, state_dict):
    r"""Returns a new state dict where the state of a conv layer specified in layer
    is replaced by a state that accounts for batch normalisation. The batch normalisation 
    is subsequently removed from the state dictionary. Assumes that the batch norm layer
    directly follows the layer specified using nn.Sequential model."""

    new_state_dict = deepcopy(state_dict)
    names = get_batch_norm_tensors_names(layer)
    # Remove batch norm state
    for value in names.values():
        del new_state_dict[value]
    del new_state_dict[layer + '.weight']
    # Compute scaled weights and biases
    layer_new_state = scale_weights(layer, state_dict)
    new_state_dict[layer + '.weight'] = layer_new_state['scaled_weights']
    new_state_dict[layer + '.bias'] = layer_new_state['biases']
    
    return new_state_dict

# Load original model
checkpoint_path = 'mobilenet_v2.pth.tar'
orig_model, state_dict = load_weights(MobileNetV2(combine_batch_norm=False), checkpoint_path, device='cpu')
orig_model.eval()
layer = 'features.0.0'

# Create a new state dictionary with scaled weights and save the weights
new_state_dict = create_new_state_dict(layer, state_dict)
torch.save(new_state_dict, 'mobilenet_v2_scaled_w.pth.tar')

#################################Testing ###############################################
# NB: In the interest of time I won't write proper unit tests, just a quick sanity check

# Load modified model
checkpoint_path = 'mobilenet_v2_scaled_w.pth.tar'
new_model, _ = load_weights(MobileNetV2(combine_batch_norm=True), checkpoint_path, device='cpu')
new_model.eval()

# Load image
image_path = 'test_img.JPG'
img = load_img(image_path)
transformed_img = preprocess_img(img)

# Run the model on the transformed image and compute probabilities
out_new = classify_img(transformed_img, new_model)
out_orig = classify_img(transformed_img, orig_model)

# Test that e.g., we return the same classes
for new, orig in zip(out_new.top5_classes, out_orig.top5_classes):
    assert new == orig
print("Top 5 class probs: {}".format(['%.5f' % elem for elem in out_new.top5_probs]))
print("For class indices: {}".format(out_new.top5_indices))
print("With labels {}".format(out_new.top5_classes))


