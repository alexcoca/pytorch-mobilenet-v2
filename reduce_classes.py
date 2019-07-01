import torch
from copy import deepcopy
from models.mobile_net import MobileNetV2
from utils.data_utils import classify_img, get_imagenet_labels, load_img, preprocess_img
from utils.model_utils import load_weights, load_weights_from_dict

def get_class_idxs(all_classes, desired_classes):
	r"""Returns the indices of the desired classes"""
	
	class_idxs = []

	for desired_class in desired_classes:
		for idx, label in enumerate(all_classes):
			if desired_class == label:
				class_idxs.append(idx)

	return class_idxs

def get_layer_state(layer, state_dict):
	r"""Returns a dict with the weights and biases of a linear layer """
	return {'weights': state_dict[layer + '.weight'],
			'biases': state_dict[layer + '.bias'],
			}

def discard_parameters(layer_state, keep):
	r"""Discards the all the units not contained in keep from the 
	weight matrix and bias vector. """

	keep_dims = torch.LongTensor(keep)
	return {'weights': layer_state['weights'][keep_dims],
		    'biases': layer_state['biases'][keep_dims],
			}

def modify_state_dict(layer, parameters, state_dict):
	r""" Returns a dict where the weights and biases of the 
	layer specified in layer are replaced with a set of
    weights and biases specified in parameters['weights'] and 
    parameters['biases']. """

	new_state_dict = deepcopy(state_dict) 
	new_state_dict[layer + '.weight'] = parameters['weights']
	new_state_dict[layer + '.bias'] = parameters['biases']
	
	return new_state_dict

def reduce_classes(keep, layer, state_dict, labels_path='imagenet_class_index.json'):
	r""" Returns a state dictionary where only the weights of 
   the classes specified in the keep list are specified. """

	all_classes = get_imagenet_labels(labels_path)
	class_idxs = get_class_idxs(all_classes, keep_classes)
	layer_state = get_layer_state(layer, state_dict)
	filtered_parameters = discard_parameters(layer_state, class_idxs)
	new_state_dict = modify_state_dict(layer, filtered_parameters, state_dict)
	
	return new_state_dict

# Load model and specify which classes to retain, and modify state dictionary
checkpoint_path = 'mobilenet_v2.pth.tar'
keep_classes = ['pickup', 'pier', 'piggy_bank', 
				'pill_bottle', 'pillow','ping-pong_ball', 
				'pinwheel', 'pirate', 'pitcher', 'plane',
				]
_, state_dict = load_weights(MobileNetV2(), checkpoint_path, device='cpu')
layer = 'classifier.1'
new_state_dict = reduce_classes(keep_classes, layer, state_dict)

# Reload the weights
model = load_weights_from_dict(MobileNetV2(n_class=len(keep_classes)), new_state_dict)
model.eval()

# Perform classification
image_path = 'test_img.JPG'
img = load_img(image_path)
transformed_img = preprocess_img(img)
out = classify_img(transformed_img, model, labels=keep_classes)
print("Top 5 class probs {}".format(['%.5f' % elem for elem in out.top5_probs]))
print("For cls repr by idx {}".format(out.top5_indices))
print("Top 5 classes {}".format(out.top5_classes))
