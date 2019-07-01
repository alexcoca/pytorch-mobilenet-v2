import io
import json
import torch
import torch.nn.functional as F
from collections import namedtuple
from PIL import Image
from torchvision import transforms

def classify_img(img, model, labels=None, labels_path='imagenet_class_index.json'):
    r"""Classifies the image img using model.
   
    Arguments:
        img (PIL.Image) : the image to be classified
        model: the model with chis to be used for classification
        labels (list): If classification with less than 1000 classes is performed, then
            labels should contain str representing the names of the labels considered.
            Set to None if all labels are used.
        labels_path (str): path to a json containing a mapping from integers to human
            readable labels"""

    out = namedtuple('out', 'top5_probs, top5_indices, top5_classes')

    logits = model(img)
    class_probs = F.softmax(logits, dim=1)
    top5_probs, top5_indices = torch.topk(class_probs, 5)
    top5_probs = top5_probs.squeeze().tolist()
    top5_indices = top5_indices.squeeze().tolist()
    # Deal with classification on a subset of classes
    if labels is None:
        idx2classlabel = get_imagenet_labels(labels_path)
    else:
        idx2classlabel = {i: label for i, label in enumerate(labels)}
    top5_predicted_classes = [idx2classlabel[index] for index in top5_indices]

    return out._make((top5_probs, top5_indices, top5_predicted_classes))

def get_imagenet_labels(path):
    r""" Returns human readable labels for ImageNet. Download mapping
    from https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
    
    Arguments:
        path (str): path where the mapping has been downloaded 
        
    Returns:
        idx2label (list): a list of 1000 str, with the string at index i 
            representing a human readble label corresponding to the class 
            index i  
    """
    
    with open("imagenet_class_index.json", "r") as read_file:
        class_idx = json.load(read_file)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

    return idx2label

def load_img(path):
    r"""Loads an image from the disk.
    
    Arguments:
        path (str): full path of the image to be loaded
    
    Returns:
        img (PIL.Image): a PIL Image object converted to RGB format """

    with open(path, 'rb') as f:
        img = Image.open(io.BytesIO(f.read()))
   
    return img.convert('RGB')

def preprocess_img(img, input_size=224):
    """Convert a PIL image to a [1, C, H, W] tensor and normalise using
   ImageNet corpus statistics"""

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Reshape the image to size of input images. Instead of cropping the center, I 
    # reshape to input_size x input_size - it doesn't make sense to discard most of the test image
    # by cropping. Apply the same normalisation as the training data. 
    transform_seq = transforms.Compose([transforms.Resize(((int(input_size)), int(input_size))),
                                        transforms.ToTensor(),
                                        normalize,
                                        ])

    return transform_seq(img).unsqueeze(0)
