from models.mobile_net import MobileNetV2
from utils.model_utils import load_weights
from utils.data_utils import classify_img, load_img, preprocess_img

# Load model and restore weights
checkpoint_path = 'mobilenet_v2.pth.tar'
model, _ = load_weights(MobileNetV2(), checkpoint_path, device='cpu')
model.eval()

# Load image
image_path = 'test_img.JPG'
img = load_img(image_path)
transformed_img = preprocess_img(img)

# Run the model on the transformed image and compute probabilities
out = classify_img(transformed_img, model)
print("Top 5 class probs {}".format(['%.5f' % elem for elem in out.top5_probs]))
print("For cls repr by idx {}".format(out.top5_indices))
print("Top 5 classes {}".format(out.top5_classes))



