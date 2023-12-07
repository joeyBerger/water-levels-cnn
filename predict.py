import json
import sys

import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torchvision import datasets, models, transforms

from get_cli_args import get_predict_cli_args


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    image.thumbnail((256, 256))

    width, height = image.size

    # Calculate the dimensions for the center crop
    size = 224
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    # Crop the image using the calculated dimensions
    image = image.crop((left, top, right, bottom))

    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])

    numpy_array = np.array(image)
    numpy_array = (numpy_array / 255 - means) / stds

    # Reorder the dimensions to match PyTorch's expectation (from HWC to CHW)
    return torch.from_numpy(numpy_array.transpose((2, 0, 1)))


def predict(image_path, model, topk, device):
    topk = 3 if topk < 1 else topk

    model.eval()

    img = process_image(Image.open(image_path))

    with torch.no_grad():
        img = img.to(device)
        output = model.forward(torch.unsqueeze(img.float(), 0))

    ps = torch.exp(output)

    top_p, top_class = ps.topk(topk, dim=1)

    return top_p, top_class


def get_categories_keys(classes):
    keys = []
    for item in classes[0]:
        keys.append(str(item.item() + 1))
    return keys


def get_categories_names(keys):
    return [cat_to_name[key] for key in keys]


# Get user cli inputs
image_path, checkpoint_path, category_names, top_k, enable_gpu = get_predict_cli_args()

checkpoint = torch.load(checkpoint_path)

model = models.densenet121(
    pretrained=True) if checkpoint['arch'] == 'densenet121' else models.vgg16(pretrained=True)

model.classifier = nn.Sequential(nn.Linear(checkpoint['classifier_inputs'], checkpoint['hidden_units']),
                                 nn.ReLU(),
                                 nn.Dropout(checkpoint['dropout']),
                                 nn.Linear(
                                     checkpoint['hidden_units'], checkpoint['outputs']),
                                 nn.LogSoftmax(dim=1))

model.load_state_dict(checkpoint['state_dict'], strict=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available()
                      and checkpoint['device'] == "cuda" else "cpu")
model.to(device)

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

probs, classes = predict(image_path, model, top_k, device)

# Get category namess
categories_keys = get_categories_keys(classes)
categories_names = get_categories_names(categories_keys)

print(f"\n\nTop predicted image using {checkpoint['arch']}:"
      f"\n\t{categories_names[0]}"
      f"\n\nWith a predicted probability of:"
      f"\n\t{probs[0][0]: .4f}%\n\n"
      )

if top_k > 1:
    print("\nThe {topk} highest predicted outcomes are: ")
    for i in range(len(categories_names)):
        print(f"\t{categories_names[i]} - {probs[0][i]: .4f}%")

    print('\n')
