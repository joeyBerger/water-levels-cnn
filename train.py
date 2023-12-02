import sys
import torch
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from get_cli_args import get_train_cli_args

def testModel(loader):
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    return test_loss, accuracy

# Get user cli inputs
data_dir, arch_id, learning_rate, hidden_units, epochs, enable_gpu = get_train_cli_args()

# Set directories
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# Define transforms for the training and validation
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
validate_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(validate_data, batch_size=64)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() and enable_gpu else "cpu")

# Setup model and classifier details
if arch_id == 0:
    model = models.densenet121(pretrained=True)
    classifier_inputs = 1024
    arch = 'densenet121'
else:
    model = models.vgg16(pretrained=True)
    classifier_inputs = 25088
    arch = 'vgg16'

# Build and train network
for param in model.parameters():
    param.requires_grad = False
    
from collections import OrderedDict
model.classifier = nn.Sequential(nn.Linear(classifier_inputs, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, 3),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

model.to(device);

steps = 0
validate_model = True
running_loss = 0
validate_every = 5
for epoch in range(epochs):
       
    for inputs, labels in trainloader:
        steps += 1
        
        print('current step:', steps)
        
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if validate_model and steps % validate_every == 0:
            model.eval()
    
            test_loss, accuracy = testModel(validationloader)
    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/validate_every:.3f}.. "
                  f"Test loss: {test_loss/len(validationloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validationloader):.3f}")
            running_loss = 0
            model.train()

# Setup info to save            
model_dict={'arch': arch,
            'state_dict': model.state_dict(), 
            'epochs': epochs,
            'device': str(device), 
            'classifier_inputs': classifier_inputs,
            'hidden_units': hidden_units,
            'dropout': 0.2,
            'outputs': 3
           }

# Save model info
torch.save(model_dict, 'checkpoint.pth')

print('Save Complete')
