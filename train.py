import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision.models import ResNet50_Weights
import wandb

from tqdm import tqdm

def train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [TRAIN, VAL]:
            if phase == TRAIN:
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Use tqdm to track progress and display loss
            data_loader = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch + 1}/{num_epochs}', leave=False)

            # Iterate over data
            for inputs, labels in data_loader:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only in training phase
                with torch.set_grad_enabled(phase == TRAIN):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

                # Update tqdm description with the current loss
                data_loader.set_postfix({'Loss': loss.item(), 'Acc': running_corrects.double().item() / total_samples})

                # WandB logging of batch loss and accuracy
                wandb.log({f"{phase}_batch_loss": loss.item(), f"{phase}_batch_accuracy": running_corrects.double() / total_samples})

            if phase == TRAIN:
                exp_lr_scheduler.step()

            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # WandB logging
            wandb.log({f"{phase}_loss": epoch_loss, f"{phase}_accuracy": epoch_acc})

            # Deep copy the model if it has the best validation accuracy
            if phase == VAL and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Save best validation model
    torch.save(best_model_wts, 'best_model.pt')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


data_dir = 'data/'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transforms = {
    TRAIN: transforms.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we randomly crop the image to 224x224 and
        # randomly flip it horizontally. 
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
    TEST: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
}

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x), 
        transform=data_transforms[x]
    )
    for x in [TRAIN, VAL, TEST]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=512,
        shuffle=True, num_workers=20
    )
    for x in [TRAIN, VAL, TEST]
}

dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}

for x in [TRAIN, VAL, TEST]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))
    
print("Classes: ")
class_names = image_datasets[TRAIN].classes
print(image_datasets[TRAIN].classes)

# Initialize the model
model = models.resnet50()
# We want to fine tune the last layer
# for param in model.parameters():
    # param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

if torch.cuda.is_available():
    model = model.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
wandb.init(project="technion-projects",name="yaniv_romano_gender_clasiification_no_pretrain")
wandb.watch(model)

config = wandb.config          # Initialize config
config.batch_size = 8          # input batch size for training (default: 64)
config.test_batch_size = 1000  # input batch size for testing (default: 1000)
config.epochs = 100             # number of epochs to train (default: 10)
config.lr = 0.01              # learning rate (default: 0.01)
config.momentum = 0.5        

# Train and evaluate
model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=150)

# WandB – Initialize a new run

# WandB – Config is a variable that holds and saves hyperparameters and inputs
# Save the final model
torch.save(model.state_dict(), 'final_model.pt')

# Test the model
model.eval()
test_loss = 0
correct = 0
total = 0

for inputs, labels in dataloaders[TEST]:
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()

    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print('Test Accuracy: {:.2f}%'.format(accuracy))
wandb.run.summary["Test Accuracy"] = accuracy
