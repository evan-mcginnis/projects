import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from  torchvision.datasets import ImageFolder
import timm

import os
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser("Pytorch test")

parser.add_argument("-i", "--input", action="store", required=False, default="d:/kaggle/cards", help="Base directory of dataset")
parser.add_argument("-lg", "--logging", action="store", required=False, default="logging.ini", help="Logging configuration file")
parser.add_argument("-o", "--output", action="store", required=False, default=".", help="Output directory")
parser.add_argument("-g", "--graph", action="store_true", required=False, default=False, help="Graph loss")
parser.add_argument("-e", "--epochs", action="store", required=False, default=5, type=int,  help="Epochs")
parser.add_argument("-m", "--model", action="store", required=False, help="Model file")
parser.add_argument("-t", "--test", action="store", required=False, help="Test model with image")

arguments = parser.parse_args()

model = None
trainModel = False
saveModel = False

# In the user specified a model file that does not exist, the implication is to save it there
if arguments.model is not None:
    # If the model is not there, train and save
    if not os.path.exists(arguments.model):
        saveModel = True
        trainModel = True
    # If the model is there, do not train and save
    else:
        saveModel = False
        trainModel = False

else:
    saveModel = False
    trainModel = True

#
# M O D E L  P R O D U C T I O N
#
class PlayingCardDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


try:
    dataset = PlayingCardDataset(
        data_dir=arguments.input + '/train'
    )
except FileNotFoundError:
    print(f"Unable to find train dataset at {arguments.input}/train")
    sys.exit(-1)

#print(f"Dataset loaded: {len(dataset)}")

# image, label = dataset[6000]
# print(label)
# image
# The mapping between class and numeric
data_dir='d:/kaggle/cards/train'
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
# debug
#print(target_to_class)

# Transform images to the same size
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = PlayingCardDataset(data_dir, transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class SimpleCardClassifer(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifer, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        # Feature size
        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output

# Suppress warnings about symlinks
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
model = SimpleCardClassifer(num_classes=53)

# T R A I N I N G

# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if trainModel:
    base = arguments.input
    train_folder = base + '/train/'
    valid_folder = base + '/valid/'
    test_folder = base + '/test/'

    train_dataset = PlayingCardDataset(train_folder, transform=transform)
    val_dataset = PlayingCardDataset(valid_folder, transform=transform)
    test_dataset = PlayingCardDataset(test_folder, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # %%
    # Simple training loop
    train_losses, val_losses = [], []


    model = SimpleCardClassifer(num_classes=53)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(arguments.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc='Training loop'):
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation loop'):
                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{arguments.epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

else:
    model = torch.load(arguments.model, weights_only=False)

if arguments.graph:
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.title("Loss over epochs")
    plt.show()

if saveModel:
    torch.save(model, arguments.model)

if arguments.test is not None:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np


    # Load and preprocess the image
    def preprocess_image(image_path, transform):
        image = Image.open(image_path).convert("RGB")
        return image, transform(image).unsqueeze(0)


    # Predict using the model
    def predict(model, image_tensor, device):
        model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities.cpu().numpy().flatten()


    # Visualization
    def visualize_predictions(original_image, probabilities, class_names):
        fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

        # Display image
        axarr[0].imshow(original_image)
        axarr[0].axis("off")

        # Display predictions
        axarr[1].barh(class_names, probabilities)
        axarr[1].set_xlabel("Probability")
        axarr[1].set_title("Class Predictions")
        axarr[1].set_xlim(0, 1)

        plt.tight_layout()
        plt.show()


    # Example usage
    if not os.path.isfile(arguments.test):
        print(f"Test file {arguments.test} does not exist")
        sys.exit(-1)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    original_image, image_tensor = preprocess_image(arguments.test, transform)
    probabilities = predict(model, image_tensor, device)

    # Assuming dataset.classes gives the class names
    class_names = dataset.classes
    visualize_predictions(original_image, probabilities, class_names)