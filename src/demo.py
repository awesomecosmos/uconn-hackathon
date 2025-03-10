import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from efficientnet_pytorch import EfficientNet

# Function to convert tensor to a NumPy array
def imshow(image):
    image = image / 2 + 0.5  # Unnormalize
    image = torch.clamp(image, 0, 1)  # Ensure it's in the valid range
    np_image = image.numpy()  # Convert to numpy array
    plt.imshow(np.transpose(np_image, (1, 2, 0)))  # Convert from CHW to HWC format
    plt.show()


# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for EfficientNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset using ImageFolder
data_dir = "../data/lfw-deepfunneled/"  # Path to your dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print("Classes:", dataset.classes)

# Load pre-trained EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')
num_classes = len(dataset.classes)
model._fc = nn.Linear(model._fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(model)  # Check model architecture

# Setting loss and optimization function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
print("Training complete!")

# Validation loop
model.eval()
correct, total = 0, 0

with torch.no_grad():
    # Visualize one batch of validation images
    data_iter = iter(val_loader)
    images, labels = next(data_iter)  # Get one batch of images and labels
    images, labels = images.to(device), labels.to(device)

    # Get model predictions
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Convert predictions to list
    predicted_labels = predicted.cpu().numpy()
    true_labels = labels.cpu().numpy()

    # Visualize the batch of images with predictions
    fig = plt.figure(figsize=(10, 10))
    for i in range(min(16, len(images))):  # Show up to 16 images in a grid
        ax = fig.add_subplot(4, 4, i+1)
        imshow(images[i].cpu())  # Show image
        ax.set_title(f"True: {dataset.classes[true_labels[i]]}\nPred: {dataset.classes[predicted_labels[i]]}")
        ax.axis('off')
    plt.savefig("../figs/model_viz.jpg", dpi=300)
    plt.show()

    # Calculate accuracy
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total:.2f}%")

def fgsm_attack(model, images, labels, epsilon=0.1):
    # Ensure that input images require gradients
    # images.requires_grad = True

    # Forward pass to get the model's output
    outputs = model(images)

    # Calculate the loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)

    # Zero all existing gradients
    model.zero_grad()

    # Backward pass to calculate gradients
    loss.backward()

    # Get the sign of the gradients to create the adversarial perturbation
    perturbation = epsilon * images.grad.sign()

    # Create the adversarial images
    adv_images = images + perturbation

    # Clamp the adversarial images to be within valid image range
    adv_images = torch.clamp(adv_images, 0, 1)  # Ensure values are in [0, 1] range

    return adv_images

# Ensure directory exists for saving figures
if not os.path.exists("../figs"):
    os.makedirs("../figs")

# Run the adversarial attack and test model accuracy
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:  # Evaluate on validation set
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial images using FGSM
        adv_images = fgsm_attack(model, images, labels, epsilon=1)
        
        # Get the model predictions for adversarial images
        outputs = model(adv_images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on adversarial examples: {100 * correct / total:.2f}%")
