from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations (resize + convert to tensor)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),  # Convert to PyTorch tensor
])

# Load dataset
lfw_dataset = datasets.ImageFolder(root="../data/", transform=transform)

# Create a DataLoader
lfw_loader = DataLoader(lfw_dataset, batch_size=32, shuffle=True)

# Check some samples
for images, labels in lfw_loader:
    print(f"Batch shape: {images.shape}")  # (batch_size, channels, height, width)
    print(f"Labels: {labels[:5]}")
    break


# import fiftyone as fo
# import fiftyone.zoo as foz
# dataset = fo.Dataset.from_images_dir("../data/")
# session = fo.launch_app(dataset, port=5151)