import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from model_arch_emir import DigitClassifier

import urllib.request
import ssl

#  To bypass server blocking and Mac SSL errors:
ssl._create_default_https_context = ssl._create_unverified_context
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


transform = transforms.Compose([
    transforms.ToTensor()  # 0-255 → 0-1 (min-max scaling)
])


full_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)


train_size = int(0.6 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
remaining = len(full_dataset) - train_size - val_size

train_subset, val_subset, _ = random_split(
    full_dataset,
    [train_size, val_size, remaining]
)


train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def rakamlari_goster(dataset):
    plt.figure(figsize=(12, 3))
    for i in range(10):
        image, label = dataset[i]
        image = image.squeeze().numpy()

        plt.subplot(1, 10, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(label)
        plt.axis('off')
    plt.show()

rakamlari_goster(full_dataset)

print("Train:", len(train_subset))
print("Validation:", len(val_subset))
print("Test:", len(test_dataset))

# --- INTEGRATION TEST ---
# Initialize Emir's model
model = DigitClassifier()

# Get a single batch of data from Esra's DataLoader
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Pass the images through the model (Forward Pass)
outputs = model(images)

# Check the dimensions
print("Input Shape (from Data Engineer):", images.shape)
print("Output Shape (from Model Architect):", outputs.shape)