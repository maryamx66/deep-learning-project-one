import torch.nn as nn
import torch.nn.functional as F


class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()

        # 1. Input Layer: Flattening 28x28 images (784 neurons)
        self.fc1 = nn.Linear(28 * 28, 128)

        # 2. Hidden Layer: Applying the "depth > width" principle (128 -> 64 neurons)
        self.fc2 = nn.Linear(128, 64)

        # 3. Output Layer: 10 classes for MNIST digits (0-9)
        self.fc3 = nn.Linear(64, 10)

        # Regularization: Dropout to prevent overfitting
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Flatten the input data (2D to 1D vector)
        x = x.view(-1, 28 * 28)

        # First hidden layer -> ReLU -> Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Second hidden layer -> ReLU -> Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Output layer (Logits only, no Softmax since CrossEntropyLoss handles it)
        x = self.fc3(x)
        return x