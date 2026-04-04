import torch.nn as nn
import torch.nn.functional as F

class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()

        # 1. Input Layer: Flattening 28x28 images (784 neurons)
        self.fc1 = nn.Linear(28 * 28, 128)
        self.bn1 = nn.BatchNorm1d(128) # Added Batch Normalization

        # 2. Hidden Layer: Applying the "depth > width" principle (128 -> 64 neurons)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)  # Added Batch Normalization

        # 3. Output Layer: 10 classes for MNIST digits (0-9)
        self.fc3 = nn.Linear(64, 10)

        # Regularization: Dropout to prevent overfitting
        self.dropout = nn.Dropout(p=0.5)
        
        # Apply He (Kaiming) Initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He/Kaiming init tailored for ReLU activations
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Flatten the input data (2D to 1D vector)
        x = x.view(-1, 28 * 28)

        # First hidden layer -> BatchNorm -> ReLU -> Dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Second hidden layer -> BatchNorm -> ReLU -> Dropout
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output layer (Logits only)
        x = self.fc3(x)
        return x