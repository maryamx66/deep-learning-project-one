import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

device = torch.device('cpu')

# UPDATED: Architecture must match Emir's new Optimized model
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.bn1 = nn.BatchNorm1d(128) # Added Batch Normalization
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)  # Added Batch Normalization
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        
        x = self.fc1(x)
        x = self.bn1(x)                # Added BatchNorm forward pass
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)                # Added BatchNorm forward pass
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

model = DigitClassifier()
try:
    state_dict = torch.load('best_model_zehra.pth', map_location=device)
    model.load_state_dict(state_dict, strict=False)
    # model.eval() is already here, which is crucial because it tells BatchNorm 
    # and Dropout to behave in "testing" mode rather than "training" mode.
    model.eval()
    print('[Model] Loaded successfully.')
except Exception as e:
    print(f'[Error] {e}')
    exit()

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

all_preds, all_labels, all_images = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        _, predicted = torch.max(model(images), 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())
        all_images.extend(images)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
accuracy = (all_preds == all_labels).sum() / len(all_labels) * 100
print(f'[Result] Accuracy: {accuracy:.2f}%')

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print('[Saved] confusion_matrix.png')

print(classification_report(all_labels, all_preds, target_names=[f'Digit {i}' for i in range(10)], digits=4))

wrong_idx = np.where(all_preds != all_labels)[0]
print(f'[Errors] {len(wrong_idx)} misclassified')
fig, axes = plt.subplots(4, 4, figsize=(10,10))
fig.suptitle('Misclassified Images', fontsize=14, fontweight='bold')
for i, ax in enumerate(axes.flat):
    if i >= min(16, len(wrong_idx)):
        ax.axis('off')
        continue
    idx = wrong_idx[i]
    ax.imshow(all_images[idx].squeeze().numpy(), cmap='gray')
    ax.set_title(f'True:{all_labels[idx]}  Pred:{all_preds[idx]}', color='red', fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.savefig('misclassified_samples.png', dpi=150)
plt.close()
print('[Saved] misclassified_samples.png')