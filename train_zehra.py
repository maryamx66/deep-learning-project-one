import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

# Downloading Emir's and Esra's files first:
from data_loader_esra import train_loader, val_loader  # Esra's Loader
from model_arch_emir import DigitClassifier           # Emir's Model

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitClassifier().to(device)

# Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# ADDED: Learning Rate Scheduler (Step Decay)
# Drops the learning rate by a factor of 0.1 every 10 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Model Selection and Early Stopping Variables
best_val_loss = float('inf') 
best_model_wts = copy.deepcopy(model.state_dict())
patience = 5  # Stop learning after 5 epoch if there's no improvement 
counter = 0
train_losses, val_losses = [], []

print(f" Training starts on the {device}...")

# Training
for epoch in range(30): # Max 30 
    #(TRAIN)
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # ADDED: Gradient Clipping
        # Rescales the gradient to prevent exploding updates before stepping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item()
    
    # Validation
    model.eval()
    v_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            v_loss += criterion(model(images), labels).item()
            
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = v_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # ADDED: Step the learning rate scheduler at the end of each epoch
    scheduler.step()

    #  Model Selection and Early Stopping 
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_wts = copy.deepcopy(model.state_dict()) # Saving the best to memory
        counter = 0
        torch.save(best_model_wts, "best_model_zehra.pth")
        print("New best model weights have been recorded!")
    else:
        counter += 1
        if counter >= patience:
            print(f" Early Stopping! The model is not evolving any further.")
            break

# Performance Graph
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='training error')
plt.plot(val_losses, label='validation error')
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.title('Zehra - Model Training Process (Optimized)')
plt.legend()
plt.show()

print(" The process is complete. The best model is ready under the name 'best_model_zehra.pth'.")