# model/model.py
import torch
import torch.nn as nn
from torchvision import models

# Model initiation
# model/model.py
import torch
import torch.nn as nn
from torchvision import models

# Model initiation
from torchvision import models
import torch.nn as nn

def initiate_model(model_type='resnet101'):
    """
    Initialize a model with random weights.

    Parameters:
    - model_type: str, type of model to initialize (e.g., 'resnet101')
    
    Returns:
    - model: Initialized model.
    """
    if model_type == 'resnet101':
        model = models.resnet101(weights=None)  # No pre-trained weights
    elif model_type == 'resnet50':
        model = models.resnet50(weights=None)  # No pre-trained weights
    else:
        model = models.resnet18(weights=None)  # No pre-trained weights

    # Modify the last fully connected layer to output 2 features (Pneumonia/Normal)
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Trains the model using the provided data and optimizes its parameters.

    Parameters:
    - model: The PyTorch model to train.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - criterion: Loss function.
    - optimizer: Optimizer for model parameters.
    - num_epochs: Number of epochs for training.
    - device: Device to run the model on ('cuda' or 'cpu').
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * imgs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        
        # Validation at the end of each epoch
        validate_model(model, val_loader, criterion, device)

# Validation function
def validate_model(model, val_loader, criterion, device):
    """
    Validates the model on the validation set.
    
    Parameters:
    - model: PyTorch model.
    - val_loader: DataLoader for validation data.
    - criterion: Loss function.
    - device: Device to run the model on ('cuda' or 'cpu').
    """
    model.eval()
    running_loss, running_corrects = 0.0, 0.0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * imgs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects / len(val_loader.dataset)

    print(f'Validation Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

# Save model
def save_model(model, path):
    """
    Saves the model to the specified path.
    
    Parameters:
    - model: PyTorch model.
    - path: str, path where the model should be saved.
    """
    torch.save(model.state_dict(), path)

# Load model
def load_model(model, path):
    """
    Loads a model's state dictionary from the specified path.
    
    Parameters:
    - model: PyTorch model structure.
    - path: str, path to the saved model.
    
    Returns:
    - model: PyTorch model with the loaded state dictionary.
    """
    model.load_state_dict(torch.load(path))
    return model
