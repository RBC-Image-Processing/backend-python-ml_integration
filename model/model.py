# model/model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Model initiation
def initiate_model(model_type='resnet18', path=None):
    if path:
        # Load model from pre-trained weights
        model = torch.load(path)
    else:
        # Initialize a new model
        if model_type == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif model_type == 'resnet50':
            model = models.resnet50(pretrained=True)
        else:
            model = models.resnet101(pretrained=True)

        # Modify the last fully connected layer to output 2 features (Pneumonia/Normal)
        model.fc = nn.Linear(model.fc.in_features, 2)

    return model

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
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

        validate_model(model, val_loader, criterion, device)

# Validation function
def validate_model(model, val_loader, criterion, device):
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
    torch.save(model, path)

# Load model
def load_model(path):
    model = torch.load(path)
    return model
