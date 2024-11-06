# model/model.py
import torch
import torch.nn as nn
from torchvision import models

def initiate_model(model_type='resnet101'):
    if model_type == 'resnet101':
        model = models.resnet101(weights=None)  # No pre-trained weights
    elif model_type == 'resnet50':
        model = models.resnet50(weights=None)  # No pre-trained weights
    else:
        model = models.resnet18(weights=None)  # No pre-trained weights

    model.fc = nn.Linear(model.fc.in_features, 2)  # For binary classification
    return model

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

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model