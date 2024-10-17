# train/train.py
import torch
from model.model import train_model, validate_model

def process(action, model, model_type, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs):
    if action == 'train':
        # Start the training process
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
        torch.save(model, "/kaggle/working/res_net_model.pth")

    elif action == 'evaluate':
        # Evaluate the model
        validate_model(model, test_loader, criterion, device)
