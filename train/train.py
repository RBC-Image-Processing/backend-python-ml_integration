# train/train.py
import torch
from model.model import train_model, validate_model, save_model

def process(action, model, model_type, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs):
    """
    Handles the training and evaluation processes based on the action parameter.

    Parameters:
    - action: str, 'train' or 'evaluate'.
    - model: PyTorch model to be trained or evaluated.
    - model_type: str, type of model (e.g., 'resnet101').
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - test_loader: DataLoader for test data.
    - criterion: Loss function.
    - optimizer: Optimizer for model parameters.
    - device: Device to run the model on ('cuda' or 'cpu').
    - num_epochs: Number of epochs for training.
    """
    if action == 'train':
        # Start the training process
        print(f"Starting training for {model_type}...")
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
        
        # Save the model after training
        model_save_path = f"/kaggle/working/{model_type}_model.pth"
        print(f"Training completed. Saving model to {model_save_path}...")
        save_model(model, model_save_path)
        print(f"Model saved successfully to {model_save_path}.")

    elif action == 'evaluate':
        # Evaluate the model on the test data
        print(f"Evaluating {model_type} on the test set...")
        validate_model(model, test_loader, criterion, device)
        print(f"Evaluation completed for {model_type}.")
