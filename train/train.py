# train/train.py
import torch
import logging
from model.model import initiate_model, save_model, validate_model

# Set up logging for train
logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Trains the model and provides metrics after each epoch.

    Parameters:
    - model: The model to be trained.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - criterion: Loss function.
    - optimizer: Optimizer.
    - num_epochs: Number of epochs.
    - device: Device to train on ('cuda' or 'cpu').
    """
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        logger.info(f"Epoch {epoch + 1}/{num_epochs} started.")
        
        # Training phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

            # Log batch metrics
            logger.info(f"Batch Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        # Validation phase
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, f"./output/best_{model.__class__.__name__}_model.pth")
            logger.info(f"Best model saved with validation accuracy: {best_val_acc:.4f}")

def validate_model(model, loader, criterion, device):
    """
    Evaluates the model on a given dataset.

    Parameters:
    - model: The model to evaluate.
    - loader: DataLoader for validation or test data.
    - criterion: Loss function.
    - device: Device to evaluate on ('cuda' or 'cpu').

    Returns:
    - Average loss and accuracy on the dataset.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects / len(loader.dataset)
    return epoch_loss, epoch_acc

def process(action, model, model_type, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs):
    if action == 'train':
        logger.info(f"Starting training for {model_type}...")
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
        model_save_path = f"./output/{model_type}_model.pth"
        save_model(model, model_save_path)
        logger.info(f"Model saved successfully to {model_save_path}.")
    elif action == 'evaluate':
        logger.info(f"Evaluating {model_type} on the test set...")
        test_loss, test_acc = validate_model(model, test_loader, criterion, device)
        logger.info(f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

# ======================

# # train/train.py
# import torch
# from model.model import train_model, validate_model, save_model

# def process(action, model, model_type, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs):
#     """
#     Handles the training and evaluation processes based on the action parameter.

#     Parameters:
#     - action: str, 'train' or 'evaluate'.
#     - model: PyTorch model to be trained or evaluated.
#     - model_type: str, type of model (e.g., 'resnet101').
#     - train_loader: DataLoader for training data.
#     - val_loader: DataLoader for validation data.
#     - test_loader: DataLoader for test data.
#     - criterion: Loss function.
#     - optimizer: Optimizer for model parameters.
#     - device: Device to run the model on ('cuda' or 'cpu').
#     - num_epochs: Number of epochs for training.
#     """
#     if action == 'train':
#         # Start the training process
#         print(f"Starting training for {model_type}...")
#         train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
        
#         # Save the model after training
#         model_save_path = f"/kaggle/working/{model_type}_model.pth"
#         print(f"Training completed. Saving model to {model_save_path}...")
#         save_model(model, model_save_path)
#         print(f"Model saved successfully to {model_save_path}.")

#     elif action == 'evaluate':
#         # Evaluate the model on the test data
#         print(f"Evaluating {model_type} on the test set...")
#         validate_model(model, test_loader, criterion, device)
#         print(f"Evaluation completed for {model_type}.")
