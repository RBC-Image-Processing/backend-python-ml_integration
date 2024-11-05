# automate_training.py
import os
import sys
import torch
import logging

# Set the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)  # Add the project root directory to sys.path

# Now import the necessary modules
from preprocess.preprocess import get_dataloaders
from model.model import initiate_model, save_model
from train.train import process
from scripts.download_dataset import main as download_datasets

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directories and Parameters
INPUT_DIR = os.path.join(os.getcwd(), 'input', 'data')
RSNA_DIR = os.path.join(INPUT_DIR, 'rsna-pneumonia')
CHEST_XRAY_DIR = os.path.join(INPUT_DIR, 'chest-xray')
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MODEL_TYPE = 'resnet50'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def main():
    # Step 1: Download datasets (if not already downloaded)
    if not os.listdir(RSNA_DIR) or not os.listdir(CHEST_XRAY_DIR):
        logger.info("Downloading datasets...")
        download_datasets()
    else:
        logger.info("Datasets found, skipping download.")

    # Step 2: Prepare data loaders for both datasets
    logger.info("Preparing data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(INPUT_DIR, BATCH_SIZE)

    # Step 3: Initialize model
    logger.info(f"Initializing model {MODEL_TYPE}...")
    model = initiate_model(MODEL_TYPE)
    model = model.to(device)

    # Step 4: Set up loss function and optimizer
    logger.info("Setting up loss function and optimizer...")
    class_counts = [sum(1 for _, label in train_loader.dataset if label == i) for i in range(2)]
    weights = torch.tensor([len(train_loader.dataset) / c for c in class_counts], dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Step 5: Train the model
    logger.info("Starting training process...")
    process(
        action='train',
        model=model,
        model_type=MODEL_TYPE,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS
    )

    # Step 6: Save the trained model
    model_save_path = os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_model.pth")
    save_model(model, model_save_path)
    logger.info(f"Model saved successfully to {model_save_path}")

if __name__ == "__main__":
    main()
