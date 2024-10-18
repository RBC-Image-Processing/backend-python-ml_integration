import nbformat as nbf
import os
import sys
import inspect
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure that the root project directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now import the required modules from different parts of the project
try:
    from preprocess.preprocess import get_dataloaders, PneumoniaDataset, dataset_insights
    from model.model import initiate_model, load_model, train_model, validate_model, save_model
    from train.train import process  # Importing process function
except ImportError as e:
    logging.error(f"Error importing modules: {e}")
    sys.exit(1)

def create_jupyter_notebook():
    logging.info("Starting notebook creation process.")

    # Ensure the output directory exists
    output_dir = "kaggle_integration/output"
    os.makedirs(output_dir, exist_ok=True)

    # Create a new notebook object
    nb = nbf.v4.new_notebook()

    # Add the kernel name explicitly
    nb['metadata']['kernelspec'] = {
        "name": "python3",  # Kernel name
        "language": "python",
        "display_name": "Python 3"
    }

    logging.info("Notebook metadata set with Python 3 kernel.")

    # Markdown content
    markdown_1 = """
    # Pneumonia Detection Using Machine Learning and Deep Learning

    This notebook contains the complete pipeline for training a deep learning model to classify DICOM images into 'NORMAL' and 'PNEUMONIA' categories.
    
    ## Steps:
    - Preprocessing DICOM images
    - Loading models (CNN, ViT)
    - Training and evaluating the models
    - Dataset Insights
    """

    # Code cell 1: Import necessary libraries
    code_1 = """
    # Import necessary libraries
    import os
    import pydicom
    import torch
    from torch.utils.data import Dataset, DataLoader
    import cv2
    import torch.optim as optim
    from sklearn.metrics import classification_report
    import torch.nn as nn
    import torchvision.models as models
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from transformers import ViTForImageClassification, ViTFeatureExtractor

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    num_features = 2
    """

    logging.info("Preparing to generate code cells from project modules.")

    try:
        # Code cell 2: Dataset creation class (PneumoniaDataset)
        dataset_class_code = inspect.getsource(PneumoniaDataset)

        # Code cell 3: Dataset and DataLoader creation (from preprocess module)
        dataset_code = inspect.getsource(get_dataloaders)

        # Code cell 4: Dataset insights function (from preprocess module)
        insights_code = inspect.getsource(dataset_insights)

        # Code cell 5: Dataset insights execution
        insights_exec = """
        # Get insights about the dataset
        dataset_insights("/kaggle/input/chest-xray-pneumonia/chest_xray")
        """

        # Code cell 6: Add explicit initiate_model definition in the notebook
        initiate_model_code = inspect.getsource(initiate_model)

        # Code cell 7: Full train_model function from model module
        train_model_code = inspect.getsource(train_model)

        # Code cell 8: Full validate_model function from model module
        validate_model_code = inspect.getsource(validate_model)

        # Code cell 9: Full save_model function from model module
        save_model_code = inspect.getsource(save_model)

        # Code cell 10: Full process function from train module (depends on train_model and validate_model)
        process_code = inspect.getsource(process)

        # Code cell 11: Training process with the explicit use of initiate_model and process
        training_code = """
        model = initiate_model('resnet101')

        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Load the dataset
        train_loader, test_loader, val_loader = get_dataloaders('/kaggle/input/chest-xray-pneumonia/chest_xray', ['NORMAL', 'PNEUMONIA'], batch_size)

        # Train the model
        process('train', model, 'resnet101', train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs)
        """

        # Code cell 12: Validation process (depends on the trained model)
        validate_code = inspect.getsource(validate_model)

    except Exception as e:
        logging.error(f"Error generating code cells: {e}")
        sys.exit(1)

    logging.info("Successfully generated code cells.")

    # Add markdown and code cells to the notebook
    nb['cells'] = [
        nbf.v4.new_markdown_cell(markdown_1),
        nbf.v4.new_code_cell(code_1),
        nbf.v4.new_code_cell(dataset_class_code),
        nbf.v4.new_code_cell(dataset_code),
        nbf.v4.new_code_cell(insights_code),
        nbf.v4.new_code_cell(insights_exec),
        nbf.v4.new_code_cell(initiate_model_code),  # Ensure initiate_model code is included in the notebook
        nbf.v4.new_code_cell(train_model_code),  # Ensure train_model is included
        nbf.v4.new_code_cell(validate_model_code),  # Ensure validate_model is included
        nbf.v4.new_code_cell(save_model_code),  # Ensure save_model is included
        nbf.v4.new_code_cell(process_code),  # Ensure process code is included before training
        nbf.v4.new_code_cell(training_code),  # Training code to execute model training
        nbf.v4.new_code_cell(validate_code)  # Validation code after training
    ]

    logging.info("Notebook cells populated.")

    # Save the notebook with the name 'notebook_v3.ipynb' in the 'output' folder
    output_notebook_path = os.path.join(output_dir, "notebook_v3.ipynb")
    try:
        with open(output_notebook_path, "w") as f:
            nbf.write(nb, f)
        logging.info(f"Notebook generated: {output_notebook_path}")
    except Exception as e:
        logging.error(f"Error saving the notebook: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_jupyter_notebook()
