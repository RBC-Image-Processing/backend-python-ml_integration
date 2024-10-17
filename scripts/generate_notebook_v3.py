import nbformat as nbf
import os
import sys
import inspect

# Ensure that the root project directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the required modules from different parts of the project
from preprocess.preprocess import get_dataloaders, PneumoniaDataset, dataset_insights
from model.model import initiate_model, load_model, train_model, validate_model
from train.train import process

def create_jupyter_notebook():
    # Ensure the output directory exists
    output_dir = "kaggle_integration/output"
    os.makedirs(output_dir, exist_ok=True)

    # Create a new notebook object
    nb = nbf.v4.new_notebook()

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
    learnin_rate = 0.001
    num_epochs = 10
    num_features = 2
    """

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

    # Code cell 6: Model initialization
    model_code = inspect.getsource(initiate_model)

    # Code cell 7: Training process
    train_code = inspect.getsource(train_model)

    # Code cell 8: Validation process
    validate_code = inspect.getsource(validate_model)

    # Code cell 9: Full process function
    process_code = inspect.getsource(process)

    # Add markdown and code cells to the notebook
    nb['cells'] = [
        nbf.v4.new_markdown_cell(markdown_1),
        nbf.v4.new_code_cell(code_1),
        nbf.v4.new_code_cell(dataset_class_code),
        nbf.v4.new_code_cell(dataset_code),
        nbf.v4.new_code_cell(insights_code),
        nbf.v4.new_code_cell(insights_exec),
        nbf.v4.new_code_cell(model_code),
        nbf.v4.new_code_cell(train_code),
        nbf.v4.new_code_cell(validate_code),
        nbf.v4.new_code_cell(process_code)
    ]

    # Save the notebook with the name 'notebook_v3.ipynb' in the 'output' folder
    output_notebook_path = os.path.join(output_dir, "notebook_v3.ipynb")
    with open(output_notebook_path, "w") as f:
        nbf.write(nb, f)

    print(f"Notebook generated: {output_notebook_path}")


if __name__ == "__main__":
    create_jupyter_notebook()
