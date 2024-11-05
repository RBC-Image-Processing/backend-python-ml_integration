import nbformat as nbf
import os
import sys
import inspect

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import your modules
from preprocess.preprocess import process_dicom_with_clahe
from model.model import load_model

def create_jupyter_notebook():
    # Ensure the output directory exists
    output_dir = "kaggle_integration/output"
    os.makedirs(output_dir, exist_ok=True)

    # Create a new notebook object
    nb = nbf.v4.new_notebook()

    # Markdown content
    markdown_1 = """
    # Pneumonia Detection Using Machine Learning and Deep Learning

    This notebook contains the pipeline for training a deep learning model to classify DICOM images into 'NORMAL' and 'PNEUMONIA' categories.
    
    ## Steps:
    - Preprocessing DICOM images
    - Loading models (CNN, ViT)
    - Training and evaluating the models
    """

    # Code cell 1: Import necessary libraries
    code_1 = """
    # import os
        import pydicom
        import torch
        from torch.utils.data import Dataset
        import cv2
        from torch.utils.data import DataLoader
        import torch.optim as optim
        from sklearn.metrics import classification_report
        from torch.utils.data import DataLoader
        import torch
        import torch.nn as nn
        import torchvision.models as models
        from sklearn.metrics import classification_report
        from PIL import Image
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        import numpy as np
        import matplotlib.pyplot as plt
        from transformers import ViTForImageClassification
        from transformers import ViTFeatureExtractor
        batch_size = 32
        learnin_rate= 0.001
        num_epochs=10
        num_features=2
    """

    # Dynamically extract the function code for `process_dicom_with_clahe`
    preprocess_code = inspect.getsource(process_dicom_with_clahe)

    # Dynamically extract the function code for `load_model`
    model_code = inspect.getsource(load_model)

    # Code cell 2: Preprocessing code extracted from the module
    code_2 = f"""
    # Preprocessing function for DICOM images
    {preprocess_code}

    # Example usage
    dicom_data = open('path_to_dicom.dcm', 'rb').read()  # Replace with the path to your DICOM file
    image_tensor = process_dicom_with_clahe(dicom_data)
    print(image_tensor.shape)
    """

    # Code cell 3: Model loading code extracted from the module
    code_3 = f"""
    # Model loading function
    {model_code}

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './models/cnn_impr.pth'  # Replace with your model path
    model = load_model(model_path, device)

    # Example forward pass
    if model:
        with torch.no_grad():
            output = model(image_tensor)
            print(output)
    """

    # Add markdown and code cells to the notebook
    nb['cells'] = [
        nbf.v4.new_markdown_cell(markdown_1),
        nbf.v4.new_code_cell(code_1),
        nbf.v4.new_code_cell(code_2),
        nbf.v4.new_code_cell(code_3)
    ]

    # Save the notebook with the name 'notebook_v2.ipynb' in the 'output' folder
    output_notebook_path = os.path.join(output_dir, "notebook_v2.ipynb")
    with open(output_notebook_path, "w") as f:
        nbf.write(nb, f)

    print(f"Notebook generated: {output_notebook_path}")


if __name__ == "__main__":
    create_jupyter_notebook()
