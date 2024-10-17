import nbformat as nbf
import os

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

    # Code snippets
    code_1 = """
    # Import necessary libraries
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    import pydicom
    import cv2
    import io
    """

    code_2 = """
    # Define a preprocessing function for DICOM images
    def process_dicom_with_clahe(dicom_data):
        dicom_file = pydicom.dcmread(io.BytesIO(dicom_data))
        pixel_array = dicom_file.pixel_array.astype(np.uint8)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed_image = clahe.apply(pixel_array)
        
        # Convert to PIL Image and resize
        image = Image.fromarray(processed_image).convert('RGB')
        image = image.resize((224, 224))
        
        # Convert to PyTorch tensor
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return image_tensor
    """

    code_3 = """
    # Model loading function
    def load_model(path, device):
        try:
            model = torch.load(path, map_location=device)
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
    """

    # Add markdown and code cells to the notebook
    nb['cells'] = [
        nbf.v4.new_markdown_cell(markdown_1),
        nbf.v4.new_code_cell(code_1),
        nbf.v4.new_code_cell(code_2),
        nbf.v4.new_code_cell(code_3)
    ]

    # Save the notebook with a new name in the 'output' folder
    output_notebook_path = os.path.join(output_dir, "notebook_v1.ipynb")
    with open(output_notebook_path, "w") as f:
        nbf.write(nb, f)

    print(f"Notebook generated: {output_notebook_path}")


if __name__ == "__main__":
    create_jupyter_notebook()
