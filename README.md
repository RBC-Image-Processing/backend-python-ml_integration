# Project Name: Machine Learning Model Training and API Deployment

## Overview


This project is designed to train machine learning models using modular components for preprocessing, model creation, training, and integration with Kaggle for remote training. Additionally, the project exposes a REST API to serve the trained model for predictions, enabling external systems to send data for inference.

## Directory Structure

        .
        ├── README.md
        ├── __init__.py
        ├── api
        │   ├── __init__.py
        │   ├── api_endpoints.py
        │   ├── dependencies.py
        │   ├── error_handlers.py
        │   ├── models
        │   │   ├── __init__.py
        │   │   └── prediction.py
        │   └── utils
        │       ├── __init__.py
        │       └── image_processing.py
        ├── config.py
        ├── kaggle_integration
        │   ├── __init__.py
        │   └── output
        │       ├── kernel-metadata.json
        │       ├── notebook_v3.ipynb
        │       └── output
        │           └── kernel-metadata.json
        ├── last_run.txt
        ├── main.py
        ├── model
        │   ├── __init__.py
        │   ├── __pycache__
        │   │   ├── __init__.cpython-311.pyc
        │   │   └── model.cpython-311.pyc
        │   └── model.py
        ├── models
        │   └── __init__.py
        ├── output
        │   └── kernel-metadata.json
        ├── preprocess
        │   ├── __init__.py
        │   ├── __pycache__
        │   │   ├── __init__.cpython-311.pyc
        │   │   └── preprocess.cpython-311.pyc
        │   └── preprocess.py
        ├── requirements.txt
        ├── scripts
        │   ├── __init__.py
        │   ├── generators
        │   │   ├── __init__.py
        │   │   ├── generate_notebook_v1.py
        │   │   ├── generate_notebook_v2.py
        │   │   └── generate_notebook_v3.py
        │   ├── pull_trained_model_from_kaggle.py
        │   └── send_notebook_to_kaggle.py
        ├── test_main.http
        └── train
            ├── __init__.py
            ├── __pycache__
            │   ├── __init__.cpython-311.pyc
            │   └── train.cpython-311.pyc
            └── train.py

        17 directories, 40 files


## Modules

### 1. `api/`
- **Purpose**: Contains REST API code, exposing the trained model via endpoints for prediction.
- **Key Features**:
  - Endpoints for model predictions.
  - Integration with model inference logic.

### 2. `config.py`
- **Purpose**: Centralized configuration settings for the project.
- **Key Features**:
  - Manages environment variables and global settings.

### 3. `kaggle_integration/`
- **Purpose**: Manages integration with Kaggle for training.
- **Key Features**:
  - Pushes training code to Kaggle.
  - Retrieves trained models from Kaggle outputs.
  - `last_run.txt` stores the last executed kernel's name for tracking.

### 4. `last_run.txt`
- **Purpose**: Tracks the name of the last Kaggle kernel executed.

### 5. `main.py`
- **Purpose**: Entry point for running the machine learning pipeline.
- **Key Features**:
  - Coordinates preprocessing, model training, and API serving.

### 6. `model/`
- **Purpose**: Defines model architecture (e.g., ResNet).
- **Key Features**:
  - Model creation with random weight initialization.
  - Save and load model functions.

### 7. `models/`
- **Purpose**: Directory to store trained models.
- **Key Features**:
  - Stores `.pth` files for model inference.

### 8. `preprocess/`
- **Purpose**: Contains data preprocessing functions.
- **Key Features**:
  - Data normalization, augmentation, and loader functions.

### 9. `scripts/`
- **Purpose**: Scripts for Kaggle integration and training.
- **Key Features**:
  - `send_notebook_to_kaggle.py`: Sends notebook to Kaggle.
  - `pull_trained_model_from_kaggle.py`: Retrieves the trained model from Kaggle.

### 10. `test_main.http`
- **Purpose**: HTTP testing script for API endpoints.

### 11. `train/`
- **Purpose**: Handles model training logic.
- **Key Features**:
  - Training loop, evaluation, and checkpointing functions.

## New Features and Updates

1. **Kaggle Integration for Remote Training**:
   - Automates notebook pushing to Kaggle for training.
   - Retrieves trained models from Kaggle.
   - Tracks kernel status and stores kernel names in `last_run.txt`.

2. **Model Architecture**:
   - Models initialized with random weights (ResNet101, ResNet50).
   - Flexible architecture for different training runs.

3. **Notebook Generation**:
   - Automatically generates notebooks for Kaggle training.
   - Notebook generator scripts are available in `scripts/generators`.

## Getting Started

### Prerequisites
- Python 3.7+
- Kaggle API
- Flask or FastAPI
- PyTorch

### Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt


1.  Set up Kaggle API credentials:
    -   Create a `kaggle.json` with your API credentials and place it in `~/.kaggle/`.

### Usage

1.  **Train on Kaggle**:

    -   Generate a notebook using `generate_notebook_v*.py`.
    -   Push the notebook using `send_notebook_to_kaggle.py`.
    -   Track the kernel and download the model using `pull_trained_model_from_kaggle.py`.
2.  **Serve API**:

    -   Start the API using FastAPI or Flask.
    -   Use `test_main.http` for testing API predictions.