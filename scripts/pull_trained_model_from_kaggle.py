import os
import time
import logging
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get model filename from .env or default to 'resnet101_model.pth'
model_filename = os.getenv('MODEL_FILENAME', 'resnet101_model.pth')

# Define file to store the last run kernel name
last_run_file = 'last_run.txt'

def get_last_run_kernel_name():
    """
    Reads the last run kernel name from the file.
    """
    if os.path.exists(last_run_file):
        with open(last_run_file, 'r') as f:
            return f.read().strip()
    else:
        logging.error(f"File '{last_run_file}' does not exist. Please ensure the kernel name is saved.")
        exit(1)

# Get the last run kernel name
kernel_name = get_last_run_kernel_name()

# Initialize Kaggle API
logging.info("Authenticating Kaggle API...")
api = KaggleApi()
api.authenticate()

# Create directory to store the downloaded model in the correct 'models' folder
output_dir = os.path.join(os.path.dirname(__file__), '..', 'models')  # Correctly pointing to the 'models' directory
os.makedirs(output_dir, exist_ok=True)

def check_for_model():
    logging.info(f"Checking for model '{model_filename}' in kernel '{kernel_name}' outputs on Kaggle...")

    while True:
        try:
            # Download the kernel output files to the 'models' directory
            logging.info(f"Fetching output files for kernel '{kernel_name}'...")
            api.kernels_output(kernel_name, path=output_dir)

            # Check if the model file exists in the output directory
            model_file_path = os.path.join(output_dir, model_filename)
            if os.path.exists(model_file_path):
                logging.info(f"Model '{model_filename}' downloaded successfully to '{model_file_path}'.")
                break
            else:
                logging.warning(f"Model '{model_filename}' not found yet. Retrying in 1 minute...")

        except Exception as e:
            logging.error(f"Unexpected error: {e}")

        time.sleep(60)  # Wait for 1 minute before checking again

# Start checking for the trained model file
check_for_model()
