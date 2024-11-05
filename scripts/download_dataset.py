# download_dataset.py
import os
import subprocess
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory paths
INPUT_DIR = os.path.join(os.getcwd(), 'input')
DATA_DIR = os.path.join(INPUT_DIR, 'data')

# Ensure the directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# Function to download dataset using Kaggle API
def download_kaggle_dataset(dataset_name, download_path, is_competition=True):
    """
    Downloads a Kaggle dataset or competition and extracts it to a specific path.

    Parameters:
    - dataset_name (str): The name of the Kaggle dataset or competition.
    - download_path (str): The path where the dataset should be downloaded.
    - is_competition (bool): If True, downloads a competition dataset; otherwise, a general dataset.
    """
    logger.info(f"Starting download for dataset '{dataset_name}' into {download_path}...")

    # Determine command based on dataset type
    if is_competition:
        command = ['kaggle', 'competitions', 'download', '-c', dataset_name, '-p', download_path]
    else:
        command = ['kaggle', 'datasets', 'download', dataset_name, '-p', download_path]
    
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to download dataset '{dataset_name}': {result.stderr}")
        return
    
    # Extract zip files
    for item in os.listdir(download_path):
        if item.endswith('.zip'):
            zip_path = os.path.join(download_path, item)
            logger.info(f"Extracting {zip_path}...")
            subprocess.run(['unzip', '-o', zip_path, '-d', download_path])
            os.remove(zip_path)  # Clean up the zip file after extraction
    
    # Handle nested directory structure for 'chest-xray-pneumonia' dataset
    if dataset_name == "paultimothymooney/chest-xray-pneumonia":
        nested_path = os.path.join(download_path, 'chest_xray')
        if os.path.isdir(nested_path):
            for item in os.listdir(nested_path):
                item_path = os.path.join(nested_path, item)
                if os.path.isdir(item_path) and item in ["train", "test", "val"]:
                    target_path = os.path.join(download_path, item)
                    shutil.move(item_path, target_path)  # Move 'train', 'test', 'val' to main directory
            shutil.rmtree(nested_path)  # Remove the now-empty nested folder
            logger.info(f"Reorganized nested folders for dataset '{dataset_name}'.")

    logger.info(f"Dataset '{dataset_name}' downloaded and extracted successfully.")

def main():
    # Ensure the Kaggle API key is configured
    kaggle_config_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_config_path):
        logger.error("Kaggle API credentials not found. Please place 'kaggle.json' in the ~/.kaggle directory.")
        return

    # Download the RSNA Pneumonia Detection Challenge dataset
    rsna_path = os.path.join(DATA_DIR, 'rsna-pneumonia')
    os.makedirs(rsna_path, exist_ok=True)
    download_kaggle_dataset("rsna-pneumonia-detection-challenge", rsna_path, is_competition=True)

    # Download the Chest X-Ray Images (Pneumonia) dataset
    chest_xray_path = os.path.join(DATA_DIR, 'chest-xray')
    os.makedirs(chest_xray_path, exist_ok=True)
    download_kaggle_dataset("paultimothymooney/chest-xray-pneumonia", chest_xray_path, is_competition=False)

if __name__ == "__main__":
    main()
