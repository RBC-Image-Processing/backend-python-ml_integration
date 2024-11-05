# import os
# import json
# import subprocess
# import logging
# import time
# from dotenv import load_dotenv
# from datetime import datetime

# # Load environment variables from the .env file
# load_dotenv()

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Get variables from .env
# kaggle_username = os.getenv('KAGGLE_USERNAME')
# kaggle_key = os.getenv('KAGGLE_KEY')
# notebook_path = os.getenv('NOTEBOOK_PATH')
# kernel_base_name = os.getenv('KAGGLE_KERNEL_NAME')  # Base name for the kernel
# dataset_path = os.getenv('KAGGLE_DATASET_PATH')
# machine_type = os.getenv('MACHINE_TYPE', 'GPU')  # Default to GPU
# polling_interval = int(os.getenv('POLLING_INTERVAL', 60))  # Default to 60 seconds

# # Generate a unique timestamp for the kernel name
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# unique_kernel_name = f"{kernel_base_name}_{timestamp}"  # Unique kernel name

# # Validate that critical environment variables are set
# if not all([kaggle_username, kaggle_key, notebook_path, unique_kernel_name, dataset_path]):
#     logging.error("One or more required environment variables are missing. Please check your .env file.")
#     exit(1)

# # Ensure notebook path is valid
# if not os.path.exists(notebook_path):
#     logging.error(f"Notebook path '{notebook_path}' does not exist. Please check the path.")
#     exit(1)

# # Setup Kaggle API credentials
# kaggle_dir = os.path.expanduser('~/.kaggle')
# os.makedirs(kaggle_dir, exist_ok=True)
# kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')

# with open(kaggle_json_path, 'w') as f:
#     f.write(f'{{"username":"{kaggle_username}", "key":"{kaggle_key}"}}')

# os.chmod(kaggle_json_path, 0o600)
# logging.info(f"Kaggle API credentials created at {kaggle_json_path}")

# # Create metadata for the Kaggle kernel
# metadata = {
#     "id": f"{kaggle_username}/{unique_kernel_name}",
#     "title": f"{kernel_base_name} {timestamp}",
#     "code_file": os.path.basename(notebook_path),
#     "language": "python",
#     "kernel_type": "notebook",
#     "is_private": "true",
#     "enable_gpu": machine_type == "GPU",
#     "enable_tpu": machine_type == "TPU",
#     "dataset_sources": [dataset_path],  # Link to the dataset
#     "kernel_sources": [],
#     "accelerator": machine_type,  # Use GPU or TPU based on the setting
# }

# # Write the kernel metadata to a JSON file
# kernel_metadata_dir = os.path.dirname(notebook_path)
# kernel_metadata_file = os.path.join(kernel_metadata_dir, 'kernel-metadata.json')
# os.makedirs(kernel_metadata_dir, exist_ok=True)

# with open(kernel_metadata_file, 'w') as f:
#     json.dump(metadata, f, indent=4)

# logging.info(f"Kernel metadata file created at {kernel_metadata_file}")

# # Save the kernel name to last_run.txt for future use
# last_run_file = 'last_run.txt'
# with open(last_run_file, 'w') as f:
#     f.write(f'{kaggle_username}/{unique_kernel_name}')

# logging.info(f"Kernel name '{unique_kernel_name}' saved to {last_run_file}")

# try:
#     # Push the notebook to Kaggle
#     logging.info(f"Pushing notebook '{notebook_path}' to Kaggle as kernel '{unique_kernel_name}'...")
#     result = subprocess.run(
#         ['kaggle', 'kernels', 'push', '-p', os.path.dirname(notebook_path)],
#         capture_output=True,
#         text=True
#     )

#     if result.returncode != 0:
#         logging.error(f"Failed to push notebook to Kaggle: {result.stderr}")
#         logging.info(f"Stdout: {result.stdout}")
#         exit(1)
#     else:
#         logging.info(f"Notebook {notebook_path} has been successfully pushed to Kaggle as '{unique_kernel_name}'.")

#     # Optionally start the training (depending on your workflow)
#     logging.info(f"Starting training on Kaggle for kernel '{unique_kernel_name}'...")

#     while True:
#         result_status = subprocess.run(
#             ['kaggle', 'kernels', 'status', f'{kaggle_username}/{unique_kernel_name}'],
#             capture_output=True,
#             text=True
#         )

#         if result_status.returncode != 0:
#             if "403 - Forbidden" in result_status.stderr:
#                 logging.error(f"Permission denied: Ensure your API key has permission to retrieve kernel status.")
#                 exit(1)
#             else:
#                 logging.error(f"Failed to check kernel status: {result_status.stderr}")
#                 logging.info(f"Stdout: {result_status.stdout}")
#         else:
#             logging.info(f"Kernel '{unique_kernel_name}' status: {result_status.stdout}")

#         # Check if the kernel is done (you could add logic here to break the loop)
#         if 'complete' in result_status.stdout.lower():
#             logging.info(f"Kernel '{unique_kernel_name}' has completed.")
#             break
#         elif 'error' in result_status.stdout.lower():
#             logging.error(f"Kernel '{unique_kernel_name}' encountered an error.")
#             break

#         # Wait for the next check
#         logging.info(f"Waiting for {polling_interval} seconds before checking status again...")
#         time.sleep(polling_interval)

# except Exception as e:
#     logging.error(f"An error occurred: {e}")
import os
import json
import subprocess
import logging
import time
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from the .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get variables from .env
kaggle_username = os.getenv('KAGGLE_USERNAME')
kaggle_key = os.getenv('KAGGLE_KEY')
notebook_path = os.getenv('NOTEBOOK_PATH')
kernel_base_name = os.getenv('KAGGLE_KERNEL_NAME')  # Base name for the kernel
dataset_path = os.getenv('KAGGLE_DATASET_PATH')
machine_type = os.getenv('MACHINE_TYPE', 'GPU')  # Default to GPU
output_model_filename = os.getenv('OUTPUT_MODEL_FILENAME', 'resnet101_model.pth')  # Example model output file

# Set polling interval to 2 minutes
polling_interval = 120  # 120 seconds (2 minutes)

# Generate a unique timestamp for the kernel name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
unique_kernel_name = f"{kernel_base_name}_{timestamp}"  # Unique kernel name

# Validate that critical environment variables are set
if not all([kaggle_username, kaggle_key, notebook_path, unique_kernel_name, dataset_path]):
    logging.error("One or more required environment variables are missing. Please check your .env file.")
    exit(1)

# Ensure notebook path is valid
if not os.path.exists(notebook_path):
    logging.error(f"Notebook path '{notebook_path}' does not exist. Please check the path.")
    exit(1)

# Setup Kaggle API credentials
kaggle_dir = os.path.expanduser('~/.kaggle')
os.makedirs(kaggle_dir, exist_ok=True)
kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')

with open(kaggle_json_path, 'w') as f:
    f.write(f'{{"username":"{kaggle_username}", "key":"{kaggle_key}"}}')

os.chmod(kaggle_json_path, 0o600)
logging.info(f"Kaggle API credentials created at {kaggle_json_path}")

# Create metadata for the Kaggle kernel
metadata = {
    "id": f"{kaggle_username}/{unique_kernel_name}",
    "title": f"{kernel_base_name} {timestamp}",
    "code_file": os.path.basename(notebook_path),
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": machine_type == "GPU",
    "enable_tpu": machine_type == "TPU",
    "dataset_sources": [dataset_path],  # Link to the dataset
    "kernel_sources": [],
    "accelerator": machine_type,  # Use GPU or TPU based on the setting
}

# Write the kernel metadata to a JSON file
kernel_metadata_dir = os.path.dirname(notebook_path)
kernel_metadata_file = os.path.join(kernel_metadata_dir, 'kernel-metadata.json')
os.makedirs(kernel_metadata_dir, exist_ok=True)

with open(kernel_metadata_file, 'w') as f:
    json.dump(metadata, f, indent=4)

logging.info(f"Kernel metadata file created at {kernel_metadata_file}")

# Save the kernel name to last_run.txt for future use
last_run_file = 'last_run.txt'
with open(last_run_file, 'w') as f:
    f.write(f'{kaggle_username}/{unique_kernel_name}')

logging.info(f"Kernel name '{unique_kernel_name}' saved to {last_run_file}")

try:
    # Push the notebook to Kaggle
    logging.info(f"Pushing notebook '{notebook_path}' to Kaggle as kernel '{unique_kernel_name}'...")
    result = subprocess.run(
        ['kaggle', 'kernels', 'push', '-p', os.path.dirname(notebook_path)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logging.error(f"Failed to push notebook to Kaggle: {result.stderr}")
        logging.info(f"Stdout: {result.stdout}")
        exit(1)
    else:
        logging.info(f"Notebook {notebook_path} has been successfully pushed to Kaggle as '{unique_kernel_name}'.")

    # Define the Kaggle output URL (update it to match your actual URL)
    download_link = f"https://www.kaggle.com/code/{kaggle_username}/{unique_kernel_name}/output?select={output_model_filename}"

    # Polling mechanism to retry download every 2 minutes
    models_dir = os.path.join('models')
    os.makedirs(models_dir, exist_ok=True)

    while True:
        try:
            logging.info(f"Attempting to download model file from: {download_link}")
            download_command = f"wget -O {models_dir}/{output_model_filename} {download_link}"
            subprocess.run(download_command, shell=True, check=True)
            logging.info(f"Model file '{output_model_filename}' has been successfully downloaded to {models_dir}.")
            break  # Exit loop if download succeeds
        except subprocess.CalledProcessError as e:
            logging.error(f"Download failed: {e}. Retrying in {polling_interval // 60} minutes...")
            time.sleep(polling_interval)  # Wait for 2 minutes before retrying

except Exception as e:
    logging.error(f"An error occurred: {e}")
