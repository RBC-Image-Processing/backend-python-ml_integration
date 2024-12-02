import torch
from torchvision import transforms
from PIL import Image
import pydicom
import numpy as np
import cv2
from io import BytesIO
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define transformations once to avoid redundancy
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing for standard images
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Specific transformation for CLAHE images (no resize needed for DICOM images)
transform_dicom = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_dicom_with_clahe(dicom_data: bytes):
    try:
        logger.info("Starting DICOM image processing with CLAHE.")

        # Read DICOM data
        dicom = pydicom.dcmread(BytesIO(dicom_data))
        img = dicom.pixel_array
        logger.info("DICOM data loaded successfully.")

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)
        logger.info("CLAHE applied to DICOM image.")

        # Resize and convert to RGB
        img_resized = cv2.resize(img_clahe, (224, 224))
        img_rgb = np.stack([img_resized] * 3, axis=-1)
        img_pil = Image.fromarray(img_rgb)
        logger.info("Image resized and converted to RGB.")

        # Transform and normalize
        processed_image = transform_dicom(img_pil).unsqueeze(0)
        logger.info("Image transformed and normalized.")

        return processed_image

    except Exception as e:
        logger.error(f"Error processing DICOM image: {e}")
        raise

def process_image_file(image_data: bytes):
    try:
        logger.info("Starting standard image file processing.")
        
        # Open image and validate format
        img_pil = Image.open(BytesIO(image_data))
        
        # Ensure image is in RGB mode
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")
        
        logger.info("Image file loaded and converted to RGB.")
        
        # Apply transformations and normalization
        processed_image = transform(img_pil).unsqueeze(0)
        logger.info("Image transformed and normalized.")

        return processed_image

    except Exception as e:
        logger.error(f"Error processing image file: {e}")
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")