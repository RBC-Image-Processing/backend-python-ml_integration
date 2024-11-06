import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import cv2
import pydicom
import numpy as np
from torchvision import transforms
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PneumoniaDataset(Dataset):
    def __init__(self, paths, labels, transform=None, img_size=224):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]

        # Load DICOM or image file
        try:
            if img_path.endswith(".dcm"):
                dicom = pydicom.dcmread(img_path)
                img = dicom.pixel_array
            else:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Check if image loaded correctly
            if img is None:
                raise ValueError(f"Failed to load image at path: {img_path}")
                
            # Resize and normalize
            img_resized = cv2.resize(img, (self.img_size, self.img_size))
            img_normalized = img_resized / 255.0
            img_rgb = np.stack([img_normalized] * 3, axis=-1)
            img_pil = Image.fromarray((img_rgb * 255).astype(np.uint8))

            if self.transform:
                img_pil = self.transform(img_pil)
            img_tensor = transforms.ToTensor()(img_pil)

            return img_tensor, label
        except Exception as e:
            logger.error(f"Error processing file {img_path}: {e}")
            return None, None  # You might want to handle these cases in your training loop

def load_rsna_dataset(rsna_dir):
    labels_df = pd.read_csv(os.path.join(rsna_dir, 'stage_2_train_labels.csv'))
    labels_df['path'] = labels_df['patientId'].apply(lambda x: os.path.join(rsna_dir, 'stage_2_train_images', f"{x}.dcm"))
    pneumonia_df = labels_df[labels_df['Target'] == 1]
    normal_df = labels_df[labels_df['Target'] == 0]

    pneumonia_paths = pneumonia_df['path'].tolist()
    normal_paths = normal_df['path'].tolist()

    logger.info(f"RSNA Dataset: {len(pneumonia_paths)} pneumonia cases, {len(normal_paths)} normal cases.")
    return pneumonia_paths, normal_paths

def load_chest_xray_dataset(chest_xray_dir):
    pneumonia_dir = os.path.join(chest_xray_dir, 'train', 'PNEUMONIA')
    normal_dir = os.path.join(chest_xray_dir, 'train', 'NORMAL')

    pneumonia_paths = [os.path.join(pneumonia_dir, fname) for fname in os.listdir(pneumonia_dir) if fname.endswith(('.jpeg', '.jpg', '.png'))]
    normal_paths = [os.path.join(normal_dir, fname) for fname in os.listdir(normal_dir) if fname.endswith(('.jpeg', '.jpg', '.png'))]

    return pneumonia_paths, normal_paths

def get_dataloaders(data_dir, batch_size, img_size=224):
    # Paths for RSNA and Chest X-Ray datasets
    rsna_dir = os.path.join(data_dir, 'rsna-pneumonia')
    chest_xray_dir = os.path.join(data_dir, 'chest-xray')

    # Load RSNA dataset
    rsna_pneumonia_paths, rsna_normal_paths = load_rsna_dataset(rsna_dir)
    rsna_dataset = PneumoniaDataset(rsna_pneumonia_paths + rsna_normal_paths, [1] * len(rsna_pneumonia_paths) + [0] * len(rsna_normal_paths), img_size=img_size)

    # Load Chest X-Ray dataset
    chest_xray_pneumonia_paths, chest_xray_normal_paths = load_chest_xray_dataset(chest_xray_dir)
    chest_xray_dataset = PneumoniaDataset(chest_xray_pneumonia_paths + chest_xray_normal_paths, [1] * len(chest_xray_pneumonia_paths) + [0] * len(chest_xray_normal_paths), img_size=img_size)

    # Combine both datasets
    combined_dataset = ConcatDataset([rsna_dataset, chest_xray_dataset])

    # Split combined dataset
    train_size = int(0.7 * len(combined_dataset))
    val_size = int(0.15 * len(combined_dataset))
    test_size = len(combined_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Data split: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")
    return train_loader, val_loader, test_loader
