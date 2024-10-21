import os
import pydicom
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Dataset creation class
class PneumoniaDataset(Dataset):
    def __init__(self, data_dir, labels, transform=None, img_size=128):
        self.data_dir = data_dir
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
        self.data = []

        # Process the data image paths and add labels to them
        for label in labels:
            label_dir = os.path.join(data_dir, label)
            class_idx = labels.index(label)
            print(label_dir)

            # Create image directories
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                self.data.append((img_path, class_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        try:
            # Attempt to read as a DICOM file
            dicom = pydicom.dcmread(img_path)
            img = dicom.pixel_array
        except Exception as e:
            # If DICOM read fails, fallback to standard image reading
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to read image file {img_path}: {e}")

        # Resize and normalize
        resized_image = cv2.resize(img, (self.img_size, self.img_size))
        img_normalized = resized_image / 255.0

        # Convert to 3 channels by repeating the single channel for grayscale images
        img_rgb = np.stack([img_normalized] * 3, axis=-1)  # Convert to 3-channel RGB image

        # Convert to tensor
        img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).float()

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label

# Data loader initialization
def get_dataloaders(data_dir, labels, batch_size):
    train_dataset = PneumoniaDataset(data_dir=os.path.join(data_dir, 'train'), labels=labels)
    test_dataset = PneumoniaDataset(data_dir=os.path.join(data_dir, 'test'), labels=labels)
    val_dataset = PneumoniaDataset(data_dir=os.path.join(data_dir, 'val'), labels=labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader

# Dataset insights function
def dataset_insights(data_dir):
    """
    Function to print insights about the dataset like the number of images in each class (NORMAL and PNEUMONIA).
    """
    labels = ['NORMAL', 'PNEUMONIA']
    
    for label in labels:
        label_dir = os.path.join(data_dir, 'train', label)
        num_images = len(os.listdir(label_dir))
        print(f"Number of {label} images in train set: {num_images}")
    
    for label in labels:
        label_dir = os.path.join(data_dir, 'test', label)
        num_images = len(os.listdir(label_dir))
        print(f"Number of {label} images in test set: {num_images}")
    
    for label in labels:
        label_dir = os.path.join(data_dir, 'val', label)
        num_images = len(os.listdir(label_dir))
        print(f"Number of {label} images in validation set: {num_images}")
