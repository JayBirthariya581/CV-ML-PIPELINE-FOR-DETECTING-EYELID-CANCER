# PROJECT: A COMPUTER VISION/MACHINE LEARNING PIPELINE FOR DETECTING EYELID CANCER
# Supervisor: Prof. Khurshid Ahmad (Trinity College Dublin)
import os
import cv2
from albumentations import (
    Compose, Rotate, HorizontalFlip, ShiftScaleRotate,
    RandomBrightnessContrast, HueSaturationValue, ToGray, CLAHE
)
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import random

# Set seed for reproducibility
random.seed(42)

# ======================== CONFIG ========================
INPUT_FOLDER = "./From Front"  # Path to original images
OUTPUT_FOLDER = "./augmented_images"
AUGMENTATIONS_PER_IMAGE = 5
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')
# ========================================================

# Create output directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define medically safe augmentation pipeline
augmentation_pipeline = Compose([
    Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.7),
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, border_mode=cv2.BORDER_REFLECT, p=0.7),
    RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
    HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.7),
    CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
], p=1.0)


# Get all image files
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(IMG_EXTENSIONS)]

# Process images
for img_name in tqdm(image_files, desc="Augmenting Images"):
    img_path = os.path.join(INPUT_FOLDER, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to read image: {img_path}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    base_name = os.path.splitext(img_name)[0]

    # Save original
    orig_out_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_original.jpg")
    cv2.imwrite(orig_out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Generate augmented versions
    for i in range(AUGMENTATIONS_PER_IMAGE):
        augmented = augmentation_pipeline(image=image)['image']
        aug_out_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_aug_{i+1}.jpg")
        cv2.imwrite(aug_out_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

print(f"\nAugmentation complete. Files saved in '{OUTPUT_FOLDER}'")
