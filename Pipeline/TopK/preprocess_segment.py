# PROJECT: A COMPUTER VISION/MACHINE LEARNING PIPELINE FOR DETECTING EYELID CANCER
# Supervisor: Prof. Khurshid Ahmad (Trinity College Dublin)
import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops, label
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- Configuration ----------
COVERED_DIR = 'covered_divided_eye'           # Root folder from one_circle.py
QUADRANTS_SUBFOLDER = 'quadrants'             # Subfolder holding divided images per split
OUTPUT_DIR = 'segmented_outputs_combined'     # Where to save preprocessed outputs
SPLITS = ['train', 'val', 'test']
CLASSES = ['benign', 'malignant']

# Processing flags
APPLY_HAIR_REMOVAL = False
APPLY_CONTRAST_ENHANCEMENT = True

# Superpixel segmentation parameters
SLIC_SEGMENTS = 150
SLIC_COMPACTNESS = 10
TOP_K_SUPERPIXELS = 5

# ---------- Hair Removal ----------
def remove_hair(img_bgr, kernel_size=7, inpaint_radius=6):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    edges = cv2.Canny(gray, 50, 150)
    hair_mask = cv2.bitwise_or(blackhat, edges)
    hair_mask = cv2.dilate(hair_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    _, hair_mask = cv2.threshold(hair_mask, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_bgr, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)

# ---------- Contrast Enhancement ----------
def enhance_contrast(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# ---------- Directory Setup ----------
def create_output_dirs():
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# ---------- Image Processing Pipeline ----------
def process_image(filepath):
    fname = os.path.basename(filepath)
    key = fname.lower()
    # Determine class from filename prefix
    cls = 'benign' if key.startswith('b') else 'malignant'

    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        print(f"Could not read {filepath}")
        return None

    # Hair removal
    if APPLY_HAIR_REMOVAL:
        img_bgr = remove_hair(img_bgr)

    # Contrast enhancement
    if APPLY_CONTRAST_ENHANCEMENT:
        img_bgr = enhance_contrast(img_bgr)

    # Convert to RGB for segmentation
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Superpixel segmentation
    segments = slic(img_rgb, n_segments=SLIC_SEGMENTS, compactness=SLIC_COMPACTNESS, start_label=1)
    seg_vis = mark_boundaries(img_rgb, segments)

    # Build mask of top-k largest superpixels
    labeled = label(segments)
    props = sorted(regionprops(labeled), key=lambda p: p.area, reverse=True)
    mask = np.zeros_like(segments, dtype=bool)
    for region in props[:TOP_K_SUPERPIXELS]:
        mask |= (segments == region.label)

    # Apply mask to get foreground-only image
    foreground = img_rgb.copy()
    foreground[~mask] = 0

    return cls, fname, seg_vis, foreground

# ---------- Main ----------
def main():
    create_output_dirs()

    for split in SPLITS:
        in_dir = os.path.join(COVERED_DIR, split, QUADRANTS_SUBFOLDER)
        if not os.path.isdir(in_dir):
            print(f"Missing input directory: {in_dir}")
            continue

        pattern = os.path.join(in_dir, '*', '*.*')
        files = [f for f in glob(pattern) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Processing {len(files)} images for split '{split}'...")

        for fp in tqdm(files):
            result = process_image(fp)
            if result is None:
                continue
            cls, fname, seg_vis, foreground = result
            base_name = os.path.splitext(fname)[0]

            # Save segmentation visual
            out_seg = os.path.join(OUTPUT_DIR, split, cls, f"{base_name}_segmented.png")
            plt.imsave(out_seg, seg_vis)
            # Save foreground-only
            out_fg = os.path.join(OUTPUT_DIR, split, cls, f"{base_name}_foreground.png")
            plt.imsave(out_fg, foreground)

    print("\nAll preprocessing, segmentation complete!")

if __name__ == '__main__':
    main()
