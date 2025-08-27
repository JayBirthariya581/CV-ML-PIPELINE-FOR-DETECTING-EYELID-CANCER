# PROJECT: A COMPUTER VISION/MACHINE LEARNING PIPELINE FOR DETECTING EYELID CANCER
# Supervisor: Prof. Khurshid Ahmad (Trinity College Dublin)
import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops, label
from skimage.color import rgb2gray, rgb2hsv
from skimage.morphology import closing, disk, remove_small_objects
from sklearn.cluster import KMeans
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
APPLY_CONTRAST_ENHANCEMENT = False

# Hair removal parameters
HAIR_KERNEL_SIZE = 7
HAIR_INPAINT_RADIUS = 6

# Superpixel & graph-cut segmentation parameters
SLIC_SEGMENTS = 100
SLIC_COMPACTNESS = 10
GRABCUT_ITER = 5
GRABCUT_MARGIN = 10  # margin from image borders for initial rectangle

# Segmentation cleanup parameters
CLOSING_DISK_RADIUS = 7
MIN_COMPONENT_SIZE = 500

# ---------- Hair Removal ----------
def remove_hair(img_bgr, kernel_size=HAIR_KERNEL_SIZE, inpaint_radius=HAIR_INPAINT_RADIUS):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
    _, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)

# ---------- Contrast Enhancement ----------
def enhance_contrast(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# ---------- Baseline Lesion Segmentation (HSV+SLIC) ----------
def segment_lesion_baseline(img_rgb,
                             n_segments=SLIC_SEGMENTS,
                             compactness=SLIC_COMPACTNESS,
                             closing_disk_radius=CLOSING_DISK_RADIUS,
                             min_component_size=MIN_COMPONENT_SIZE):
    img = img_rgb.astype(float) / 255.0
    segments = slic(img, n_segments=n_segments, compactness=compactness, start_label=1)
    hsv = rgb2hsv(img)
    labels = np.unique(segments)
    feats = np.zeros((len(labels), 3), float)
    for i, lbl in enumerate(labels):
        mask = segments == lbl
        feats[i, 0] = hsv[..., 0][mask].mean()
        feats[i, 1] = hsv[..., 1][mask].mean()
        feats[i, 2] = hsv[..., 2][mask].mean()
    km = KMeans(n_clusters=2, random_state=0).fit(feats)
    cluster_vs = [feats[km.labels_ == i, 2].mean() for i in (0, 1)]
    lesion_cluster = int(np.argmin(cluster_vs))
    lesion_mask = np.isin(segments, labels[km.labels_ == lesion_cluster])
    lesion_mask = closing(lesion_mask, disk(closing_disk_radius))
    lesion_mask = remove_small_objects(lesion_mask, min_size=min_component_size)
    lbls = label(lesion_mask)
    if lbls.max() == 0:
        raise RuntimeError("Baseline segmentation failed: no lesion component found.")
    props = regionprops(lbls)
    largest = max(props, key=lambda r: r.area).label
    return lbls == largest

# ---------- Lesion Segmentation (GraphCut + SLIC with fallback) ----------
def segment_lesion_superpixel(img_rgb,
                              n_segments=SLIC_SEGMENTS,
                              compactness=SLIC_COMPACTNESS,
                              grabcut_iter=GRABCUT_ITER,
                              margin=GRABCUT_MARGIN):
    """
    Use GrabCut to initialize a lesion mask, then refine with SLIC superpixels.
    Falls back to HSV+SLIC baseline if GrabCut yields no region.
    Returns a boolean mask of the segmented lesion.
    """
    try:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, grabcut_iter, cv2.GC_INIT_WITH_RECT)
        binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
        img_float = img_rgb.astype(float) / 255.0
        segments = slic(img_float, n_segments=n_segments, compactness=compactness, start_label=1)
        lesion_labels = np.unique(segments[binary == 1])
        if len(lesion_labels) == 0:
            raise ValueError("No GrabCut overlap segments")
        refined = np.zeros_like(segments)
        for lbl in lesion_labels:
            refined[segments == lbl] = lbl
        lesion_mask = refined > 0
        lesion_mask = closing(lesion_mask, disk(CLOSING_DISK_RADIUS))
        lesion_mask = remove_small_objects(lesion_mask, min_size=MIN_COMPONENT_SIZE)
        labeled = label(lesion_mask)
        if labeled.max() == 0:
            raise ValueError("No lesion after cleanup")
        regions = regionprops(labeled)
        largest = max(regions, key=lambda r: r.area).label
        return labeled == largest
    except Exception as e:
        print(f"GraphCut segmentation failed ({e}); falling back to baseline.")
        return segment_lesion_baseline(img_rgb)

# ---------- Directory Setup ----------
def create_output_dirs():
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# ---------- Image Processing Pipeline ----------
def process_image(filepath):
    fname = os.path.basename(filepath)
    cls = 'benign' if fname.lower().startswith('b') else 'malignant'
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        print(f"Could not read {filepath}")
        return None
    if APPLY_HAIR_REMOVAL:
        img_bgr = remove_hair(img_bgr)
    if APPLY_CONTRAST_ENHANCEMENT:
        img_bgr = enhance_contrast(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lesion_mask = segment_lesion_superpixel(img_rgb)
    seg_vis = mark_boundaries(img_rgb, lesion_mask.astype(int))
    foreground = img_rgb.copy()
    foreground[~lesion_mask] = 0
    return cls, fname, seg_vis, foreground

# ---------- Main ----------
def main():
    create_output_dirs()
    for split in SPLITS:
        in_dir = os.path.join(COVERED_DIR, split, QUADRANTS_SUBFOLDER)
        if not os.path.isdir(in_dir):
            print(f"Missing input directory: {in_dir}")
            continue
        files = [f for f in glob(os.path.join(in_dir, '*', '*.*'))
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Processing {len(files)} images for split '{split}'...")
        for fp in tqdm(files):
            result = process_image(fp)
            if result is None:
                continue
            cls, fname, seg_vis, foreground = result
            base = os.path.splitext(fname)[0]
            plt.imsave(os.path.join(OUTPUT_DIR, split, cls, f"{base}_segmented.png"), seg_vis)
            plt.imsave(os.path.join(OUTPUT_DIR, split, cls, f"{base}_foreground.png"), foreground)
            
    print("\nAll preprocessing and segmentation complete!")

if __name__ == '__main__':
    main()
