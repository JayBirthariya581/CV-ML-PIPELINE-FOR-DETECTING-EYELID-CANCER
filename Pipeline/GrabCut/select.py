# PROJECT: A COMPUTER VISION/MACHINE LEARNING PIPELINE FOR DETECTING EYELID CANCER
# Supervisor: Prof. Khurshid Ahmad (Trinity College Dublin)
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
from tqdm import tqdm

# ---------- Configuration ----------
COVERED_DIR = 'covered_divided_eye'   # root folder created by one_circle.py
SPLITS = ['train', 'val', 'test']
QUADRANTS_SUBFOLDER = 'quadrants2'
SELECTED_SUBFOLDER = 'quadrants/selected_images'  # where selected parts are saved
SAVE_OPTION = 0  # 0: draw bbox, 1: blackout background, 2: crop only
OUTPUT_EXCEL = 'part_selection_results.xlsx'

# ---------- Lesion Detection & Scoring ----------
def detect_lesion_box(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    _, saturation, _ = cv2.split(hsv)
    blurred = cv2.GaussianBlur(saturation, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 4
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = img_rgb.shape[:2]
    min_area = 0.01 * h * w
    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        ar = bw / (bh + 1e-6)
        if area > min_area and 0.2 < ar < 5:
            valid.append(cnt)
    if not valid:
        return None
    hull = cv2.convexHull(max(valid, key=cv2.contourArea))
    x, y, bw, bh = cv2.boundingRect(hull)
    return (x, y, x + bw, y + bh)


def calculate_score(img_rgb, box):
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    crop = img_rgb[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    img_area = img_rgb.shape[0] * img_rgb.shape[1]
    # size score
    area = w * h
    size_score = min(area / img_area, 0.5) * 2
    # contrast score
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    contrast_score = np.std(gray) / 128.0
    # shape score
    ar = w / (h + 1e-6)
    shape_score = 1.0 - abs(1.0 - ar) * 0.5
    shape_score = max(0.0, min(1.0, shape_score))
    # edge score
    edges = cv2.Canny(gray, 50, 150)
    edge_score = np.sum(edges > 0) / (w * h)
    edge_score = min(edge_score, 1.0)
    # color variation
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    color_var = np.std(hsv[:, :, 1]) / 128.0
    # weighted sum
    total = (size_score * 0.3 + contrast_score * 0.25 + 
             shape_score * 0.2 + edge_score * 0.15 + color_var * 0.1)
    return total


def main():
    results = []

    for split in SPLITS:
        in_base = os.path.join(COVERED_DIR, split, QUADRANTS_SUBFOLDER)
        out_base = os.path.join(COVERED_DIR, split, SELECTED_SUBFOLDER)
        os.makedirs(out_base, exist_ok=True)
        # gather all parts
        part_files = glob(os.path.join(in_base, '*', '*.jpg')) + glob(os.path.join(in_base, '*', '*.png'))
        groups = defaultdict(list)
        for fp in part_files:
            name = os.path.splitext(os.path.basename(fp))[0]
            if '_' in name:
                base, part = name.rsplit('_', 1)
            else:
                base, part = name, ''
            groups[base].append((fp, part))

        print(f"Selecting best parts for {split} ({len(groups)} images)")
        for base, items in tqdm(groups.items()):
            best_score = -1.0
            best = None
            for fp, part in items:
                img = cv2.imread(fp)
                if img is None: continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                box = detect_lesion_box(img_rgb)
                score = calculate_score(img_rgb, box)
                results.append({
                    'split': split,
                    'base_name': base,
                    'part': part,
                    'filepath': fp,
                    'score': score,
                    'selected': False
                })
                if score > best_score:
                    best_score = score
                    best = (fp, part, box, img_rgb)

            if best is None:
                print(f"No valid part for {base} in {split} — using full covered image")

                # Try to load the full covered image from the covered_eyes folder
                covered_fp = os.path.join(COVERED_DIR, split, 'covered_eyes', f"{base}_covered.jpg")
                if not os.path.exists(covered_fp):
                    print(f"Covered image not found for {base} in split {split}")
                    continue

                img = cv2.imread(covered_fp)
                if img is None:
                    print(f"Failed to load covered image: {covered_fp}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                prefix = 'benign' if base.lower().startswith('b') else 'malignant'
                out_fp = os.path.join(out_base, f"{prefix}_{base}_selected.png")

                if SAVE_OPTION == 0:
                    plt.imsave(out_fp, img_rgb)
                elif SAVE_OPTION == 1:
                    # Save as-is (no lesion box to blackout)
                    plt.imsave(out_fp, np.zeros_like(img_rgb))
                else:
                    # Save entire image (no cropping)
                    cv2.imwrite(out_fp, img)

                results.append({
                    'split': split,
                    'base_name': base,
                    'part': 'full',
                    'filepath': covered_fp,
                    'score': 0.0,
                    'selected': True
                })
                continue
            fp_best, part_best, box_best, img_best = best
            x1, y1, x2, y2 = box_best if box_best else (0,0,0,0)
            # determine label prefix
            prefix = 'benign' if base.lower().startswith('b') else 'malignant'
            fname = f"{base}_{part_best}_selected.png" if part_best else f"{base}_selected.png"
            out_fp = os.path.join(out_base, f"{prefix}_{fname}")

            if SAVE_OPTION == 0:
                vis = img_best.copy()
                # cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                plt.imsave(out_fp, vis)
            elif SAVE_OPTION == 1:
                mask = np.zeros_like(img_best)
                mask[y1:y2, x1:x2] = img_best[y1:y2, x1:x2]
                plt.imsave(out_fp, mask)
            else:
                crop = img_best[y1:y2, x1:x2]
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_fp, crop_bgr)

            for r in results:
                if r['filepath'] == fp_best and r['base_name'] == base:
                    r['selected'] = True
                    break


if __name__ == '__main__':
    main()
