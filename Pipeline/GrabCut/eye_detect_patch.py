# PROJECT: A COMPUTER VISION/MACHINE LEARNING PIPELINE FOR DETECTING EYELID CANCER
# Supervisor: Prof. Khurshid Ahmad (Trinity College Dublin)
import os
import cv2
import numpy as np
import multiprocessing
import time

# Configuration
class Config:
    # Paths
    input_folder = os.path.join('..', '../labelled_dataset')
    output_folder = 'covered_divided_eye'
    debug_folder = 'debug_output'

    # Subfolder names
    covered_subfolder = 'covered_eyes'
    quadrants_subfolder = 'quadrants2'
    splits = ['train', 'val', 'test']

    # Division mode
    division_mode = 1
    division_folders = {
        0: ['whole'],
        1: ['Top', 'Bottom'],
        2: ['NW', 'NE', 'SW', 'SE']
    }

    # Detection parameters
    show_before_and_after = False
    min_contour_area = 100
    max_area_ratio = 0.8
    min_axis_length = 5
    
    # Patch configuration
    patch_shape = 'ellipse'  # 'circle' or 'ellipse'
    max_size_ratio = 0.35  # Maximum size relative to image
    
    # Timeout settings
    hough_timeout = 6      # Seconds for Hough Circle detection
    ellipse_timeout = 10   # Seconds for ellipse fitting
    total_timeout = 16     # Total timeout per image

    @classmethod
    def create_directories(cls):
        os.makedirs(cls.debug_folder, exist_ok=True)
        for split in cls.splits:
            base_out = os.path.join(cls.output_folder, split)
            os.makedirs(base_out, exist_ok=True)
            os.makedirs(os.path.join(base_out, cls.covered_subfolder), exist_ok=True)
            for folder in cls.division_folders.get(cls.division_mode, []):
                os.makedirs(os.path.join(base_out, cls.quadrants_subfolder, folder), exist_ok=True)


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    return gray, enhanced


def run_hough_circle(enhanced, queue):
    """Run Hough Circle detection with timeout"""
    h, w = enhanced.shape[:2]
    try:
        circles = cv2.HoughCircles(
            enhanced, cv2.HOUGH_GRADIENT, 1, int(min(h, w)/4),
            param1=50, param2=30,
            minRadius=int(min(h, w)/10), 
            maxRadius=int(min(h, w)/2)
        )
        queue.put(circles)
    except Exception as e:
        queue.put(None)


def detect_eye(image, filename):
    h, w = image.shape[:2]
    gray, enhanced = preprocess_image(image)

    # Save debug images
    cv2.imwrite(os.path.join(Config.debug_folder, f"{filename}_gray.jpg"), gray)
    cv2.imwrite(os.path.join(Config.debug_folder, f"{filename}_enhanced.jpg"), enhanced)

    best_ellipse, found = None, False
    start_time = time.time()
    
    # Phase 1: Try Hough Circle with timeout
    hough_queue = multiprocessing.Queue()
    hough_process = multiprocessing.Process(
        target=run_hough_circle, 
        args=(enhanced, hough_queue)
    )
    hough_process.start()
    hough_process.join(Config.hough_timeout)
    
    circles = None
    if hough_process.is_alive():
        hough_process.terminate()
        print(f"Hough Circle timeout for {filename}")
    elif not hough_queue.empty():
        circles = hough_queue.get()
    
    roi_offset = (0, 0)
    roi = enhanced
    
    # If Hough Circle found something, define ROI
    if circles is not None:
        try:
            x, y, r = np.uint16(np.around(circles))[0][0]
            x1 = max(x - r, 0)
            y1 = max(y - r, 0)
            x2 = min(x + r, w)
            y2 = min(y + r, h)
            
            # Check for valid ROI dimensions
            if x2 > x1 and y2 > y1:
                roi = enhanced[y1:y2, x1:x2]
                roi_offset = (x1, y1)
                print(f"Hough Circle found ROI for {filename}")
            else:
                print(f"Invalid ROI dimensions for {filename}")
        except Exception as e:
            print(f"Hough Circle processing error for {filename}: {str(e)}")
            circles = None
    
    # Check total time used so far
    time_used = time.time() - start_time
    remaining_time = max(0, Config.total_timeout - time_used)
    
    # Phase 2: Ellipse fitting in ROI
    if roi.size == 0:  # Handle empty ROI
        print(f"Empty ROI for {filename}, using full image")
        roi = enhanced
        roi_offset = (0, 0)
    
    _, thresh1_roi = cv2.threshold(roi, 70, 255, cv2.THRESH_BINARY_INV)
    _, thresh2_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Create process for ellipse fitting
    ellipse_queue = multiprocessing.Queue()
    ellipse_process = multiprocessing.Process(
        target=run_ellipse_fitting,
        args=(thresh1_roi, thresh2_roi, h, w, roi_offset, ellipse_queue)
    )
    ellipse_process.start()
    ellipse_process.join(remaining_time)
    
    if ellipse_process.is_alive():
        ellipse_process.terminate()
        print(f"Ellipse fitting timeout for {filename}")
        found = False
    elif not ellipse_queue.empty():
        best_ellipse, found = ellipse_queue.get()
    
    return best_ellipse, found, gray


def run_ellipse_fitting(thresh1, thresh2, full_h, full_w, roi_offset, queue):
    """Run ellipse fitting in ROI"""
    best_ellipse = None
    found = False
    
    # Process both thresholds
    for thresh in (thresh1, thresh2):
        if thresh is None or thresh.size == 0:
            continue
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        try:
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        except:
            continue
            
        cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if Config.min_contour_area < area < Config.max_area_ratio * full_h * full_w and len(cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                except:
                    continue
                (cx, cy), (MA, ma), _ = ellipse
                if MA > Config.min_axis_length and ma > Config.min_axis_length:
                    aspect = min(MA, ma) / max(MA, ma)
                    score = area * aspect
                    candidates.append((ellipse, score))
        
        if candidates:
            # Pick best in ROI and convert to full image coordinates
            ellipse_roi, _ = max(candidates, key=lambda x: x[1])
            (cx_roi, cy_roi), axes, ang = ellipse_roi
            best_ellipse = (
                (cx_roi + roi_offset[0], cy_roi + roi_offset[1]),
                axes,
                ang
            )
            found = True
            break
    
    queue.put((best_ellipse, found))


def cover_eye_with_skin(image, ellipse):
    result = image.copy()
    h, w = image.shape[:2]
    (x, y), (width, height), angle = ellipse
    
    # Apply size limits based on patch shape
    max_size = min(h, w) * Config.max_size_ratio
    
    if Config.patch_shape == 'circle':
        # Convert to circle: use average of axes as diameter
        diameter = (width + height) / 2
        radius = diameter / 2
        
        # Apply size limit
        if diameter > max_size:
            scale_factor = max_size / diameter
            diameter *= scale_factor
            radius = diameter / 2
            print(f"Reduced circle size for eye patch (diameter: {diameter:.1f}px)")
            
        # Create circular patch
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(radius), 255, thickness=-1)
        
        # Create circular sampling area
        outer_radius = int(radius * 3)  # 3x radius for sampling ring
        outer_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(outer_mask, (int(x), int(y)), outer_radius, 255, thickness=-1)
        sampling_mask = cv2.subtract(outer_mask, mask)
    else:  # ellipse
        # Apply size limit to ellipse
        current_max_axis = max(width, height)
        if current_max_axis > max_size:
            scale_factor = max_size / current_max_axis
            new_width = width * scale_factor
            new_height = height * scale_factor
            ellipse = ((x, y), (new_width, new_height), angle)
            width, height = new_width, new_height
            print(f"Reduced ellipse size for eye patch (axes: {width:.1f}x{height:.1f}px)")

        # Create elliptical patch
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, ellipse, 255, thickness=-1)
        
        # Create elliptical sampling area
        sampling_radius = int(max(width, height) * 1.5)
        outer_mask = np.zeros_like(mask)
        if width > height:
            sample_ellipse = ((x, y), (width, sampling_radius), angle)
        else:
            sample_ellipse = ((x, y), (sampling_radius, height), angle)
        cv2.ellipse(outer_mask, sample_ellipse, 255, thickness=-1)
        sampling_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(mask))

    # Get sampling pixels and apply skin color
    sampling_pixels = image[sampling_mask > 0]
    if sampling_pixels.size:
        skin_color = np.mean(sampling_pixels, axis=0).astype(np.uint8)
        mask_3ch = cv2.merge([mask] * 3)
        result = cv2.bitwise_and(result, cv2.bitwise_not(mask_3ch))
        result = cv2.bitwise_or(result, cv2.bitwise_and(
            np.full_like(image, skin_color), mask_3ch
        ))

    return result, mask


def create_quadrants(image, cx, cy, base_filename, split):
    h, w = image.shape[:2]
    cx, cy = int(cx), int(cy)
    if not (0 < cx < w and 0 < cy < h):
        print(f"Invalid center ({cx},{cy}) for {base_filename}")
        return

    regions = {}
    if Config.division_mode == 2:
        regions = {
            'NW': image[0:cy, 0:cx],
            'NE': image[0:cy, cx:w],
            'SW': image[cy:h, 0:cx],
            'SE': image[cy:h, cx:w]
        }
    elif Config.division_mode == 1:
        regions = {'Top': image[0:cy, :], 'Bottom': image[cy:h, :]}
    else:
        regions = {'whole': image}

    for name, region in regions.items():
        if region.size == 0:
            continue
        out_dir = os.path.join(Config.output_folder, split, Config.quadrants_subfolder, name)
        out_path = os.path.join(out_dir, f"{base_filename}_{name}.jpg")
        cv2.imwrite(out_path, region)


def process_image(image_path, base_filename, split):
    print(f"Starting processing for {base_filename} in {split}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load {image_path}")
        return False

    ellipse, found, _ = detect_eye(image, base_filename)
    if not found:
        if split == 'test':
            print(f"No eye detected in {base_filename}, applying fallback using image center (test only)")

            h, w = image.shape[:2]
            cx, cy = w // 2, h // 2

            max_size = min(h, w) * Config.max_size_ratio
            width = height = max_size  
            angle = 0 

            fallback_ellipse = ((cx, cy), (width, height), angle)

            covered, _ = cover_eye_with_skin(image, fallback_ellipse)

            fallback_dir = os.path.join(Config.output_folder, split, Config.covered_subfolder)
            os.makedirs(fallback_dir, exist_ok=True)
            cv2.imwrite(os.path.join(fallback_dir, f"{base_filename}_fallback_covered.jpg"), covered)

            create_quadrants(covered, cx, cy, base_filename, split)

            debug_dir = os.path.join(Config.output_folder, split, 'undetected')
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"{base_filename}_original.jpg"), image)

            return True
        else:
            print(f"No eye detected in {base_filename}, skipping (fallback only for test split)")
            return False

    # Cover the eye
    covered, _ = cover_eye_with_skin(image, ellipse)
    covered_dir = os.path.join(Config.output_folder, split, Config.covered_subfolder)
    cv2.imwrite(os.path.join(covered_dir, f"{base_filename}_covered.jpg"), covered)

    # Optional debug before/after
    if Config.show_before_and_after:
        before = image.copy()
        cv2.ellipse(before, ellipse, (0,255,0), 2)
        comp = np.hstack((before, covered))
        cv2.imwrite(os.path.join(Config.debug_folder, f"{base_filename}_before_after.jpg"), comp)

    # Divide into regions
    (cx, cy), _, _ = ellipse
    create_quadrants(covered, cx, cy, base_filename, split)
    print(f"[{split}] Successfully processed {base_filename}")
    return True


def target_process_image(image_path, base_filename, split, queue):
    try:
        res = process_image(image_path, base_filename, split)
        queue.put(res)
    except Exception as e:
        print(f"Error processing {base_filename}: {e}")
        queue.put(False)


def process_image_with_timeout(image_path, base_filename, split, timeout=16):
    print(f"Processing {base_filename} with timeout of {timeout}s")
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=target_process_image,
        args=(image_path, base_filename, split, queue)
    )
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        print(f"Total timeout: skipped {base_filename}")
        if split == 'test':
            print(f"Applying fallback in MAIN process for {base_filename} (test only)")
            image = cv2.imread(image_path)
            if image is not None:
                h, w = image.shape[:2]
                cx, cy = w // 2, h // 2
                max_size = min(h, w) * Config.max_size_ratio

                if Config.patch_shape == 'circle':
                    fallback_axes = (max_size, max_size)
                    angle = 0
                else:
   
                    width = max_size
                    height = max_size * 0.6
                    fallback_axes = (width, height)
                    angle = 0

                fallback_ellipse = ((cx, cy), fallback_axes, angle)

                covered, _ = cover_eye_with_skin(image, fallback_ellipse)

                covered_dir = os.path.join(Config.output_folder, split, Config.covered_subfolder)
                os.makedirs(covered_dir, exist_ok=True)
                cv2.imwrite(os.path.join(covered_dir, f"{base_filename}_fallback_covered.jpg"), covered)

                debug_dir = os.path.join(Config.output_folder, split, 'undetected')
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"{base_filename}_original.jpg"), image)

                create_quadrants(covered, cx, cy, base_filename, split)

                print(f"[test-fallback] Successfully processed {base_filename} with fallback (main process)")
                return True
            else:
                print(f"Could not load image for fallback: {base_filename}")
                return False
        return False
    return queue.get() if not queue.empty() else False


def batch_process():
    Config.create_directories()
    print("Created all output directories")
    
    success, failed = 0, 0
    for split in Config.splits:
        print(f"\nProcessing split: {split}")
        split_dir = os.path.join(Config.input_folder, split)
        if not os.path.isdir(split_dir):
            print(f"Input folder not found: {split_dir}")
            continue
        
        files = [f for f in os.listdir(split_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        print(f"Found {len(files)} images in {split}")
        
        for i, fname in enumerate(files):
            path = os.path.join(split_dir, fname)
            base = os.path.splitext(fname)[0]
            print(f"\n[{i+1}/{len(files)}] Processing {base}")
            
            if process_image_with_timeout(path, base, split, Config.total_timeout):
                success += 1
                print(f"Successfully processed {base}")
            else:
                failed += 1
                print(f"Failed to process {base}")

    print("\nAll done!")
    print(f"Success: {success}, Failed: {failed}")
    print(f"Patch shape: {Config.patch_shape}, Max size ratio: {Config.max_size_ratio}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    print("="*50)
    print(f"Starting eye covering with {Config.patch_shape} patches")
    print(f"Timeouts: Hough={Config.hough_timeout}s, Ellipse={Config.ellipse_timeout}s, Total={Config.total_timeout}s")
    print("="*50)
    batch_process()