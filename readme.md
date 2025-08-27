# Project Overview

This is an ongoing project under **Prof. Khurshid Ahmad** (Trinity College Dublin). It focuses on developing a COMPUTER VISION/MACHINE LEARNING PIPELINE FOR DETECTING EYELID CANCER



## Project Structure

The project is organized into the following directories and files:

```
[Root Directory]
    │
    ├── labelled_dataset/
    │   ├── test/       # Test images (benign: b*.jpg, malignant: m*.jpg)
    │   ├── train/      # Training images
    │   └── val/        # Validation images
    │
    ├── pipeline/
    │   ├── GrabCut/
    │   │   ├── eye_detect_patch.py         # Detects eyes, applies patch, and performs image division.
    │   │   ├── select.py                   # Script for selecting image parts after division.
    │   │   ├── preprocess_segment.py       # Performs contrast enhancement, hair removal, and GrabCut+SLIC segmentation.
    │   │   ├── train.py                    # Trains and saves CNN models, plots results.
    │   │   ├── run_saved.py                # Executes a previously saved model on test data.
    │   │   ├── run_best.py                 # Executes the best-performing model on test data.
    │   │   └── run_pipeline.py             # Orchestrator to run the entire pipeline automatically.
    │   │
    │   └── TopK/
    │       ├── eye_detect_patch.py         # (Same as above)
    │       ├── select.py                   # (Same as above)
    │       ├── preprocess_segment.py       # Performs contrast enhancement, hair removal, and Top-K+SLIC segmentation.
    │       ├── train.py                    # (Same as above)
    │       ├── run_saved.py                # (Same as above)
    │       ├── run_best.py                 # (Same as above)
    │       └── run_pipeline.py             # (Same as above)
    │
    └── augmentation.py                     # Script to augment the training dataset.
```

## Key Pipeline Configurations

You can configure the preprocessing steps by modifying the variables in the following files within either the `pipeline/GrabCut` or `pipeline/TopK` directory.

-   **`eye_detect_patch.py`**
    -   `patch_shape`: Set the shape of the patch applied over the eye region.
        -   Options: `'ellipse'` or `'circle'`.
    -   `division_mode`: Set the image division mode.
        -   Options: `0` (no division), `1`, or `2`.

-   **`preprocess_segment.py`**
    -   `APPLY_CONTRAST_ENHANCEMENT`: Enable or disable contrast enhancement.
        -   Options: `True` or `False`.
    -   `APPLY_HAIR_REMOVAL`: Enable or disable hair removal.
        -   Options: `True` or `False`.

## Execution Instructions

### 1. Data Preparation

1.  Place your training, validation, and testing images in the respective `train`, `val`, and `test` folders inside `labelled_dataset`.
2.  Ensure images are named with a prefix `b` for benign (e.g., `b1.jpg`) and `m` for malignant (e.g., `m1.jpg`).

### 2. Training a New Model

1.  Choose a pipeline (`GrabCut` or `TopK`) and set your desired configurations as described in the "Key Pipeline Configurations" section.
2.  Execute the pipeline using either the **Manual** or **Automated** method.

#### Manual Execution
Run the scripts in the following order:
1.  `python eye_detect_patch.py`
2.  `python select.py` (Only if `division_mode` is not `0`)
3.  `python preprocess_segment.py`
4.  `python train.py`

#### Automated Execution
1.  Open `run_pipeline.py`.
2.  Set `TYPE = "TRAIN"`.
3.  If you enabled image division, set `USING_IMAGE_DIVISION = True`, otherwise set it to `False`.
4.  Run the script: `python run_pipeline.py`.

### 3. Running a Saved Model

After training, models are saved in the `saved_models` folder. To test these models:

1.  Ensure your test images are in the `labelled_dataset/test` folder.
2.  Configure the pipeline with the **exact same settings** used during the model's training.
3.  Execute the pipeline.

#### Manual Execution
Run the scripts in the following order:
1.  `python eye_detect_patch.py`
2.  `python select.py` (Only if `division_mode` is not `0`)
3.  `python preprocess_segment.py`
4.  `python run_saved.py`

#### Automated Execution
1.  Open `run_pipeline.py`.
2.  Set `TYPE = "SAVED"`.
3.  Set `USING_IMAGE_DIVISION` according to your configuration.
4.  Run the script: `python run_pipeline.py`.

### 4. Running the Best Model

The best-performing models identified during our study are stored in the `best_model` folder.

1.  Ensure your test images are in the `labelled_dataset/test` folder.
2.  Configure the pipeline with the specific settings for the best model:
    -   **Top-K Pipeline**: Enable Patch, Division (mode 1), and Contrast Enhancement.
    -   **GrabCut Pipeline**: Enable Patch and Division (mode 1).
3.  Execute the pipeline.

#### Manual Execution
Run the scripts in the following order:
1.  `python eye_detect_patch.py`
2.  `python select.py`
3.  `python preprocess_segment.py`
4.  `python run_best.py`

#### Automated Execution
1.  Open `run_pipeline.py`.
2.  Set `TYPE = "BEST"`.
3.  Set `USING_IMAGE_DIVISION = True`.
4.  Run the script: `python run_pipeline.py`.


## Credits
**Supervisor:** Prof. Khurshid Ahmad (Professor, School of Computer Science & Statistics, Trinity College Dublin)

**Team:**
- Jay Birthariya (M.Sc. Student, Computer Science – Intelligent Systems, Trinity College Dublin)
- Dr. Qirat Qurban (Oculoplastics, Royal Victoria Eye and Ear Hospital; PhD)
- Dr. Tracey Hilton (PhD in Artificial Intelligence; Neural Networks)
- Prof. Lorraine Cassidy (Consultant Ophthalmologist; Oculoplastic & Eyelid Reconstructive Surgery)
- Dr. Mahmood (Ophthalmologist, Royal Victoria Eye and Ear Hospital, Dublin)