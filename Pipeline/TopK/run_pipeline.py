#!/usr/bin/env python3
# PROJECT: A COMPUTER VISION/MACHINE LEARNING PIPELINE FOR DETECTING EYELID CANCER
# Supervisor: Prof. Khurshid Ahmad (Trinity College Dublin)
import os
import sys
import subprocess

# Toggle this to include/exclude select.py in the pipeline
USING_IMAGE_DIVISION = True  # set to False to skip select.py

TYPE = "SAVED" # TRAIN or SAVED or BEST

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name: str) -> bool:
    script_path = os.path.join(BASE_DIR, script_name)
    print(f"\nRunning {script_name} ...")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=BASE_DIR,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            text=True,
            capture_output=True,
            check=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"Finished: {script_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in {script_name}\n{'='*60}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False

def main():
    script_order = ["eye_detect_patch.py"]
    if USING_IMAGE_DIVISION:
        script_order.append("select.py")
    script_order += ["preprocess_segment.py"]

    if TYPE == "SAVED":
        script_order += ["run_saved.py"]
    elif TYPE == "BEST":
        script_order += ["run_best.py"]
    else:  # TRAIN
        script_order += ["train.py"]

    for script in script_order:
        if not run_script(script):
            print("\nStopping pipeline due to the error above.")
            sys.exit(1)
    print("\n🎉 All steps completed successfully.")

if __name__ == "__main__":
    main()
