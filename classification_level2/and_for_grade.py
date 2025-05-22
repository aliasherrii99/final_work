import os
import re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Define paths
and_result_dir = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\classification_level2\AND_result")
ulcer_cornea_dir = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\classification_level2\ulcer_cornea_result")
output_dir = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\classification_level2\images_for_classification_grade")
output_dir.mkdir(parents=True, exist_ok=True)

# Supported image extensions
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

# Function to extract ID from file name
def get_id(p: Path) -> int:
    try:
        match = re.findall(r'\d+', p.stem)
        if not match:
            print(f"Warning: No numeric ID found in {p.name}. Skipping.")
            return None
        return int(match[0])
    except Exception as e:
        print(f"Error extracting ID from {p.name}: {e}")
        return None

# Create pairs of images based on IDs
and_map = {}
for p in and_result_dir.glob("*.*"):
    if p.suffix.lower() not in VALID_EXTENSIONS:
        print(f"Skipping non-image file: {p.name}")
        continue
    fid = get_id(p)
    if fid is not None:
        and_map[fid] = p

pairs = []
for ulcer_path in sorted(ulcer_cornea_dir.glob("*.*"), key=lambda x: get_id(x) or -1):
    if ulcer_path.suffix.lower() not in VALID_EXTENSIONS:
        print(f"Skipping non-image file: {ulcer_path.name}")
        continue
    fid = get_id(ulcer_path)
    if fid is not None and fid in and_map:
        pairs.append((and_map[fid], ulcer_path))

# Check if pairs are found
if not pairs:
    print("Error: No valid image pairs found. Check directories and file naming.")
    exit(1)

# Process image pairs
for and_img_path, ulcer_img_path in pairs:
    file_id = get_id(and_img_path)
    if file_id is None:
        continue

    try:
        # Load images
        and_img = cv2.imread(str(and_img_path), cv2.IMREAD_COLOR)
        ulcer_img = cv2.imread(str(ulcer_img_path), cv2.IMREAD_GRAYSCALE)

        if and_img is None or ulcer_img is None:
            print(f"Error: Could not load images for ID {file_id}")
            continue

        # Ensure images are 256x256
        if and_img.shape[:2] != (256, 256) or ulcer_img.shape != (256, 256):
            print(f"Warning: Image size mismatch for ID {file_id}. Skipping.")
            continue

        # Step 1: Change black background to white
        mask = np.any(and_img > 10, axis=2).astype(np.uint8) * 255  # Mask for non-black regions
        white_background = np.ones_like(and_img) * 255  # Create white background
        white_background[mask == 255] = and_img[mask == 255]  # Keep colored region
        and_img = white_background  # Update image with white background

        # Step 2: Convert colored region to grayscale
        gray_and_img = cv2.cvtColor(and_img, cv2.COLOR_BGR2GRAY)
        gray_region = cv2.bitwise_and(gray_and_img, gray_and_img, mask=mask)  # Grayscale only in colored region
        result = np.copy(and_img)  # Initialize result with white background
        result[mask == 255] = cv2.cvtColor(gray_region, cv2.COLOR_GRAY2BGR)[mask == 255]  # Apply grayscale region

        # Step 3: Apply black spots from ulcer_img onto grayscale region
        black_spots_mask = (ulcer_img < 50).astype(np.uint8) * 255  # Mask for black spots (threshold < 50)
        black_spots_mask = cv2.bitwise_and(black_spots_mask, mask)  # Restrict to grayscale region
        result[black_spots_mask == 255] = [0, 0, 0]  # Set black spots

        # Save the result
        out_path = output_dir / f"{file_id}.png"
        cv2.imwrite(str(out_path), result)
        print(f"Saved result for ID {file_id} to {out_path}")

    except Exception as e:
        print(f"Error processing ID {file_id}: {e}")

print(f"Done! Results saved in {output_dir}")