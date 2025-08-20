import os
import re
from pathlib import Path
import cv2
import numpy as np

# Define paths
and_result_dir = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\classification_level2\AND_result")
ulcer_cornea_dir = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\classification_level2\ulcer_cornea_result")
output_dir = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\classification_level2\images_for_classification_grade")
output_dir.mkdir(parents=True, exist_ok=True)

# Supported image extensions
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

def get_id(p: Path) -> int:
    try:
        return int(re.findall(r'\d+', p.stem)[0])
    except:
        return None

# Create pairs of images based on IDs
and_map = {}
for p in and_result_dir.glob("*.*"):
    if p.suffix.lower() in VALID_EXTENSIONS:
        if (fid := get_id(p)) is not None:
            and_map[fid] = p

pairs = []
for ulcer_path in ulcer_cornea_dir.glob("*.*"):
    if ulcer_path.suffix.lower() in VALID_EXTENSIONS:
        if (fid := get_id(ulcer_path)) is not None and fid in and_map:
            pairs.append((and_map[fid], ulcer_path))

# Process image pairs
for and_img_path, ulcer_img_path in pairs:
    try:
        file_id = get_id(and_img_path)
        and_img = cv2.imread(str(and_img_path), cv2.IMREAD_COLOR)
        ulcer_img = cv2.imread(str(ulcer_img_path), cv2.IMREAD_GRAYSCALE)

        # Resize and validate images
        and_img = cv2.resize(and_img, (256, 256)) if and_img.shape != (256, 256) else and_img
        ulcer_img = cv2.resize(ulcer_img, (256, 256)) if ulcer_img.shape != (256, 256) else ulcer_img

        # Step 1: White background conversion
        mask = np.any(and_img > 10, axis=2).astype(np.uint8) * 255
        white_background = np.full_like(and_img, 255)
        white_background[mask == 255] = and_img[mask == 255]

        # Step 2: Grayscale conversion
        gray_and = cv2.cvtColor(white_background, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray_and, cv2.COLOR_GRAY2BGR)

        # Step 3: Apply red ulcers
        _, ulcer_mask = cv2.threshold(ulcer_img, 50, 255, cv2.THRESH_BINARY_INV)
        ulcer_mask = cv2.bitwise_and(ulcer_mask, mask)
        result[ulcer_mask == 255] = [0, 0, 255]

        # Save result
        cv2.imwrite(str(output_dir / f"{file_id}.png"), result)
        print(f"Processed ID {file_id}")

    except Exception as e:
        print(f"Error processing {and_img_path.name}: {str(e)}")

print("Processing completed!")