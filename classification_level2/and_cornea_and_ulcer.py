import os
import re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from attention_unet import AttentionUNet

images_dir = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\classification_level2\AND_result")
labels_dir = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\classification_level2\ulcer_images")
model_path = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\classification_level2\attention_unet_ulcer.pth")
output_dir = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\classification_level2\ulcer_cornea_result")
output_dir.mkdir(parents=True, exist_ok=True)

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

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

mask_map = {}
for p in images_dir.glob("*.*"):
    if p.suffix.lower() not in VALID_EXTENSIONS:
        print(f"Skipping non-image file: {p.name}")
        continue
    fid = get_id(p)
    if fid is not None:
        mask_map[fid] = p

pairs = []
for ulcer_path in sorted(labels_dir.glob("*.*"), key=lambda x: get_id(x) or -1):
    if ulcer_path.suffix.lower() not in VALID_EXTENSIONS:
        print(f"Skipping non-image file: {ulcer_path.name}")
        continue
    fid = get_id(ulcer_path)
    if fid is not None and fid in mask_map:
        pairs.append((mask_map[fid], ulcer_path))

# Check if pairs are found
if not pairs:
    print("Error: No valid image pairs found. Check directories and file naming.")
    exit(1)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = AttentionUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Define image transformations
tfm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Process images
for eye_img_path, ulcer_img_path in pairs:
    file_id = get_id(ulcer_img_path)
    if file_id is None:
        continue

    try:
        # Load and preprocess eye image
        eye_img_pil = Image.open(eye_img_path).convert("RGB")
        eye_img_t = tfm(eye_img_pil).unsqueeze(0).to(device)

        # Load and preprocess ulcer image
        ulcer_img_pil = Image.open(ulcer_img_path).convert("RGB")
        ulcer_img_t = tfm(ulcer_img_pil).unsqueeze(0).to(device)

        # Get cornea mask using the model
        with torch.no_grad():
            pred = torch.sigmoid(model(eye_img_t))

        # Binarize the predicted mask
        pred_bin = (pred.cpu().squeeze().numpy() > 0.5).astype(np.uint8)
        cornea_mask = pred_bin * 255

        # Convert ulcer image to numpy
        ulcer_img_np = (ulcer_img_t.cpu().squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Perform AND operation
        cornea_result = cv2.bitwise_and(ulcer_img_np, ulcer_img_np, mask=cornea_mask)

        # Save the result
        out_path = output_dir / f"{file_id}.png"
        Image.fromarray(cornea_result).save(out_path)
        print(f"Saved result for ID {file_id} to {out_path}")

    except Exception as e:
        print(f"Error processing ID {file_id}: {e}")

print(f"Done! Results saved in {output_dir}")