import os
import re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from attention_unet import AttentionUNet

# Define paths
images_dir = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\classification_level2\images")
labels_dir = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\classification_level2\ulcer_images")
model_path = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\classification_level2\attention_unet_ulcer.pth")
output_dir = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\classification_level2\segmented_ulcer")
output_dir.mkdir(parents=True, exist_ok=True)

# Function to extract ID from file name
def get_id(p: Path) -> int:
    return int(re.findall(r'\d+', p.stem)[0])

# Create pairs of images and masks based on IDs
mask_map = {get_id(p): p for p in labels_dir.glob("*.*")}

pairs = []
for img_path in sorted(images_dir.glob("*.*"), key=get_id):
    fid = get_id(img_path)
    if fid in mask_map:
        pairs.append((img_path, mask_map[fid]))

# Unzip pairs into lists
images_list, masks_list = zip(*pairs)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionUNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define image transformations
tfm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Process all images
for idx in range(len(images_list)):
    img_path = Path(images_list[idx])
    mask_path = Path(masks_list[idx])
    file_id = get_id(img_path)

    # Load and preprocess image
    img_pil = Image.open(img_path).convert("RGB")
    img_t = tfm(img_pil).unsqueeze(0).to(device)

    # Get predicted mask
    with torch.no_grad():
        pred = torch.sigmoid(model(img_t))

    # Binarize the predicted mask
    pred_bin = (pred.cpu().squeeze().numpy() > 0.5).astype(np.uint8)
    inv_mask = (1 - pred_bin) * 255

    # Perform AND operation
    img_np = (img_t.cpu().squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    and_res = cv2.bitwise_and(img_np, img_np, mask=inv_mask)

    # Save the result
    out_path = output_dir / f"{file_id}.png"
    Image.fromarray(and_res).save(out_path)

print(f"Done! Results saved in {output_dir}")