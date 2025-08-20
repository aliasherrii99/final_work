import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from attention_unet import AttentionUNet

# مسیر تصاویر ورودی
images_dir = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\train_together\images")

# مسیر مدل ذخیره‌شده
model_path = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\segment\attention_unet_fold_2.pth")

# مسیر ذخیره خروجی
output_dir = Path(r"C:\Users\Meta Pc\PycharmProjects\pythonProject\myproject\payanName\my_work\train_together\AND_result")
output_dir.mkdir(parents=True, exist_ok=True)

# لود مدل
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionUNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# تغییر سایز و تبدیل به تنسور
tfm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# پردازش تمام تصاویر
for img_path in sorted(images_dir.glob("*.*")):
    file_id = img_path.stem

    # لود و پیش‌پردازش تصویر
    img_pil = Image.open(img_path).convert("RGB")
    img_t = tfm(img_pil).unsqueeze(0).to(device)

    # پیش‌بینی ماسک
    with torch.no_grad():
        pred = torch.sigmoid(model(img_t))

    # باینری کردن ماسک
    pred_bin = (pred.cpu().squeeze().numpy() > 0.5).astype(np.uint8)
    inv_mask = (1 - pred_bin) * 255

    # اعمال AND روی تصویر اصلی
    img_np = (img_t.cpu().squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    and_res = cv2.bitwise_and(img_np, img_np, mask=inv_mask)

    # ذخیره خروجی
    out_path = output_dir / f"{file_id}.png"
    Image.fromarray(and_res).save(out_path)

print(f"Done! Results saved in {output_dir}")
