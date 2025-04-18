import os
from PIL import Image
import torch
from transformers import ViTModel, AutoFeatureExtractor
from tqdm import tqdm
import numpy as np

def load_vit(device):
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device).eval()
    extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    return model, extractor

def extract_and_save(model, extractor, img_path, save_root, device, root_dir, mode="cls"):
    img = Image.open(img_path).convert("RGB")
    inputs = extractor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

        relative_path = os.path.relpath(img_path, start=root_dir)

        if mode in ("cls", "both"):
            feature_cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            save_path_cls = os.path.join(save_root, "cls", relative_path).replace(".jpg", ".npy").replace(".png", ".npy")
            os.makedirs(os.path.dirname(save_path_cls), exist_ok=True)
            np.save(save_path_cls, feature_cls)

        if mode in ("mean", "both"):
            feature_mean = outputs.last_hidden_state[:, 1:, :].mean(dim=1).cpu().numpy()
            save_path_mean = os.path.join(save_root, "mean", relative_path).replace(".jpg", ".npy").replace(".png", ".npy")
            os.makedirs(os.path.dirname(save_path_mean), exist_ok=True)
            np.save(save_path_mean, feature_mean)

def run_inference(image_root, save_root, device="cuda", mode="cls"):
    model, extractor = load_vit(device)

    # 모든 이미지 경로 미리 수집
    img_paths = []
    for dirpath, _, filenames in os.walk(image_root):
        for fname in sorted(filenames):
            if fname.lower().endswith((".jpg", ".png")):
                img_paths.append(os.path.join(dirpath, fname))

    # tqdm 적용하여 진행률 표시
    for img_path in tqdm(img_paths, desc="Extracting features"):
        extract_and_save(model, extractor, img_path, save_root, device, image_root, mode)

if __name__ == "__main__":
    IMAGE_ROOT = "data/raw/image"
    SAVE_ROOT = "data/features/image"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    run_inference(IMAGE_ROOT, SAVE_ROOT, DEVICE, mode="both")  # cls, mean, or both