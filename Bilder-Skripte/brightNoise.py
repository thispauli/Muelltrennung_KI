# Set 1 Helligkeit / noise

import os
import numpy as np
from PIL import Image, ImageEnhance

# Basis
BASE_PATH = os.getcwd()

SOURCE_DIR = os.path.join(BASE_PATH, "CompleteDataset")
TARGET_DIR = os.path.join(BASE_PATH, "augmented_noise_brightness")

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Parameter
BRIGHTNESS_FACTORS = [0.7, 1.3]   # dunkler / heller
NOISE_LEVEL = 0.05                # Stärke des Noise

print("Quelle:", SOURCE_DIR)
print("Ziel:", TARGET_DIR)

os.makedirs(TARGET_DIR, exist_ok=True)

for folder in os.listdir(SOURCE_DIR):
    source_path = os.path.join(SOURCE_DIR, folder)

    if not os.path.isdir(source_path):
        continue

    target_path = os.path.join(TARGET_DIR, folder)
    os.makedirs(target_path, exist_ok=True)

    print(f"\nVerarbeite: {folder}")

    for file in os.listdir(source_path):
        if not file.lower().endswith(EXTENSIONS):
            continue

        input_path = os.path.join(source_path, file)

        try:
            img = Image.open(input_path).convert("RGB")

            name, ext = os.path.splitext(file)

            # -------------------
            # 1. Helligkeit
            # -------------------
            for factor in BRIGHTNESS_FACTORS:
                enhancer = ImageEnhance.Brightness(img)
                bright_img = enhancer.enhance(factor)

                bright_path = os.path.join(
                    target_path,
                    f"{name}_b{str(factor).replace('.', '')}{ext}"
                )

                bright_img.save(bright_path)

            # -------------------
            # 2. Noise hinzufügen
            # -------------------
            img_np = np.array(img).astype(np.float32) / 255.0

            noise = np.random.normal(0, NOISE_LEVEL, img_np.shape)
            noisy = img_np + noise

            noisy = np.clip(noisy, 0, 1)
            noisy = (noisy * 255).astype(np.uint8)

            noisy_img = Image.fromarray(noisy)

            noise_path = os.path.join(
                target_path,
                f"{name}_noise{ext}"
            )

            noisy_img.save(noise_path)

        except Exception as e:
            print(f"Fehler bei {file}: {e}")

print("\nFertig: Noise + Helligkeit erstellt ✅")