# Original und erster flip vertikal flippen, gesamt zu Set 1

import os
from PIL import Image

# Basis: Script liegt im data-Ordner
BASE_PATH = os.getcwd()

SOURCE_DIR = os.path.join(BASE_PATH, "CompleteDataset")
TARGET_DIR = os.path.join(BASE_PATH, "flipped_vertical")

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

print("Quelle:", SOURCE_DIR)
print("Ziel:", TARGET_DIR)

# Zielordner erstellen
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

            # ✅ VERTIKALES SPIEGELN (oben ↔ unten)
            flipped = img.transpose(Image.FLIP_TOP_BOTTOM)

            name, ext = os.path.splitext(file)
            output_path = os.path.join(target_path, f"{name}_vflip{ext}")

            flipped.save(output_path)

        except Exception as e:
            print(f"Fehler bei {file}: {e}")

print("\nFertig: Vertikal gespiegelte Bilder erstellt ✅")