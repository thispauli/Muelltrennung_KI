import os
from PIL import Image

BASE_PATH = os.getcwd()

SOURCE_DIR = os.path.join(BASE_PATH, "CompleteDataset")
TARGET_DIR = os.path.join(BASE_PATH, "rotated_45")

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

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

            # ✅ Rotation um 45°
            rotated = img.rotate(45, expand=True)

            name, ext = os.path.splitext(file)
            output_path = os.path.join(target_path, f"{name}_rot45{ext}")

            rotated.save(output_path)

        except Exception as e:
            print(f"Fehler bei {file}: {e}")

print("\nFertig: 45° gedrehte Bilder erstellt ✅")