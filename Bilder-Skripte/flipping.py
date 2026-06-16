import os
from PIL import Image

# Basis: dein DATA-Ordner (Script liegt ja dort)
BASE_PATH = os.getcwd()

SOURCE_DIR = os.path.join(BASE_PATH, "dataset")
TARGET_DIR = os.path.join(BASE_PATH, "flipped")

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

            # SPIEGELN
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)

            name, ext = os.path.splitext(file)
            output_path = os.path.join(target_path, f"{name}_flip{ext}")

            flipped.save(output_path)

        except Exception as e:
            print(f"Fehler bei {file}: {e}")

print("\nFertig: Nur gespiegelte Bilder erstellt ✅")