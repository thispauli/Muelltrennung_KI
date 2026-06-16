import os
import random
import shutil

BASE_PATH = os.getcwd()

SOURCE_BASE = BASE_PATH
TARGET_BASE = os.path.join(BASE_PATH, "dataset")

folders = ["Sonder", "Rest", "Papier", "Plastik", "Bio", "Glas"]
N = 1500

random.seed(42)

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

print("Arbeitsverzeichnis:", BASE_PATH)

# WICHTIG: dataset IMMER vorher erstellen
if not os.path.exists(TARGET_BASE):
    os.makedirs(TARGET_BASE)
    print("dataset-Ordner erstellt")

for folder in folders:
    source_path = os.path.join(SOURCE_BASE, folder)
    target_path = os.path.join(TARGET_BASE, folder)

    print(f"\nVerarbeite: {folder}")

    if not os.path.exists(source_path):
        print(f"Fehler: {folder} existiert nicht")
        continue

    # Zielordner erstellen
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    files = [
        f for f in os.listdir(source_path)
        if os.path.isfile(os.path.join(source_path, f))
        and f.lower().endswith(extensions)
    ]

    print(f"{folder}: {len(files)} Bilder gefunden")

    selected_files = files if len(files) <= N else random.sample(files, N)

    for file in selected_files:
        src = os.path.join(source_path, file)
        dst = os.path.join(target_path, file)

        shutil.copy2(src, dst)

    print(f"{folder}: {len(selected_files)} kopiert")

print("\nFertig")