import os
import random
import shutil

BASE_PATH = os.getcwd()

SOURCE_BASE = BASE_PATH
REST_BASE = os.path.join(BASE_PATH, "dataset_rest")

folders = ["Sonder", "Rest", "Papier", "Plastik", "Bio", "Glas"]
N = 1500

random.seed(42)

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

os.makedirs(REST_BASE, exist_ok=True)

for folder in folders:
    source_path = os.path.join(SOURCE_BASE, folder)
    rest_path = os.path.join(REST_BASE, folder)

    print(f"\nVerarbeite: {folder}")

    if not os.path.exists(source_path):
        print(f"Fehler: {folder} existiert nicht")
        continue

    os.makedirs(rest_path, exist_ok=True)

    files = [
        f for f in os.listdir(source_path)
        if os.path.isfile(os.path.join(source_path, f))
        and f.lower().endswith(extensions)
    ]

    print(f"{folder}: {len(files)} Bilder gefunden")

    # gleiche Auswahl wie im Originalskript
    selected_files = files if len(files) <= N else random.sample(files, N)

    # nur Rest
    rest_files = list(set(files) - set(selected_files))

    print(f"{folder}: {len(rest_files)} Restbilder")

    for file in rest_files:
        src = os.path.join(source_path, file)
        dst = os.path.join(rest_path, file)
        shutil.copy2(src, dst)

print("\nFertig ✅")