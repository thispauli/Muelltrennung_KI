import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

# =========================
# Pfade
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "test_data")  # DEIN TESTSET
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
CLASSES_PATH = os.path.join(PROJECT_ROOT, "models", "classes.json")

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Klassen laden
# =========================
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# =========================
# Transformation
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset laden
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# =========================
# Modell laden
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

# =========================
# Evaluation
# =========================
correct = 0
total = 0

class_correct = [0] * len(class_names)
class_total = [0] * len(class_names)

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for i in range(len(labels)):
            label = labels[i]
            pred = preds[i]

            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

# =========================
# Ergebnisse
# =========================
accuracy = correct / total
print(f"\nGesamt-Accuracy: {accuracy:.4f}\n")

print("Klassen-Accuracy:")
for i, name in enumerate(class_names):
    if class_total[i] > 0:
        acc = class_correct[i] / class_total[i]
        print(f"{name}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")
    else:
        print(f"{name}: keine Testdaten")
