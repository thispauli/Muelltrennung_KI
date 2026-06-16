import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from datetime import datetime

# =========================
# Pfade
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # src/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))   # Projektwurzel
DATA_DIR = os.path.join(PROJECT_ROOT, "data")                  # data/
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")              # models/

os.makedirs(MODELS_DIR, exist_ok=True)

# =========================
# Parameter
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.0005
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)

# =========================
# Gerät
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
start = datetime.now()
# =========================
# Transformationen
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# =========================
# Datensatz laden
# =========================
# Achtung: ImageFolder erwartet direkt Klassenordner unter data/
full_dataset = datasets.ImageFolder(DATA_DIR)

class_names = full_dataset.classes
num_classes = len(class_names)

print("Gefundene Klassen:", class_names)
print("Anzahl Klassen:", num_classes)
print("Gesamtzahl Bilder:", len(full_dataset))

# Split in Train / Validation
train_size = int(TRAIN_SPLIT * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

# WICHTIG:
# random_split teilt nur Indizes, deshalb Transform separat setzen
train_dataset.dataset = copy.deepcopy(full_dataset)
val_dataset.dataset = copy.deepcopy(full_dataset)

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train-Bilder: {len(train_dataset)}")
print(f"Val-Bilder:   {len(val_dataset)}")

# =========================
# Modell
# =========================
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Letzte Schicht ersetzen
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# =========================
# Loss / Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Optional einfacher Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# =========================
# Hilfsfunktion Accuracy
# =========================
def evaluate(model, dataloader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# =========================
# Training
# =========================
best_val_acc = 0.0
best_model_path = os.path.join(MODELS_DIR, "best_model.pth")
last_model_path = os.path.join(MODELS_DIR, "last_model.pth")
classes_path = os.path.join(MODELS_DIR, "classes.json")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    val_loss, val_acc = evaluate(model, val_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # bestes Modell speichern
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"Bestes Modell gespeichert: {best_model_path}")

    scheduler.step()

# letztes Modell speichern
torch.save(model.state_dict(), last_model_path)

# Klassennamen speichern
with open(classes_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print("\nTraining abgeschlossen.")
print(f"Bestes Modell: {best_model_path}")
print(f"Letztes Modell: {last_model_path}")
print(f"Klassennamen: {classes_path}")
print(f"Beste Validierungs-Accuracy: {best_val_acc:.4f}")
print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
end = datetime.now()
print("Dauer:", end - start)