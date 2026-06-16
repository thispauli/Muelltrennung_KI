import os
import json
import copy
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# =========================
# Pfade
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # src/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))    # Projektwurzel
DATA_DIR = os.path.join(PROJECT_ROOT, "Training")               # Training/
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")               # models/

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(MODELS_DIR, "last_model.pth")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# =========================
# Parameter
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001   # kleiner als beim Initialtraining
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
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Training-Ordner nicht gefunden: {DATA_DIR}")

full_dataset = datasets.ImageFolder(DATA_DIR)

current_class_names = full_dataset.classes
num_classes = len(current_class_names)

print("Gefundene Klassen im Training-Ordner:", current_class_names)
print("Anzahl Klassen:", num_classes)
print("Gesamtzahl Bilder:", len(full_dataset))

# =========================
# Klassenprüfung
# =========================
if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"classes.json nicht gefunden: {CLASSES_PATH}")

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    saved_class_names = json.load(f)

print("Gespeicherte Klassen aus classes.json:", saved_class_names)

if current_class_names != saved_class_names:
    raise ValueError(
        "Die Klassen im Training-Ordner stimmen nicht mit classes.json überein.\n"
        f"Aktuell: {current_class_names}\n"
        f"Gespeichert: {saved_class_names}"
    )

# =========================
# Split in Train / Validation
# =========================
train_size = int(TRAIN_SPLIT * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

# random_split teilt nur Indizes, daher separates Setzen der Transform
train_dataset.dataset = copy.deepcopy(full_dataset)
val_dataset.dataset = copy.deepcopy(full_dataset)

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train-Bilder: {len(train_dataset)}")
print(f"Val-Bilder:   {len(val_dataset)}")

# =========================
# Modell laden
# =========================
if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(f"best_model.pth nicht gefunden: {BEST_MODEL_PATH}")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)

print(f"Bestehendes Modell geladen: {BEST_MODEL_PATH}")

# =========================
# Loss / Optimizer / Scheduler
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# =========================
# Evaluate
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
# Optional: Startwert auf Validation prüfen
# =========================
initial_val_loss, initial_val_acc = evaluate(model, val_loader)
print(f"Start-Validation | Loss: {initial_val_loss:.4f} | Acc: {initial_val_acc:.4f}")

best_val_acc = initial_val_acc

continued_best_model_path = os.path.join(MODELS_DIR, "continued_best_model.pth")
continued_last_model_path = os.path.join(MODELS_DIR, "continued_last_model.pth")

# =========================
# Weitertraining
# =========================
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

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), continued_best_model_path)
        print(f"Neues bestes weitertrainiertes Modell gespeichert: {continued_best_model_path}")

    scheduler.step()

# letztes weitertrainiertes Modell speichern
torch.save(model.state_dict(), continued_last_model_path)

print("\nWeitertraining abgeschlossen.")
print(f"Weitertrainiertes bestes Modell: {continued_best_model_path}")
print(f"Weitertrainiertes letztes Modell: {continued_last_model_path}")
print(f"Beste Validierungs-Accuracy: {best_val_acc:.4f}")
print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
end = datetime.now()
print("Dauer:", end - start)
