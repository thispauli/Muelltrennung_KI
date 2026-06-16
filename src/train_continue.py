import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from datetime import datetime
import shutil

# =========================
# Pfade
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))   
TRAINING_DIR = os.path.join(PROJECT_ROOT, "Training")          # <-- Hier liegen die neuen Bilder
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")              

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
BACKUP_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_backup.pth")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")

# =========================
# Parameter fürs Fine-Tuning
# =========================
IMG_SIZE = 224
BATCH_SIZE = 16          # Kleinerer Batch, da wir evtl. wenig neue Bilder haben
EPOCHS = 5               # Weniger Epochen reichen meist beim Weitertrainieren
LEARNING_RATE = 0.0001   # WICHTIG: Kleinere Lernrate, damit altes Wissen nicht zerstört wird
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- Starte Fine-Tuning ---")
print(f"Device: {device}")
start = datetime.now()

# =========================
# Klassen sichern & Ordner vorbereiten
# =========================
# WICHTIG: PyTorch ImageFolder vergibt IDs basierend auf alphabetisch sortierten Ordnern.
# Wenn im 'Training' Ordner noch nicht für JEDE Klasse ein Bild liegt, verschieben sich die IDs!
# Lösung: Wir laden die originalen Klassen und erstellen leere Ordner.
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    original_classes = json.load(f)

for class_name in original_classes:
    os.makedirs(os.path.join(TRAINING_DIR, class_name), exist_ok=True)

num_classes = len(original_classes)

# =========================
# Datensatz laden
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

try:
    full_dataset = datasets.ImageFolder(TRAINING_DIR)
except Exception as e:
    print(f"Fehler beim Laden des Datensatzes: {e}")
    exit()

if len(full_dataset) == 0:
    print("❌ Keine Bilder im Ordner 'Training' gefunden. Beende Skript.")
    exit()

print(f"Gefundene Klassen: {full_dataset.classes}")
print(f"Anzahl neuer Bilder fürs Training: {len(full_dataset)}")

# Absicherung bei sehr wenigen Bildern (Split funktioniert sonst nicht)
if len(full_dataset) < 5:
    print("⚠️ Sehr wenige Bilder! Verwende alle Bilder für Training UND Validierung (nur zum Testen!).")
    train_dataset = copy.deepcopy(full_dataset)
    val_dataset = copy.deepcopy(full_dataset)
else:
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    # Transforms anwenden
    train_dataset.dataset = copy.deepcopy(full_dataset)
    val_dataset.dataset = copy.deepcopy(full_dataset)

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# Bisheriges Modell laden
# =========================
print(f"Lade bestehendes Modell von: {BEST_MODEL_PATH}")
model = models.resnet18(weights=None) 
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Lade die Gewichte des bereits trainierten Modells
if os.path.exists(BEST_MODEL_PATH):
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
else:
    print("❌ Kein best_model.pth gefunden! Bitte zuerst train.py ausführen.")
    exit()

model = model.to(device)

# =========================
# Loss / Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def evaluate(model, dataloader):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

# =========================
# Training (Fine-Tuning)
# =========================
best_val_acc = 0.0

# Backup des alten Modells erstellen, falls das Fine-Tuning in die Hose geht
shutil.copy(BEST_MODEL_PATH, BACKUP_MODEL_PATH)
print(f"Sicherungskopie des alten Modells erstellt: {BACKUP_MODEL_PATH}")

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

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

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)

print("\n✅ Fine-Tuning abgeschlossen.")
print(f"Das aktualisierte Modell wurde gespeichert unter: {BEST_MODEL_PATH}")
print("Dauer:", datetime.now() - start)