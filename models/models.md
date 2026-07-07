# Modellübersicht

## Ziel

Diese Datei dokumentiert die verwendeten Modellartefakte des Mülltrennungs-KI-Projekts sowie deren Herkunft, Trainingsgrundlage und Verwendungszweck.

---

## best_model.pth

**Beschreibung:**
Das beim Training mit ~180.000 Bildern beste erzielte Modell. Wird nur bei höherer Validierungs-Accuracy überschrieben.

**Trainingsgrundlage:**

- Eigener Datensatz mit ca. **180.000 Bildern** (aus Original-Kaggle-Datensätzen zusammengestellt)
- 6 Klassen: Bio, Glas, Papier, Plastik, Rest, Sonder
- ResNet18, Fine-Tuning auf ImageNet-Pretrained-Gewichten
- 8 Epochen, Batch Size 32, Lernrate 0.0005

**Ergebnis:**

- Beste Validierungs-Accuracy: **99.54%** (Epoch 8)
- Training auf CPU, Dauer: ~1 Tag 7 Stunden

**Verwendung:**

- Standardmodell in der Gradio-UI (`app.py`)
- Referenzmodell für Vergleiche

**Aktualisierung:**

- Wird ausschließlich überschrieben, wenn ein neuer Trainingslauf eine höhere Validation Accuracy erzielt

---

## best_model_20x.pth

**Beschreibung:**
Das beste Modell nach dem Training mit dem 20-fach augmentierten Datensatz (~180.000 Bilder).

**Trainingsgrundlage:**

- Erweiterter Datensatz mit ca. **180.000 Bildern** (20-fache Augmentierung der ~9.000 Basisbilder)
- Augmentierung: Horizontal Flip, Vertikal Flip, Rotation 45°, Helligkeitsvariation, Noise
- 6 Klassen: Bio, Glas, Papier, Plastik, Rest, Sonder
- ResNet18, Fine-Tuning auf ImageNet-Pretrained-Gewichten
- 8 Epochen, Batch Size 32, Lernrate 0.0005

**Ergebnis:**

- Beste Validierungs-Accuracy: **99.54%** (Epoch 8)
- Training auf CPU, Dauer: ~1 Tag 7 Stunden

**Verwendung:**

- Vergleich und Zwischenspeicherung

---

## last_model.pth

**Beschreibung:**
Das letzte Modell des Trainingslaufs mit ~9.000 Bildern (nicht unbedingt das beste).

**Trainingsgrundlage:**

- Eigener Datensatz mit ca. **9.000 Bildern**
- Gleiche Konfiguration wie `best_model.pth`
- Speichert den Zustand nach der letzten Epoche (unabhängig von der Accuracy)

**Verwendung:**

- Nachvollziehbarkeit des letzten Trainingslaufs
- Vergleich zwischen bestem und letztem Modell
- Analyse der Auswirkungen der Datensatzskalierung

**Aktualisierung:**

- Wird nicht automatisch aktualisiert
- Statisch – entspricht dem Stand des Trainings mit 9k Bildern

---

## model.pth

**Beschreibung:**
Altbestand – älteres Modell aus einem früheren Trainingslauf, vor der Einführung der aktuellen Datensätze (9k / 180k Bilder).

**Trainingsgrundlage:**

- Unbekannte/ältere Trainingsdaten (vor der eigenen Datenaugmentierung)
- Nicht mehr aktiv verwendet

**Verwendung:**

- Aktuell keine – wird von keinem Skript referenziert
- Kann bei Bedarf gelöscht werden

---

## classes.json

**Inhalt:**

```json
["Bio", "Glas", "Papier", "Plastik", "Rest", "Sonder"]
```

**Verwendung:**

- Mapping der Klassen-IDs (0–5) zu den sprechenden Klassennamen
- Wird von allen Skripten (`app.py`, `train.py`, `evaluate.py`) verwendet

---

## Zusammenfassung

| Datei | Trainingsgrundlage | Status |
|-------|-------------------|--------|
| `best_model.pth` | ~180.000 Bilder (20x augmentiert) | ✅ Aktiv – Standardmodell in der UI |
| `best_model_20x.pth` | ~180.000 Bilder (20x augmentiert) | ⚠️ Statisch – Vergleich/Zwischenspeicherung |
| `last_model.pth` | ~9.000 Bilder | ⚠️ Statisch – letzter Stand des 9k-Trainings |
| `model.pth` | Ältere Trainingsdaten (vor Datenaugmentierung) | ❌ Altbestand – nicht mehr verwendet |
| `classes.json` | Klassenmapping | ✅ Aktiv |