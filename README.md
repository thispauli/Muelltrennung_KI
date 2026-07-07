# 🗑️ Mülltrennung KI

Universitätsprojekt zur automatisierten Klassifikation von Müllarten mittels Deep Learning (CNN – ResNet18).

## Klassen

Das Modell unterscheidet **6 Müllkategorien**:

| Klasse | Beschreibung |
|--------|-------------|
| **Bio** | Biotonne (organische Abfälle) |
| **Glas** | Glasverpackungen |
| **Papier** | Papier & Pappe |
| **Plastik** | Plastikverpackungen |
| **Rest** | Restmüll |
| **Sonder** | Sondermüll |

---

## Verwendete Datensätze (Original)

Die Bilddaten stammen aus folgenden Kaggle-Datensätzen:

- [RealWaste](https://www.kaggle.com/datasets/joebeachcapital/realwaste)
- [Garbage Classification (1)](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- [Garbage Classification (2)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- [New Trash Classification Dataset](https://www.kaggle.com/datasets/glhdamar/new-trash-classfication-dataset)

---

## Datenaugmentierung

Aus den Originaldatensätzen wurde ein eigener Trainingsdatensatz erstellt. Die Skripte im Ordner `Bilder-Skripte/` wenden folgende Transformationen an:

### Pipeline

1. **Zufallsauswahl** (`getRandom.py`) – Aus den zusammengesetzten Originaldaten werden pro Klasse eine zufällige Teilmenge ausgewählt.
2. **Horizontaler Flip** (`flipping.py`) – Jedes ausgewählte Bild wird horizontal gespiegelt.
3. **Vertikaler Flip** (`vflip.py`) – Original + horizontaler Flip werden vertikal gespiegelt.

**→ Zusammen = Set 1**

4. **Rotation** (`rotate.py`) – Set 1 wird um 45° rotiert.
5. **Helligkeit / Noise** (`brightNoise.py`) – Set 1 wird mit variierter Helligkeit (dunkler/heller) und Rauschen augmentiert.

**→ Alles zusammen = 20 Varianten pro Originalbild**

### Ergebnis

- **~9.000 Bilder** – Erster Trainingsdatensatz
- **~180.000 Bilder** – Erweiterter Trainingsdatensatz (20-fache Augmentierung)

---

## Projektstruktur

```
├── Bilder-Skripte/        # Datenaugmentierungsskripte
│   ├── getRandom.py       # Zufallsauswahl aus Originaldaten
│   ├── flipping.py        # Horizontaler Flip
│   ├── vflip.py           # Vertikaler Flip
│   ├── rotate.py          # Rotation um 45°
│   └── brightNoise.py     # Helligkeit + Noise Variation
├── data/                  # Trainingsdaten (klassenbasiert)
├── test_data/             # Testdaten für Evaluation
├── models/                # Gespeicherte Modellgewichte
│   ├── best_model.pth     # Bestes Modell (9k Bilder)
│   ├── best_model_20x.pth # Bestes Modell (180k Bilder, 20x augmentiert)
│   ├── last_model.pth     # Letztes Modell (9k Bilder)
│   └── classes.json       # Klassenmapping
├── outputs/               # Grad-CAM Visualisierungen
├── src/                   # Quellcode
│   ├── app.py             # Gradio-UI (Web-Interface mit Admin-Bereich)
│   ├── train.py           # Erstes Training vom Grundmodell
│   ├── train_continue.py  # Weitertraining mit Korrekturen (Admin-Bereich)
│   └── evaluate.py        # Evaluation mit Testdaten
└── backup/                # Alte Versionen
```

---

## Modelle

| Datei | Beschreibung | Trainingsgrundlage |
|-------|-------------|-------------------|
| `best_model.pth` | Bestes Modell – wird nur bei höherer Validierungs-Accuracy überschrieben | ~9.000 Bilder (unser erster Datensatz) |
| `best_model_20x.pth` | Bestes Modell nach Training mit 20x augmentierten Daten | ~180.000 Bilder (20-fache Augmentierung) |
| `last_model.pth` | Letztes Modell des Trainingslaufs mit 9k Bildern | ~9.000 Bilder |
| `classes.json` | Klassen-IDs → Namen (Bio, Glas, Papier, Plastik, Rest, Sonder) | – |

---

## Modellarchitektur

**ResNet18** (Pre-trained auf ImageNet, Fine-Tuning)

- Convolutional Neural Network (CNN) für Bilddaten
- Residual Connections gegen Vanishing Gradient
- Transfer Learning: vortrainiert auf ImageNet, letzte Schicht angepasst
- Gutes Verhältnis aus Leistung und Rechenaufwand

→ Details in [`model-warum.md`](model-warum.md)

---

## Schnellstart

### Voraussetzungen

- Python 3.x
- PyTorch, torchvision, gradio, Pillow, matplotlib, numpy

### Training

```bash
# Erstes Training (vom Grundmodell)
python src/train.py

# Weitertraining mit Korrekturen aus dem Admin-Bereich
python src/train_continue.py
```

### UI starten

```bash
python src/app.py
```

Startet ein Gradio-Web-Interface mit:

- **Normaler Modus:** Bild hochladen → Klassifikation + Grad-CAM Visualisierung
- **Admin-Bereich:** Falsch klassifizierte Bilder korrigieren (Bilder werden im `Training/`-Ordner gespeichert und können für das Weitertraining verwendet werden)

### Evaluation

```bash
python src/evaluate.py
```

Testet das Modell mit den Daten im `test_data/`-Ordner und gibt Accuracy, Precision, Recall und Confusion Matrix aus, ohne weiterzutrainieren.

---

## Trainingsverlauf (180k Bilder)

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1/8 | 86.94% | 90.16% |
| 2/8 | 93.85% | 96.54% |
| 3/8 | 95.83% | 96.79% |
| 4/8 | 98.53% | 98.52% |
| 5/8 | 98.80% | 99.04% |
| 6/8 | 98.97% | 98.86% |
| 7/8 | 99.56% | 99.41% |
| 8/8 | 99.64% | **99.54%** |

**Gesamtdauer:** ~1 Tag 7 Stunden (CPU)

---

## Lizenz

Universitätsprojekt – kein kommerzieller Einsatz.
