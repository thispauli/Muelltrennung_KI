# Vorgehen nach CRISP-DM

## 1. Business Understanding

### Ziel
- Entwicklung eines Modells zur automatischen Klassifikation von Müllarten  
  (Bio, Glas, Papier, Plastik, Rest, Sonder)

### Anforderungen
- möglichst hohe Klassifikationsgenauigkeit  
- robust gegenüber unterschiedlichen Bildbedingungen  
- Erkennung auch bei variierenden Perspektiven und Objekten  

---

## 2. Data Understanding

### Ausgangsdaten
- Bilder nach Klassen sortiert (Ordnerstruktur)
- unterschiedliche Bildqualitäten und Perspektiven

### Beobachtungen
- Klassen waren anfangs ungleich verteilt  
  → z. B. deutlich mehr Papierbilder als andere Klassen  

### Erkenntnisse
- Ungleichgewicht der Daten führt zu schlechterer Modellleistung  
- Modell wurde bei mehreren Objekten im Bild ungenauer  

---

## 3. Data Preparation

### Datenbereinigung
- Auswahl von Bildern pro Klasse  
- Vereinheitlichung der Struktur  

### Balancing
- Reduktion auf **1500 Bilder pro Klasse**  
- Ziel: gleichverteilte Datenbasis  

### Augmentation
Eingesetzte Methoden:
- horizontales Spiegeln  
- vertikales Spiegeln  
- Rotation  
- Helligkeitsanpassung  
- Rauschen (Noise)  

### Erkenntnisse
- Augmentation verbessert Robustheit  
- Kombination aus originalen und augmentierten Daten führt zu besseren Ergebnissen  

---

## 4. Modeling

### Modellwahl
- ResNet18 (vortrainiert auf ImageNet)

### Vorgehen
- Transfer Learning (Fine-Tuning der letzten Schicht)  
- Training mit:
  - Datenaugmentation  
  - balancierten Klassen  

### Experimente
1. **Ungleich verteilte Daten (z. B. mehr Papier)**
   - schlechtere Ergebnisse  
   - Modell bias-lastig  

2. **Gleich verteilte Daten (1500 pro Klasse)**
   - bessere Basisleistung  
   - stabilere Klassifikation  

3. **Mit Augmentation**
   - bessere Generalisierung  
   - verbesserte Erkennung bei mehreren Objekten  

---

## 5. Evaluation

### Bewertungsmethoden
- Accuracy (Train und Validation)  
- Vergleich verschiedener Trainingsläufe  

### Beobachtungen
- Ungleichgewicht der Klassen verschlechtert Ergebnisse  
- Modell erkennt einzelne Objekte besser als Szenen mit mehreren Objekten  
- Augmentierte Daten verbessern die Robustheit deutlich  

### Zentrale Erkenntnis
- **Datenqualität und -verteilung haben größeren Einfluss als Modellwahl**  

---

## 6. Deployment / Anwendung

### Nutzung
- Klassifikation einzelner Bilder über `app.py`  
- Ausgabe:
  - Kategorie  
  - Confidence  
  - visuelle Erklärung (Heatmap / Grad-CAM)  

### Erweiterungen
- Anwendung auf neue Datensätze zur Evaluation  
- Visualisierung der Entscheidungsbereiche im Bild  

---

## Gesamtfazit

- Balancierte Daten sind entscheidend für stabile Modelle  
- Augmentation verbessert die Generalisierung  
- Mehr Daten allein (z. B. Papier) führt nicht automatisch zu besseren Ergebnissen  
- Kombination aus:
  - gleichverteilten Daten  
  - Augmentation  
führt zu den besten Resultaten  

- Modellleistung sinkt bei:
  - unbalancierten Datensätzen  
  - Bildern mit mehreren Objekten  

- Visuelle Erklärbarkeit (Heatmaps) hilft bei der Analyse von Fehlklassifikationen  