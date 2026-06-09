# Begründung für das Basismodell (ResNet18)

## Grundlegende Wahl

### Convolutional Neural Network (CNN)
- speziell für Bilddaten entwickelt  
- extrahiert automatisch relevante visuelle Merkmale  

### ResNet18
- bewährte Standardarchitektur  
- in vielen Anwendungen eingesetzt  
- gute Balance zwischen Leistung und Rechenaufwand  

---

## Architektur-Vorteile

### Residual Learning (Skip Connections)
- reduziert das Vanishing-Gradient-Problem  
- ermöglicht stabiles und tiefes Training  

### Tiefe: 18 Layer
- ausreichend für komplexe Muster  
- gleichzeitig effizient und schnell trainierbar  

### Feature-Hierarchie
- frühe Layer: Kanten und Farben  
- mittlere Layer: Formen  
- späte Layer: Objekte und Strukturen (z. B. Flaschen, Papier)  

---

## Praktische Vorteile für das Projekt

- schnelle Trainingszeiten (auch ohne GPU)  
- geringer Speicherbedarf  
- robust gegenüber Overfitting  

**Geeignet für:**
- mittelgroße Datensätze  
- mehrere Klassen (z. B. Müllarten)  

---

## Transfer Learning

### Vortraining auf ImageNet
- Millionen Bilder  
- tausende Klassen  

**Gelerntes Wissen:**
- Kanten  
- Texturen  
- Formen  

### Vorgehen
- Anpassung der letzten Schicht (Fine-Tuning)  

### Vorteile
- weniger Trainingsdaten erforderlich  
- bessere Ergebnisse bei begrenzten Daten  

---

## Abgrenzung zu größeren Modellen

Größere Modelle (z. B. ResNet50, EfficientNet):
- höherer Rechenaufwand  
- längere Trainingszeit  
- erhöhtes Overfitting-Risiko  

**Einordnung:**
- kein signifikanter Mehrwert im aktuellen Setup  

**Fazit:**  
ResNet18 bietet das beste Verhältnis aus Aufwand und Nutzen  

---

## Eignung für Müllklassifikation

Relevante visuelle Unterschiede:
- Form (z. B. Flasche vs. Papier)  
- Textur (z. B. Plastik vs. Glas)  
- Struktur (z. B. glatt vs. zerknüllt)  

→ CNN kann diese Merkmale zuverlässig erkennen  

---

## Einschränkungen

- kein Verständnis von Regeln oder Kontext  
- basiert ausschließlich auf visuellen Mustern  
- anfällig für irrelevanten Hintergrund  

---

## Fazit

ResNet18 ist geeignet, weil es:
- effizient und stabil trainiert  
- gute Generalisierung liefert  
- Transfer Learning nutzt  
- ideal für mittelgroße Bilddatensätze ist  

---
---

# Datenaugmentation

## Helligkeit

**Ursachen für Variation:**
- unterschiedliche Lichtverhältnisse (Tag/Nacht)  
- Schatten  
- Innen- vs. Außenaufnahmen  

**Effekt:**
- höhere Robustheit gegenüber Beleuchtung  

---

## Noise

**Simuliert:**
- geringe Kameraqualität  
- Kompressionsartefakte  
- Unschärfe  

**Effekt:**
- stabilere Vorhersagen bei variierender Bildqualität  