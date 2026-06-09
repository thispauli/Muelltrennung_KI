1. Begründung für das Basismodell (ResNet18)
🔧 Grundlegende Wahl

CNN (Convolutional Neural Network)

speziell für Bilddaten entwickelt
kann automatisch relevante Merkmale aus Bildern extrahieren


ResNet18 = bewährte Standardarchitektur

in vielen realen Anwendungen verwendet
gute Balance zwischen Leistung und Aufwand




🧠 Architektur-Vorteile

Residual Learning (Skip Connections)

vermeidet Vanishing Gradient Problem
ermöglicht stabileres Training


Tiefe: 18 Layer

tief genug für komplexe Muster
aber nicht zu groß → effizient


Feature-Hierarchie

frühe Layer: Kanten, Farben
mittlere Layer: Formen
späte Layer: Objekte (z. B. Flasche, Papierstruktur)




⚡ Praktische Vorteile für dein Projekt

Schnell trainierbar

wichtig ohne GPU


geringer Speicherbedarf
robust gegen Overfitting (im Vergleich zu größeren Modellen)
gute Wahl für:

mittelgroße Datensätze (wie deiner)
viele Klassen (6 Müllarten)




🔁 Transfer Learning (extrem wichtig)

Modell ist vortrainiert auf ImageNet

Millionen Bilder
tausende Klassen


bedeutet:

kennt schon:

Kanten
Texturen
Formen




du machst nur:

Fine-Tuning der letzten Schicht



👉 Vorteil:

viel weniger Trainingsdaten nötig
bessere Ergebnisse bei wenig Daten


📊 Warum nicht größere Modelle?
(z. B. ResNet50, EfficientNet)

brauchen:

mehr Rechenleistung
längere Trainingszeit


höheres Risiko von:

Overfitting


kaum Vorteil bei deinem Setup

👉 ResNet18 = beste Kosten-Nutzen-Wahl

🔍 Warum geeignet für Müllklassifikation?

Müll unterscheidet sich stark über:

Form (Flasche vs Papier)
Textur (Plastik vs Glas)
Struktur (zerknüllt vs glatt)



👉 CNN erkennt genau diese Eigenschaften sehr gut

⚠️ Einschränkungen (ehrlich)

Modell „versteht“ keine Regeln

z. B. verschmutztes Papier → Problem


basiert nur auf:

visuellen Mustern


anfällig für:

irrelevanten Hintergrund




✅ Fazit (kompakt)
ResNet18 ist gewählt, weil es:

effizient und stabil trainiert
gute Generalisierung liefert
Transfer Learning nutzt
optimal für mittelgroße Bilddatensätze ist