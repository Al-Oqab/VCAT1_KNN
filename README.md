# VCAT1 - Übung 1: Bildklassifikation mit K-Nearest-Neighbor

Dieses Projekt implementiert eine Bildklassifikation auf dem CIFAR-10 Datensatz unter Verwendung des K-Nearest-Neighbor (KNN) Algorithmus.

## Voraussetzungen
- Python 3.x
- NumPy (`pip install numpy`)
- Matplotlib (`pip install matplotlib`)
- CIFAR-10 Datensatz: [Download hier](https://www.cs.toronto.edu/~kriz/cifar.html)

## Projektstruktur
- `main.py`: Enthält den gesamten ausführbaren Code.
- `cifar-10-batches-py/`: Der Ordner, der die extrahierten CIFAR-10 Batch-Dateien enthalten muss.

## Ausführung
1. Stellen Sie sicher, dass der CIFAR-10 Datensatz im Hauptverzeichnis entpackt ist.
2. Führen Sie das Skript mit folgendem Befehl aus:
   ```bash
   python main.py
3. Das Skript lädt die Daten, zeigt ein Beispielbild und berechnet KNN für die ersten 10 Bilder.
4. Die Ausgabe zeigt die vorhergesagten Labels für die ersten 10 Bilder im Testset.
5. Die Ausgabe enthält auch die Zeit, die für die Berechnung der KNN-Vorhersagen benötigt wurde.
6. Nach dem Schließen der ersten Visualisierung startet automatisch die Evaluation aller 10.000 Testbilder (dies kann je nach Hardware einige Minuten dauern).