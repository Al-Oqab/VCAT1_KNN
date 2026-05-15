import pickle
import numpy as np
import matplotlib.pyplot as plt


# Hilfsfunktion zum Laden der CIFAR-10 Batch-Dateien
def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch[b'data'], np.array(batch[b'labels'])


# 1a) Trainingsdaten einladen und in die Variable Xtr schreiben
Xtr_list = []
Ytr_labels_list = []  # Wir speichern die Trainings-Labels, da wir sie später für KNN benötigen

# Alle Batches (1 bis 5) durchlaufen und in die Liste einfügen
for i in range(1, 6):
    file = f'cifar-10-batches-py/data_batch_{i}'
    X_data, Y_labels = load_cifar_batch(file)
    Xtr_list.append(X_data)
    Ytr_labels_list.append(Y_labels)

# Alle Batches in eine einzige Variable 'Xtr' zusammenfügen
Xtr = np.concatenate(Xtr_list)
Y_train_labels = np.concatenate(Ytr_labels_list)

# 1b) Testdaten einladen und in die Variable Y schreiben
Y_test_data, Y_test_labels = load_cifar_batch('cifar-10-batches-py/test_batch')
Y = Y_test_data

print("Daten erfolgreich geladen!")
print(f"Shape von Xtr (Trainingsdaten): {Xtr.shape}")
print(f"Shape von Y (Testdaten): {Y.shape}")


# 1c) Funktion, die ein Bild der Größe 32x32x3 aus einem 3.072 Vektor erstellt und visualisiert
def visualize_image(image_vector):
    # Umwandlung von (3072,) zu (3, 32, 32) und Rotation (Transponieren) zu (32, 32, 3)
    img = image_vector.reshape(3, 32, 32).transpose(1, 2, 0)

    # Bild mit matplotlib anzeigen
    plt.imshow(img)
    plt.title("CIFAR-10 Bild")
    plt.axis('off')  # Achsen ausblenden
    plt.show()


# Testaufruf für das erste Trainingsbild
print("Erstes Bild wird angezeigt...")
visualize_image(Xtr[0])


# --- 2. Algorithmus ---

# 2a) K-Nearest-Neighbor Funktion mit L1-Distanzmaß
def predict_knn_l1(X_train, Y_train_labels, x_test_image, k):
    # L1-Distanz berechnen: Summe der absoluten Differenzen zwischen dem Testbild und allen Trainingsbildern
    # np.abs() berechnet den absoluten Wert, np.sum(..., axis=1) summiert die Zeilen
    distances = np.sum(np.abs(X_train - x_test_image), axis=1)

    # Indizes der k kleinsten Distanzen finden (die k nächsten Nachbarn)
    min_indices = np.argsort(distances)[:k]

    # Labels der k nächsten Nachbarn holen
    k_nearest_labels = Y_train_labels[min_indices]

    # Majority Vote: Das häufigste Label ermitteln
    # np.bincount zählt die Häufigkeit, argmax gibt den Index des Maximums zurück
    majority_label = np.bincount(k_nearest_labels).argmax()

    return majority_label


# 2b) Vorhersage für die ersten 10 Testbilder durchführen
print("\nStarte KNN Vorhersage für die ersten 10 Testbilder (Das kann ein paar Sekunden dauern)...")
k_value = 3  # Wir testen es erstmal mit K=3
vorhersagen = []

# Wir nehmen die ersten 10 Bilder aus Y (Testdaten)
for i in range(10):
    test_bild = Y[i]
    # Vorhersage aufrufen
    vorhergesagtes_label = predict_knn_l1(Xtr, Y_train_labels, test_bild, k=k_value)
    vorhersagen.append(vorhergesagtes_label)

    wahres_label = Y_test_labels[i]
    print(f"Bild {i + 1} | Vorhersage: {vorhergesagtes_label} | Wahres Label: {wahres_label}")