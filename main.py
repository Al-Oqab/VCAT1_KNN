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
    distances = np.sum(np.abs(X_train - x_test_image), axis=1)
    min_indices = np.argsort(distances)[:k]
    k_nearest_labels = Y_train_labels[min_indices]
    majority_label = np.bincount(k_nearest_labels).argmax()
    return majority_label


# 2b) Vorhersage und Visualisierung für die ersten 10 Testbilder
print("\nStarte KNN Vorhersage für die ersten 10 Testbilder (Das kann ein paar Sekunden dauern)...")
k_value = 3
vorhersagen = []

# Schritt 1: Schleife für die Vorhersagen der 10 Bilder
for i in range(10):
    test_bild = Y[i]
    vorhergesagtes_label = predict_knn_l1(Xtr, Y_train_labels, test_bild, k=k_value)
    vorhersagen.append(vorhergesagtes_label)

    wahres_label = Y_test_labels[i]
    print(f"Bild {i + 1} | Vorhersage: {vorhergesagtes_label} | Wahres Label: {wahres_label}")

# Schritt 2: Visualisierung (Läuft erst, wenn die Schleife oben komplett fertig ist)
print("\nErstelle die Visualisierung für die 10 Bilder...")
klassen_namen = ['Flugzeug', 'Auto', 'Vogel', 'Katze', 'Reh', 'Hund', 'Frosch', 'Pferd', 'Schiff', 'LKW']

plt.figure(figsize=(15, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    img = Y[i].reshape(3, 32, 32).transpose(1, 2, 0)
    plt.imshow(img)

    vorhersage_name = klassen_namen[vorhersagen[i]]
    wahres_name = klassen_namen[Y_test_labels[i]]

    plt.title(f"Vorhersage: {vorhersage_name}\nWahr: {wahres_name}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()


# --- 2c & 2d) Genauigkeit für K=1, 3, 5 und 7 bestimmen und plotten ---

print("\n--- Starte Evaluation für alle Testdaten ---")
print("ACHTUNG: Das Berechnen von 10.000 Bildern dauert einige Minuten. Bitte warten...")


def evaluate_all_k(X_train, Y_train_labels, X_test, Y_test_labels, k_values):
    # Ein Dictionary, um die Anzahl der korrekten Vorhersagen pro K zu speichern
    correct_predictions = {k: 0 for k in k_values}
    num_test_images = len(X_test)  # Das sind alle 10.000 Bilder

    # Maximales K finden, damit wir das Array nur einmal durchsuchen müssen
    max_k = max(k_values)

    for i in range(num_test_images):
        # 1. Distanz zu ALLEN Trainingsbildern berechnen (L1-Distanz)
        distances = np.sum(np.abs(X_train - X_test[i]), axis=1)

        # 2. Nur die Indizes der 'max_k' kleinsten Distanzen finden
        min_indices = np.argsort(distances)[:max_k]
        nearest_labels = Y_train_labels[min_indices]

        # 3. Auswertung für jedes gewünschte K durchführen
        for k in k_values:
            k_labels = nearest_labels[:k]
            majority_label = np.bincount(k_labels).argmax()

            # Überprüfen, ob die Vorhersage richtig war
            if majority_label == Y_test_labels[i]:
                correct_predictions[k] += 1

        # Fortschrittsanzeige alle 500 Bilder
        if (i + 1) % 500 == 0:
            print(f"Fortschritt: {i + 1} / {num_test_images} Bilder verarbeitet...")

    # Genauigkeit (Accuracy) in Prozent berechnen
    accuracies = []
    print("\n--- Endergebnisse ---")
    for k in k_values:
        acc = correct_predictions[k] / num_test_images
        accuracies.append(acc)
        print(f"Genauigkeit für K={k}: {acc * 100:.2f}%")

    return accuracies


# Die geforderten K-Werte
k_list = [1, 3, 5, 7]

# Funktion aufrufen
genauigkeiten = evaluate_all_k(Xtr, Y_train_labels, Y, Y_test_labels, k_list)

# Ergebnisse graphisch darstellen
plt.figure(figsize=(8, 5))
plt.plot(k_list, genauigkeiten, marker='o', linestyle='-', color='b', markersize=8)
plt.title('KNN Klassifikationsgenauigkeit (L1-Distanz)')
plt.xlabel('Anzahl der Nachbarn (K)')
plt.ylabel('Genauigkeit')
plt.xticks(k_list)  # Damit nur 1, 3, 5, 7 auf der X-Achse stehen
plt.grid(True)
plt.show()


# --- 2e) L2-Distanzmaß durchführen und vergleichen ---

print("\n--- Starte Evaluation für alle Testdaten mit L2-Distanz ---")


def evaluate_all_k_l2(X_train, Y_train_labels, X_test, Y_test_labels, k_values):
    correct_predictions = {k: 0 for k in k_values}
    num_test_images = len(X_test)
    max_k = max(k_values)

    # WICHTIG: Datentyp auf float32 ändern, um Überläufe (Overflow) bei der Quadrierung zu vermeiden!
    X_train_float = X_train.astype(np.float32)
    X_test_float = X_test.astype(np.float32)

    for i in range(num_test_images):
        # L2-Distanz: Wurzel aus der Summe der quadrierten Differenzen
        # np.square quadriert die Differenzen, np.sum summiert sie, np.sqrt zieht die Wurzel
        distances = np.sqrt(np.sum(np.square(X_train_float - X_test_float[i]), axis=1))

        min_indices = np.argsort(distances)[:max_k]
        nearest_labels = Y_train_labels[min_indices]

        for k in k_values:
            k_labels = nearest_labels[:k]
            majority_label = np.bincount(k_labels).argmax()

            if majority_label == Y_test_labels[i]:
                correct_predictions[k] += 1

        if (i + 1) % 500 == 0:
            print(f"L2 Fortschritt: {i + 1} / {num_test_images} Bilder verarbeitet...")

    accuracies = []
    print("\n--- Endergebnisse L2 ---")
    for k in k_values:
        acc = correct_predictions[k] / num_test_images
        accuracies.append(acc)
        print(f"L2 Genauigkeit für K={k}: {acc * 100:.2f}%")

    return accuracies


# Funktion aufrufen
genauigkeiten_l2 = evaluate_all_k_l2(Xtr, Y_train_labels, Y, Y_test_labels, k_list)

# Ergebnisse graphisch darstellen (Vergleich L1 vs L2)
plt.figure(figsize=(8, 5))
# Wir plotten L1 (aus der vorherigen Berechnung) und L2 im selben Diagramm zum besseren Vergleich
plt.plot(k_list, genauigkeiten, marker='o', linestyle='-', color='b', label='L1-Distanz')
plt.plot(k_list, genauigkeiten_l2, marker='s', linestyle='--', color='r', label='L2-Distanz')
plt.title('KNN Klassifikationsgenauigkeit (L1 vs L2)')
plt.xlabel('Anzahl der Nachbarn (K)')
plt.ylabel('Genauigkeit')
plt.xticks(k_list)
plt.legend()  # Zeigt die Legende an
plt.grid(True)
plt.show()