# Import delle librerie necessarie
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Per la standardizzazione delle features
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_iris # Useremo il dataset Iris

# --- 1. Caricamento e Preparazione dei Dati ---
print("--- K-Nearest Neighbors (KNN) ---")
print("Caricamento del dataset Iris...")
iris = load_iris()
X = iris.data # Features: lunghezza sepalo, larghezza sepalo, lunghezza petalo, larghezza petalo
y = iris.target # Target: specie di Iris (0: setosa, 1: versicolor, 2: virginica)
feature_names = iris.feature_names
target_names = iris.target_names

# Visualizziamo le dimensioni dei dati
print(f"Numero di campioni: {X.shape[0]}")
print(f"Numero di features: {X.shape[1]}")
print(f"Nomi delle features: {feature_names}")
print(f"Classi target: {target_names} (corrispondenti a {np.unique(y)})")

# --- 2. Divisione dei Dati in Training Set e Test Set ---
# Dividiamo i dati per addestrare il modello e per testarne le prestazioni.
# test_size=0.3 significa che il 30% dei dati sarà usato per il test.
# random_state assicura che la divisione sia sempre la stessa.
# stratify=y è importante per i problemi di classificazione, per mantenere
# la stessa proporzione di classi nel training e nel test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nDimensioni Training Set: {X_train.shape[0]} campioni")
print(f"Dimensioni Test Set: {X_test.shape[0]} campioni")

# --- 3. Standardizzazione delle Features ---
# KNN è un algoritmo basato sulla distanza, quindi è sensibile alla scala delle features.
# È buona pratica standardizzare le features (media 0, deviazione standard 1).
print("\nStandardizzazione delle features...")
scaler = StandardScaler()
# Adattiamo lo scaler SOLO sui dati di training per evitare data leakage
X_train_scaled = scaler.fit_transform(X_train)
# Applichiamo la trasformazione sia ai dati di training che di test
X_test_scaled = scaler.transform(X_test)

# --- 4. Creazione e Addestramento del Modello ---
# Scegliamo un valore per k (numero di vicini). Un valore comune è 5.
k = 5
print(f"\nCreazione del modello KNN con k={k}")
model = KNeighborsClassifier(n_neighbors=k)

# Addestriamo il modello utilizzando i dati di training standardizzati
print("Addestramento del modello KNN...")
model.fit(X_train_scaled, y_train)
print("Modello addestrato.")

# --- 5. Effettuare Predizioni ---
# Usiamo il modello addestrato per fare predizioni sul test set (standardizzato)
print("\nEffettuare predizioni sul Test Set...")
y_pred = model.predict(X_test_scaled)

# --- 6. Valutazione del Modello ---
# Valutiamo le prestazioni del modello sul test set.

# Accuratezza (Accuracy): la proporzione di predizioni corrette.
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza del modello KNN sul Test Set: {accuracy:.2f} (ovvero {accuracy*100:.2f}%)")

# Matrice di Confusione: mostra il numero di predizioni corrette e errate per ciascuna classe.
print("\nGenerazione della Matrice di Confusione...")
cm = confusion_matrix(y_test, y_pred)
# print("Matrice di Confusione:")
# print(cm)

# Visualizzazione della Matrice di Confusione
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Matrice di Confusione KNN (k={k}) - Dataset Iris")
plt.show()

# --- 7. Visualizzazione dei Dati di Test (opzionale, solo per 2 features) ---
# Per visualizzare i risultati, usiamo solo le prime due features del dataset Iris
# (lunghezza sepalo e larghezza sepalo) per semplicità.

if X_test_scaled.shape[1] >= 2:
    plt.figure(figsize=(10, 7))

    # Colori per le classi
    cmap_light = plt.cm.get_cmap('viridis', 3) # Per le regioni di decisione (non mostrate qui per semplicità)
    cmap_bold = plt.cm.get_cmap('viridis', 3)  # Per i punti

    # Plot dei punti del test set, colorati in base alla classe reale
    scatter = plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=60, alpha=0.8)

    plt.xlabel(f"{feature_names[0]} (standardizzata)")
    plt.ylabel(f"{feature_names[1]} (standardizzata)")
    plt.title(f"Classificazione KNN (k={k}) - Dati di Test (Prime due features Iris)")

    # Creazione di una legenda per le classi
    handles, _ = scatter.legend_elements(prop="colors")
    legend_labels = [f"Specie: {name}" for name in target_names]
    plt.legend(handles, legend_labels, title="Classi Reali")
    plt.grid(True)
    plt.show()

    # Potremmo anche visualizzare i punti colorati in base alla predizione y_pred
    # e magari evidenziare gli errori, ma per semplicità lo omettiamo.
    # Esempio per evidenziare errori:
    # errori = X_test_scaled[y_test != y_pred]
    # if errori.shape[0] > 0:
    #     plt.scatter(errori[:, 0], errori[:, 1], facecolors='none', edgecolors='red', s=150, linewidths=2, label='Errori')


print("\nEsecuzione script KNN completata.")