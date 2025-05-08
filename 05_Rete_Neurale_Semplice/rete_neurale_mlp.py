# Import delle librerie necessarie
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Fondamentale per le reti neurali
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_digits # Useremo il dataset Digits

# --- 1. Caricamento e Preparazione dei Dati ---
print("--- Rete Neurale Semplice (MLPClassifier) ---")
print("Caricamento del dataset Digits...")
digits = load_digits()
X = digits.data # Features: immagini 8x8 appiattite (64 pixels)
y = digits.target # Target: cifre da 0 a 9

# Visualizziamo le dimensioni dei dati
print(f"Numero di campioni: {X.shape[0]}")
print(f"Numero di features (pixels per immagine): {X.shape[1]}")
print(f"Classi target: {np.unique(y)}")

# --- 2. Divisione dei Dati in Training Set e Test Set ---
# test_size=0.3 significa che il 30% dei dati sarà usato per il test.
# random_state assicura che la divisione sia sempre la stessa.
# stratify=y è importante per i problemi di classificazione.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nDimensioni Training Set: {X_train.shape[0]} campioni")
print(f"Dimensioni Test Set: {X_test.shape[0]} campioni")

# --- 3. Standardizzazione delle Features ---
# Le reti neurali sono molto sensibili alla scala delle features.
# È cruciale standardizzare i dati (media 0, deviazione standard 1).
print("\nStandardizzazione delle features...")
scaler = StandardScaler()
# Adattiamo lo scaler SOLO sui dati di training
X_train_scaled = scaler.fit_transform(X_train)
# Applichiamo la trasformazione sia ai dati di training che di test
X_test_scaled = scaler.transform(X_test)

# --- 4. Creazione e Addestramento del Modello ---
# Inizializziamo il Multi-Layer Perceptron Classifier.
# - hidden_layer_sizes: tupla, es. (100,) per un layer nascosto da 100 neuroni,
#   (64, 32) per due layer nascosti.
# - activation: funzione di attivazione per i layer nascosti ('relu', 'logistic', 'tanh').
# - solver: algoritmo per l'ottimizzazione dei pesi ('adam' è spesso una buona scelta).
# - alpha: parametro di regolarizzazione L2.
# - max_iter: numero massimo di iterazioni (epoche).
# - learning_rate_init: tasso di apprendimento iniziale (per solver 'sgd' o 'adam').
# - early_stopping: se True, interrompe l'addestramento quando il punteggio di validazione
#   non migliora, per prevenire l'overfitting.
# - validation_fraction: proporzione di dati di training da usare come set di validazione
#   per l'early stopping.
# - n_iter_no_change: numero di iterazioni senza miglioramento sul validation set
#   prima di fermare l'addestramento con early_stopping.
# - random_state: per riproducibilità.

print("\nCreazione del modello MLPClassifier...")
model = MLPClassifier(hidden_layer_sizes=(100, 50), # Due layer nascosti
                      activation='relu',
                      solver='adam',
                      alpha=0.0001,
                      learning_rate_init=0.001,
                      max_iter=300, # Aumentato per dare più tempo, ma early stopping aiuta
                      early_stopping=True,
                      validation_fraction=0.1,
                      n_iter_no_change=10,
                      random_state=42,
                      verbose=False) # Imposta a True per vedere il progresso dell'addestramento

print("Addestramento del modello MLPClassifier (potrebbe richiedere un po' di tempo)...")
model.fit(X_train_scaled, y_train)
print("Modello addestrato.")
print(f"Numero di iterazioni eseguite: {model.n_iter_}")
print(f"Numero di layers (incluso input e output): {model.n_layers_}")

# --- 5. Effettuare Predizioni ---
# Usiamo il modello addestrato per fare predizioni sul test set (standardizzato)
print("\nEffettuare predizioni sul Test Set...")
y_pred = model.predict(X_test_scaled)

# --- 6. Valutazione del Modello ---
# Accuratezza
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza del modello MLP sul Test Set: {accuracy:.3f} (ovvero {accuracy*100:.2f}%)")

# Matrice di Confusione
print("\nGenerazione della Matrice di Confusione...")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice di Confusione MLPClassifier - Dataset Digits")
plt.show()

# --- 7. Visualizzazione della Curva di Loss ---
# La curva di loss mostra come l'errore del modello diminuisce durante l'addestramento.
# È disponibile se il solver la traccia (es. 'adam', 'sgd')
if hasattr(model, 'loss_curve_'):
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_)
    plt.title("Curva di Loss durante l'Addestramento MLP")
    plt.xlabel("Iterazioni (Epoche)")
    plt.ylabel("Loss (Errore)")
    plt.grid(True)
    plt.show()
else:
    print("\nCurva di loss non disponibile per questo solver o configurazione.")

# --- 8. Visualizzazione di Alcune Predizioni (Specifico per dataset di immagini) ---
# Mostriamo alcune immagini dal test set con le loro etichette vere e predette.
n_images_to_show = 15
# Selezioniamo indici casuali dal test set
random_indices = np.random.choice(X_test.shape[0], size=n_images_to_show, replace=False)

fig, axes = plt.subplots(3, 5, figsize=(12, 8), subplot_kw={'xticks':[], 'yticks':[]})
fig.suptitle('Esempi di Predizioni MLP sul Dataset Digits', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < n_images_to_show:
        idx = random_indices[i]
        # Mostriamo l'immagine originale (non scalata)
        ax.imshow(X_test[idx].reshape(8, 8), cmap='binary', interpolation='nearest')
        true_label = y_test[idx]
        predicted_label = y_pred[idx]
        ax.set_title(f"Vero: {true_label}\nPred: {predicted_label}",
                     color='green' if true_label == predicted_label else 'red')
    else:
        ax.axis('off') # Nasconde gli assi vuoti se n_images_to_show non è un multiplo di 5*3

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\nEsecuzione script MLPClassifier completata.")