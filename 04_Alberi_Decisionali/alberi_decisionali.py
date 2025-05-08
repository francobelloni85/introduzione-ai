# Import delle librerie necessarie
import numpy as np
import pandas as pd # Pandas è utile per visualizzare le feature importances
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_breast_cancer # Useremo il dataset Breast Cancer

# --- 1. Caricamento e Preparazione dei Dati ---
print("--- Alberi Decisionali (Decision Tree Classifier) ---")
print("Caricamento del dataset Breast Cancer...")
cancer = load_breast_cancer()
X = cancer.data # Features
y = cancer.target # Target (0: maligno, 1: benigno)
feature_names = cancer.feature_names
target_names = cancer.target_names

# Visualizziamo le dimensioni dei dati
print(f"Numero di campioni: {X.shape[0]}")
print(f"Numero di features: {X.shape[1]}")
# print(f"Nomi delle features: {feature_names}") # Molte features, commentato per brevità
print(f"Classi target: {target_names} (corrispondenti a {np.unique(y)})")

# --- 2. Divisione dei Dati in Training Set e Test Set ---
# test_size=0.3 significa che il 30% dei dati sarà usato per il test.
# random_state assicura che la divisione sia sempre la stessa.
# stratify=y è importante per i problemi di classificazione.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nDimensioni Training Set: {X_train.shape[0]} campioni")
print(f"Dimensioni Test Set: {X_test.shape[0]} campioni")

# --- 3. Creazione e Addestramento del Modello ---
# Inizializziamo il classificatore ad albero decisionale.
# - criterion: la funzione per misurare la qualità di una divisione ('gini' o 'entropy').
# - max_depth: la profondità massima dell'albero. Utile per prevenire l'overfitting
#   e per rendere l'albero visualizzabile.
# - random_state: per riproducibilità.
max_tree_depth = 4 # Limitiamo la profondità per una migliore visualizzazione
print(f"\nCreazione del modello Decision Tree con max_depth={max_tree_depth}")
model = DecisionTreeClassifier(criterion='gini',
                               max_depth=max_tree_depth,
                               random_state=42)

# Addestriamo il modello utilizzando i dati di training
print("Addestramento del modello Decision Tree...")
model.fit(X_train, y_train)
print("Modello addestrato.")
print(f"Profondità effettiva dell'albero: {model.get_depth()}")

# --- 4. Effettuare Predizioni ---
# Usiamo il modello addestrato per fare predizioni sul test set
print("\nEffettuare predizioni sul Test Set...")
y_pred = model.predict(X_test)

# --- 5. Valutazione del Modello ---
# Accuratezza
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza del modello Decision Tree sul Test Set: {accuracy:.3f} (ovvero {accuracy*100:.2f}%)")

# Matrice di Confusione
print("\nGenerazione della Matrice di Confusione...")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Matrice di Confusione (max_depth={max_tree_depth})")
plt.show()

# --- 6. Visualizzazione dell'Albero Decisionale ---
print("\nVisualizzazione dell'Albero Decisionale...")
plt.figure(figsize=(20,12)) # Imposta dimensioni più grandi per la figura
plot_tree(model,
          filled=True, # Colora i nodi per indicare la classe maggioritaria
          rounded=True, # Usa angoli arrotondati per i box dei nodi
          class_names=target_names, # Nomi delle classi target
          feature_names=feature_names, # Nomi delle features
          fontsize=10, # Dimensione del font
          proportion=False, # Mostra il numero di campioni invece che le proporzioni
          precision=2) # Numero di decimali per i valori
plt.title(f"Albero Decisionale (max_depth={max_tree_depth}) - Dataset Breast Cancer", fontsize=16)
plt.show()

# --- 7. Importanza delle Features ---
# Gli alberi decisionali possono fornire una stima dell'importanza di ciascuna feature.
print("\nImportanza delle Features:")
importances = model.feature_importances_
# Creiamo un DataFrame Pandas per una visualizzazione più chiara
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

print(feature_importance_df.head(10)) # Mostra le 10 features più importanti

# Grafico dell'importanza delle features (prime 10)
plt.figure(figsize=(10, 6))
plt.title("Importanza delle Features (prime 10)")
plt.bar(feature_importance_df['feature'][:10], feature_importance_df['importance'][:10], color='skyblue')
plt.xlabel("Feature")
plt.ylabel("Importanza")
plt.xticks(rotation=45, ha="right")
plt.tight_layout() # Aggiusta il layout per evitare sovrapposizioni
plt.show()

print("\nEsecuzione script Alberi Decisionali completata.")