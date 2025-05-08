# Import delle librerie necessarie
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # Per la standardizzazione (buona pratica)
from sklearn.datasets import make_blobs # Per generare dati di esempio per il clustering

# --- 1. Generazione dei Dati di Esempio ---
print("--- K-Means Clustering ---")
print("Generazione dati di esempio con make_blobs...")

# Creiamo dei "blob" di punti per simulare cluster naturali.
# n_samples: numero totale di punti.
# centers: numero di centri (cluster) da generare o coordinate dei centri.
# cluster_std: deviazione standard dei cluster (quanto sono sparsi).
# random_state: per riproducibilità.
n_samples = 300
n_features = 2
n_clusters_dati = 15 # Numero di cluster che vogliamo generare
random_seed = 42

X, y_true = make_blobs(n_samples=n_samples,
                       n_features=n_features,
                       centers=n_clusters_dati,
                       cluster_std=0.7, # Cluster abbastanza definiti
                       random_state=random_seed)

print(f"Generati {X.shape[0]} campioni con {X.shape[1]} features.")
# y_true contiene le etichette vere dei cluster, ma K-Means non le userà (è non supervisionato).
# Le useremo solo alla fine per confrontare visivamente il risultato.

# --- (Opzionale ma consigliato) Standardizzazione delle Features ---
# Anche se K-Means può funzionare senza, se le feature hanno scale molto diverse
# la standardizzazione può migliorare i risultati.
print("\nStandardizzazione delle features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Per questo esempio con make_blobs e cluster_std simili, l'effetto potrebbe non essere drastico,
# ma è una buona pratica. Useremo X_scaled da ora.

# --- 2. Creazione e Addestramento del Modello K-Means ---
# Scegliamo il numero di cluster (k) per l'algoritmo K-Means.
# In un caso reale, non conosceremmo n_clusters_dati e dovremmo usare metodi
# come l'Elbow method o Silhouette score per stimare il k ottimale.
# Qui, per semplicità didattica, usiamo il numero di cluster che sappiamo essere presenti.
k_kmeans = n_clusters_dati
print(f"\nCreazione del modello K-Means con k={k_kmeans} cluster...")

# random_state nel KMeans assicura che l'inizializzazione dei centroidi sia la stessa,
# portando a risultati riproducibili.
# n_init='auto' è l'impostazione predefinita nelle versioni recenti di scikit-learn
# per eseguire l'algoritmo più volte con diverse inizializzazioni dei centroidi.
kmeans = KMeans(n_clusters=k_kmeans, random_state=random_seed, n_init='auto')

print("Addestramento del modello K-Means...")
# Addestriamo il modello K-Means sui dati (standardizzati)
# K-Means assegna ogni punto a un cluster e calcola i centroidi.
kmeans.fit(X_scaled)
print("Modello addestrato.")

# --- 3. Ottenere le Etichette dei Cluster e i Centroidi ---
# Etichette dei cluster assegnate a ciascun punto dati
labels_pred = kmeans.labels_

# Coordinate dei centroidi dei cluster trovati
centroids = kmeans.cluster_centers_

print(f"\nEtichette dei cluster predette per i primi 10 punti: {labels_pred[:10]}")
print(f"Coordinate dei centroidi dei {k_kmeans} cluster:\n{centroids}")

# --- 4. Valutazione del Modello (Principalmente Visiva in questo script) ---
# In pratica, si userebbero metriche come:
# - Inertia (WCSS - Within-Cluster Sum of Squares): kmeans.inertia_
#   Misura la somma delle distanze al quadrato dei campioni dal centro del loro cluster.
#   Tende a diminuire all'aumentare di k. Usata nell'Elbow Method.
# - Silhouette Score: misura quanto un campione sia simile al proprio cluster
#   rispetto agli altri cluster. Valori vicini a +1 indicano buona clusterizzazione.
#   (from sklearn.metrics import silhouette_score)

print(f"\nInertia (WCSS) del modello: {kmeans.inertia_:.2f}")
# silhouette_avg = silhouette_score(X_scaled, labels_pred)
# print(f"Silhouette Score medio: {silhouette_avg:.2f}") # Richiede sklearn.metrics

# --- 5. Visualizzazione dei Risultati ---

# Grafico 1: Dati clusterizzati da K-Means
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Punti dati, colorati in base al cluster assegnato da K-Means
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_pred, s=50, cmap='viridis', alpha=0.7)
# Centroidi dei cluster
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', edgecolor='black', label='Centroidi')
plt.title(f"K-Means Clustering (k={k_kmeans}) - Predizioni")
plt.xlabel("Feature 1 (standardizzata)")
plt.ylabel("Feature 2 (standardizzata)")
plt.legend()
plt.grid(True)

# Grafico 2: Dati originali con le etichette vere (per confronto)
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, s=50, cmap='viridis', alpha=0.7)
plt.title("Dati Originali - Etichette Vere")
plt.xlabel("Feature 1 (standardizzata)")
plt.ylabel("Feature 2 (standardizzata)")
plt.grid(True)

plt.suptitle("Confronto K-Means Clustering vs Etichette Vere", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Aggiusta layout per il suptitle
plt.show()

print("\nEsecuzione script K-Means completata.")