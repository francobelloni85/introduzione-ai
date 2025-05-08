#  K-Means Clustering

Questo esempio illustra l'algoritmo K-Means, uno dei più noti algoritmi di **clustering non supervisionato**. Il suo obiettivo è partizionare un dataset in 'K' gruppi (cluster) distinti, dove ogni punto dati appartiene al cluster il cui centroide (media) è più vicino.

## 📚 Breve Descrizione Teorica

L'algoritmo **K-Means** funziona in modo iterativo per raggruppare i dati:

1.  **Inizializzazione:** Si scelgono casualmente 'K' punti dal dataset come centroidi iniziali (oppure si usano strategie di inizializzazione più intelligenti come k-means++).
2.  **Assegnazione:** Ogni punto dati viene assegnato al cluster il cui centroide è più vicino (solitamente usando la distanza Euclidea).
3.  **Aggiornamento dei Centroidi:** Per ogni cluster, il centroide viene ricalcolato come la media di tutti i punti dati assegnati a quel cluster.
4.  **Iterazione:** I passaggi 2 e 3 vengono ripetuti finché i centroidi non si stabilizzano (cioè, si muovono molto poco tra un'iterazione e l'altra) o finché non viene raggiunto un numero massimo di iterazioni.

**Caratteristiche principali:**
* **Tipo di Apprendimento:** Non Supervisionato (non usa etichette predefinite per l'addestramento).
* **Scelta di 'K':** Il numero di cluster 'K' deve essere specificato a priori. La scelta di un 'K' appropriato è cruciale e spesso si usano metodi euristici come l'Elbow Method (basato sull'inerzia) o il Silhouette Score.
* **Inizializzazione dei Centroidi:** Il risultato di K-Means può dipendere dall'inizializzazione casuale dei centroidi. Per questo, l'algoritmo viene spesso eseguito più volte con diverse inizializzazioni (`n_init` in scikit-learn).
* **Forma dei Cluster:** K-Means tende a funzionare meglio quando i cluster sono sferici, di dimensioni simili e ben separati.

## 📊 Dataset Utilizzato nello Script

Nello script `kmeans_clustering.py`, i dati sono **generati sinteticamente** utilizzando la funzione `make_blobs` di `sklearn.datasets`. Questo ci permette di creare "blob" di punti che formano cluster naturali e visivamente distinti.

* **Features (X):** Lo script genera dati con due features numeriche.
* **Etichette Vere (`y_true`):** La funzione `make_blobs` restituisce anche le etichette vere dei cluster a cui ogni punto appartiene. Queste etichette **non vengono utilizzate** dall'algoritmo K-Means durante l'addestramento (poiché è non supervisionato), ma le usiamo nello script solo per **confrontare visivamente** i risultati del clustering con la struttura reale dei dati.

## 🐍 Contenuto dello Script `kmeans_clustering.py`

Lo script `kmeans_clustering.py` esegue i seguenti passaggi principali:

1.  **Importazione delle Librerie:** Vengono importate `numpy`, `matplotlib.pyplot`, e moduli specifici da `sklearn` (`KMeans`, `StandardScaler`, `make_blobs`).
2.  **Generazione dei Dati:** Vengono creati i dati sintetici con `make_blobs`, specificando il numero di campioni, il numero di centri (cluster veri) e la dispersione dei cluster.
3.  **Standardizzazione delle Features (Opzionale ma Consigliata):** Le features vengono standardizzate. Anche se per dati generati in modo così controllato l'effetto potrebbe essere minimo, è una buona pratica generale per K-Means, specialmente se le features avessero scale molto diverse.
4.  **Creazione e Addestramento del Modello K-Means:**
    * Viene creata un'istanza del modello `KMeans`, specificando `n_clusters` (il numero 'K' di cluster da trovare), `random_state` per la riproducibilità, e `n_init='auto'` (per gestire automaticamente il numero di inizializzazioni).
    * Il modello viene addestrato sui dati (`kmeans.fit(X_scaled)`).
5.  **Ottenimento dei Risultati:**
    * Vengono recuperate le etichette dei cluster (`kmeans.labels_`) assegnate a ciascun punto.
    * Vengono recuperate le coordinate dei centroidi dei cluster trovati (`kmeans.cluster_centers_`).
6.  **Calcolo dell'Inerzia:** Viene calcolata l'inerzia del modello (`kmeans.inertia_`), che è la somma delle distanze al quadrato dei campioni dal centro del loro cluster assegnato (WCSS - Within-Cluster Sum of Squares).
7.  **Visualizzazione:** Vengono mostrati due grafici affiancati:
    * Il primo grafico mostra i punti dati colorati in base ai cluster assegnati da K-Means, con i centroidi finali evidenziati.
    * Il secondo grafico mostra gli stessi punti dati colorati in base alle loro etichette "vere" (generate da `make_blobs`), permettendo un confronto visivo.

## 🚀 Istruzioni per l'Esecuzione

Per eseguire lo script:

1.  Assicurati di aver attivato il tuo ambiente virtuale Python con le dipendenze installate.
2.  Apri un terminale o prompt dei comandi.
3.  Naviga fino alla directory principale del progetto e poi in questa sottocartella:
    ```bash
    cd 03_K_Means_Clustering
    ```
4.  Esegui lo script con il comando:
    ```bash
    python kmeans_clustering.py
    ```

## 📈 Output Atteso

Dopo l'esecuzione, vedrai:

* **Nel terminale:**
    * Informazioni sulla generazione dei dati.
    * Il valore di 'K' utilizzato per K-Means.
    * Le etichette predette per i primi campioni.
    * Le coordinate dei centroidi trovati.
    * Il valore dell'inerzia (WCSS) del modello.
* **Una finestra grafica:**
    * Due grafici a dispersione affiancati: uno con i cluster trovati da K-Means e i relativi centroidi, l'altro con i cluster "veri" del dataset generato.

## 💡 Possibili Esperimenti e Modifiche

Prova a modificare lo script per esplorare ulteriormente K-Means:

* **Variare 'K':** Cambia il valore di `k_kmeans` (cioè `n_clusters` passato a `KMeans`). Cosa succede se scegli un 'K' diverso dal numero di centri specificato in `make_blobs`? Come cambia l'inerzia?
* **Parametri di `make_blobs`:** Modifica i parametri di `make_blobs`, ad esempio `cluster_std` (per rendere i cluster più o meno sovrapposti) o `centers` (per cambiare il numero di cluster veri). Come si comporta K-Means?
* **Effetto della Standardizzazione:** Commenta la parte relativa alla standardizzazione con `StandardScaler`. In questo specifico esempio con `make_blobs` e `cluster_std` uniforme, la differenza potrebbe essere minima, ma prova a generare dati con scale molto diverse per le due features per vedere un impatto maggiore.
* **Scelta di 'K' Ottimale:** Ricerca e prova ad implementare l'**Elbow Method** plottando l'inerzia per diversi valori di 'K', o calcola il **Silhouette Score** (da `sklearn.metrics import silhouette_score`) per diversi 'K' per trovare un valore ottimale.
* **Dati Reali:** Prova ad applicare K-Means a un dataset reale (es. caricando un CSV con `pandas` e selezionando solo le colonne numeriche). Ricorda di pre-processare i dati (gestione valori mancanti, scaling).

Buona esplorazione del clustering con K-Means!