#  K-Nearest Neighbors (KNN)

Questo esempio illustra l'algoritmo K-Nearest Neighbors (KNN), un metodo di apprendimento supervisionato non parametrico utilizzato sia per compiti di classificazione che di regressione. In questo script, ci concentreremo sulla classificazione.

## 📚 Breve Descrizione Teorica

L'algoritmo **K-Nearest Neighbors (KNN)** classifica un nuovo punto dati basandosi sulla classe di maggioranza dei suoi 'k' vicini più prossimi nel dataset di addestramento.

**Come funziona:**
1.  **Calcolo delle Distanze:** Quando si deve classificare un nuovo punto, KNN calcola la distanza tra questo punto e tutti i punti nel dataset di addestramento. Le metriche di distanza comuni includono la distanza Euclidea, Manhattan, Minkowski.
2.  **Identificazione dei Vicini:** Vengono selezionati i 'k' punti del training set più vicini al nuovo punto (i "k vicini più prossimi").
3.  **Voto di Maggioranza (per Classificazione):** Il nuovo punto viene assegnato alla classe che è più frequente tra i suoi 'k' vicini. Ad esempio, se k=5 e 3 dei 5 vicini appartengono alla Classe A e 2 alla Classe B, il nuovo punto sarà classificato come Classe A.
4.  **(Media per Regressione):** Se usato per la regressione, il valore predetto sarebbe la media (o mediana) dei valori dei 'k' vicini.

**Caratteristiche principali:**
* **"Lazy Learner" (Apprendimento Pigro):** KNN è un algoritmo "pigro" o basato su istanze perché non costruisce un modello esplicito durante la fase di addestramento. Semplicemente memorizza l'intero dataset di training. La computazione avviene solo al momento della predizione.
* **Importanza di 'k':** La scelta del valore 'k' è cruciale. Un 'k' piccolo può rendere il modello sensibile al rumore, mentre un 'k' grande può "appiattire" le decisioni rendendo i confini tra le classi meno distinti.
* **Sensibilità alla Scala delle Features:** Poiché si basa sulle distanze, KNN è molto sensibile alla scala delle variabili. È quindi fondamentale standardizzare o normalizzare i dati prima di applicare l'algoritmo.
* **Tipo di Apprendimento:** Supervisionato.

## 📊 Dataset Utilizzato nello Script

Nello script `knn_classificazione.py`, utilizziamo il classico dataset **Iris**, fornito da `sklearn.datasets`.

* **Features (Variabili Indipendenti X):**
    1.  Lunghezza sepalo (cm)
    2.  Larghezza sepalo (cm)
    3.  Lunghezza petalo (cm)
    4.  Larghezza petalo (cm)
* **Target (Variabile Dipendente y):** Specie del fiore Iris.
    * Classe 0: Iris Setosa
    * Classe 1: Iris Versicolor
    * Classe 2: Iris Virginica

## 🐍 Contenuto dello Script `knn_classificazione.py`

Lo script `knn_classificazione.py` esegue i seguenti passaggi principali:

1.  **Importazione delle Librerie:** Vengono importate `numpy`, `matplotlib.pyplot`, e moduli specifici da `sklearn` (`KNeighborsClassifier`, `train_test_split`, `StandardScaler`, `accuracy_score`, `confusion_matrix`, `ConfusionMatrixDisplay`, `load_iris`).
2.  **Caricamento dei Dati:** Viene caricato il dataset Iris.
3.  **Divisione dei Dati:** Il dataset viene suddiviso in un **Training Set** e un **Test Set**, utilizzando `stratify=y` per assicurare che le proporzioni delle classi siano mantenute in entrambi i set.
4.  **Standardizzazione delle Features:** Le features numeriche vengono standardizzate (media 0, deviazione standard 1) utilizzando `StandardScaler`. Questo passaggio è **fondamentale** per KNN. Lo scaler viene addestrato (`fit`) solo sui dati di training e poi applicato (`transform`) sia al training che al test set per evitare *data leakage*.
5.  **Creazione e Addestramento del Modello:** Viene creata un'istanza del classificatore `KNeighborsClassifier` (con un valore di `k` specificato, ad esempio `k=5`) e viene addestrata usando i dati di training standardizzati.
6.  **Effettuare Predizioni:** Il modello addestrato viene utilizzato per predire le classi dei campioni nel test set.
7.  **Valutazione del Modello:** Le prestazioni vengono valutate tramite:
    * **Accuratezza (Accuracy):** La percentuale di classificazioni corrette.
    * **Matrice di Confusione:** Una tabella che mostra le predizioni corrette e sbagliate per ciascuna classe.
8.  **Visualizzazione:**
    * Viene mostrata graficamente la **Matrice di Confusione**.
    * Viene generato un **grafico a dispersione (scatter plot)** dei dati di test (utilizzando le prime due features) colorati in base alla loro classe reale, per dare un'idea visiva della separazione delle classi.

## 🚀 Istruzioni per l'Esecuzione

Per eseguire lo script:

1.  Assicurati di aver attivato il tuo ambiente virtuale Python con le dipendenze installate.
2.  Apri un terminale o prompt dei comandi.
3.  Naviga fino alla directory principale del progetto e poi in questa sottocartella:
    ```bash
    cd 02_K_Nearest_Neighbors
    ```
4.  Esegui lo script con il comando:
    ```bash
    python knn_classificazione.py
    ```

## 📈 Output Atteso

Dopo l'esecuzione, vedrai:

* **Nel terminale:**
    * Informazioni sul caricamento e le dimensioni del dataset.
    * Il valore di `k` utilizzato.
    * L'accuratezza del modello KNN sul test set.
* **Finestre grafiche:**
    1.  La **Matrice di Confusione** che visualizza le prestazioni di classificazione.
    2.  Un **grafico a dispersione** dei dati di test (prime due features) colorati in base alla loro classe effettiva.

## 💡 Possibili Esperimenti e Modifiche

Prova a modificare lo script per approfondire la tua comprensione di KNN:

* **Cambia il valore di `k`:** Prova diversi valori per `n_neighbors` (es. 1, 3, 7, 10, 20) e osserva come cambiano l'accuratezza e la matrice di confusione. Cosa succede con `k` molto piccoli o molto grandi?
* **Rimuovi la Standardizzazione:** Commenta le righe relative a `StandardScaler` e riesegui lo script. Come cambiano le prestazioni del modello? Questo dovrebbe evidenziare l'importanza della scalatura delle features per KNN.
* **Diverse Metriche di Distanza:** Esplora il parametro `metric` di `KNeighborsClassifier` (es. `'manhattan'`, `'minkowski'` con diversi valori di `p`).
* **Altro Dataset:** Prova ad applicare KNN a un altro dataset di classificazione disponibile in `sklearn.datasets` (es. `load_wine()` o `load_breast_cancer()`). Potrebbe essere necessario adattare leggermente lo script.
* **Visualizzazione diversa:** Modifica lo scatter plot per utilizzare coppie diverse di features per la visualizzazione. Come appare la separazione delle classi?
* **Weighted KNN:** Esplora il parametro `weights` in `KNeighborsClassifier` (es. `weights='distance'`), dove i vicini più vicini hanno un'influenza maggiore sulla predizione rispetto a quelli più lontani.

Buona sperimentazione con KNN!