#  Alberi Decisionali (Decision Trees)

Questo esempio esplora gli **Alberi Decisionali**, un potente e intuitivo algoritmo di apprendimento supervisionato utilizzato sia per compiti di classificazione che di regressione. In questo script, ci concentriamo sulla classificazione utilizzando il `DecisionTreeClassifier`.

## 📚 Breve Descrizione Teorica

Un **Albero Decisionale** è un modello predittivo che assomiglia a una struttura ad albero o a un diagramma di flusso. Ogni:
* **Nodo interno** rappresenta un "test" su una specifica feature (attributo).
* **Ramo** che si diparte da un nodo rappresenta l'esito di quel test.
* **Nodo foglia** (o nodo terminale) rappresenta una decisione finale, ovvero l'etichetta di classe (in classificazione) o un valore continuo (in regressione).

**Come funziona (semplificato):**
L'algoritmo costruisce l'albero partendo dalla radice e suddividendo ricorsivamente i dati in sottoinsiemi più piccoli e omogenei. Ad ogni passo, seleziona la migliore feature e il miglior punto di divisione (threshold) per quella feature, basandosi su un criterio di "purezza" o "impurità" dei nodi risultanti. I criteri comuni includono:
* **Impurità di Gini:** Misura la probabilità di classificare erroneamente un elemento scelto a caso se fosse etichettato casualmente secondo la distribuzione delle etichette nel nodo.
* **Guadagno di Informazione (Information Gain, basato sull'Entropia):** Misura la riduzione dell'entropia (incertezza) dopo una divisione.

**Vantaggi:**
* **Facilità di interpretazione e visualizzazione:** La logica decisionale è esplicita.
* Richiedono poca preparazione dei dati (es. non necessitano di scaling delle feature).
* Possono gestire sia dati numerici che categorici (anche se `scikit-learn` richiede che i dati categorici siano prima codificati numericamente).
* Possono catturare relazioni non lineari tra le feature e il target.

**Svantaggi:**
* **Tendenza all'overfitting:** Gli alberi possono diventare molto complessi e adattarsi eccessivamente ai dati di training, generalizzando male su dati nuovi. Tecniche di "pruning" (potatura) come limitare la profondità massima (`max_depth`) o il numero minimo di campioni per foglia (`min_samples_leaf`) aiutano a mitigare questo problema.
* Possono essere instabili: piccole variazioni nei dati di training possono portare a strutture d'albero molto diverse.

**Tipo di Apprendimento:** Supervisionato.

## 📊 Dataset Utilizzato nello Script

Nello script `alberi_decisionali.py`, utilizziamo il dataset **Breast Cancer Wisconsin (Diagnostic)**, fornito da `sklearn.datasets`.

* **Features (X):** Contiene 30 feature numeriche calcolate da immagini digitalizzate di aspirati con ago sottile (FNA) di masse tumorali al seno. Queste feature descrivono caratteristiche dei nuclei cellulari presenti nell'immagine (es. raggio, texture, perimetro, area, levigatezza, ecc.).
* **Target (y):** La diagnosi della massa.
    * Classe 0: Maligno (M)
    * Classe 1: Benigno (B)

## 🐍 Contenuto dello Script `alberi_decisionali.py`

Lo script `alberi_decisionali.py` esegue i seguenti passaggi principali:

1.  **Importazione delle Librerie:** Vengono importate `numpy`, `pandas` (per la gestione delle feature importances), `matplotlib.pyplot`, e moduli specifici da `sklearn` (`DecisionTreeClassifier`, `plot_tree`, `train_test_split`, `accuracy_score`, `confusion_matrix`, `ConfusionMatrixDisplay`, `load_breast_cancer`).
2.  **Caricamento dei Dati:** Viene caricato il dataset Breast Cancer.
3.  **Divisione dei Dati:** Il dataset viene suddiviso in Training Set e Test Set, utilizzando `stratify=y`.
4.  **Creazione e Addestramento del Modello:**
    * Viene creata un'istanza del classificatore `DecisionTreeClassifier`. Nello script, si specificano:
        * `criterion='gini'` (si potrebbe usare anche `'entropy'`).
        * `max_depth`: per limitare la profondità dell'albero e renderlo più interpretabile e meno prono all'overfitting.
        * `random_state` per la riproducibilità.
    * Il modello viene addestrato sui dati di training.
5.  **Effettuare Predizioni:** Il modello addestrato viene utilizzato per predire le diagnosi per i campioni nel test set.
6.  **Valutazione del Modello:** Le prestazioni vengono valutate tramite:
    * **Accuratezza (Accuracy).**
    * **Matrice di Confusione.**
7.  **Visualizzazione dell'Albero:** La struttura dell'albero decisionale addestrato viene visualizzata graficamente utilizzando la funzione `plot_tree`, mostrando le decisioni prese a ogni nodo.
8.  **Importanza delle Features:** Viene calcolata e visualizzata l'importanza di ciascuna feature nel contribuire alle decisioni dell'albero. Le feature più in alto nell'albero o usate per split che riducono maggiormente l'impurità sono generalmente più importanti.

## 🚀 Istruzioni per l'Esecuzione

Per eseguire lo script:

1.  Assicurati di aver attivato il tuo ambiente virtuale Python con le dipendenze installate (incluso `pandas`).
2.  Apri un terminale o prompt dei comandi.
3.  Naviga fino alla directory principale del progetto e poi in questa sottocartella:
    ```bash
    cd 04_Alberi_Decisionali
    ```
4.  Esegui lo script con il comando:
    ```bash
    python alberi_decisionali.py
    ```

## 📈 Output Atteso

Dopo l'esecuzione, vedrai:

* **Nel terminale:**
    * Informazioni sul caricamento e le dimensioni del dataset.
    * La profondità massima impostata e quella effettiva dell'albero.
    * L'accuratezza del modello sul test set.
    * Un elenco delle feature più importanti secondo il modello.
* **Finestre grafiche:**
    1.  La **Matrice di Confusione**.
    2.  Una visualizzazione dettagliata della **struttura dell'Albero Decisionale**.
    3.  Un **grafico a barre** che mostra l'importanza relativa delle feature più rilevanti.

## 💡 Possibili Esperimenti e Modifiche

Prova a modificare lo script per approfondire la tua comprensione degli Alberi Decisionali:

* **Variare `max_depth`:** Prova diversi valori per `max_depth` (es. 2, 3, 5, None per un albero completo). Osserva come cambiano la struttura dell'albero, l'accuratezza sul test set e (se la calcolassi) l'accuratezza sul training set. Come si manifesta l'overfitting?
* **Cambiare Criterio di Split:** Modifica `criterion` da `'gini'` a `'entropy'`. Ci sono differenze significative nelle prestazioni o nella struttura dell'albero?
* **Altri Parametri di Pruning:** Esplora altri parametri come `min_samples_split` (numero minimo di campioni richiesti per splittare un nodo interno) o `min_samples_leaf` (numero minimo di campioni richiesti in un nodo foglia).
* **Altro Dataset:** Prova ad applicare l'albero decisionale al dataset Iris (usato per KNN) o ad un altro dataset di classificazione.
* **Regressione:** Prova ad usare `DecisionTreeRegressor` (da `sklearn.tree`) su un problema di regressione (es. i dati generati per la regressione lineare, o il dataset Boston Housing se disponibile/permesso).
* **Analisi Feature Importance:** Confronta le feature più importanti identificate dall'albero con la tua eventuale conoscenza del dominio del problema (es. quali feature mediche ti aspetteresti siano più rilevanti per la diagnosi del tumore al seno?).

Buona esplorazione con gli Alberi Decisionali!