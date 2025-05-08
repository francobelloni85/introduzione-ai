#  Rete Neurale Semplice (Multi-Layer Perceptron - MLP)

Questo esempio introduce il **Multi-Layer Perceptron (MLP)**, un tipo fondamentale di Rete Neurale Artificiale (ANN) feedforward. Gli MLP sono utilizzati per una vasta gamma di compiti di apprendimento supervisionato, inclusa la classificazione (come in questo script) e la regressione.

## 📚 Breve Descrizione Teorica

Una **Rete Neurale Artificiale** è un modello computazionale ispirato alla struttura e al funzionamento delle reti neurali biologiche del cervello. Un **Multi-Layer Perceptron (MLP)** è composto da più strati (layer) di nodi (neuroni):

1.  **Layer di Input:** Riceve le features del dataset (un neurone per ogni feature).
2.  **Layer Nascosti (Hidden Layers):** Uno o più strati intermedi che eseguono trasformazioni non lineari sui dati. Ogni neurone in un layer nascosto è tipicamente connesso a tutti i neuroni del layer precedente e del layer successivo. Il numero di layer nascosti e il numero di neuroni in ciascuno di essi definiscono l'architettura della rete.
3.  **Layer di Output:** Produce il risultato finale (es. probabilità per ciascuna classe in un problema di classificazione).

**Componenti chiave:**
* **Neuroni:** Unità computazionali che ricevono input, applicano una somma pesata, aggiungono un bias e poi passano il risultato attraverso una **funzione di attivazione**.
* **Pesi e Bias:** Parametri del modello che vengono appresi durante la fase di addestramento.
* **Funzioni di Attivazione:** Introducono non linearità nella rete, permettendo di apprendere relazioni complesse. Esempi comuni sono:
    * **ReLU (Rectified Linear Unit):** `f(x) = max(0, x)`. Molto usata nei layer nascosti.
    * **Sigmoide (Logistic):** `f(x) = 1 / (1 + exp(-x))`. Produce output tra 0 e 1, usata spesso nel layer di output per classificazione binaria.
    * **Softmax:** Usata tipicamente nel layer di output per classificazione multi-classe, produce una distribuzione di probabilità sulle classi.
    * **Tanh (Hyperbolic Tangent):** `f(x) = tanh(x)`. Produce output tra -1 e 1.
* **Apprendimento (Training):**
    * **Feedforward:** I dati passano dal layer di input, attraverso i layer nascosti, fino al layer di output per generare una predizione.
    * **Funzione di Loss (o Costo):** Misura la discrepanza tra le predizioni del modello e i valori reali (target).
    * **Backpropagation (Retropropagazione dell'Errore):** Algoritmo che calcola il gradiente della funzione di loss rispetto a ciascun peso e bias nella rete.
    * **Ottimizzazione (es. Discesa del Gradiente):** I pesi e i bias vengono aggiornati iterativamente in direzione opposta al gradiente per minimizzare la funzione di loss. Varianti comuni includono Adam, SGD (Stochastic Gradient Descent).
* **Scaling delle Features:** Le reti neurali sono molto sensibili alla scala delle features di input. È cruciale standardizzare (media 0, deviazione standard 1) o normalizzare (range 0-1) i dati.
* **Overfitting:** Le reti neurali, specialmente quelle complesse, possono facilmente andare in overfitting. Tecniche per mitigarlo includono la regolarizzazione (es. L2, `alpha` in `MLPClassifier`), il dropout (non direttamente in `MLPClassifier` base di sklearn), e l'**early stopping** (interrompere l'addestramento quando le prestazioni su un set di validazione iniziano a peggiorare).

**Tipo di Apprendimento:** Supervisionato.

## 📊 Dataset Utilizzato nello Script

Nello script `rete_neurale_mlp.py`, utilizziamo il dataset **Digits**, fornito da `sklearn.datasets`.

* **Features (X):** Immagini 8x8 di cifre scritte a mano (da 0 a 9). Ogni immagine è appiattita in un vettore di 64 pixel (features), dove ogni pixel ha un valore di intensità (da 0 a 16).
* **Target (y):** La cifra effettiva (0, 1, ..., 9) rappresentata nell'immagine.

## 🐍 Contenuto dello Script `rete_neurale_mlp.py`

Lo script `rete_neurale_mlp.py` esegue i seguenti passaggi principali:

1.  **Importazione delle Librerie:** Vengono importate `numpy`, `matplotlib.pyplot`, e moduli specifici da `sklearn` (`MLPClassifier`, `train_test_split`, `StandardScaler`, metriche di valutazione, `load_digits`).
2.  **Caricamento dei Dati:** Viene caricato il dataset Digits.
3.  **Divisione dei Dati:** Il dataset viene suddiviso in Training Set e Test Set, utilizzando `stratify=y`.
4.  **Standardizzazione delle Features:** Le features (i valori dei pixel) vengono standardizzate. **Questo passaggio è cruciale per le prestazioni delle reti neurali.**
5.  **Creazione e Addestramento del Modello `MLPClassifier`:**
    * Viene creata un'istanza del modello, specificando parametri chiave come:
        * `hidden_layer_sizes`: Architettura dei layer nascosti (es. due layer con 100 e 50 neuroni).
        * `activation`: Funzione di attivazione (es. `'relu'`).
        * `solver`: Ottimizzatore (es. `'adam'`).
        * `alpha`: Termine di regolarizzazione L2.
        * `learning_rate_init`: Tasso di apprendimento iniziale.
        * `max_iter`: Numero massimo di epoche di addestramento.
        * `early_stopping`, `validation_fraction`, `n_iter_no_change`: Parametri per l'interruzione anticipata dell'addestramento per prevenire l'overfitting.
        * `random_state` per la riproducibilità.
    * Il modello viene addestrato sui dati di training standardizzati.
6.  **Effettuare Predizioni:** Il modello addestrato viene utilizzato per predire le cifre per i campioni nel test set.
7.  **Valutazione del Modello:** Le prestazioni vengono valutate tramite:
    * **Accuratezza (Accuracy).**
    * **Matrice di Confusione.**
8.  **Visualizzazione:**
    * Viene mostrata graficamente la **Matrice di Confusione**.
    * Viene plottata la **Curva di Loss** del modello durante l'addestramento, se disponibile.
    * Vengono visualizzate alcune **immagini di cifre** dal test set con le relative etichette vere e quelle predette dal modello.

## 🚀 Istruzioni per l'Esecuzione

Per eseguire lo script:

1.  Assicurati di aver attivato il tuo ambiente virtuale Python con le dipendenze installate.
2.  Apri un terminale o prompt dei comandi.
3.  Naviga fino alla directory principale del progetto e poi in questa sottocartella:
    ```bash
    cd 05_Rete_Neurale_Semplice
    ```
4.  Esegui lo script con il comando:
    ```bash
    python rete_neurale_mlp.py
    ```
L'addestramento potrebbe richiedere qualche secondo.

## 📈 Output Atteso

Dopo l'esecuzione, vedrai:

* **Nel terminale:**
    * Informazioni sul caricamento e le dimensioni del dataset.
    * Dettagli sull'architettura della rete e sull'addestramento (es. numero di iterazioni).
    * L'accuratezza del modello MLP sul test set.
* **Finestre grafiche:**
    1.  La **Matrice di Confusione**.
    2.  La **Curva di Loss** che mostra l'andamento dell'errore durante l'addestramento.
    3.  Una griglia di **immagini di cifre** dal test set con le etichette vere e predette.

## 💡 Possibili Esperimenti e Modifiche

Prova a modificare lo script per approfondire la tua comprensione delle Reti Neurali:

* **Architettura della Rete:** Cambia `hidden_layer_sizes`. Prova con un solo layer nascosto, più layer, o un numero diverso di neuroni per layer. Come influisce sulle prestazioni e sul tempo di addestramento?
* **Funzioni di Attivazione:** Sperimenta con diverse `activation` (es. `'logistic'`, `'tanh'`) per i layer nascosti.
* **Ottimizzatori e Tasso di Apprendimento:** Prova diversi `solver` (es. `'sgd'`) e aggiusta `learning_rate_init`.
* **Parametri di Addestramento:** Modifica `max_iter` e i parametri relativi a `early_stopping`. Cosa succede se disabiliti l'early stopping e aumenti `max_iter` (rischio di overfitting)?
* **Regolarizzazione:** Varia il parametro `alpha` (regolarizzazione L2).
* **Scaling:** Prova a commentare la standardizzazione delle features. Come cambiano drasticamente (in peggio) le prestazioni?
* **Altri Dataset:** Applica l'MLP ai dataset usati negli esempi precedenti (es. Breast Cancer, Iris), ricordando di scalarli sempre. Come si confrontano le prestazioni con gli altri algoritmi?
* **(Avanzato) Librerie Dedicate:** Per reti neurali più complesse e un controllo più granulare, esplora librerie come TensorFlow (con Keras) o PyTorch.

Buona esplorazione del mondo delle Reti Neurali!