#  Regressione Lineare

Questo esempio dimostra l'utilizzo della Regressione Lineare, un algoritmo fondamentale di machine learning supervisionato, per modellare la relazione tra una variabile indipendente (feature) e una variabile dipendente (target) continua.

## 📚 Breve Descrizione Teorica

La **Regressione Lineare** ha lo scopo di trovare la migliore linea retta (o iperpiano, in caso di più features) che descrive la relazione tra le variabili. L'equazione di una retta è tipicamente $y = mx + q$, dove:
* $y$ è la variabile dipendente (quella che vogliamo predire).
* $x$ è la variabile indipendente (la feature che usiamo per la predizione).
* $m$ è il coefficiente angolare (o pendenza) della retta, che indica come $y$ cambia al variare di $x$.
* $q$ è l'intercetta, ovvero il valore di $y$ quando $x$ è zero.

L'algoritmo cerca di trovare i valori di $m$ e $q$ che minimizzano l'errore tra i valori predetti dalla retta e i valori reali osservati nel dataset di training. È un algoritmo di **apprendimento supervisionato** perché impara da dati etichettati (coppie di input $X$ e output $y$ noti).

## 📊 Dataset Utilizzato nello Script

Nello script `regressione_lineare.py`, i dati sono **generati sinteticamente** utilizzando la libreria `numpy`. Questo ci permette di avere un dataset semplice e controllabile per illustrare il funzionamento dell'algoritmo.

* **Feature (Variabile Indipendente X):** "Dimensione Casa (mq)" - Rappresenta la superficie di un'abitazione, generata casualmente ma con una tendenza.
* **Target (Variabile Dipendente y):** "Prezzo Stimato (migliaia di €)" - Rappresenta il prezzo della casa, calcolato come una funzione lineare della dimensione più un certo "rumore" casuale per rendere i dati più realistici.

## 🐍 Contenuto dello Script `regressione_lineare.py`

Lo script `regressione_lineare.py` esegue i seguenti passaggi:

1.  **Importazione delle Librerie:** Vengono importate `numpy` per la manipolazione numerica, `matplotlib.pyplot` per la visualizzazione, e moduli specifici da `sklearn` (`LinearRegression`, `train_test_split`, `mean_squared_error`, `r2_score`).
2.  **Generazione dei Dati:** Vengono creati i dati sintetici per la dimensione delle case (X) e i loro prezzi (y). Viene usato `np.random.seed(42)` per garantire che i dati generati siano sempre gli stessi ad ogni esecuzione, rendendo l'esempio riproducibile.
3.  **Divisione dei Dati:** Il dataset viene suddiviso in un **Training Set** (usato per addestrare il modello) e un **Test Set** (usato per valutare le prestazioni del modello su dati mai visti prima). Questo è un passaggio cruciale per una valutazione oggettiva del modello.
4.  **Creazione e Addestramento del Modello:** Viene creata un'istanza del modello `LinearRegression` e viene addestrata utilizzando i dati del training set (`model.fit(X_train, y_train)`).
5.  **Effettuare Predizioni:** Il modello addestrato viene utilizzato per predire i prezzi delle case basandosi sulle dimensioni presenti nel test set (`model.predict(X_test)`).
6.  **Valutazione del Modello:** Le prestazioni del modello vengono valutate utilizzando due metriche comuni:
    * **Mean Squared Error (MSE):** L'errore quadratico medio. Più basso è, meglio è.
    * **R-squared (R²):** Il coefficiente di determinazione. Indica quanta parte della varianza della variabile dipendente è spiegata dal modello. Un valore più vicino a 1 indica un modello migliore.
7.  **Visualizzazione:** Viene generato un grafico che mostra:
    * I punti dati del training set (blu).
    * I punti dati del test set (verdi).
    * La linea di regressione trovata dal modello (rossa).
8.  **Predizione su Nuovi Dati:** Viene mostrato un esempio di come utilizzare il modello addestrato per predire il prezzo di una nuova casa con una dimensione specifica.

## 🚀 Istruzioni per l'Esecuzione

Per eseguire lo script:

1.  Assicurati di aver attivato il tuo ambiente virtuale Python (se ne stai usando uno) dove hai installato le dipendenze.
2.  Apri un terminale o prompt dei comandi.
3.  Naviga fino alla directory principale del progetto (es. `IA_Python_Lezione/`) e poi in questa sottocartella:
    ```bash
    cd 01_Regressione_Lineare
    ```
4.  Esegui lo script con il comando:
    ```bash
    python regressione_lineare.py
    ```

## 📈 Output Atteso

Dopo l'esecuzione, vedrai:

* **Nel terminale:**
    * I valori del coefficiente (pendenza) e dell'intercetta della retta di regressione.
    * L'equazione della retta.
    * Il Mean Squared Error (MSE) e il coefficiente R-squared (R²) calcolati sul test set.
    * Un esempio di predizione per una nuova dimensione di casa.
* **Una finestra grafica:**
    * Un grafico a dispersione con i punti dati di training e test, e la linea di regressione sovrapposta.

## 💡 Possibili Esperimenti e Modifiche

Prova a modificare lo script per esplorare ulteriormente:

* Cambia il valore di `np.random.seed()` o i parametri nella generazione dei dati (es. la quantità di "rumore" aggiunto a `y`) e osserva come cambiano i risultati e il grafico.
* Modifica la proporzione tra training e test set cambiando il parametro `test_size` in `train_test_split()`. Come influisce sulle metriche di valutazione?
* Prova a commentare la parte di divisione dei dati e ad addestrare il modello sull'intero dataset (`model.fit(X, y)`). Come cambiano le metriche se valuti sullo stesso dataset usato per l'addestramento? (Attenzione: questo porta a overfitting e non è una buona pratica per la valutazione reale!).
* Aggiungi più "rumore" ai dati `y` (aumentando il fattore moltiplicativo di `np.random.randn()`) e osserva come R² e MSE peggiorano.
* (Più avanzato) Prova ad aggiungere una seconda feature sintetica a `X` e adatta il modello per una regressione lineare multipla. Come cambia l'interpretazione dei coefficienti?

Buona sperimentazione!