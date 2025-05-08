# Lezione Pratica di Intelligenza Artificiale con Python

Benvenuti a questa raccolta di script Python ed esempi pratici pensati per introdurre i concetti fondamentali dell'Intelligenza Artificiale e del Machine Learning. Questo repository è stato creato per accompagnare una lezione interattiva, permettendo di esplorare e sperimentare direttamente con il codice.

## 🎯 Obiettivi

* Fornire una panoramica dei concetti chiave dell'IA e del Machine Learning.
* Mostrare l'implementazione di alcuni algoritmi fondamentali utilizzando Python e la libreria `scikit-learn`.
* Incoraggiare la sperimentazione e l'apprendimento pratico.

##  Prerequisites

Prima di iniziare, assicurati di avere installato sul tuo sistema:

* **Python** (versione 3.8 o superiore consigliata). Puoi scaricarlo da [python.org](https://www.python.org/).
* **pip** (il gestore di pacchetti Python, solitamente installato con Python).
* **Git** (per clonare il repository). Puoi scaricarlo da [git-scm.com](https://git-scm.com/).

## 🚀 Setup dell'Ambiente

Segui questi passaggi per configurare l'ambiente di lavoro sul tuo computer:

1.  **Clona il Repository:**
    Apri un terminale o prompt dei comandi e digita:
    ```bash
    git clone https://github.com/francobelloni85/introduzione-ai.git
    ```
    ```bash
    cd introduzione-ai
    ```
   

2.  **Crea un Ambiente Virtuale (Consigliato):**
    È buona pratica utilizzare un ambiente virtuale per isolare le dipendenze del progetto.
    ```bash
    python -m venv venv
    ```
    Attiva l'ambiente virtuale:
    * **Su Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **Su macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    Dovresti vedere `(venv)` all'inizio del prompt del terminale, a indicare che l'ambiente virtuale è attivo.

3.  **Installa le Dipendenze:**
    Con l'ambiente virtuale attivo, installa tutte le librerie necessarie eseguendo:
    ```bash
    pip install -r requirements.txt
    ```

## 📁 Struttura del Repository

Il repository è organizzato come segue:

* **`README.md`**: Questo file che stai leggendo.
* **`requirements.txt`**: Elenco delle librerie Python necessarie.
* **`.gitignore`**: Specifica i file ignorati da Git.
* **`LICENSE`**: La licenza sotto cui è rilasciato questo codice.
* **`00_Introduzione_IA/`**: (Opzionale) Materiale introduttivo sull'Intelligenza Artificiale.
* **`01_Regressione_Lineare/`**: Script ed esempi per la Regressione Lineare.
    * `regressione_lineare.py`
    * `README.md` (dettagli specifici sull'algoritmo)
* **`02_K_Nearest_Neighbors/`**: Script ed esempi per l'algoritmo K-Nearest Neighbors (KNN).
    * `knn_classificazione.py`
    * `README.md`
* **`03_K_Means_Clustering/`**: Script ed esempi per l'algoritmo K-Means Clustering.
    * `kmeans_clustering.py`
    * `README.md`
* **`04_Alberi_Decisionali/`**: Script ed esempi per gli Alberi Decisionali.
    * `alberi_decisionali.py`
    * `README.md`
* **`05_Rete_Neurale_Semplice/`**: (Opzionale) Script ed esempi per una Rete Neurale Semplice (MLP).
    * `rete_neurale_mlp.py`
    * `README.md`
* **`utils/`**: (Opzionale) Eventuali script di utilità condivisi.

## 💻 Come Eseguire gli Script

1.  Assicurati che il tuo ambiente virtuale sia attivo.
2.  Naviga nella cartella dell'algoritmo che ti interessa (es. `cd 01_Regressione_Lineare/`).
3.  Esegui lo script Python corrispondente (es. `python regressione_lineare.py`).
4.  Consulta il file `README.md` specifico all'interno di ogni cartella per maggiori dettagli sull'algoritmo e sullo script.

## 🛠️ Sperimenta!

Sentiti libero di modificare gli script, cambiare i parametri degli algoritmi, provare con dataset diversi (molti sono disponibili in `sklearn.datasets`) o integrare nuove funzionalità. L'obiettivo è imparare sperimentando!

## 🤝 Contributi

Se hai suggerimenti, correzioni o miglioramenti, sentiti libero di aprire una "Issue" o una "Pull Request" su GitHub.

## 📜 Licenza

Questo progetto è rilasciato sotto la Licenza MIT. Vedi il file `LICENSE` per maggiori dettagli.

---

Buon apprendimento e buona programmazione!
