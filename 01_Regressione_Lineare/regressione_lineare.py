# Import delle librerie necessarie
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Preparazione dei Dati (Esempio Semplice) ---
# Supponiamo di avere dati sulla dimensione delle case (X) e il loro prezzo (y)
# X = np.array([[50], [60], [70], [80], [90], [100], [110], [120], [130], [140]]) # Mq
# y = np.array([150, 180, 210, 240, 270, 300, 330, 360, 390, 420]) # Prezzo in migliaia di €

# Generiamo dati casuali più realistici per l'esempio
np.random.seed(42) # Per riproducibilità
X = np.sort(np.random.rand(50, 1) * 100 + 50, axis=0) # Dimensioni case tra 50 e 150 mq
y = (2.5 * X.flatten() + np.random.randn(50) * 50 + 20).flatten() # Prezzo con un po' di rumore

# --- 2. Divisione dei Dati in Training Set e Test Set ---
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Per questo esempio semplice con pochi dati, usiamo tutti i dati per il training,
# ma in pratica la divisione è FONDAMENTALE.
# Qui usiamo X e y direttamente per semplicità didattica.
# Se vuoi mostrare la divisione, decommenta la riga sopra e usa X_train, y_train per fit()
# e X_test, y_test per predict() e score().

# --- 3. Creazione e Addestramento del Modello ---
model = LinearRegression()
model.fit(X, y) # Addestriamo il modello

# --- 4. Effettuare Predizioni ---
y_pred = model.predict(X) # Prediciamo i prezzi usando le dimensioni X

# --- 5. Valutazione del Modello ---
# (Se avessimo usato train_test_split, valuteremmo su X_test, y_test)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("--- Regressione Lineare ---")
print(f"Coefficiente (pendenza): {model.coef_[0]:.2f}")
print(f"Intercetta: {model.intercept_:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# --- 6. Visualizzazione ---
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Dati Reali')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regressione Lineare')
plt.xlabel("Dimensione Casa (mq)")
plt.ylabel("Prezzo (migliaia di €)")
plt.title("Regressione Lineare: Prezzo Casa vs. Dimensione")
plt.legend()
plt.grid(True)
plt.show()

# Predizione per una nuova casa
nuova_casa_mq = np.array([[105]])
prezzo_predetto = model.predict(nuova_casa_mq)
print(f"\nPrezzo predetto per una casa di {nuova_casa_mq[0][0]} mq: {prezzo_predetto[0]:.2f} mila €")