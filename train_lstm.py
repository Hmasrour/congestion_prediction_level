import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import sys

# Assurer l'encodage correct pour l'affichage console
sys.stdout.reconfigure(encoding='utf-8')

# 1. Configuration des chemins et paramètres
DATA_PATH = "data_cleaned/RP3323_August2024_hourly.csv"
RESULTS_DIR = "results/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)

LOOKBACK = 24  # Utiliser les 24h précédentes pour prédire la prochaine
TRAIN_SPLIT = 0.8
EPOCHS = 50
BATCH_SIZE = 16

print("========================================")
print("  Entraînement du Modèle LSTM - RP3323")
print("========================================")

# 2. Chargement et Prétraitement
print("Chargement des données...")
df = pd.read_csv(DATA_PATH)

# Interpolation des valeurs manquantes (au cas où il y a des NaN)
df['average_speed_kph'] = df['average_speed_kph'].interpolate(method='linear')
df['average_speed_kph'] = df['average_speed_kph'].bfill().ffill()

data = df['average_speed_kph'].values.reshape(-1, 1)

# Normalisation entre 0 et 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. Création des séquences X, Y
X, y = [], []
for i in range(LOOKBACK, len(scaled_data)):
    X.append(scaled_data[i-LOOKBACK:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
# Le LSTM de Keras attend une forme [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Séparation Train / Test
split_idx = int(len(X) * TRAIN_SPLIT)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Forme de X_train : {X_train.shape}")
print(f"Forme de X_test  : {X_test.shape}")

# 4. Construction de l'Architecture LSTM
print("\nConstruction du modèle LSTM...")
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Entraînement
print("Début de l'entraînement...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# 6. Prédiction et Évaluation
print("\nÉvaluation sur l'ensemble de test...")
predictions = model.predict(X_test)
# Inverser la normalisation pour avoir des km/h
predictions_inv = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
mae = mean_absolute_error(y_test_inv, predictions_inv)

print(f"RMSE (Erreur quadratique moyenne) : {rmse:.2f} km/h")
print(f"MAE (Erreur absolue moyenne)      : {mae:.2f} km/h")

# 7. Visualisation
plt.figure(figsize=(14, 6))
plt.plot(y_test_inv, color='blue', label='Vitesse Réelle (km/h)')
plt.plot(predictions_inv, color='red', linestyle='--', label='Prédiction LSTM (km/h)')
plt.title(f'Prédiction de la Congestion (Vitesse) - Route RP3323\nRMSE: {rmse:.2f} | MAE: {mae:.2f}')
plt.xlabel('Heures (Ensemble de test)')
plt.ylabel('Vitesse Moyenne (km/h)')
plt.legend()
plt.grid(True)

plot_path = os.path.join(RESULTS_DIR, "lstm_prediction_RP3323.png")
plt.savefig(plot_path)
plt.close()

print(f"Graphique sauvegardé dans : {plot_path}")
print("Terminé !")
