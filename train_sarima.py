import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
import warnings
warnings.filterwarnings("ignore")

# Assurer l'encodage correct pour l'affichage console
sys.stdout.reconfigure(encoding='utf-8')

# 1. Configuration des chemins et paramètres
DATA_PATH = "data_cleaned/RP3323_August2024_hourly.csv"
RESULTS_DIR = "results/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_SPLIT = 0.8
# L'ordre SARIMA inclut la saisonnalité s=24 (24 heures dans une journée)
ORDER = (2, 0, 1)            # (p, d, q)
SEASONAL_ORDER = (1, 1, 0, 24) # (P, D, Q, s)

print("========================================")
print("  Entraînement Modèle SARIMA - RP3323")
print("========================================")

# 2. Chargement et Prétraitement
print("Chargement des données...")
df = pd.read_csv(DATA_PATH)

# Gestion des valeurs manquantes
df['average_speed_kph'] = df['average_speed_kph'].interpolate(method='linear')
df['average_speed_kph'] = df['average_speed_kph'].bfill().ffill()

data = df['average_speed_kph'].values

# 3. Séparation Train / Test
split_idx = int(len(data) * TRAIN_SPLIT)
train_data, test_data = data[:split_idx], data[split_idx:]

print(f"Taille de l'ensemble d'entraînement : {len(train_data)} heures")
print(f"Taille de l'ensemble de test        : {len(test_data)} heures")

# 4. Construction et Entraînement du Modèle SARIMA
print(f"\nEntraînement du modèle SARIMA avec l'ordre {ORDER} et saisonnalité {SEASONAL_ORDER}...")
print("Cela peut prendre quelques instants (le calcul de la saisonnalité sur 24h est gourmand)...")

model = SARIMAX(train_data, order=ORDER, seasonal_order=SEASONAL_ORDER, enforce_stationarity=False, enforce_invertibility=False)
fitted_model = model.fit(disp=False)

# 5. Prédiction et Évaluation
print("\nÉvaluation sur l'ensemble de test...")
# Prédiction séquentielle (Multi-step forecast)
predictions = fitted_model.forecast(steps=len(test_data))

rmse = np.sqrt(mean_squared_error(test_data, predictions))
mae = mean_absolute_error(test_data, predictions)

print(f"RMSE (Erreur quadratique moyenne) : {rmse:.2f} km/h")
print(f"MAE (Erreur absolue moyenne)      : {mae:.2f} km/h")

# 6. Visualisation
plt.figure(figsize=(14, 6))
plt.plot(test_data, color='blue', label='Vitesse Réelle (km/h)')
plt.plot(predictions, color='orange', linestyle='--', label='Prédiction SARIMA (km/h)')
plt.title(f'Prédiction de la Congestion (Vitesse) - SARIMA - Route RP3323\nRMSE: {rmse:.2f} | MAE: {mae:.2f}')
plt.xlabel('Heures (Ensemble de test)')
plt.ylabel('Vitesse Moyenne (km/h)')
plt.legend()
plt.grid(True)

plot_path = os.path.join(RESULTS_DIR, "sarima_prediction_RP3323.png")
plt.savefig(plot_path)
plt.close()

print(f"\nGraphique sauvegardé dans : {plot_path}")
print("Terminé !")
