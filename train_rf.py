import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys

# Assurer l'encodage correct pour l'affichage console
sys.stdout.reconfigure(encoding='utf-8')

# 1. Configuration des chemins et paramètres
DATA_PATH = "data_cleaned/RP3323_August2024_hourly.csv"
RESULTS_DIR = "results/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)

LOOKBACK = 24  # Utiliser les 24h précédentes pour prédire la prochaine
TRAIN_SPLIT = 0.8
N_ESTIMATORS = 100

print("========================================")
print("  Entraînement Modèle Random Forest - RP3323")
print("========================================")

# 2. Chargement et Prétraitement
print("Chargement des données...")
df = pd.read_csv(DATA_PATH)

# Interpolation des valeurs manquantes
df['average_speed_kph'] = df['average_speed_kph'].interpolate(method='linear')
df['average_speed_kph'] = df['average_speed_kph'].bfill().ffill()

# 3. Ingénierie des Caractéristiques (Feature Engineering pour Random Forest)
print("Création des caractéristiques temporelles (Lags)...")
# On crée 24 colonnes, chacune décalée d'une heure supplémentaire
for i in range(1, LOOKBACK + 1):
    df[f'lag_{i}'] = df['average_speed_kph'].shift(i)

# On supprime les 24 premières lignes qui contiennent des NaN à cause du décalage
df = df.dropna().reset_index(drop=True)

# Définition des variables explicatives (X) et de la variable cible (y)
feature_cols = [f'lag_{i}' for i in range(1, LOOKBACK + 1)]
X = df[feature_cols].values
y = df['average_speed_kph'].values

# Séparation Train / Test sans mélanger (pour respecter l'ordre chronologique)
split_idx = int(len(X) * TRAIN_SPLIT)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Forme de X_train : {X_train.shape}")
print(f"Forme de X_test  : {X_test.shape}")

# 4. Construction et Entraînement du Modèle Random Forest
print("\nEntraînement du modèle Random Forest...")
model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Prédiction et Évaluation
print("\nÉvaluation sur l'ensemble de test...")
predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print(f"RMSE (Erreur quadratique moyenne) : {rmse:.2f} km/h")
print(f"MAE (Erreur absolue moyenne)      : {mae:.2f} km/h")

# 6. Visualisation
plt.figure(figsize=(14, 6))
plt.plot(y_test, color='blue', label='Vitesse Réelle (km/h)')
plt.plot(predictions, color='red', linestyle='--', label='Prédiction Random Forest (km/h)')
plt.title(f'Prédiction de la Congestion (Vitesse) - Route RP3323\nRMSE: {rmse:.2f} | MAE: {mae:.2f}')
plt.xlabel('Heures (Ensemble de test)')
plt.ylabel('Vitesse Moyenne (km/h)')
plt.legend()
plt.grid(True)

plot_path = os.path.join(RESULTS_DIR, "rf_prediction_RP3323.png")
plt.savefig(plot_path)
plt.close()

print(f"\nGraphique sauvegardé dans : {plot_path}")
print("Terminé !")
