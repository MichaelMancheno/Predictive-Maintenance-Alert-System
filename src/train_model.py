# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Cargar datos procesados
# Asegúrate de que la ruta sea correcta según tu estructura de carpetas
if os.path.exists('data/processed/processed_data.csv'):
    df = pd.read_csv('data/processed/processed_data.csv')
else:
    # Intento alternativo de ruta si lo corres desde la raíz
    df = pd.read_csv('../data/processed/processed_data.csv')

# --- CORRECCIÓN CRÍTICA AQUÍ ---
# Quitamos 'failure' de esta lista. X solo debe tener los datos de entrada.
features_cols = [
    'max_vibration', 
    'mean_vibration', 
    'std_vibration', 
    'rms_vibration', 
    'hours_operation', 
    'rms_diff', 
    'severity_ratio', 
    'relative_std'
]

# Definir X (Entradas) e y (Objetivo) por separado
X = df[features_cols]  # X tiene 8 columnas
y = df['failure']      # y tiene 1 columna

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalar
# El scaler ahora aprenderá SOLO sobre las 8 columnas de vibración/horas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)

# Evaluar
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.2%}")
print(classification_report(y_test, y_pred))

# Guardar modelos
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Modelo entrenado y guardado correctamente en models/")