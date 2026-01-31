"""
Training script for Predictive Maintenance Model
Entrena el modelo usando las features generadas en prepare_data.py
"""
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

print("="*60)
print("ENTRENAMIENTO DEL MODELO PREDICTIVO")
print("="*60)

# 1. Cargar datos procesados
file_path = 'data/processed/processed_data.csv'

if not os.path.exists(file_path):
    print(f"❌ Error: No existe {file_path}. Ejecuta primero prepare_data.py")
    exit()

df = pd.read_csv(file_path)
print(f"📂 Datos cargados: {df.shape}")

# 2. Separar Features (X) y Target (y)
target = 'RUL_hours'
X = df.drop(columns=[target])
y = df[target]

print("\nFeatures seleccionadas para entrenamiento:")
print(list(X.columns)) 
# ESTO DEBE IMPRIMIR: ['max_vibration', ..., 'kurtosis', 'crest_factor', ...]

# 3. Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Escalar datos (IMPORTANTE: El scaler guardará los nombres de estas columnas)
scaler = StandardScaler()
# Ajustamos el scaler con los datos de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Entrenar Modelo
print("\n🤖 Entrenando Random Forest...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# 6. Evaluar
predictions = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\n📊 Resultados:")
print(f"   MAE: {mae:.2f} horas")
print(f"   R² Score: {r2:.4f}")

# 7. Guardar Modelos
os.makedirs('models', exist_ok=True)

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅ Modelos guardados en carpeta 'models/':")
print("   - model.pkl")
print("   - scaler.pkl")
print("="*60)