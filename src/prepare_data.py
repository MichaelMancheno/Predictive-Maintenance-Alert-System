# src/prepare_data.py
import pandas as pd
import numpy as np
import os

# --- 1. CARGA DE DATOS ---
# Ajusta la ruta si es necesario
file_path = 'data/raw/bearing_data.txt' 
print(f"📂 Cargando datos desde: {file_path}")

try:
    df = pd.read_csv(file_path, sep='\t', header=None)
except FileNotFoundError:
    # Intento alternativo por si la ruta cambia
    df = pd.read_csv('../data/raw/bearing_data.txt', sep='\t', header=None)

df.columns = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']
print(f"✅ Datos cargados: {df.shape}")

# --- 2. INGENIERÍA DE VARIABLES ---
bearing_cols = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']

df['max_vibration'] = df[bearing_cols].max(axis=1)
df['mean_vibration'] = df[bearing_cols].mean(axis=1)
df['std_vibration'] = df[bearing_cols].std(axis=1)
df['rms_vibration'] = np.sqrt((df[bearing_cols] ** 2).mean(axis=1))

# Variables temporales simuladas para el dataset NASA
df['time_step'] = range(len(df))
df['hours_operation'] = df['time_step'] * 10 / 60 

# Variables avanzadas (Contexto para el modelo)
df['rms_diff'] = df['rms_vibration'].diff().fillna(0)
# Evitamos división por cero sumando un epsilon (1e-6)
df['severity_ratio'] = df['max_vibration'] / (df['rms_vibration'] + 1e-6)
df['relative_std'] = df['std_vibration'] / (df['mean_vibration'] + 1e-6)

# --- 3. DEFINICIÓN ROBUSTA DEL TARGET (FALLO) ---
# ESTRATEGIA: "Run-to-Failure"
# En este dataset, sabemos que el rodamiento se rompe al final.
# Vamos a etiquetar el último 5% de los datos como "Fallo", sin importar el valor.
# Esto GARANTIZA que tengas clase 1.

punto_de_corte = int(len(df) * 0.95) # El 95% del tiempo es normal
df['failure'] = 0
df.loc[punto_de_corte:, 'failure'] = 1

# --- RED DE SEGURIDAD ---
# Verificamos si logramos crear fallas.
conteo_fallas = df['failure'].sum()

if conteo_fallas == 0:
    print("⚠️ ADVERTENCIA: No se detectaron fallas con el método del 95%.")
    print("🔧 APLICANDO MÉTODO DE RESPALDO: Forzando las últimas 50 filas como falla.")
    df.iloc[-50:, df.columns.get_loc('failure')] = 1

# --- 4. GUARDADO ---
print(f"\n📊 Distribución de clases FINAL:")
print(f"   Normal (0): {sum(df['failure']==0)}")
print(f"   Falla  (1): {sum(df['failure']==1)}")

# Seleccionamos solo las columnas que el modelo usará
features_cols = ['max_vibration', 'mean_vibration', 'std_vibration', 
                 'rms_vibration', 'hours_operation', 'rms_diff', 
                 'severity_ratio', 'relative_std', 'failure']

df_final = df[features_cols]

# Asegurar que la carpeta existe
os.makedirs('data/processed', exist_ok=True)
df_final.to_csv('data/processed/processed_data.csv', index=False)

print(f"\n✅ Dataset procesado guardado correctamente.")