"""
Data preparation script for NASA bearing dataset
Converts raw vibration data to engineered features and RUL targets
"""
import pandas as pd
import numpy as np
import os

print("="*60)
print("NASA BEARING DATASET - DATA PREPARATION")
print("="*60)

# ===== CONFIGURACIÓN =====
# Cargar datos raw de NASA (ajusta el nombre según tu archivo)
file_path = 'data/raw/bearing_data.txt'

if not os.path.exists(file_path):
    print(f"\n❌ ERROR: No se encontró {file_path}")
    print("Por favor coloca el archivo de datos NASA en data/raw/")
    exit(1)

print(f"\n📂 Cargando datos desde: {file_path}")

# NASA bearing data usa tabulaciones y no tiene headers
df = pd.read_csv(file_path, sep='\t', header=None)

# Asignar nombres de columnas (4 canales de acelerómetros)
df.columns = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']

print(f"✅ Datos cargados: {df.shape[0]:,} mediciones x {df.shape[1]} bearings")
print(f"\nRango de valores:")
print(df.describe())

# ===== FEATURE ENGINEERING =====
print("\n🔧 Generando features estadísticas...")

bearing_cols = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']

# Features de vibración (valores en g - aceleración)
# Convertir de g a mm/s (multiplicar por ~9.81 para aproximación)
# Nota: Los valores NASA están normalizados, usamos directamente

df['max_vibration'] = df[bearing_cols].max(axis=1)
df['mean_vibration'] = df[bearing_cols].mean(axis=1)
df['std_vibration'] = df[bearing_cols].std(axis=1)

# RMS (Root Mean Square) - Métrica estándar ISO 10816
df['rms_vibration'] = np.sqrt((df[bearing_cols] ** 2).mean(axis=1))

# Peak-to-Peak
df['peak_to_peak'] = df[bearing_cols].max(axis=1) - df[bearing_cols].min(axis=1)

# Kurtosis (detecta impulsos - indicador temprano de falla en bearings)
df['kurtosis'] = df[bearing_cols].apply(lambda x: x.kurtosis(), axis=1)

# Crest Factor (peak / RMS) - Alto crest factor indica impactos
df['crest_factor'] = df['max_vibration'] / (df['rms_vibration'] + 1e-10)

print("✅ Features creadas:")
print(f"  - RMS Vibration (ISO 10816 metric)")
print(f"  - Max, Mean, Std, Peak-to-Peak")
print(f"  - Kurtosis (bearing fault indicator)")
print(f"  - Crest Factor (impact detection)")

# ===== VARIABLE TEMPORAL =====
# Asumiendo mediciones cada 10 minutos (ajusta según tu dataset)
df['time_step'] = range(len(df))
df['hours_operation'] = df['time_step'] * 10 / 60  # Convertir a horas

print(f"\n⏱️  Duración total del experimento: {df['hours_operation'].max():.1f} horas")

# ===== CALCULAR RUL (Remaining Useful Life) =====
print("\n🎯 Calculando RUL (Remaining Useful Life)...")

# Vida total del bearing (hasta falla)
total_life_hours = df['hours_operation'].max()

# RUL = Tiempo total - Tiempo transcurrido
df['RUL_hours'] = total_life_hours - df['hours_operation']

# Asegurar que RUL no sea negativo
df['RUL_hours'] = df['RUL_hours'].clip(lower=0)

print(f"✅ RUL calculado:")
print(f"  - Vida total del bearing: {total_life_hours:.1f} horas")
print(f"  - RUL máximo: {df['RUL_hours'].max():.1f} horas")
print(f"  - RUL medio: {df['RUL_hours'].mean():.1f} horas")
print(f"  - RUL mínimo: {df['RUL_hours'].min():.1f} horas")

# ===== CLASIFICACIÓN ISO 10816 (para referencia) =====
# Grupo 2: Máquinas medianas (15-75 kW) en fundaciones rígidas
# Límites en mm/s RMS
def classify_iso10816(rms_value):
    """Clasificación según ISO 10816-1 Grupo 2"""
    if rms_value < 2.8:
        return 'A'
    elif rms_value < 4.5:
        return 'B'
    elif rms_value < 7.1:
        return 'C'
    else:
        return 'D'

df['iso_class'] = df['rms_vibration'].apply(classify_iso10816)

print("\n📊 Distribución ISO 10816 (Grupo 2):")
print(df['iso_class'].value_counts().sort_index())

# ===== GUARDAR DATASET PROCESADO =====
# Seleccionar features finales
feature_cols = [
    'max_vibration', 
    'mean_vibration', 
    'std_vibration', 
    'rms_vibration',
    'peak_to_peak',
    'kurtosis',
    'crest_factor',
    'hours_operation',
    'RUL_hours'
]

df_final = df[feature_cols]

# Crear carpeta si no existe
os.makedirs('data/processed', exist_ok=True)

# Guardar
output_path = 'data/processed/processed_data.csv'
df_final.to_csv(output_path, index=False)

print(f"\n✅ Dataset procesado guardado en: {output_path}")
print(f"📦 Total de muestras: {len(df_final):,}")
print(f"📊 Features: {len(feature_cols) - 1} (+ 1 target)")
print("\n" + "="*60)
print("PREPARACIÓN COMPLETADA")
print("="*60)