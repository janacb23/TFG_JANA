import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- CAMBIO: Rutas base para procesar todos los usuarios ---
BASE_INPUT_DIR = "landmarks_csvs"
BASE_OUTPUT_DIR = "landmarks_csvs_ojos_boca"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Índices de landmarks (basado en FAN de 68 puntos)
ojos_boca_indices = list(range(36, 48)) + list(range(48, 68))  # ojos + boca

print(f"Procesando landmarks de ojos y boca...")
print(f"Índices seleccionados: {ojos_boca_indices}")
print(f"Total puntos por frame: {len(ojos_boca_indices)} puntos")

# Estadísticas globales
total_files_processed = 0
total_frames_processed = 0
total_nan_frames = 0

# --- CAMBIO: Bucle para recorrer cada carpeta de usuario ---
user_folders = [d for d in os.listdir(BASE_INPUT_DIR) if os.path.isdir(os.path.join(BASE_INPUT_DIR, d))]

for user_folder in tqdm(user_folders, desc="Procesando Usuarios"):
    user_input_path = os.path.join(BASE_INPUT_DIR, user_folder)
    user_output_path = os.path.join(BASE_OUTPUT_DIR, user_folder)
    os.makedirs(user_output_path, exist_ok=True)

    csv_files = [f for f in os.listdir(user_input_path) if f.endswith(".csv")]

    for filename in tqdm(csv_files, desc=f"CSVs de {user_folder}", leave=False):
        file_path = os.path.join(user_input_path, filename)
        
        try:
            df = pd.read_csv(file_path)
            
            # Verificar que existan las columnas necesarias
            required_columns = ["frame", "quality"]
            required_columns += [f"x{i+1}" for i in ojos_boca_indices]
            required_columns += [f"y{i+1}" for i in ojos_boca_indices]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                # Si falta la columna 'quality', no es un error fatal, simplemente no la incluimos.
                if 'quality' in missing_columns:
                    columns_to_keep = ["frame"]
                else: # Si faltan otras columnas, sí es un problema.
                    print(f" Columnas faltantes en {filename}: {missing_columns}. Saltando archivo.")
                    continue
            else:
                 columns_to_keep = ["frame", "quality"]

            # Generar el resto de columnas a mantener
            for i in ojos_boca_indices:
                columns_to_keep.append(f"x{i+1}")
            for i in ojos_boca_indices:
                columns_to_keep.append(f"y{i+1}")

            # Crear y guardar nuevo DataFrame
            reduced_df = df[columns_to_keep]
            
            # Estadísticas del archivo
            file_frames = len(reduced_df)
            file_nan_frames = reduced_df.drop(columns=['frame', 'quality'], errors='ignore').isnull().any(axis=1).sum()
            
            total_frames_processed += file_frames
            total_nan_frames += file_nan_frames
            total_files_processed += 1
            
            # Guardar archivo procesado
            output_path = os.path.join(user_output_path, filename) # Se guarda con el mismo nombre en la nueva carpeta
            reduced_df.to_csv(output_path, index=False)
            
        except Exception as e:
            print(f" Error procesando {filename}: {e}")
            continue

# Reporte final
print(f"\n\n=== REPORTE FINAL ===")
print(f"Archivos CSV procesados: {total_files_processed}")
print(f"Total de frames procesados: {total_frames_processed}")
valid_frames = total_frames_processed - total_nan_frames
print(f"Frames válidos (sin NaN en landmarks): {valid_frames}")
print(f"Frames con problemas (con NaN en landmarks): {total_nan_frames}")
if total_frames_processed > 0:
    quality_percentage = (valid_frames / total_frames_processed) * 100
    print(f"Calidad general de los datos: {quality_percentage:.1f}%")