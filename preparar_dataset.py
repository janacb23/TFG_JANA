import os
import pandas as pd
import numpy as np
from glob import glob

# --- CONFIGURACIÓN ---
# Elige qué grupo de usuarios procesar: "manual", "kmeans" o "all"
EXPERIMENT_GROUP = "kmeans"
# Elige la estrategia para etiquetar las ventanas de somnolencia
LABELING_STRATEGY = "majority_weighted"  # Opciones: "mode", "threshold", "any_presence", "majority_weighted"

# Rutas principales
LANDMARKS_BASE_DIR = "landmarks_csvs_ojos_boca"
MANUAL_LABELS_DIR = "dataset"
KMEANS_LABELS_DIR = "csvs_advanced"
OUTPUT_DIR = "entrada_CNN"

# Parámetros del preprocesamiento
FPS = 30
WINDOW_SIZE = 30

# Listas de usuarios para cada grupo
MANUAL_USER_IDS = ['user_id_24026', 'user_id_24491', 'user_id_35133', 'user_id_38058', 'user_id_49381']
KMEANS_USER_IDS = ['user_id_42271', 'user_id_56306', 'user_id_65818', 'user_id_72519', 'user_id_79336']

# Crear directorio de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- FUNCIONES ---
def time_to_seconds(t):
    """Convierte tiempo HH:MM:SS a segundos"""
    try:
        if pd.isna(t) or t == '' or str(t).strip() == '':
            return 0
        t_str = str(t).strip()
        if ':' in t_str:
            parts = t_str.split(":")
            if len(parts) == 3:
                h, m, s = parts
                return int(float(h)) * 3600 + int(float(m)) * 60 + int(float(s))
            elif len(parts) == 2:
                m, s = parts
                return int(float(m)) * 60 + int(float(s))
        return float(t_str)
    except:
        return 0

def normalize_landmarks(row):
    """Normaliza landmarks por centroide y escala"""
    # Extraer coordenadas x e y (excluyendo frame y quality)
    coords_cols = [col for col in row.index if col.startswith(('x', 'y')) and not col in ['frame', 'quality']]
    
    # Verificar si hay NaN
    if row[coords_cols].isnull().any():
        return [np.nan] * len(coords_cols)
    
    try:
        xs = [row[c] for c in coords_cols if c.startswith("x")]
        ys = [row[c] for c in coords_cols if c.startswith("y")]
        
        if not xs or not ys:
            return [np.nan] * len(coords_cols)
        
        # Calcular centroide
        cx, cy = np.mean(xs), np.mean(ys)
        
        # Centrar en el origen
        xs_centered = [(x - cx) for x in xs]
        ys_centered = [(y - cy) for y in ys]
        
        # Calcular escala (distancia máxima desde el centro)
        distances = np.sqrt(np.array(xs_centered)**2 + np.array(ys_centered)**2)
        scale = np.max(distances)
        if scale == 0 or np.isnan(scale):
            scale = 1
        
        # Normalizar por la escala
        xs_normalized = [x / scale for x in xs_centered]
        ys_normalized = [y / scale for y in ys_centered]
        
        return xs_normalized + ys_normalized
    except:
        return [np.nan] * len(coords_cols)

def estrategia_etiquetado(labels_ventana, strategy="mode"):
    """Diferentes estrategias para etiquetar ventanas"""
    if len(labels_ventana) == 0:
        return 0
        
    if strategy == "mode":
        # Método original: usar la moda
        mode_result = labels_ventana.mode()
        return mode_result.iloc[0] if len(mode_result) > 0 else 0
    
    elif strategy == "threshold":
        # Si >50% de frames son somnolencia → etiqueta como somnolencia  
        somnolencia_ratio = (labels_ventana == 14).mean()
        return 14 if somnolencia_ratio > 0.5 else 0
    
    elif strategy == "any_presence":
        # Si hay cualquier frame de somnolencia → etiqueta como somnolencia
        return 14 if (labels_ventana == 14).any() else 0
    
    elif strategy == "majority_weighted":
        # Dar más peso a la somnolencia para balancear clases (umbral más bajo)
        somnolencia_ratio = (labels_ventana == 14).mean()
        return 14 if somnolencia_ratio > 0.3 else 0
    
    else:
        mode_result = labels_ventana.mode()
        return mode_result.iloc[0] if len(mode_result) > 0 else 0

def load_labels_csv(user_id, is_manual=True):
    """Carga el CSV de etiquetas para un usuario específico"""
    if is_manual:
        # Buscar CSV en la carpeta del usuario manual
        user_dir = os.path.join(MANUAL_LABELS_DIR, user_id)
        if not os.path.exists(user_dir):
            print(f"⚠️  Directorio no encontrado: {user_dir}")
            return None
        
        csv_files = glob(os.path.join(user_dir, "*.csv"))
        if not csv_files:
            print(f"⚠️  No se encontraron CSVs en: {user_dir}")
            return None
            
        csv_path = csv_files[0]
    else:
        # Buscar CSV en la carpeta de k-means
        csv_path = os.path.join(KMEANS_LABELS_DIR, f"{user_id}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️  Archivo no encontrado: {csv_path}")
            return None
    
    try:
        # Intentar con diferentes separadores
        for sep in [';', ',', '\t']:
            try:
                df = pd.read_csv(csv_path, sep=sep, encoding='utf-8')
                if len(df.columns) >= 7:
                    break
            except:
                try:
                    df = pd.read_csv(csv_path, sep=sep, encoding='latin-1')
                    if len(df.columns) >= 7:
                        break
                except:
                    continue
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip()
        
        # Mapear columnas estándar
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'label' in col_lower or 'class' in col_lower:
                column_mapping[col] = 'Label/Class ID'
            elif 'start' in col_lower and 'time' in col_lower:
                column_mapping[col] = 'Start Time'
            elif 'end' in col_lower and 'time' in col_lower:
                column_mapping[col] = 'End Time'
            elif 'filename' in col_lower or ('file' in col_lower and 'name' in col_lower):
                column_mapping[col] = 'Filename'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Forward fill para nombres de archivo
        if 'Filename' in df.columns:
            df["Filename"] = df["Filename"].ffill()
            df["Filename"] = df["Filename"].astype(str).str.strip()
        
        print(f"✅ CSV cargado: {csv_path} - {len(df)} filas")
        return df
        
    except Exception as e:
        print(f"❌ Error cargando {csv_path}: {e}")
        return None

def label_for_frame(frame, df_labels, filename):
    """Determina la etiqueta para un frame específico"""
    second = frame // FPS
    
    # Buscar etiquetas para este archivo
    video_labels = df_labels[df_labels["Filename"].str.contains(filename.replace('.csv', ''), na=False, case=False)]
    
    if len(video_labels) == 0:
        # Si no hay coincidencia exacta, usar todas las etiquetas
        video_labels = df_labels
    
    for _, row in video_labels.iterrows():
        try:
            start_sec = time_to_seconds(row["Start Time"])
            end_sec = time_to_seconds(row["End Time"])
            
            if start_sec <= second <= end_sec:
                label_id = str(row["Label/Class ID"]).strip()
                try:
                    return int(float(label_id))
                except:
                    return 0
        except:
            continue
    
    return 0

def get_users_to_process():
    """Determina qué usuarios procesar según el grupo de experimento"""
    if EXPERIMENT_GROUP == "manual":
        return [(user_id, True) for user_id in MANUAL_USER_IDS]
    elif EXPERIMENT_GROUP == "kmeans":
        return [(user_id, False) for user_id in KMEANS_USER_IDS]
    elif EXPERIMENT_GROUP == "all":
        manual_users = [(user_id, True) for user_id in MANUAL_USER_IDS]
        kmeans_users = [(user_id, False) for user_id in KMEANS_USER_IDS]
        return manual_users + kmeans_users
    else:
        print(f"❌ EXPERIMENT_GROUP '{EXPERIMENT_GROUP}' no válido. Use 'manual', 'kmeans' o 'all'")
        return []

def find_landmark_files(user_id):
    """Encuentra archivos de landmarks para un usuario"""
    user_landmarks_dir = os.path.join(LANDMARKS_BASE_DIR, user_id)
    
    if not os.path.exists(user_landmarks_dir):
        print(f"⚠️  Directorio de landmarks no encontrado: {user_landmarks_dir}")
        return []
    
    csv_files = glob(os.path.join(user_landmarks_dir, "*.csv"))
    print(f"📄 Archivos de landmarks encontrados para {user_id}: {len(csv_files)}")
    
    return csv_files

def convert_numpy_types(obj):
    """Convierte tipos numpy a tipos serializables para JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convertir las claves que sean numpy types
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# --- FUNCIÓN PRINCIPAL ---
def main():
    print("🚀 INICIANDO PREPARACIÓN DE DATASET")
    print("=" * 60)
    print(f"📊 Grupo de experimento: {EXPERIMENT_GROUP}")
    print(f"🎯 Estrategia de etiquetado: {LABELING_STRATEGY}")
    print(f"🪟 Tamaño de ventana: {WINDOW_SIZE} frames ({WINDOW_SIZE/FPS:.1f}s)")
    print(f"📁 Directorio de salida: {OUTPUT_DIR}")
    print()
    
    # Obtener usuarios a procesar
    users_to_process = get_users_to_process()
    if not users_to_process:
        return
    
    print(f"👥 Usuarios a procesar: {len(users_to_process)}")
    for user_id, is_manual in users_to_process:
        source = "Manual" if is_manual else "K-means"
        print(f"  - {user_id} ({source})")
    print()
    
    # Variables para estadísticas globales
    all_X, all_y = [], []
    global_stats = {
        'users_processed': 0,
        'files_processed': 0,
        'total_frames': 0,
        'total_windows': 0,
        'label_distribution': {},
        'user_stats': {}
    }
    
    # Procesar cada usuario
    for user_id, is_manual in users_to_process:
        print(f"\n{'='*20} PROCESANDO {user_id} {'='*20}")
        source_type = "Manual" if is_manual else "K-means"
        print(f"📋 Fuente de etiquetas: {source_type}")
        
        # Cargar etiquetas
        df_labels = load_labels_csv(user_id, is_manual)
        if df_labels is None:
            print(f"⏭️  Saltando {user_id} - No se pudieron cargar las etiquetas")
            continue
        
        # Mostrar información de las etiquetas
        print(f"📈 Distribución de etiquetas en {user_id}:")
        if 'Label/Class ID' in df_labels.columns:
            label_counts = df_labels['Label/Class ID'].value_counts()
            for label, count in label_counts.head(10).items():
                print(f"  {label}: {count}")
        
        # Encontrar archivos de landmarks
        landmark_files = find_landmark_files(user_id)
        if not landmark_files:
            print(f"⏭️  Saltando {user_id} - No se encontraron archivos de landmarks")
            continue
        
        # Estadísticas del usuario
        user_stats = {
            'source_type': source_type,
            'files_processed': 0,
            'total_frames': 0,
            'valid_frames': 0,
            'total_windows': 0,
            'label_distribution': {}
        }
        
        # Procesar cada archivo de landmarks
        for landmark_file in landmark_files:
            filename = os.path.basename(landmark_file)
            print(f"\n📄 Procesando: {filename}")
            
            try:
                # Cargar landmarks
                df_landmarks = pd.read_csv(landmark_file)
                print(f"  📊 Frames en archivo: {len(df_landmarks)}")
                
                # Verificar calidad de datos
                frames_with_nan = df_landmarks.isnull().any(axis=1).sum()
                valid_frames = len(df_landmarks) - frames_with_nan
                print(f"  ✅ Frames válidos: {valid_frames}")
                print(f"  ⚠️  Frames con NaN: {frames_with_nan}")
                
                if valid_frames < WINDOW_SIZE:
                    print(f"  ⏭️  Muy pocos frames válidos. Saltando archivo.")
                    continue
                
                # Normalizar landmarks
                print("  🔄 Normalizando landmarks...")
                norm_points = df_landmarks.apply(normalize_landmarks, axis=1, result_type="expand")
                
                if norm_points.empty or norm_points.isnull().all().all():
                    print("  ❌ Error en normalización. Saltando archivo.")
                    continue
                
                norm_points.columns = [f"p{i}" for i in range(norm_points.shape[1])]
                
                # Generar etiquetas por frame
                print("  🏷️  Generando etiquetas por frame...")
                labels = df_landmarks["frame"].apply(lambda f: label_for_frame(f, df_labels, filename))
                df_landmarks["label"] = labels
                
                # Estadísticas de etiquetas
                label_counts = labels.value_counts()
                print(f"  📈 Distribución de etiquetas:")
                for label, count in label_counts.items():
                    print(f"    {label}: {count} ({count/len(labels)*100:.1f}%)")
                
                # Crear ventanas deslizantes
                print(f"  🪟 Creando ventanas de {WINDOW_SIZE} frames...")
                file_windows = 0
                
                for i in range(0, len(df_landmarks) - WINDOW_SIZE + 1, WINDOW_SIZE):
                    window_points = norm_points.iloc[i:i + WINDOW_SIZE].values
                    window_labels = df_landmarks["label"].iloc[i:i + WINDOW_SIZE]
                    
                    # Verificar validez de la ventana
                    if np.isnan(window_points).all():
                        continue
                    
                    # Aplicar estrategia de etiquetado
                    label_final = estrategia_etiquetado(window_labels, strategy=LABELING_STRATEGY)
                    
                    all_X.append(window_points)
                    all_y.append(label_final)
                    
                    file_windows += 1
                    
                    # Actualizar estadísticas
                    if label_final not in user_stats['label_distribution']:
                        user_stats['label_distribution'][label_final] = 0
                    user_stats['label_distribution'][label_final] += 1
                
                print(f"  ✅ Ventanas creadas: {file_windows}")
                
                # Actualizar estadísticas del usuario
                user_stats['files_processed'] += 1
                user_stats['total_frames'] += len(df_landmarks)
                user_stats['valid_frames'] += valid_frames
                user_stats['total_windows'] += file_windows
                
                # Actualizar estadísticas globales
                global_stats['files_processed'] += 1
                global_stats['total_frames'] += len(df_landmarks)
                
            except Exception as e:
                print(f"  ❌ Error procesando {filename}: {e}")
                continue
        
        # Guardar estadísticas del usuario
        if user_stats['files_processed'] > 0:
            global_stats['users_processed'] += 1
            global_stats['user_stats'][user_id] = user_stats
            
            print(f"\n📊 Resumen para {user_id}:")
            print(f"  📁 Archivos procesados: {user_stats['files_processed']}")
            print(f"  🎞️  Frames procesados: {user_stats['total_frames']}")
            print(f"  🪟 Ventanas creadas: {user_stats['total_windows']}")
            print(f"  📈 Distribución de etiquetas:")
            for label, count in user_stats['label_distribution'].items():
                percentage = count / user_stats['total_windows'] * 100 if user_stats['total_windows'] > 0 else 0
                print(f"    {label}: {count} ({percentage:.1f}%)")
        else:
            print(f"⚠️  No se procesaron archivos para {user_id}")
    
    # --- ANÁLISIS FINAL Y GUARDADO ---
    if not all_X:
        print("\n❌ No se crearon ventanas. Verifica los datos de entrada.")
        return
    
    print(f"\n{'='*60}")
    print("📊 ANÁLISIS FINAL")
    print(f"{'='*60}")
    
    # Convertir a arrays numpy
    X = np.array(all_X)
    y = np.array(all_y)
    
    print(f"📐 Forma del dataset:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    
    # Actualizar estadísticas globales
    global_stats['total_windows'] = len(all_y)
    
    # Distribución final de clases
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📈 Distribución final de clases:")
    for label, count in zip(unique, counts):
        percentage = count / len(y) * 100
        global_stats['label_distribution'][str(label)] = int(count)  # Convertir clave a string
        print(f"  Clase {label}: {count} ventanas ({percentage:.1f}%)")
    
    # Estadísticas por tipo de fuente
    manual_windows = sum(stats['total_windows'] for user_id, stats in global_stats['user_stats'].items() 
                        if stats['source_type'] == 'Manual')
    kmeans_windows = sum(stats['total_windows'] for user_id, stats in global_stats['user_stats'].items() 
                        if stats['source_type'] == 'K-means')
    
    print(f"\n📊 Distribución por fuente:")
    print(f"  Manual: {manual_windows} ventanas")
    print(f"  K-means: {kmeans_windows} ventanas")
    
    # Guardar datasets
    experiment_suffix = f"{EXPERIMENT_GROUP}_{LABELING_STRATEGY}"
    output_X = os.path.join(OUTPUT_DIR, f"X_{experiment_suffix}.npy")
    output_y = os.path.join(OUTPUT_DIR, f"y_{experiment_suffix}.npy")
    output_stats = os.path.join(OUTPUT_DIR, f"stats_{experiment_suffix}.json")
    
    print(f"\n💾 GUARDANDO DATASETS:")
    np.save(output_X, X)
    np.save(output_y, y)
    
    # Guardar estadísticas - aplicar conversión completa
    import json
    
    # Convertir todas las estadísticas a tipos serializables
    stats_for_json = convert_numpy_types(global_stats)
    
    with open(output_stats, 'w') as f:
        json.dump(stats_for_json, f, indent=2)
    
    print(f"  ✅ X guardado en: {output_X}")
    print(f"  ✅ y guardado en: {output_y}")
    print(f"  ✅ Estadísticas en: {output_stats}")
    
    # Verificación final
    try:
        X_loaded = np.load(output_X)
        y_loaded = np.load(output_y)
        print(f"\n✅ VERIFICACIÓN EXITOSA:")
        print(f"   X shape: {X_loaded.shape}")
        print(f"   y shape: {y_loaded.shape}")
        print(f"   Contiene NaN: {np.isnan(X_loaded).any()}")
        print(f"   Experimento: {EXPERIMENT_GROUP}")
        print(f"   Estrategia: {LABELING_STRATEGY}")
    except Exception as e:
        print(f"\n❌ Error en verificación: {e}")

if __name__ == "__main__":
    main()