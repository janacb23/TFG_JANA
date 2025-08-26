import os
import cv2
import csv
import face_alignment
import torch
import numpy as np
from tqdm import tqdm

# Configuración
BASE_INPUT_DIR = "dataset"
BASE_OUTPUT_DIR  = "landmarks_csvs"
os.makedirs(BASE_OUTPUT_DIR , exist_ok=True)

# Evitar error libiomp5md.dll en Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Inicializar modelo FAN
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)
    print(f"FaceAlignment inicializado en dispositivo: {device}")
except Exception as e:
    print(f"Error inicializando FaceAlignment: {e}")
    exit()

def validate_landmarks(landmarks, frame_shape):
    """
    Validar calidad de landmarks detectados
    """
    if not landmarks:
        return False, "No landmarks detected"
    
    points = landmarks[0]
    h, w = frame_shape[:2]
    
    # 1. Verificar que están dentro del frame
    if np.any(points[:, 0] < 0) or np.any(points[:, 0] > w):
        return False, "Points outside frame width"
    if np.any(points[:, 1] < 0) or np.any(points[:, 1] > h):
        return False, "Points outside frame height"
    
    # 2. Verificar tamaño mínimo de cara
    x_range = np.max(points[:, 0]) - np.min(points[:, 0])
    y_range = np.max(points[:, 1]) - np.min(points[:, 1])
    
    min_face_size = min(w, h) * 0.05  # Al menos 5% del frame (más permisivo)
    if x_range < min_face_size or y_range < min_face_size:
        return False, f"Face too small: {x_range:.1f}x{y_range:.1f}"
    
    # 3. Verificar que no hay puntos agrupados (landmarks colapsados)
    distances = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            distances.append(dist)
    
    mean_dist = np.mean(distances)
    if mean_dist < 2.0:  # Landmarks muy juntos
        return False, f"Collapsed landmarks: mean_dist={mean_dist:.2f}"
    
    return True, "OK"

# Estadísticas de calidad globales
total_frames = 0
valid_frames = 0
quality_issues = {}

# Bucle principal para recorrer cada carpeta de usuario
for user_folder in os.listdir(BASE_INPUT_DIR):
    user_input_path = os.path.join(BASE_INPUT_DIR, user_folder)
    user_output_path = os.path.join(BASE_OUTPUT_DIR, user_folder)

    if not os.path.isdir(user_input_path):
        continue
    os.makedirs(user_output_path, exist_ok=True)

    print(f"\n--- Procesando Usuario: {user_folder} ---")

    # Bucle para procesar cada vídeo dentro de la carpeta del usuario
    for filename in os.listdir(user_input_path):
        if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
            continue

        video_path = os.path.join(user_input_path, filename)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Error: No se pudo abrir el vídeo {filename}")
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_csv = os.path.join(user_output_path, os.path.splitext(filename)[0] + ".csv")
        
        video_total = 0
        video_valid = 0
        video_issues = {}
        
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["frame", "quality"] + [f"x{i+1}" for i in range(68)] + [f"y{i+1}" for i in range(68)]
            writer.writerow(header)

            frame_id = 0
            last_valid_landmarks = None
            
            with tqdm(total=frame_count, desc=f"  Procesando {filename[:30]}...") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    landmarks = fa.get_landmarks(frame)
                    is_valid, quality_msg = validate_landmarks(landmarks, frame.shape)
                    
                    total_frames += 1
                    video_total += 1
                    
                    if is_valid:
                        points = landmarks[0]
                        row = [frame_id, "OK"] + points[:, 0].tolist() + points[:, 1].tolist()
                        valid_frames += 1
                        video_valid += 1
                        last_valid_landmarks = points.copy()
                        
                        # Dibuja landmarks en verde para detecciones válidas
                        for (x, y) in points:
                            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
                            
                    else:
                        if quality_msg not in quality_issues: quality_issues[quality_msg] = 0
                        if quality_msg not in video_issues: video_issues[quality_msg] = 0
                        quality_issues[quality_msg] += 1
                        video_issues[quality_msg] += 1
                        
                        if last_valid_landmarks is not None:
                            row = [frame_id, f"INTERPOLATED_{quality_msg}"] + last_valid_landmarks[:, 0].tolist() + last_valid_landmarks[:, 1].tolist()
                            # Dibuja landmarks en amarillo para datos interpolados
                            for (x, y) in last_valid_landmarks:
                                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), -1)
                        else:
                            row = [frame_id, quality_msg] + [np.nan] * 136
                        
                        # Escribe "LOW QUALITY" en el frame
                        cv2.putText(frame, "LOW QUALITY", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    writer.writerow(row)
                    
                    # --- CÓDIGO DE VISUALIZACIÓN REINCORPORADO ---
                    #cv2.imshow('Landmarks', frame)
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #    break
                    # --- FIN DEL CÓDIGO DE VISUALIZACIÓN ---
                    
                    frame_id += 1
                    pbar.update(1)

        cap.release()
        #cv2.destroyAllWindows() # <-- Se añade para cerrar la ventana después de cada vídeo
        
        if video_total > 0:
            print(f"    - {filename}: {video_valid}/{video_total} frames válidos ({video_valid/video_total*100:.1f}%)")
# Reporte final de calidad
print(f"\n=== REPORTE FINAL DE CALIDAD ===")
print(f"Total frames procesados: {total_frames}")
if total_frames > 0:
    print(f"Frames válidos: {valid_frames} ({valid_frames/total_frames*100:.1f}%)")
    print(f"Frames con problemas: {total_frames - valid_frames}")

if quality_issues:
    print("\nTipos de problemas encontrados:")
    for issue, count in quality_issues.items():
        print(f"  {issue}: {count} frames ({count/total_frames*100:.1f}%)")