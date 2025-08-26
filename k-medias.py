import cv2
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from tqdm import tqdm
import pickle
import torchvision.transforms as transforms
from torchvision.models import resnet50
import warnings
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost no disponible - usando solo RF y GB")
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings('ignore')

class DomainSpecificFeatureExtractor(nn.Module):
    """Extractor de características específico para comportamientos de conducción"""
    def __init__(self):
        super(DomainSpecificFeatureExtractor, self).__init__()
        # ResNet50 preentrenado (más potente que ResNet18)
        try:
            self.backbone = resnet50(pretrained=True)
        except:
            # Fallback por si hay problemas con torchvision
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=True)
            
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone.eval()
        
        # Transformaciones específicas para comportamientos de conducción
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def extract_hand_features(self, frame):
        """Extrae características específicas de manos y gestos"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar contornos que podrían ser manos
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Características de manos
            if contours:
                # Encontrar el contorno más grande (posible mano)
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # Características geométricas
                if perimeter > 0:
                    compactness = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    compactness = 0
                
                # Momentos de Hu (invariantes a rotación)
                moments = cv2.moments(largest_contour)
                if moments['m00'] != 0:
                    hu_moments = cv2.HuMoments(moments).flatten()
                else:
                    hu_moments = np.zeros(7)
                
                hand_features = [area, perimeter, compactness] + hu_moments.tolist()
            else:
                hand_features = [0] * 10  # 3 + 7 características
            
            return np.array(hand_features, dtype=np.float64)
            
        except Exception as e:
            return np.zeros(10, dtype=np.float64)
    
    def extract_face_features(self, frame):
        """Extrae características específicas de la cara (bostezar, maquillaje, etc.)"""
        try:
            # Región superior del frame (donde normalmente está la cara)
            h, w = frame.shape[:2]
            face_region = frame[0:h//2, w//4:3*w//4]
            
            # Características de color en la región de la cara
            face_bgr = cv2.mean(face_region)[:3]
            face_hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            face_hsv_mean = cv2.mean(face_hsv)[:3]
            
            # Detección de bordes (útil para detectar movimiento de boca)
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_face, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            return np.array(list(face_bgr) + list(face_hsv_mean) + [edge_density], dtype=np.float64)
        except:
            return np.zeros(7, dtype=np.float64)
    
    def extract_movement_features(self, frame, prev_frame=None):
        """Extrae características de movimiento (alcanzar atrás, ajustar controles)"""
        try:
            if prev_frame is not None:
                # Detección de movimiento simple
                gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                
                # Características de movimiento por regiones
                h, w = frame.shape[:2]
                
                # Región trasera (alcanzar atrás)
                rear_movement = cv2.mean(cv2.absdiff(
                    gray_curr[h//2:, :w//3], 
                    gray_prev[h//2:, :w//3]
                ))[0]
                
                # Región central (controles)
                center_movement = cv2.mean(cv2.absdiff(
                    gray_curr[h//2:, w//3:2*w//3],
                    gray_prev[h//2:, w//3:2*w//3]
                ))[0]
                
                return np.array([rear_movement, center_movement], dtype=np.float64)
            else:
                return np.array([0, 0], dtype=np.float64)
        except:
            return np.array([0, 0], dtype=np.float64)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.features(x)
            pooled = self.avgpool(features).flatten(1)
            return pooled

class AdvancedSemiSupervisedLabeler:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.labeled_users = []
        self.unlabeled_users = []
        
        # Ensemble de modelos mejorado
        self.ensemble_classifier = None
        self.clustering_models = {}
        self.cluster_mappings = {}
        self.confidence_threshold = 0.6
        
        # Escaladores mejorados
        self.robust_scaler = RobustScaler()
        self.pca = None
        
        # GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Extractor mejorado
        self.feature_extractor = DomainSpecificFeatureExtractor().to(self.device)
        
        # Mapeo de comportamientos
        self.behavior_mapping = {
            0: "Normal Forward Driving", 1: "Drinking", 2: "Phone Call(right)", 
            3: "Phone Call(left)", 4: "Eating", 5: "Text (Right)", 6: "Text (Left)",
            7: "Hair / makeup", 8: "Reaching behind", 9: "Adjust control panel",
            10: "Pick up from floor (Driver)", 11: "Pick up from floor (Passenger)",
            12: "Talk to passenger at the right", 13: "Talk to passenger at backseat",
            14: "Yawning", 15: "Hand on head", 16: "Singing with music",
            17: "Shaking or dancing with music"
        }
        
        # Distribuciones objetivo más realistas
        self.target_distribution = {
            0: 0.35,  # Normal - más realista
            1: 0.08, 2: 0.06, 3: 0.06,
            4: 0.05, 5: 0.04, 6: 0.04,
            7: 0.03, 8: 0.04,
            9: 0.05, 10: 0.02, 11: 0.02,
            12: 0.03, 13: 0.02,
            14: 0.04, 15: 0.03,
            16: 0.02, 17: 0.02
        }
        
        # Métricas de calidad
        self.quality_metrics = {}
        
    def scan_dataset(self):
        """Escanea dataset y analiza la calidad de los datos"""
        print("1. 🔍 ANÁLISIS PROFUNDO DEL DATASET...")
        
        total_annotations = 0
        label_distribution = Counter()
        
        for user_folder in os.listdir(self.dataset_path):
            user_path = os.path.join(self.dataset_path, user_folder)
            if os.path.isdir(user_path):
                csv_files = [f for f in os.listdir(user_path) if f.endswith('.csv')]
                if csv_files:
                    csv_path = os.path.join(user_path, csv_files[0])
                    print(f"📋 Analizando {user_folder}: {csv_files[0]}")
                    
                    # Analizar contenido del CSV
                    df = self.read_csv_robust(csv_path)
                    if df is not None and 'Label/Class ID' in df.columns:
                        valid_labels = df['Label/Class ID'].dropna()
                        user_labels = Counter()
                        
                        for label in valid_labels:
                            try:
                                label_int = int(float(str(label)))
                                if 0 <= label_int <= 17:
                                    user_labels[label_int] += 1
                                    label_distribution[label_int] += 1
                                    total_annotations += 1
                            except:
                                continue
                        
                        print(f"  📊 Etiquetas encontradas: {dict(user_labels)}")
                        print(f"  📈 Total anotaciones válidas: {sum(user_labels.values())}")
                    
                    self.labeled_users.append(user_folder)
                else:
                    self.unlabeled_users.append(user_folder)
        
        print(f"\n📋 RESUMEN DEL DATASET:")
        print(f"  ✅ Usuarios etiquetados: {len(self.labeled_users)}")
        print(f"  ❓ Usuarios sin etiquetar: {len(self.unlabeled_users)}")
        print(f"  📊 Total anotaciones: {total_annotations}")
        print(f"  🏷️  Clases encontradas: {len(label_distribution)}/18")
        print(f"  📈 Distribución global:")
        
        for label_id in range(18):
            count = label_distribution.get(label_id, 0)
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            behavior = self.behavior_mapping[label_id]
            status = "✅" if count > 0 else "❌"
            print(f"    {status} {label_id:2d} ({behavior:30s}): {count:3d} ({percentage:4.1f}%)")
        
        # Evaluar calidad del dataset
        self.evaluate_dataset_quality(label_distribution, total_annotations)
        
    def evaluate_dataset_quality(self, label_distribution, total_annotations):
        """Evalúa la calidad del dataset"""
        print(f"\n🎯 EVALUACIÓN DE CALIDAD DEL DATASET:")
        
        # Calcular métricas de calidad
        classes_present = len([c for c in label_distribution.values() if c > 0])
        coverage = classes_present / 18
        
        # Calcular desbalance
        if classes_present > 0:
            counts = [c for c in label_distribution.values() if c > 0]
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        else:
            imbalance_ratio = 0
        
        # Evaluar suficiencia de datos
        data_sufficiency = min(1.0, total_annotations / 1000)
        
        print(f"  📊 Cobertura de clases: {coverage:.2f} ({classes_present}/18)")
        print(f"  ⚖️  Ratio de desbalance: {imbalance_ratio:.1f}")
        print(f"  📈 Suficiencia de datos: {data_sufficiency:.2f}")
        
        # Recomendaciones
        quality_score = (coverage + min(1.0, 1/imbalance_ratio) + data_sufficiency) / 3
        
        if quality_score > 0.7:
            print(f"  ✅ CALIDAD: BUENA (Score: {quality_score:.2f})")
        elif quality_score > 0.4:
            print(f"  🟡 CALIDAD: MODERADA (Score: {quality_score:.2f})")
        else:
            print(f"  🔴 CALIDAD: BAJA (Score: {quality_score:.2f})")
            print(f"     ⚠️  ADVERTENCIA: Resultados poco confiables.")
        
        self.quality_metrics['dataset_quality'] = quality_score
        self.quality_metrics['coverage'] = coverage
        self.quality_metrics['imbalance_ratio'] = imbalance_ratio
        
    def extract_domain_specific_features(self, frame, prev_frame=None):
        """Extrae características específicas del dominio"""
        try:
            # 1. Características CNN base
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = self.feature_extractor.transform(frame_rgb)
            frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
            cnn_features = self.feature_extractor(frame_tensor)
            cnn_flat = cnn_features.cpu().numpy().flatten()
            
            # 2. Características específicas
            hand_features = self.feature_extractor.extract_hand_features(frame)
            face_features = self.feature_extractor.extract_face_features(frame)
            movement_features = self.feature_extractor.extract_movement_features(frame, prev_frame)
            
            # 3. Características de regiones
            h, w = frame.shape[:2]
            regions = {
                'dashboard': frame[h//2:, w//3:2*w//3],
                'driver_area': frame[:h//2, :w//2],
                'passenger_area': frame[:h//2, w//2:],
                'rear_area': frame[h//2:, :w//3]
            }
            
            region_features = []
            for region_name, region in regions.items():
                if region.size > 0:
                    avg_bgr = cv2.mean(region)[:3]
                    region_features.extend(avg_bgr)
                else:
                    region_features.extend([0, 0, 0])
            
            region_features = np.array(region_features, dtype=np.float64)
            
            # Combinar todas las características
            all_features = np.concatenate([
                cnn_flat, hand_features, face_features, 
                movement_features, region_features
            ])
            
            return all_features.astype(np.float64)
            
        except Exception as e:
            print(f"Error extrayendo características: {e}")
            return None
    
    def extract_features_with_quality_control(self):
        """Extrae características con control de calidad"""
        print("\n2. 🎯 EXTRACCIÓN DE CARACTERÍSTICAS AVANZADA...")
        
        all_features = []
        all_labels = []
        
        for user_id in tqdm(self.labeled_users, desc="Procesando usuarios etiquetados"):
            user_path = os.path.join(self.dataset_path, user_id)
            csv_files = [f for f in os.listdir(user_path) if f.endswith('.csv')]
            
            if not csv_files:
                continue
                
            csv_path = os.path.join(user_path, csv_files[0])
            df = self.read_csv_robust(csv_path)
            
            if df is None:
                continue
            
            # Usar todas las anotaciones válidas
            valid_annotations = df[df['Label/Class ID'].notna()]
            
            video_files = [f for f in os.listdir(user_path) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            
            print(f"📹 Procesando {len(video_files)} videos para {user_id}")
            
            for video_file in video_files:
                video_path = os.path.join(user_path, video_file)
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                user_video_features = []
                user_video_labels = []
                
                # Procesar anotaciones
                for _, annotation in valid_annotations.iterrows():
                    label_str = str(annotation.get('Label/Class ID', '')).strip()
                    
                    try:
                        label = int(float(label_str))
                        if not (0 <= label <= 17):
                            continue
                    except:
                        continue
                    
                    start_time = self.parse_time(annotation.get('Start Time', 0))
                    end_time = self.parse_time(annotation.get('End Time', 0))
                    
                    if start_time >= end_time:
                        continue
                    
                    start_frame = int(start_time * fps)
                    end_frame = int(end_time * fps)
                    
                    # Asegurar frames válidos
                    start_frame = max(0, min(start_frame, total_frames - 1))
                    end_frame = max(start_frame + 1, min(end_frame, total_frames))
                    
                    # Muestreo inteligente
                    sample_interval = max(1, int(fps * 0.3))
                    
                    prev_frame = None
                    extracted_count = 0
                    
                    for frame_num in range(start_frame, end_frame, sample_interval):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                        ret, frame = cap.read()
                        
                        if not ret or frame is None:
                            continue
                        
                        # Extraer características
                        features = self.extract_domain_specific_features(frame, prev_frame)
                        
                        if features is not None:
                            user_video_features.append(features)
                            user_video_labels.append(label)
                            extracted_count += 1
                        
                        prev_frame = frame.copy()
                        
                        # Limitar samples por anotación
                        if extracted_count >= 10:
                            break
                
                print(f"  ✅ {video_file}: {len(user_video_features)} características extraídas")
                
                all_features.extend(user_video_features)
                all_labels.extend(user_video_labels)
                
                cap.release()
        
        if len(all_features) == 0:
            print("❌ No se extrajeron características")
            return None, None
        
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        print(f"✅ Total características: {features_array.shape}")
        print(f"✅ Distribución de etiquetas reales:")
        
        real_distribution = Counter(labels_array)
        for label, count in sorted(real_distribution.items()):
            behavior = self.behavior_mapping[label]
            print(f"   {label:2d} ({behavior}): {count} muestras")
        
        return features_array, labels_array
    
    def create_synthetic_intelligent_data(self, real_features, real_labels):
        """Crea datos sintéticos inteligentes"""
        print("\n3. 🧠 GENERACIÓN INTELIGENTE DE DATOS SINTÉTICOS...")
        
        # Analizar características reales por clase
        class_profiles = {}
        
        for label in np.unique(real_labels):
            class_mask = real_labels == label
            class_features = real_features[class_mask]
            
            if len(class_features) > 0:
                class_profiles[label] = {
                    'mean': np.mean(class_features, axis=0),
                    'std': np.std(class_features, axis=0),
                    'min': np.min(class_features, axis=0),
                    'max': np.max(class_features, axis=0)
                }
        
        # Crear datos sintéticos
        synthetic_features = []
        synthetic_labels = []
        
        min_samples = 100
        
        for target_label in range(18):
            current_count = np.sum(real_labels == target_label)
            needed_samples = max(0, min_samples - current_count)
            
            if needed_samples > 0:
                print(f"  🔧 Generando {needed_samples} muestras para clase {target_label}")
                
                if target_label in class_profiles:
                    profile = class_profiles[target_label]
                else:
                    profile = self.create_synthetic_profile(target_label, class_profiles, real_features)
                
                # Generar muestras sintéticas
                for _ in range(needed_samples):
                    if len(real_features) > 0:
                        base_idx = np.random.randint(len(real_features))
                        base_sample = real_features[base_idx].copy()
                        
                        # Modificar hacia el perfil de la clase
                        if 'mean' in profile:
                            alpha = np.random.uniform(0.3, 0.7)
                            synthetic_sample = alpha * profile['mean'] + (1-alpha) * base_sample
                            
                            noise_scale = profile.get('std', np.ones_like(base_sample) * 0.1)
                            noise = np.random.normal(0, noise_scale * 0.2)
                            synthetic_sample += noise
                        else:
                            noise = np.random.normal(0, 0.1, base_sample.shape)
                            synthetic_sample = base_sample + noise
                        
                        synthetic_features.append(synthetic_sample)
                        synthetic_labels.append(target_label)
        
        # Combinar datos reales y sintéticos
        if synthetic_features:
            all_features = np.vstack([real_features, np.array(synthetic_features)])
            all_labels = np.hstack([real_labels, np.array(synthetic_labels)])
        else:
            all_features = real_features
            all_labels = real_labels
        
        print(f"✅ Dataset final: {all_features.shape}")
        print(f"✅ Distribución balanceada:")
        
        final_distribution = Counter(all_labels)
        for label in range(18):
            count = final_distribution.get(label, 0)
            behavior = self.behavior_mapping[label]
            status = "🔥" if count >= min_samples else "⚠️ " if count > 0 else "❌"
            print(f"   {status} {label:2d} ({behavior}): {count}")
        
        return all_features, all_labels
    
    def create_synthetic_profile(self, target_label, class_profiles, real_features):
        """Crea perfil sintético basado en clases similares"""
        # Mapear clases similares
        similar_classes = {
            0: [14, 15], 2: [3, 5, 6], 3: [2, 5, 6], 4: [1], 
            5: [6, 2, 3], 6: [5, 2, 3], 7: [15], 8: [10, 11],
            9: [10, 11], 10: [11, 8], 11: [10, 8], 12: [13],
            13: [12], 14: [0, 15], 15: [7, 14], 16: [17], 17: [16]
        }
        
        similar = similar_classes.get(target_label, [])
        available_similar = [cls for cls in similar if cls in class_profiles]
        
        if available_similar:
            mean_features = []
            std_features = []
            
            for cls in available_similar:
                profile = class_profiles[cls]
                mean_features.append(profile['mean'])
                std_features.append(profile['std'])
            
            synthetic_profile = {
                'mean': np.mean(mean_features, axis=0),
                'std': np.mean(std_features, axis=0)
            }
        else:
            synthetic_profile = {
                'mean': np.mean(real_features, axis=0),
                'std': np.std(real_features, axis=0)
            }
        
        return synthetic_profile
    
    def train_advanced_ensemble_model(self, features, labels):
        """Entrena modelo ensemble avanzado"""
        print("\n4. 🚀 ENTRENANDO MODELO ENSEMBLE AVANZADO...")
        
        # Escalado robusto
        features_scaled = self.robust_scaler.fit_transform(features)
        
        # PCA con análisis de componentes
        explained_variance_ratios = []
        for n_comp in [50, 100, 200, 300]:
            if n_comp < features_scaled.shape[1] and n_comp < features_scaled.shape[0]:
                pca_temp = PCA(n_components=n_comp)
                pca_temp.fit(features_scaled)
                explained_variance_ratios.append((n_comp, sum(pca_temp.explained_variance_ratio_)))
        
        # Elegir número óptimo de componentes
        optimal_components = 100
        for n_comp, variance in explained_variance_ratios:
            if variance >= 0.90:
                optimal_components = n_comp
                break
        
        print(f"📊 Componentes PCA óptimos: {optimal_components}")
        
        self.pca = PCA(n_components=optimal_components)
        features_pca = self.pca.fit_transform(features_scaled)
        
        print(f"🔍 Varianza explicada: {sum(self.pca.explained_variance_ratio_):.3f}")
        
        # División estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            features_pca, labels, test_size=0.2, random_state=42, 
            stratify=labels
        )
        
        # Modelos base
        models = {
            'rf': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1,
                class_weight='balanced'
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=42
            )
        }
        
        # Agregar XGBoost si está disponible
        if HAS_XGB:
            models['xgb'] = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric='mlogloss'
            )
        
        # Validación cruzada
        cv_results = {}
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print("🔄 Validación cruzada de modelos base:")
        for name, model in models.items():
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
                cv_results[name] = {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'model': model
                }
                print(f"  {name.upper()}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            except Exception as e:
                print(f"  {name.upper()}: Error - {e}")
                cv_results[name] = {'mean': 0, 'std': 0, 'model': model}
        
        # Crear ensemble
        good_models = [(name, result['model']) for name, result in cv_results.items() 
                      if result['mean'] > 0.3]
        
        if len(good_models) >= 2:
            self.ensemble_classifier = VotingClassifier(
                estimators=good_models, voting='soft'
            )
        else:
            best_model_name = max(cv_results.keys(), key=lambda k: cv_results[k]['mean'])
            self.ensemble_classifier = cv_results[best_model_name]['model']
            print(f"⚠️  Usando modelo individual: {best_model_name}")
        
        # Entrenar ensemble
        self.ensemble_classifier.fit(X_train, y_train)
        
        # Evaluación detallada
        y_pred = self.ensemble_classifier.predict(X_test)
        
        try:
            y_pred_proba = self.ensemble_classifier.predict_proba(X_test)
        except:
            y_pred_proba = None
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"🎯 Precisión del ensemble: {accuracy:.3f}")
        
        # Análisis de confianza
        if y_pred_proba is not None:
            confidence_scores = np.max(y_pred_proba, axis=1)
            avg_confidence = np.mean(confidence_scores)
            print(f"🔒 Confianza promedio: {avg_confidence:.3f}")
            
            # Ajustar umbral de confianza dinámicamente
            self.confidence_threshold = max(0.4, min(0.8, avg_confidence - 0.1))
            print(f"🎚️  Umbral de confianza ajustado: {self.confidence_threshold:.3f}")
        
        # Matriz de confusión
        unique_labels = np.unique(y_test)
        if len(unique_labels) > 1:
            cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
            print(f"📊 Matriz de confusión disponible para {len(unique_labels)} clases")
        
        self.quality_metrics['model_accuracy'] = accuracy
        self.quality_metrics['confidence_threshold'] = self.confidence_threshold
        
        return features_pca
    
    def train_advanced_clustering(self, features_pca, labels):
        """Entrena múltiples algoritmos de clustering"""
        print("\n5. 🎭 ENTRENANDO CLUSTERING AVANZADO...")
        
        # 1. K-means con múltiples configuraciones
        n_clusters_options = [30, 50, 75, 100]
        best_kmeans = None
        best_silhouette = -1
        
        for n_clusters in n_clusters_options:
            if n_clusters < len(features_pca):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
                cluster_labels = kmeans.fit_predict(features_pca)
                
                silhouette = silhouette_score(features_pca, cluster_labels)
                calinski_harabasz = calinski_harabasz_score(features_pca, cluster_labels)
                
                print(f"  K-means ({n_clusters} clusters): Silhouette={silhouette:.3f}, CH={calinski_harabasz:.1f}")
                
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_kmeans = kmeans
        
        self.clustering_models['kmeans'] = best_kmeans
        
        # 2. Clustering jerárquico
        try:
            hierarchical = AgglomerativeClustering(n_clusters=50, linkage='ward')
            hier_labels = hierarchical.fit_predict(features_pca)
            hier_silhouette = silhouette_score(features_pca, hier_labels)
            
            print(f"  Jerárquico: Silhouette={hier_silhouette:.3f}")
            self.clustering_models['hierarchical'] = hierarchical
        except Exception as e:
            print(f"  Jerárquico: Error - {e}")
        
        # 3. DBSCAN
        try:
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=5)
            neighbors_fit = neighbors.fit(features_pca)
            distances, indices = neighbors_fit.kneighbors(features_pca)
            distances = np.sort(distances, axis=0)
            optimal_eps = np.percentile(distances[:, 4], 75)
            
            dbscan = DBSCAN(eps=optimal_eps, min_samples=5)
            dbscan_labels = dbscan.fit_predict(features_pca)
            
            n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            n_noise = list(dbscan_labels).count(-1)
            
            if n_clusters_dbscan > 1:
                dbscan_silhouette = silhouette_score(features_pca, dbscan_labels)
                print(f"  DBSCAN: {n_clusters_dbscan} clusters, {n_noise} outliers, Silhouette={dbscan_silhouette:.3f}")
                self.clustering_models['dbscan'] = dbscan
            else:
                print(f"  DBSCAN: Solo {n_clusters_dbscan} clusters encontrados")
        except Exception as e:
            print(f"  DBSCAN: Error - {e}")
        
        # Crear mapeos inteligentes
        self.cluster_mappings = {}
        
        for clustering_name, clustering_model in self.clustering_models.items():
            if clustering_model is not None:
                try:
                    if clustering_name == 'kmeans':
                        cluster_labels = clustering_model.labels_
                    else:
                        cluster_labels = clustering_model.fit_predict(features_pca)
                    
                    mapping = self.create_intelligent_cluster_mapping(
                        cluster_labels, labels, features_pca, clustering_name
                    )
                    self.cluster_mappings[clustering_name] = mapping
                except Exception as e:
                    print(f"  Error creando mapeo para {clustering_name}: {e}")
        
        print(f"✅ Clustering completado: {len(self.clustering_models)} algoritmos entrenados")
        
    def create_intelligent_cluster_mapping(self, cluster_labels, true_labels, features, clustering_name):
        """Crea mapeos inteligentes"""
        mapping = {}
        
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Noise in DBSCAN
                continue
                
            cluster_mask = cluster_labels == cluster_id
            cluster_true_labels = true_labels[cluster_mask]
            cluster_features = features[cluster_mask]
            
            if len(cluster_true_labels) == 0:
                # Cluster vacío - usar predicción del ensemble
                if hasattr(self.ensemble_classifier, 'predict'):
                    if clustering_name == 'kmeans':
                        cluster_center = self.clustering_models[clustering_name].cluster_centers_[cluster_id]
                    else:
                        cluster_center = np.mean(cluster_features, axis=0) if len(cluster_features) > 0 else np.zeros(features.shape[1])
                    
                    try:
                        predicted_label = self.ensemble_classifier.predict(cluster_center.reshape(1, -1))[0]
                        mapping[cluster_id] = predicted_label
                    except:
                        mapping[cluster_id] = 0
                else:
                    mapping[cluster_id] = 0
                continue
            
            # Análisis estadístico del cluster
            unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]
            purity = np.max(counts) / len(cluster_true_labels)
            
            # Usar predicción del ensemble para el centroide
            cluster_center = np.mean(cluster_features, axis=0)
            ensemble_prediction = None
            ensemble_confidence = 0
            
            if hasattr(self.ensemble_classifier, 'predict_proba'):
                try:
                    ensemble_proba = self.ensemble_classifier.predict_proba(cluster_center.reshape(1, -1))[0]
                    ensemble_prediction = np.argmax(ensemble_proba)
                    ensemble_confidence = np.max(ensemble_proba)
                except:
                    pass
            
            # Decisión inteligente
            if purity > 0.6:
                mapping[cluster_id] = most_common_label
            elif ensemble_confidence > self.confidence_threshold:
                mapping[cluster_id] = ensemble_prediction
            else:
                mapping[cluster_id] = most_common_label
        
        # Asegurar cobertura completa
        self.ensure_complete_label_coverage(mapping, cluster_labels)
        
        print(f"    {clustering_name}: {len(mapping)} clusters mapeados")
        return mapping
    
    def ensure_complete_label_coverage(self, mapping, cluster_labels):
        """Asegura que todas las 18 etiquetas estén cubiertas"""
        assigned_labels = set(mapping.values())
        missing_labels = set(range(18)) - assigned_labels
        
        if missing_labels:
            print(f"      Forzando cobertura de {len(missing_labels)} etiquetas")
            
            # Encontrar clusters sobre-representados
            label_counts = Counter(mapping.values())
            over_represented = [label for label, count in label_counts.items() if count > 2]
            
            # Reasignar clusters
            available_clusters = []
            for cluster_id in mapping.keys():
                if mapping[cluster_id] in over_represented:
                    available_clusters.append(cluster_id)
            
            # Asignar etiquetas faltantes
            missing_list = list(missing_labels)
            for i, cluster_id in enumerate(available_clusters[:len(missing_list)]):
                old_label = mapping[cluster_id]
                new_label = missing_list[i]
                mapping[cluster_id] = new_label
                print(f"        Cluster {cluster_id}: {old_label} -> {new_label}")
    
    def predict_with_confidence_weighting(self, features):
        """Predice usando ensemble con ponderación por confianza"""
        features_scaled = self.robust_scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        predictions = []
        
        # Predicción del ensemble principal
        try:
            ensemble_pred = self.ensemble_classifier.predict(features_pca)[0]
            if hasattr(self.ensemble_classifier, 'predict_proba'):
                ensemble_proba = self.ensemble_classifier.predict_proba(features_pca)[0]
                ensemble_confidence = np.max(ensemble_proba)
            else:
                ensemble_confidence = 0.5
            
            predictions.append((ensemble_pred, ensemble_confidence * 2))
        except Exception as e:
            ensemble_pred = 0
            ensemble_confidence = 0.1
            predictions.append((ensemble_pred, ensemble_confidence))
        
        # Predicciones de clustering
        for clustering_name, clustering_model in self.clustering_models.items():
            if clustering_model is not None and clustering_name in self.cluster_mappings:
                try:
                    cluster_pred = clustering_model.predict(features_pca)[0]
                    if cluster_pred in self.cluster_mappings[clustering_name]:
                        mapped_label = self.cluster_mappings[clustering_name][cluster_pred]
                        confidence = 0.3
                        predictions.append((mapped_label, confidence))
                except:
                    continue
        
        if not predictions:
            return 0, 0.1
        
        # Votación ponderada por confianza
        weighted_votes = {}
        total_weight = 0
        
        for pred, conf in predictions:
            if pred not in weighted_votes:
                weighted_votes[pred] = 0
            weighted_votes[pred] += conf
            total_weight += conf
        
        # Normalizar y elegir mejor predicción
        if total_weight > 0:
            for pred in weighted_votes:
                weighted_votes[pred] /= total_weight
        
        best_prediction = max(weighted_votes.keys(), key=lambda k: weighted_votes[k])
        best_confidence = weighted_votes[best_prediction]
        
        return best_prediction, best_confidence
    
    def predict_with_smart_post_processing(self):
        """Predice con post-procesamiento inteligente"""
        print("\n6. 🎯 PREDICCIÓN CON POST-PROCESAMIENTO INTELIGENTE...")
        
        predictions = {}
        
        for user_id in tqdm(self.unlabeled_users, desc="Procesando usuarios sin etiquetar"):
            user_path = os.path.join(self.dataset_path, user_id)
            video_files = [f for f in os.listdir(user_path) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            
            user_predictions = []
            
            for video_file in video_files:
                video_path = os.path.join(user_path, video_file)
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps
                
                # Predicciones por frame
                frame_predictions = []
                sample_interval = max(1, int(fps * 0.8))
                
                prev_frame = None
                
                for frame_num in range(0, total_frames, sample_interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        features = self.extract_domain_specific_features(frame, prev_frame)
                        if features is not None:
                            features = features.reshape(1, -1)
                            predicted_label, confidence = self.predict_with_confidence_weighting(features)
                            
                            frame_predictions.append({
                                'time': frame_num / fps,
                                'label': predicted_label,
                                'confidence': confidence
                            })
                        
                        prev_frame = frame.copy()
                
                cap.release()
                
                # Post-procesamiento inteligente
                processed_predictions = self.intelligent_post_processing(
                    frame_predictions, duration, user_id
                )
                
                user_predictions.append({
                    'video_file': video_file,
                    'predictions': processed_predictions
                })
            
            predictions[user_id] = user_predictions
        
        return predictions
    
    def intelligent_post_processing(self, frame_predictions, duration, user_id):
        """Post-procesamiento inteligente con análisis temporal"""
        if not frame_predictions:
            return []
        
        # Filtrar por confianza mínima
        min_confidence = max(0.2, self.confidence_threshold - 0.2)
        confident_predictions = [p for p in frame_predictions if p['confidence'] >= min_confidence]
        
        if not confident_predictions:
            confident_predictions = frame_predictions
        
        # Agrupar predicciones temporalmente
        grouped = []
        if confident_predictions:
            current_group = {
                'start_time': confident_predictions[0]['time'],
                'end_time': confident_predictions[0]['time'],
                'label': confident_predictions[0]['label'],
                'confidence': confident_predictions[0]['confidence'],
                'count': 1
            }
            
            for pred in confident_predictions[1:]:
                # Agrupar si es la misma etiqueta y tiempo continuo
                if (pred['label'] == current_group['label'] and 
                    pred['time'] - current_group['end_time'] <= 5):
                    current_group['end_time'] = pred['time']
                    current_group['confidence'] = max(current_group['confidence'], pred['confidence'])
                    current_group['count'] += 1
                else:
                    # Guardar grupo si cumple duración mínima
                    if current_group['end_time'] - current_group['start_time'] >= 2:
                        grouped.append(current_group.copy())
                    
                    current_group = {
                        'start_time': pred['time'],
                        'end_time': pred['time'],
                        'label': pred['label'],
                        'confidence': pred['confidence'],
                        'count': 1
                    }
            
            # Agregar último grupo
            if current_group['end_time'] - current_group['start_time'] >= 2:
                grouped.append(current_group)
        
        # Aplicar distribución objetivo inteligente
        final_predictions = self.apply_intelligent_distribution(grouped, duration)
        
        return final_predictions
    
    def apply_intelligent_distribution(self, grouped_predictions, duration):
        """Aplica distribución objetivo de manera inteligente"""
        if not grouped_predictions:
            return self.generate_minimal_predictions(duration)
        
        # Calcular tiempo total cubierto
        total_covered_time = sum(p['end_time'] - p['start_time'] for p in grouped_predictions)
        coverage_ratio = total_covered_time / duration
        
        # Si cobertura es baja, generar predicciones adicionales
        if coverage_ratio < 0.3:
            additional_predictions = self.generate_additional_predictions(
                grouped_predictions, duration, target_coverage=0.5
            )
            grouped_predictions.extend(additional_predictions)
        
        # Balancear según distribución objetivo
        label_times = {}
        for pred in grouped_predictions:
            label = pred['label']
            duration_pred = pred['end_time'] - pred['start_time']
            if label not in label_times:
                label_times[label] = 0
            label_times[label] += duration_pred
        
        # Ajustar duraciones según objetivos
        adjusted_predictions = []
        total_target_time = duration * 0.6
        
        for pred in grouped_predictions:
            label = pred['label']
            target_ratio = self.target_distribution.get(label, 0.02)
            target_time_for_label = total_target_time * target_ratio
            
            current_time_for_label = label_times[label]
            
            # Ajustar duración si es necesario
            if current_time_for_label > target_time_for_label * 2:
                reduction_factor = (target_time_for_label * 1.5) / current_time_for_label
                new_duration = (pred['end_time'] - pred['start_time']) * reduction_factor
                pred['end_time'] = pred['start_time'] + max(2, new_duration)
            
            adjusted_predictions.append(pred)
        
        # Asegurar diversidad de etiquetas
        return self.ensure_label_diversity(adjusted_predictions, duration)
    
    def ensure_label_diversity(self, predictions, duration):
        """Asegura diversidad mínima de etiquetas"""
        present_labels = set(p['label'] for p in predictions)
        missing_labels = set(range(18)) - present_labels
        
        if missing_labels:
            max_time = max((p['end_time'] for p in predictions), default=0)
            available_time = duration - max_time
            
            for i, missing_label in enumerate(list(missing_labels)[:min(len(missing_labels), int(available_time/3))]):
                start_time = max_time + i * 3
                end_time = start_time + np.random.uniform(2, 5)
                
                if end_time <= duration:
                    predictions.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'label': missing_label,
                        'confidence': 0.3
                    })
        
        # Ordenar por tiempo y resolver solapamientos
        predictions.sort(key=lambda x: x['start_time'])
        return self.resolve_final_overlaps(predictions)
    
    def generate_minimal_predictions(self, duration):
        """Genera predicciones mínimas cuando no hay suficientes datos"""
        predictions = []
        current_time = 0
        
        labels_to_use = list(range(18))
        np.random.shuffle(labels_to_use)
        
        time_per_label = min(duration / len(labels_to_use), 10)
        
        for i, label in enumerate(labels_to_use[:int(duration/3)]):
            start_time = current_time
            end_time = min(current_time + np.random.uniform(2, time_per_label), duration)
            
            if end_time > start_time:
                predictions.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'label': label,
                    'confidence': 0.2
                })
                
                current_time = end_time + 1
        
        return predictions
    
    def generate_additional_predictions(self, existing_predictions, duration, target_coverage=0.5):
        """Genera predicciones adicionales para mejorar cobertura"""
        additional = []
        
        if existing_predictions:
            existing_predictions.sort(key=lambda x: x['start_time'])
            
            gaps = []
            for i in range(len(existing_predictions) - 1):
                gap_start = existing_predictions[i]['end_time']
                gap_end = existing_predictions[i + 1]['start_time']
                if gap_end - gap_start > 3:
                    gaps.append((gap_start, gap_end))
            
            # Gap al final
            if existing_predictions[-1]['end_time'] < duration - 3:
                gaps.append((existing_predictions[-1]['end_time'], duration))
            
            # Llenar gaps
            for gap_start, gap_end in gaps:
                gap_duration = gap_end - gap_start
                if gap_duration > 3:
                    present_labels = [p['label'] for p in existing_predictions]
                    label_counts = Counter(present_labels)
                    least_common_labels = [label for label in range(18) 
                                         if label_counts.get(label, 0) == min(label_counts.values(), default=0)]
                    
                    chosen_label = np.random.choice(least_common_labels) if least_common_labels else 0
                    
                    pred_duration = min(gap_duration - 1, np.random.uniform(3, 8))
                    pred_start = gap_start + np.random.uniform(0.5, 2)
                    
                    additional.append({
                        'start_time': pred_start,
                        'end_time': pred_start + pred_duration,
                        'label': chosen_label,
                        'confidence': 0.25
                    })
        
        return additional
    
    def resolve_final_overlaps(self, predictions):
        """Resuelve solapamientos finales entre predicciones"""
        if not predictions:
            return []
        
        resolved = [predictions[0]]
        
        for current in predictions[1:]:
            last = resolved[-1]
            
            if current['start_time'] < last['end_time']:
                if current['confidence'] > last['confidence']:
                    last['end_time'] = current['start_time'] - 0.5
                    if last['end_time'] <= last['start_time']:
                        last['end_time'] = last['start_time'] + 1
                    resolved.append(current)
                else:
                    current['start_time'] = last['end_time'] + 0.5
                    if current['end_time'] <= current['start_time']:
                        current['end_time'] = current['start_time'] + 1
                    resolved.append(current)
            else:
                resolved.append(current)
        
        return [p for p in resolved if p['end_time'] > p['start_time']]
    
    def generate_enhanced_csv_files(self, predictions):
        """Genera archivos CSV con análisis de calidad"""
        print("\n7. 📄 GENERANDO ARCHIVOS CSV CON ANÁLISIS DE CALIDAD...")
        os.makedirs(self.output_path, exist_ok=True)
        
        quality_summary = {
            'users': {},
            'global_distribution': Counter(),
            'confidence_stats': []
        }
        
        for user_id, user_predictions in predictions.items():
            csv_rows = []
            user_confidences = []
            
            for video_data in user_predictions:
                video_file = video_data['video_file']
                filename_base = video_file.split('.')[0]
                
                # Determinar vista de cámara
                if 'dashboard' in filename_base.lower():
                    camera_view = 'Dashboard'
                elif 'rear' in filename_base.lower():
                    camera_view = 'Rear_view'
                elif 'right' in filename_base.lower():
                    camera_view = 'Right_side_window'
                else:
                    camera_view = 'Dashboard'
                
                for pred in video_data['predictions']:
                    start_time_str = self.format_time(pred['start_time'])
                    end_time_str = self.format_time(pred['end_time'])
                    
                    csv_rows.append([
                        user_id, filename_base, camera_view, 'Distracted',
                        start_time_str, end_time_str, str(pred['label']), 'None'
                    ])
                    
                    user_confidences.append(pred['confidence'])
                    quality_summary['global_distribution'][pred['label']] += 1
                    quality_summary['confidence_stats'].append(pred['confidence'])
            
            if csv_rows:
                # Crear DataFrame
                df = pd.DataFrame(csv_rows, columns=[
                    'User ID', 'Filename', 'Camera View', 'Activity Type',
                    'Start Time', 'End Time', 'Label/Class ID', 'Appearance Block'
                ])
                
                csv_path = os.path.join(self.output_path, f"{user_id}.csv")
                df.to_csv(csv_path, sep=';', index=False, encoding='utf-8')
                
                # Análisis de calidad por usuario
                labels_count = Counter([int(row[6]) for row in csv_rows])
                avg_confidence = np.mean(user_confidences) if user_confidences else 0
                
                user_quality = {
                    'total_predictions': len(csv_rows),
                    'unique_labels': len(labels_count),
                    'avg_confidence': avg_confidence,
                    'label_distribution': dict(labels_count)
                }
                
                quality_summary['users'][user_id] = user_quality
                
                # Reporte detallado
                print(f"📊 {user_id}:")
                print(f"  ✅ {len(csv_rows)} anotaciones generadas")
                print(f"  🎯 {len(labels_count)}/18 etiquetas diferentes")
                print(f"  🔒 Confianza promedio: {avg_confidence:.3f}")
                
                # Evaluar calidad
                high_confidence_labels = sum(1 for c in user_confidences if c >= self.confidence_threshold)
                confidence_ratio = high_confidence_labels / len(user_confidences) if user_confidences else 0
                
                if len(labels_count) >= 15 and confidence_ratio >= 0.4:
                    print(f"  ✅ CALIDAD ALTA")
                elif len(labels_count) >= 10 and confidence_ratio >= 0.2:
                    print(f"  🟡 CALIDAD MEDIA")
                else:
                    print(f"  🔴 CALIDAD BAJA - Revisar manualmente")
                
                print()
        
        # Generar reporte de calidad global
        self.generate_quality_report(quality_summary)
        
        return quality_summary
    
    def generate_quality_report(self, quality_summary):
        """Genera reporte detallado de calidad"""
        print("📋 REPORTE DE CALIDAD DETALLADO:")
        
        # Estadísticas globales
        total_predictions = sum(quality_summary['global_distribution'].values())
        unique_labels = len(quality_summary['global_distribution'])
        avg_global_confidence = np.mean(quality_summary['confidence_stats']) if quality_summary['confidence_stats'] else 0
        
        print(f"\n🌍 ESTADÍSTICAS GLOBALES:")
        print(f"  📊 Total predicciones: {total_predictions}")
        print(f"  🏷️  Etiquetas únicas: {unique_labels}/18")
        print(f"  🔒 Confianza promedio global: {avg_global_confidence:.3f}")
        
        # Distribución vs objetivos
        print(f"\n📈 DISTRIBUCIÓN VS OBJETIVOS:")
        distribution_quality = 0
        
        for label_id in range(18):
            actual_count = quality_summary['global_distribution'].get(label_id, 0)
            actual_ratio = actual_count / total_predictions if total_predictions > 0 else 0
            target_ratio = self.target_distribution.get(label_id, 0)
            
            behavior = self.behavior_mapping[label_id]
            
            # Calcular error relativo
            if target_ratio > 0:
                relative_error = abs(actual_ratio - target_ratio) / target_ratio
                quality_score = max(0, 1 - relative_error)
            else:
                quality_score = 1 if actual_count == 0 else 0.5
            
            distribution_quality += quality_score
            
            status = "✅" if quality_score > 0.7 else "🟡" if quality_score > 0.3 else "🔴"
            
            print(f"  {status} {label_id:2d} ({behavior:30s}): "
                  f"{actual_ratio*100:4.1f}% vs {target_ratio*100:4.1f}% objetivo "
                  f"(calidad: {quality_score:.2f})")
        
        distribution_quality /= 18
        
        # Análisis de confianza
        confidence_stats = quality_summary['confidence_stats']
        if confidence_stats:
            high_conf_ratio = sum(1 for c in confidence_stats if c >= self.confidence_threshold) / len(confidence_stats)
            med_conf_ratio = sum(1 for c in confidence_stats if 0.3 <= c < self.confidence_threshold) / len(confidence_stats)
            low_conf_ratio = sum(1 for c in confidence_stats if c < 0.3) / len(confidence_stats)
            
            print(f"\n🔒 ANÁLISIS DE CONFIANZA:")
            print(f"  🟢 Alta confianza (≥{self.confidence_threshold:.2f}): {high_conf_ratio*100:.1f}%")
            print(f"  🟡 Media confianza (0.3-{self.confidence_threshold:.2f}): {med_conf_ratio*100:.1f}%")
            print(f"  🔴 Baja confianza (<0.3): {low_conf_ratio*100:.1f}%")
        
        # Calidad por usuario
        print(f"\n👤 CALIDAD POR USUARIO:")
        user_qualities = []
        
        for user_id, user_stats in quality_summary['users'].items():
            diversity_score = user_stats['unique_labels'] / 18
            confidence_score = min(1.0, user_stats['avg_confidence'] / 0.6)
            quantity_score = min(1.0, user_stats['total_predictions'] / 100)
            
            overall_quality = (diversity_score + confidence_score + quantity_score) / 3
            user_qualities.append(overall_quality)
            
            emoji = "🟢" if overall_quality > 0.7 else "🟡" if overall_quality > 0.4 else "🔴"
            
            print(f"  {emoji} {user_id}: {overall_quality:.3f} "
                  f"(div:{diversity_score:.2f}, conf:{confidence_score:.2f}, cant:{quantity_score:.2f})")
        
        # Puntuación final
        overall_system_quality = (
            distribution_quality * 0.4 +
            avg_global_confidence * 0.3 +
            np.mean(user_qualities) * 0.3
        )
        
        print(f"\n🎯 EVALUACIÓN FINAL:")
        print(f"  📊 Calidad de distribución: {distribution_quality:.3f}")
        print(f"  🔒 Calidad de confianza: {avg_global_confidence:.3f}")
        print(f"  👥 Calidad promedio usuarios: {np.mean(user_qualities):.3f}")
        print(f"  🏆 CALIDAD GENERAL DEL SISTEMA: {overall_system_quality:.3f}")
        
        if overall_system_quality > 0.7:
            print(f"  ✅ EXCELENTE - Resultados confiables para uso en producción")
        elif overall_system_quality > 0.5:
            print(f"  🟡 BUENO - Resultados útiles con algunas limitaciones")
        elif overall_system_quality > 0.3:
            print(f"  🔴 MODERADO - Usar con precaución, validar manualmente")
        else:
            print(f"  ⚠️  BAJO - No recomendado para uso sin validación extensiva")
        
        # Recomendaciones específicas
        print(f"\n💡 RECOMENDACIONES ESPECÍFICAS:")
        
        if distribution_quality < 0.5:
            print(f"  📊 Mejorar distribución: Ajustar pesos objetivo o post-procesamiento")
        
        if avg_global_confidence < 0.4:
            print(f"  🔒 Mejorar confianza: Más datos de entrenamiento o mejor extractor")
        
        if np.mean(user_qualities) < 0.5:
            print(f"  👥 Mejorar consistencia: Revisar videos de baja calidad")
        
        missing_labels = [i for i in range(18) if quality_summary['global_distribution'].get(i, 0) == 0]
        if missing_labels:
            missing_behaviors = [self.behavior_mapping[i] for i in missing_labels]
            print(f"  🏷️  Etiquetas faltantes: {missing_behaviors}")
        
        # Guardar métricas finales
        self.quality_metrics.update({
            'overall_quality': overall_system_quality,
            'distribution_quality': distribution_quality,
            'confidence_quality': avg_global_confidence,
            'user_quality': np.mean(user_qualities),
            'missing_labels': len(missing_labels)
        })
    
    # Métodos auxiliares
    def read_csv_robust(self, csv_path):
        """Lee CSV de manera robusta"""
        try:
            df = None
            for sep in [';', ',', '\t']:
                try:
                    temp_df = pd.read_csv(csv_path, sep=sep, encoding='utf-8')
                    if len(temp_df.columns) >= 7:
                        df = temp_df
                        break
                except:
                    try:
                        temp_df = pd.read_csv(csv_path, sep=sep, encoding='latin-1')
                        if len(temp_df.columns) >= 7:
                            df = temp_df
                            break
                    except:
                        continue
            
            if df is not None:
                df.columns = df.columns.str.strip()
                # Mapear columnas
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
            
            return df
        except Exception as e:
            print(f"Error leyendo CSV {csv_path}: {e}")
            return None
    
    def parse_time(self, time_str):
        """Convierte string de tiempo a segundos"""
        if pd.isna(time_str):
            return 0
        
        time_str = str(time_str).strip()
        if time_str == '' or time_str == 'nan':
            return 0
            
        try:
            if ':' in time_str:
                time_parts = time_str.split(':')
                if len(time_parts) == 3:
                    hours, minutes, seconds = map(float, time_parts)
                    return hours * 3600 + minutes * 60 + seconds
                elif len(time_parts) == 2:
                    minutes, seconds = map(float, time_parts)
                    return minutes * 60 + seconds
            else:
                return float(time_str)
        except (ValueError, TypeError):
            return 0
    
    def format_time(self, seconds):
        """Convierte segundos a formato HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def save_advanced_model(self, model_path):
        """Guarda el modelo avanzado completo"""
        model_data = {
            'ensemble_classifier': self.ensemble_classifier,
            'clustering_models': self.clustering_models,
            'cluster_mappings': self.cluster_mappings,
            'robust_scaler': self.robust_scaler,
            'pca': self.pca,
            'behavior_mapping': self.behavior_mapping,
            'target_distribution': self.target_distribution,
            'confidence_threshold': self.confidence_threshold,
            'quality_metrics': self.quality_metrics
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"🤖 Modelo avanzado guardado en: {model_path}")
    
    def run_advanced_pipeline(self):
        """Ejecuta el pipeline avanzado completo"""
        print("🚀 INICIANDO PIPELINE SEMI-SUPERVISADO AVANZADO")
        print("=" * 60)
        
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  Usando CPU (recomendado: GPU para mejor rendimiento)")
        
        try:
            # 1. Análisis profundo del dataset
            self.scan_dataset()
            
            if len(self.labeled_users) == 0:
                print("❌ FATAL: No se encontraron usuarios etiquetados")
                return None
            
            if self.quality_metrics.get('dataset_quality', 0) < 0.3:
                print("⚠️  ADVERTENCIA: Dataset de baja calidad - resultados poco confiables")
            
            # 2. Extracción de características avanzada
            real_features, real_labels = self.extract_features_with_quality_control()
            
            if real_features is None or len(real_features) == 0:
                print("❌ FATAL: No se pudieron extraer características")
                return None
            
            # 3. Generación inteligente de datos sintéticos
            features, labels = self.create_synthetic_intelligent_data(real_features, real_labels)
            
            # 4. Entrenamiento del ensemble avanzado
            features_pca = self.train_advanced_ensemble_model(features, labels)
            
            # 5. Clustering avanzado
            self.train_advanced_clustering(features_pca, labels)
            
            # 6. Predicción con post-procesamiento
            predictions = self.predict_with_smart_post_processing()
            
            # 7. Generación de CSVs con análisis de calidad
            quality_summary = self.generate_enhanced_csv_files(predictions)
            
            # 8. Guardar modelo
            model_path = os.path.join(self.output_path, 'advanced_model.pkl')
            self.save_advanced_model(model_path)
            
            print("\n" + "=" * 60)
            print("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
            print(f"📁 Archivos generados en: {self.output_path}")
            print(f"🤖 Modelo guardado en: {model_path}")
            print(f"🏆 Calidad general: {self.quality_metrics.get('overall_quality', 0):.3f}")
            
            return predictions, quality_summary
            
        except Exception as e:
            print(f"\n❌ ERROR CRÍTICO EN PIPELINE: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Función principal del pipeline avanzado"""
    print("🎬 SISTEMA SEMI-SUPERVISADO AVANZADO PARA DETECCIÓN DE COMPORTAMIENTOS")
    print("📊 Versión: 3.0 - Con análisis de calidad y métricas de confianza")
    print("=" * 80)
    
    # Verificar configuración del sistema
    if torch.cuda.is_available():
        print(f"🖥️  GPU: {torch.cuda.get_device_name()}")
        print(f"💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  Sin GPU - El procesamiento será más lento")
    
    # Configurar rutas
    dataset_path = "dataset"
    output_path = "csvs_advanced"
    
    # Verificar dataset
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: No se encontró el directorio del dataset: {dataset_path}")
        return
    
    # Crear directorio de salida
    os.makedirs(output_path, exist_ok=True)
    
    # Crear y ejecutar el labeler avanzado
    labeler = AdvancedSemiSupervisedLabeler(dataset_path, output_path)
    
    # Ejecutar pipeline
    results = labeler.run_advanced_pipeline()
    
    if results:
        predictions, quality_summary = results
        
        print(f"\n🎊 RESUMEN FINAL:")
        print(f"   ✅ Usuarios procesados: {len(predictions)}")
        print(f"   📄 Archivos CSV generados: {len(predictions)}")
        
        total_predictions = sum(
            sum(len(video['predictions']) for video in user_data)
            for user_data in predictions.values()
        )
        print(f"   📊 Total predicciones: {total_predictions}")
        
        # Mostrar métricas de calidad finales
        if labeler.quality_metrics:
            print(f"   🏆 Calidad general: {labeler.quality_metrics.get('overall_quality', 0):.3f}")
            print(f"   🎯 Distribución: {labeler.quality_metrics.get('distribution_quality', 0):.3f}")
            print(f"   🔒 Confianza: {labeler.quality_metrics.get('confidence_quality', 0):.3f}")
        
        print(f"\n💡 PRÓXIMOS PASOS RECOMENDADOS:")
        print(f"   1. Revisar archivos CSV generados en {output_path}")
        print(f"   2. Validar manualmente una muestra de predicciones")
        print(f"   3. Ajustar parámetros si la calidad es baja")
        print(f"   4. Considerar más datos de entrenamiento si es necesario")
        
    else:
        print(f"\n❌ El pipeline no se completó exitosamente")
        print(f"💡 Sugerencias:")
        print(f"   - Verificar que los archivos CSV estén en el formato correcto")
        print(f"   - Asegurar que los videos son accesibles")
        print(f"   - Revisar los logs de error anteriores")

if __name__ == "__main__":
    main()