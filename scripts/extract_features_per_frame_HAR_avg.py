import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import json

# Configuración
frames_dir = 'frames'  # Directorio de los frames de video
features_dir = 'features'  # Directorio donde se guardarán las características
json_path = 'WLASL_v0.3.json'  # Ruta al archivo JSON
model_name = 'r3d_18'  # Modelo de clasificación de video de PyTorch
num_threads = multiprocessing.cpu_count()  # Número de hilos según los núcleos de la CPU
batch_size = 16  # Tamaño del lote para procesamiento en lotes

# Cargar el modelo y el extractor de características
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = models.video.R3D_18_Weights.DEFAULT
original_model = models.video.r3d_18(weights=weights).to(device)
original_model.eval()  # Importante para poner el modelo en modo evaluación

# Modificar el modelo para extraer características de una capa intermedia (sin logits)
class FeatureExtractor(torch.nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])  # Remover la capa 'fc'
        
        # Mostrar todas las capas del modelo
        print("Capas del modelo R3D_18:")
        for idx, layer in enumerate(self.features):
            print(f"Capa {idx}: {layer}")

    def forward(self, x):
        for idx, layer in enumerate(self.features):
            x = layer(x)
            print(f"Output de la capa {idx} ({layer}): {x.shape}")  # Para verificar el formato de los datos
        x = torch.flatten(x, 1)
        return x

# Crear el extractor de características desde la capa intermedia
model = FeatureExtractor(original_model).to(device)
model.eval()

# Transformaciones para los frames
preprocess = weights.transforms()

# Cargar el archivo JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Crear un diccionario para mapear video_id a etiquetas (gloss)
video_to_gloss = {}
for entry in data:
    gloss = entry['gloss']
    for instance in entry['instances']:
        video_id = instance['video_id']
        video_to_gloss[video_id] = gloss

def extract_features_block(frames_block):
    """Extrae características de un bloque de frames utilizando el extractor de características."""
    batch = preprocess(frames_block).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(batch)

    # Comprobar si las características están normalizadas
    print(f"Características extraídas: {features}")
    print(f"Valor máximo: {features.max()}, Valor mínimo: {features.min()}")  # Comprobar normalización
    
    return features.squeeze(0).cpu().numpy()

def process_video(video_name, hdf5_file):
    """Procesa los frames de un video y guarda las características en un único archivo HDF5."""
    video_path = os.path.join(frames_dir, video_name)
    if not os.path.isdir(video_path):
        print(f"Advertencia: No se encontraron frames para el video {video_name}")
        return
    
    video_id = os.path.splitext(video_name)[0]  # Usar el nombre del video como ID
    gloss = video_to_gloss.get(video_id)
    if gloss is None:
        print(f"Advertencia: No se encontró la etiqueta para el video {video_name}")
        return

    # Verificar si las características ya existen en el HDF5
    if video_id in hdf5_file:
        print(f"Características ya presentes para el video {video_name}. Saltando.")
        return

    print(f"Procesando vídeo: {video_name} con etiqueta: {gloss}")
    frame_paths = sorted([os.path.join(video_path, frame_name) for frame_name in os.listdir(video_path) if frame_name.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not frame_paths:
        print(f"Advertencia: No se encontraron frames válidos en {video_path}")
        return

    video_frames = []
    for frame_path in frame_paths:
        frame = Image.open(frame_path).convert('RGB')
        video_frames.append(np.array(frame))

    video_frames = np.array(video_frames)
    video_frames = torch.tensor(video_frames).permute(0, 3, 1, 2).float() / 255.0

    # Inicializar una lista para almacenar todas las características del video
    video_features = []

    # Dividir los frames en bloques de 10 con solapamiento de 5
    block_size = 10
    overlap = 5
    num_frames = video_frames.shape[0]
    
    for i in range(0, num_frames, overlap):
        if i + block_size <= num_frames:
            frames_block = video_frames[i:i+block_size]
        else:
            # Si no hay suficientes frames, repetir el último frame hasta completar el bloque
            frames_block = video_frames[i:]
            while frames_block.shape[0] < block_size:
                frames_block = torch.cat((frames_block, frames_block[-1:].clone()), dim=0)

        features = extract_features_block(frames_block)
        video_features.append(features)

    video_features = np.array(video_features)

    # Guardar las características y la etiqueta en el archivo HDF5
    group = hdf5_file.create_group(video_id)  # Crear grupo con el ID del video
    group.create_dataset("features", data=video_features)  # Guardar características
    group.create_dataset("label", data=np.string_(gloss.encode('utf-8')))  # Guardar etiqueta como dataset

# Crear el directorio de características si no existe
os.makedirs(features_dir, exist_ok=True)

# Inicializar el archivo HDF5 para todos los videos
hdf5_path = os.path.join(features_dir, 'all_videos_features_avg.h5')

# Procesamiento en paralelo
def process_missing_videos(only_missing=False):
    """Procesar los videos faltantes o todos los videos según el argumento."""
    with h5py.File(hdf5_path, 'a') as hdf5_file:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for video_name in os.listdir(frames_dir):
                if only_missing and video_name in hdf5_file:
                    continue
                futures.append(executor.submit(process_video, video_name, hdf5_file))
            
            for future in as_completed(futures):
                future.result()

# Llamar a la función para procesar los videos faltantes o todos
process_missing_videos(only_missing=True)

print("Características extraídas y guardadas correctamente en el archivo HDF5.")
