import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5py
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import argparse

# Fijar la semilla para reproducibilidad
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Verificar si CUDA está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Rutas de los archivos
hdf5_path = 'features'
train_txt = 'train_videos.txt'
test_txt = 'test_videos.txt'
validation_txt = 'validation_videos.txt'
json_path = 'WLASL_v0.3.json'

# Función para cargar videos desde los archivos de texto
def load_video_list(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f]

# Crear un diccionario de etiquetas a partir del archivo JSON
def create_labels_dict(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    labels_dict = {}
    for entry in data:
        gloss = entry['gloss']
        for instance in entry['instances']:
            video_id = instance['video_id']
            labels_dict[video_id] = gloss

    return labels_dict

# Función para encontrar las clases que aparecen tanto en train como en test
def get_common_classes(train_list, test_list, labels_dict):
    train_classes = set([labels_dict[os.path.splitext(video)[0]] for video in train_list if os.path.splitext(video)[0] in labels_dict])
    test_classes = set([labels_dict[os.path.splitext(video)[0]] for video in test_list if os.path.splitext(video)[0] in labels_dict])
    common_classes = train_classes.intersection(test_classes)
    return common_classes

# Filtrar los videos que pertenecen a clases comunes
def filter_videos_by_class(video_list, labels_dict, common_classes):
    return [video for video in video_list if os.path.splitext(video)[0] in labels_dict and labels_dict[os.path.splitext(video)[0]] in common_classes]

# Cargar listas de videos
train_list = load_video_list(train_txt)
test_list = load_video_list(test_txt)
validation_list = load_video_list(validation_txt)

# Obtener las clases comunes
labels_dict = create_labels_dict(json_path)
common_classes = get_common_classes(train_list, test_list, labels_dict)

# Filtrar las listas de videos para que solo incluyan instancias de clases comunes
train_list = filter_videos_by_class(train_list, labels_dict, common_classes)
test_list = filter_videos_by_class(test_list, labels_dict, common_classes)
validation_list = filter_videos_by_class(validation_list, labels_dict, common_classes)

# Modificar el dataset para aceptar las listas de videos filtrados
class SignLanguageDataset(Dataset):
    def __init__(self, features_dir, video_list, labels_dict, common_classes):
        self.features_dir = features_dir
        self.video_list = video_list
        self.labels_dict = labels_dict
        self.common_classes = common_classes
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(common_classes))}

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_file = self.video_list[idx]
        video_id = os.path.splitext(video_file)[0]

        try:
            with h5py.File(os.path.join(self.features_dir, 'all_videos_features_avg.h5'), 'r') as h5f:
                features = h5f[video_id]['features'][:]
                label = self.labels_dict[video_id]
                label_idx = self.label_to_idx[label]
        except KeyError:
            return None

        features = torch.tensor(features, dtype=torch.float32)

        if torch.isnan(features).any():
            print(f"NaN encontrado en características del video {video_id}")

        return features, label_idx

# Función personalizada para la colación usando pad_sequence
def collate_fn(batch):
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    features, labels = zip(*batch)
    padded_features = pad_sequence(features, batch_first=True)  # No limitamos a max_seq_len, se ajusta dinámicamente

    attention_masks = torch.zeros(padded_features.shape[:-1], dtype=torch.float32)
    for i, feature in enumerate(features):
        attention_masks[i, :feature.shape[0]] = 1  # Máscara de atención según la longitud original de la secuencia

    labels = torch.tensor(labels, dtype=torch.long)

    return padded_features, attention_masks, labels

# Modelo
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.cls_token.requires_grad = True
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))  # 1000 es un número grande para asegurar capacidad
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        torch.nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")

    def forward(self, x):
        x = x.to(device)
        batch_size = x.size(0)

        # Expandir el cls_token para cada muestra en el batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Aplicar la capa de embedding y añadir el embedding posicional
        x = self.embedding(x) + self.positional_embedding[:, :x.size(1), :]

        # Concatenar el cls_token a la secuencia
        x = torch.cat((cls_tokens, x), dim=1)

        # Pasar por el transformer encoder
        x = self.transformer_encoder(x)

        # Extraer la salida del cls_token
        cls_output = x[:, 0, :]

        # Pasar por la capa fully connected
        x = self.fc(cls_output)

        return x

# Función de entrenamiento (ajustada para mostrar los resultados en %)
def train_model(model, train_loader, validation_loader, num_epochs=50, patience=5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Crear listas para almacenar las métricas
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    # Entrenamiento del modelo
    for epoch in range(num_epochs):
        if early_stop:
            print(f"Deteniendo el entrenamiento en la epoch {epoch + 1} debido a early stopping.")
            break
        
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in train_loader:
            if batch is None:
                continue
            features, attention_masks, labels = batch
            features, attention_masks, labels = features.to(device), attention_masks.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = logits.argmax(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calcular métricas en el conjunto de entrenamiento en %
        train_accuracy = accuracy_score(all_labels, all_preds) * 100
        train_f1 = f1_score(all_labels, all_preds, average='weighted') * 100

        # Almacenar las métricas
        train_accuracies.append(train_accuracy)
        train_f1_scores.append(train_f1)
        train_losses.append(running_loss / len(train_loader))

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader):.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, Train F1-Score: {train_f1:.2f}%')

        # Evaluación en el conjunto de validación
        model.eval()
        val_running_loss = 0.0
        val_all_preds = []
        val_all_labels = []

        with torch.no_grad():
            for batch in validation_loader:
                if batch is None:
                    continue
                features, attention_masks, labels = batch
                features, attention_masks, labels = features.to(device), attention_masks.to(device), labels.to(device)

                logits = model(features)
                loss = criterion(logits, labels)
                val_running_loss += loss.item()
                predicted = logits.argmax(1)

                val_all_preds.extend(predicted.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())

        # Calcular métricas en el conjunto de validación en %
        val_accuracy = accuracy_score(val_all_labels, val_all_preds) * 100
        val_f1 = f1_score(val_all_labels, val_all_preds, average='weighted') * 100

        # Almacenar las métricas
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)
        val_losses.append(val_running_loss / len(validation_loader))

        print(f'Validation Loss: {val_running_loss / len(validation_loader):.4f}, '
              f'Validation Accuracy: {val_accuracy:.2f}%, Validation F1-Score: {val_f1:.2f}%')

        # Comprobar si se necesita early stopping
        if val_running_loss < best_val_loss:
            best_val_loss = val_running_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("Mejor modelo guardado.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True

    # Gráficas de pérdidas y métricas
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Pérdida de Entrenamiento')
    plt.plot(val_losses, label='Pérdida de Validación')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Accuracy de Entrenamiento')
    plt.plot(val_accuracies, label='Accuracy de Validación')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.show()

# Función de prueba (ajustada para mostrar los resultados en %)
def test_model(model, test_loader):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    all_preds = []
    all_labels = []
    test_running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            features, attention_masks, labels = batch
            features, attention_masks, labels = features.to(device), attention_masks.to(device), labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)
            test_running_loss += loss.item()

            predicted = logits.argmax(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcular métricas de prueba en porcentaje con dos decimales
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0.0) * 100
    recall = recall_score(all_labels, all_preds, average='weighted') * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    avg_loss = test_running_loss / len(test_loader)

    print(f'Métricas de Prueba: Accuracy: {accuracy:.2f}%, Precisión: {precision:.2f}%, '
          f'Recall: {recall:.2f}%, F1-Score: {f1:.2f}%, Pérdida Promedio: {avg_loss:.4f}')


# Función principal
def main(train_mode):
    # Crear datasets y loaders
    train_dataset = SignLanguageDataset(hdf5_path, train_list, labels_dict, common_classes)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    validation_dataset = SignLanguageDataset(hdf5_path, validation_list, labels_dict, common_classes)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    test_dataset = SignLanguageDataset(hdf5_path, test_list, labels_dict, common_classes)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Definir el modelo TransformerEncoder
    input_dim = 512
    hidden_dim = 256
    num_layers = 2
    num_heads = hidden_dim // 64
    dim_feedforward = hidden_dim * 4
    output_dim = len(common_classes)

    model = TransformerEncoder(input_dim, hidden_dim, num_layers, num_heads, output_dim, dim_feedforward)
    model.to(device)

    if train_mode:
        train_model(model, train_loader, validation_loader)
    else:
        test_model(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenamiento o prueba del modelo de clasificación de lenguaje de signos.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Modo a ejecutar: "train" o "test".')
    
    args = parser.parse_args()
    train_mode = args.mode == 'train'
    main(train_mode)
