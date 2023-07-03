import os
import cv2
import pandas as pd


def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # Redimensione a imagem para um tamanho adequado

    # Extraia o histograma de cores
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return hist


def process_images(directory):
    features = []
    labels = []

    for subdir in os.listdir(directory):
        sub_dir_path = os.path.join(directory, subdir)
        if os.path.isdir(sub_dir_path):
            for filename in os.listdir(sub_dir_path):
                if filename.endswith(".bmp"):
                    image_path = os.path.join(sub_dir_path, filename)
                    label = subdir  # O nome da subpasta é o rótulo

                    # Extraia as características da imagem
                    hist = extract_features(image_path)

                    features.append(hist)
                    labels.append(label)

    return features, labels


# Diretórios de treino e teste
train_dir = "Train"
valid_dir = "Valid"

# Processamento das imagens de treino
train_features, train_labels = process_images(train_dir)

# Salvar os dados de treino em um arquivo CSV
train_data = pd.DataFrame(train_features)
train_data["label"] = train_labels
train_data.to_csv("train_data.csv", index=False)

# Processamento das imagens de teste
valid_features, valid_labels = process_images(valid_dir)

# Salvar os dados de teste em um arquivo CSV
valid_data = pd.DataFrame(valid_features)
valid_data["label"] = valid_labels
valid_data.to_csv("valid_data.csv", index=False)
