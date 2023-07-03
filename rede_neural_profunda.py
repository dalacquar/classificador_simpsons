import os
import cv2
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.utils import to_categorical

# Configurações dos dados e diretórios
train_dir = 'Train'  # Pasta de treinamento
valid_dir = 'Valid'  # Pasta de teste
classes = ['bart', 'homer', 'lisa', 'maggie', 'marge']
image_size = (64, 64)

# Carregar imagens e rótulos do conjunto de treinamento
train_images = []
train_labels = []
for class_name in classes:
    class_dir = os.path.join(train_dir, class_name)
    for file_name in os.listdir(class_dir):
        if file_name.endswith('.bmp'):
            image_path = os.path.join(class_dir, file_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, image_size)
            train_images.append(image)
            train_labels.append(class_name)

# Carregar imagens e rótulos do conjunto de teste
test_images = []
test_labels = []
for class_name in classes:
    class_dir = os.path.join(valid_dir, class_name)
    for file_name in os.listdir(class_dir):
        if file_name.endswith('.bmp'):
            image_path = os.path.join(class_dir, file_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, image_size)
            test_images.append(image)
            test_labels.append(class_name)

# Converter listas em arrays numpy
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Codificar rótulos
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
train_labels = to_categorical(train_labels)
test_labels = label_encoder.transform(test_labels)
test_labels = to_categorical(test_labels)

# Pré-processamento das imagens para entrada no modelo VGG16
train_images = preprocess_input(train_images)
test_images = preprocess_input(test_images)

# Carregar modelo VGG16 pré-treinado (sem as camadas densas)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

# Congelar os pesos das camadas convolucionais
for layer in base_model.layers:
    layer.trainable = False

# Adicionar camadas densas no topo do modelo
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x)

# Criar modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar modelo
model.fit(train_images, train_labels, epochs=8, batch_size=32, validation_data=(test_images, test_labels))

# Avaliar modelo
_, accuracy = model.evaluate(test_images, test_labels)
print("Acurácia: {:.2f}%".format(accuracy * 100))

# Fazer previsões no conjunto de teste
y_pred = model.predict(test_images)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_labels, axis=1)

# Calcular a matriz de confusão
confusion = confusion_matrix(y_true, y_pred)
print("Matriz de Confusão:")
print(confusion)

# Exibir a matriz de confusão como uma imagem
class_names = label_encoder.classes_
plt.figure(figsize=(8, 6))
plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.tight_layout()
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Verdadeira")
plt.title("Matriz de Confusão")
plt.show()
