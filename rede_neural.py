import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados de treinamento
train_data = pd.read_csv("train_data.csv")
train_features = train_data.drop("label", axis=1)
train_labels = train_data["label"]

# Carregar os dados de teste
valid_data = pd.read_csv("valid_data.csv")
valid_features = valid_data.drop("label", axis=1)
valid_labels = valid_data["label"]

# Criar e treinar o classificador MLP
mlp = MLPClassifier(hidden_layer_sizes=(80), activation='relu', solver='adam', random_state=42)
mlp.fit(train_features, train_labels)

# Fazer previsões com o classificador treinado
predictions = mlp.predict(valid_features)

# Calcular a acurácia do classificador
accuracy = accuracy_score(valid_labels, predictions)
print("Acurácia do classificador MLP com tamanho da camada oculta {}: {:.2f}%".format(80, accuracy * 100))

# Gerar a matriz de confusão
confusion = confusion_matrix(valid_labels, predictions)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Verdadeira")
plt.title("Matriz de Confusão")
plt.show()
