import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregar os dados de treinamento
train_data = pd.read_csv("train_data.csv")
train_features = train_data.drop("label", axis=1)
train_labels = train_data["label"]

# Carregar os dados de teste
valid_data = pd.read_csv("valid_data.csv")
valid_features = valid_data.drop("label", axis=1)
valid_labels = valid_data["label"]

# Loop para testar apenas valores ímpares de k
for k in range(3, 20, 2):
    # Criar e treinar o classificador KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_features, train_labels)

    # Fazer previsões com o classificador treinado
    predictions = knn.predict(valid_features)

    # Calcular a acurácia do classificador
    accuracy = accuracy_score(valid_labels, predictions)
    print("Acurácia do classificador KNN com k={}: {:.2f}%".format(k, accuracy * 100))
