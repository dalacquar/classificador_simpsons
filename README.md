#Projeto de Classificação de Personagens dos Simpsons:

Este projeto consiste em um sistema de classificação de personagens dos Simpsons utilizando características extraídas de imagens e um classificador KNN (K-Nearest Neighbors). O objetivo é treinar o classificador com um conjunto de imagens de treino e avaliar sua acurácia usando um conjunto de imagens de teste.

##Organização do diretório
O diretório atual contém as seguintes pastas:

###Train:
  Pasta que contém as imagens utilizadas para treinamento.
###Valid:
  Pasta que contém as imagens utilizadas para teste.
###Arquivos:
extrator_caracteristicas.py
Este código é responsável por extrair as características das imagens e salvar os dados de treino e teste em arquivos CSV.

train_data.csv e valid_data.csv: Arquivos CSV contendo os dados de treino e teste, respectivamente.

Loop para testar vários valores ímpares de k.
knn = KNeighborsClassifier(n_neighbors=k): Criação do classificador KNN com o número de vizinhos igual a k.
accuracy_score(valid_labels, predictions): Cálculo da acurácia do classificador.
Execução do projeto
Certifique-se de ter as bibliotecas necessárias instaladas, como o OpenCV e o scikit-learn.
Coloque as imagens de treino na pasta Train e as imagens de teste na pasta Valid.
Execute o script extrator_caracteristicas.py para extrair as características das imagens e gerar os arquivos CSV de treino e teste.
Execute o script classificador_knn.py para carregar os dados dos arquivos CSV, treinar o classificador KNN e calcular a acurácia.
Analise os resultados obtidos.
Observações
Certifique-se de que as imagens estejam nomeadas corretamente, seguindo o formato personagemXXX.bmp, onde personagem é o nome do personagem e XXX é o número da imagem.
Os arquivos CSV de treino e teste serão gerados automaticamente pelo script extrator_caracteristicas.py e utilizados pelo script classificador_knn.py.
