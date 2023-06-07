# Projeto de Classificação de Personagens dos Simpsons

Este projeto consiste em um sistema de classificação de personagens dos Simpsons utilizando características extraídas de imagens e um classificador KNN (K-Nearest Neighbors). O objetivo é treinar o classificador com um conjunto de imagens dos simpsons
e avaliar sua acurácia usando um conjunto de imagens de teste, que tenta classificar corretamente qual personagem está presente na imagem.

## Organização do diretório

O diretório atual contém as seguintes pastas:

- `Train`: Pasta que contém as imagens utilizadas para treinamento.
- `Valid`: Pasta que contém as imagens utilizadas para teste.

## Requisitos

Certifique-se de ter o Python e as seguintes bibliotecas instaladas:

- OpenCV: `pip install opencv-python`
- scikit-learn: `pip install scikit-learn`

O OpenCV é uma biblioteca de visão computacional de código aberto. Ela oferece uma ampla gama de funções e algoritmos otimizados para processamento de imagens e visão computacional. O OpenCV é escrito em C++ e possui interfaces para várias linguagens de programação, incluindo Python.

O scikit-learn é uma biblioteca de aprendizado de máquina em Python, projetada para ser simples e eficiente de usar. Ela fornece uma ampla gama de algoritmos e ferramentas para tarefas de aprendizado de máquina, incluindo classificação, regressão, agrupamento, redução de dimensionalidade e seleção de recursos.


## Arquivos

### extrator_caracteristicas.py

Este script é responsável por extrair as características das imagens e salvar os dados de treino e teste em arquivos CSV.

- `extract_features(image_path)`: Função que recebe o caminho de uma imagem e retorna o histograma de cores normalizado.
- `process_images(directory)`: Função que processa todas as imagens do diretório, extrai as características e retorna as características e rótulos.
- `train_dir` e `valid_dir`: Diretórios de treino e teste, respectivamente.
- `train_data.csv` e `valid_data.csv`: Arquivos CSV contendo os dados de treino e teste, respectivamente. Caracteristicas nas primeiras colunas e nome do personagem na última coluna.

### classificador_knn.py

Este script carrega os dados de treinamento e teste dos arquivos CSV gerados pelo `extrator_caracteristicas.py` e realiza a classificação utilizando o algoritmo KNN.

- `train_data.csv` e `valid_data.csv`: Arquivos CSV contendo os dados de treino e teste, respectivamente.
- Loop para testar `k` de 3 a 19.
- `knn = KNeighborsClassifier(n_neighbors=k)`: Criação do classificador KNN com o número de vizinhos igual a `k`.
- `accuracy_score(valid_labels, predictions)`: Cálcula acurácia do classificador.

## Execução do projeto

1. Certifique-se de ter instalado o Python e as bibliotecas necessárias (OpenCV e scikit-learn).

2. Coloque as imagens de treino na pasta `Train` e as imagens de teste na pasta `Valid`.

3. Abra um terminal ou prompt de comando na pasta do projeto.

4. Execute o seguinte comando para extrair as características das imagens e gerar os arquivos CSV de treino e teste:

   ```shell
   python extrator_caracteristicas.py
   ```

5. Em seguida, execute o seguinte comando para treinar o classificador KNN e calcular a acurácia:

   ```shell
   python classificador_knn.py
   ```

6. Analise os resultados obtidos exibidos no terminal.

## Observações

- Certifique-se de que as imagens estejam nomeadas corretamente, seguindo o formato `personagemXXX.bmp`, onde `personagem` é o nome do personagem e `XXX` é o número da imagem.
- Os arquivos CSV de treino e teste serão gerados automaticamente pelo script `extrator_caracteristicas.py` e utilizados pelo script `classificador_knn.py`.

Certifique-se de ter as permissões adequadas para ler, gravar e executar arquivos e pastas no diretório.
