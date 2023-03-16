import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization.batch_normalization import BatchNormalization

# Importando a base e realizando a divisão de teste e treinamento
# Serão 60 mil imagens 28x28 pixels para treino e 10 mil para testes
(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()
#plt.imshow(X_treinamento[0], cmap = 'gray')
#plt.title('Classe ' + str(y_treinamento[0]))

# transformando as imagens para o keras conseguir usar e o ultimo atributo é a quantidade de canais de saida, que no caso será 1
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)

# Alterando o tipo do dado
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

# Normalizando os dados para que os valores estejam entre 0 e 1
previsores_treinamento /= 255
previsores_teste /= 255

# transformando a base para o keras conseguir fazer o treinamento
#classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
#classe_teste = np_utils.to_categorical(y_teste, 10)

classificador = Sequential()

# A cada operador de convolução é necessário ter um pooling e um flattening 


# Operador de convolução 1 etapa
"""
Parâmetros:
    1 atributo: quantidade de detectores de caracteristicas
    2 atributo: tamanho da matriz do detector de caracteristicas
    input_shape=(28, 28, 1) dimensão da imagem e seu canal
    activation = 'relu'
"""
classificador.add(Conv2D(32, (3,3), input_shape=(28, 28, 1),activation = 'relu'))

# Normalização para deixar os valores da matriz entre zero e um
classificador.add(BatchNormalization())

# Pooling 2 etapa
"""
Parâmetros:
    pool_size = (2,2): tamanho da matriz para realização do pooling
"""
classificador.add(MaxPooling2D(pool_size = (2,2)))

# Flattening 3 etapa
#classificador.add(Flatten())

# Segunda camada de convolução
classificador.add(Conv2D(32, (3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
# O flatten é usado somente na última etapa
classificador.add(Flatten())


#Redes neurais artificiais
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))

# treinamento da rede
classificador.add(Dense(units = 1, activation = 'linear'))
classificador.compile(loss = 'mean_absolute_error',
                      optimizer = 'adam', metrics = ['mean_absolute_error'])
classificador.fit(previsores_treinamento, y_treinamento,
                  batch_size = 128, epochs = 5,
                  validation_data = (previsores_teste, y_teste))

resultado = classificador.evaluate(previsores_teste, y_teste)
previsao = classificador.predict(previsores_teste)