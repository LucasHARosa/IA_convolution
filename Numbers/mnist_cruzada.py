from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 5
np.random.seed(seed)

(X, y), (X_teste, y_teste) = mnist.load_data()
previsores = X.reshape(X.shape[0], 28, 28, 1)
previsores = previsores.astype('float32')
previsores /= 255
classe = np_utils.to_categorical(y, 10)


# Controla a validação cruzada
"""
Parâmetros:
    n_splits = 5: Numero de divisões da base - Number of folds
    shuffle = True: Pega os dados aleatoriamente
    random_state = seed
"""
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
resultados = []

#a = np.zeros(5)
#b = np.zeros(shape = (classe.shape[0], 1))

# Com esse for dividiremos aleatoriamente em uma unica base os valores de teste e os valores de treino
for indice_treinamento, indice_teste in kfold.split(previsores, 
                                                    np.zeros(shape = (classe.shape[0], 1))):
    #print('Índices treinamento: ', indice_treinamento, 'Índice teste', indice_teste)
    classificador = Sequential()
    classificador.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation = 'relu'))
    classificador.add(MaxPooling2D(pool_size = (2,2)))
    classificador.add(Flatten())
    classificador.add(Dense(units = 128, activation = 'relu'))
    classificador.add(Dense(units = 10, activation = 'softmax'))
    classificador.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
                          metrics = ['accuracy'])
    
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento],
                      batch_size = 128, epochs = 5)
    precisao = classificador.evaluate(previsores[indice_teste], classe[indice_teste])
    resultados.append(precisao[1])

#media = resultados.mean()
media = sum(resultados) / len(resultados)

