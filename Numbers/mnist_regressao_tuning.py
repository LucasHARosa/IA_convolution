from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization.batch_normalization import BatchNormalization
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

# Importando a base e realizando a divisão de teste e treinamento
(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()


# transformando as imagens para o keras conseguir usar e o ultimo atributo é a quantidade de canais de saida, que no caso será 1
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)

# Alterando o tipo do dado
previsores_treinamento = previsores_treinamento.astype('float32')

# Normalizando os dados para que os valores estejam entre 0 e 1
previsores_treinamento /= 255

def criarRede(loss, activation, drop, units):
    regressor = Sequential()
    
    regressor.add(Conv2D(32, (3,3), input_shape=(28, 28, 1),activation = 'relu'))
    regressor.add(BatchNormalization())
    regressor.add(MaxPooling2D(pool_size = (2,2)))
    regressor.add(Conv2D(32, (3,3), activation = 'relu'))
    regressor.add(BatchNormalization())
    regressor.add(MaxPooling2D(pool_size = (2,2)))
    regressor.add(Flatten())
    
    regressor.add(Dense(units = units, activation = activation))
    regressor.add(Dropout(drop))
    regressor.add(Dense(units = units, activation = activation))
    regressor.add(Dropout(drop))
    
    regressor.add(Dense(units = 1, activation = 'linear'))
    regressor.compile(loss = loss, optimizer = 'adam', metrics = ['mean_absolute_error'])
    return regressor

regressor = KerasRegressor(build_fn = criarRede)
parametros={'batch_size':[128],
            'epochs':[5],
            'loss':['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error'],
            'units':[128,256,90],
            'activation':['relu','tanh'],
            'drop':[0.2,0.3]}

grid_search = GridSearchCV(estimator=regressor,
                           param_grid=parametros,
                           cv=4)

grid_search = grid_search.fit(previsores_treinamento,y_treinamento)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

