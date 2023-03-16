from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils import load_img, img_to_array


# Criação da rede neural convolucional
classificador = Sequential()
# Primeira camada de convolução
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
# Segunda camada de convolução
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
# Flatten para a realização da rede neural
classificador.add(Flatten())

# Rede neural
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))

# Compilação da rede neural e sua configurações
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])

# Variável que vai criar mais imagens para treino
gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)

# único argumento necessário é o rescale para todas as imagens ficarem em um padrão
gerador_teste = ImageDataGenerator(rescale = 1./255)

#Pegando as imagens de dentro das pastas
base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (64, 64),
                                                           batch_size = 32,
                                                           class_mode = 'binary')

base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                               target_size = (64, 64),
                                               batch_size = 32,
                                               class_mode = 'binary')

# steps_per_epoch = 4000 / 32 é a quantidade de imagens usadas por época
classificador.fit_generator(base_treinamento, steps_per_epoch = 4000 / 32,
                            epochs = 10, validation_data = base_teste,
                            validation_steps = 1000 / 32)


# Testando para 1 valor
imagem_teste = load_img('dataset/test_set/cachorro/dog.3500.jpg',target_size = (64,64))
imagem_teste = img_to_array(imagem_teste)
imagem_teste /= 255
# expandindo a dimensão porque é assim que o tensor flow trabalha com as imagens (quantidade de imagem,altura,largura,numero de canais)
imagem_teste = np.expand_dims(imagem_teste, axis = 0)
previsao = classificador.predict(imagem_teste)
previsao = (previsao > 0.5)
# Mostra oq é 0 e oq é 1
base_treinamento.class_indices























