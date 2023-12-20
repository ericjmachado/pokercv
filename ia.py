import os


def treinar_modelo():
    import numpy as np
    import cv2

    # Importar classes
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

    # Definir conjunto de dados de treinamento
    path_cards = "images/cards"
    path_half_cards = 'images/half_cards'

    # Carregar imagens de cartas completas
    imagens_cards = []
    rotulos_cards = []
    rotulos_half_cards = []

    for arquivo in os.listdir(path_cards):
        imagem = cv2.imread(os.path.join(path_cards, arquivo))
        imagem = cv2.resize(imagem, (224, 224))
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        rotulos_cards.append(arquivo.split(".")[0].split(""))

    # Carregar imagens de cartas cortadas pela metade
    imagens_half_cards = []
    for arquivo in os.listdir(path_half_cards):
        imagem = cv2.imread(os.path.join(path_half_cards, arquivo))
        imagem = cv2.resize(imagem, (224, 224))
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        imagens_half_cards.append(imagem)
        rotulos_half_cards.append([arquivo.split(".")[0].split("")])

    # Combinar imagens e rótulos
    imagens = imagens_cards + imagens_half_cards
    rotulos = rotulos_cards + rotulos_half_cards

    # Dividir dados em treinamento e teste
    X_treinamento = imagens[:int(0.8 * len(imagens))]
    y_treinamento = rotulos[:int(0.8 * len(rotulos))]
    X_teste = imagens[int(0.8 * len(imagens)):]
    y_teste = rotulos[int(0.8 * len(rotulos)):]

    # Definir modelo de rede neural convolucional
    modelo = Sequential()
    modelo.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))
    modelo.add(MaxPooling2D((2, 2), strides=(2, 2)))
    modelo.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    modelo.add(MaxPooling2D((2, 2), strides=(2, 2)))
    modelo.add(Flatten())
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dense(3, activation='softmax'))

    # Compilar modelo
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Treinar modelo
    modelo.fit(X_treinamento, y_treinamento, epochs=10)

    # Avaliar modelo
    print(modelo.evaluate(X_teste, y_teste))

if __name__ == "__main__":
