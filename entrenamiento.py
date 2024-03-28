import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
import os
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#Aproximacion solo distancias
def crear_ann(topology, learning_rate, input_shape, output_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(topology[0], activation='relu', input_shape=(input_shape,)))  #Entrada
    for layer_size in topology[1:]:
        model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
    model.add(tf.keras.layers.Dense(output_shape, activation='sigmoid'))  # Salida
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[tf.keras.metrics.MeanSquaredError()])
    return model



def gauss(data, sigma):
    # Aplicar filtro gaussiano con cv2 a las columnas de las distancias (32)
    for i in range(32):
        data[:, i] = cv2.GaussianBlur(data[:, i], (1, sigma), 0).flatten()
    return data

#Conjunto general con todos los datos suavizados sin normalizar
#SOL -> Una mierda, al no normalizar 0.5 de accuracy
def conjunto_1(data):
    # Gaussiano y datos a 10 decimales
    data = gauss(data, 21)

    data = np.round(data, 10)

    return data


def conjunto_2(data):
    # Aplicar el filtro gaussiano
    data = gauss(data, 21)

    # Normalizar las columnas 0-31 basándose en la columna 32
    for i in range(32):
        data[:, i] = data[:, i] / data[:, 32]

    # Eliminar la columna 32 y desplazar las columnas restantes una posición hacia atrás
    data = np.delete(data, 32, axis=1)

    # Redondear los datos a 10 decimales
    data = np.round(data, 10)

    return data


def conjunto_3(data):
    # Aplicar el filtro gaussiano
    data = gauss(data, 21)

    # Calcular la media de cada columna con la columna 16 y eliminar la columna 16
    for i in range(16):
        data[:, i] = np.mean(data[:, [i, 16]], axis=1)
        data = np.delete(data, 16, axis=1)

    # Normalizar las primeras 16 columnas basándose en la columna 17
    for i in range(16):
        data[:, i] = data[:, i] / data[:, 16]

    # Eliminar la columna 16
    data = np.delete(data, 16, axis=1)

    # Redondear los datos a 10 decimales
    data = np.round(data, 10)

    return data


def comparar_columnas(data_con_gauss, data_sin_gauss, columna):
    # Extraer la columna especificada de cada conjunto de datos
    columna_sin_gauss = data_sin_gauss[:, columna]
    columna_con_gauss = data_con_gauss[:, columna]

    # Crear la gráfica
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(columna_sin_gauss)
    plt.title('Sin Gauss')
    plt.xlabel('Índice de línea')
    plt.ylabel('Valor en la columna {}'.format(columna))

    plt.subplot(1, 2, 2)
    plt.plot(columna_con_gauss)
    plt.title('Con Gauss')
    plt.xlabel('Índice de línea')
    plt.ylabel('Valor en la columna {}'.format(columna))

    plt.tight_layout()
    plt.show()


#main
def main():
    # Cargar los datos
    #Datos: d0, d2, ....., d31, media_tamaño_ojos, or_X, or_Y, p_X, p_Y, ear, thresh_ear
    data_input = np.loadtxt('txts/input.txt', delimiter=',')
    output = np.loadtxt('txts/output.txt', delimiter=',')

    #Datos de test
    input_test = np.loadtxt('txts/input2.txt', delimiter=',')
    output_test = np.loadtxt('txts/output2.txt', delimiter=',')

    #comparar_columnas(conjunto_1(np.copy(data_input)), data_input, 0)




    # Normalizar los datos
    input = conjunto_3(data_input)
    input_test = conjunto_3(input_test)
    
    # Cargar la rna
    #model = tf.keras.models.load_model('rna_20k_ord.keras')
    
    #Evaluar el modelo
    #evaluar_modelo(model, input_test, output_test)

    # Entrenar y guardar las ANN
    #entrenar_y_guardar_anns(input, output, input_test, output_test)

    # Entenar ann
    model = crear_ann([20, 15, 4], 0.005, 22, 2)

    model.fit(input, output, epochs=500, verbose=1)

    model.evaluate(input_test, output_test)

    model.save('anns/ann_conj3_50k.keras')
    


if __name__ == "__main__":
    main()