import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
import os
from Conjuntos import Conjuntos

# Funcion para crear la ANN
def crear_ann(topology, learning_rate, input_shape, output_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(topology[0], activation='relu', input_shape=(input_shape,)))  #Entrada
    for layer_size in topology[1:]:
        model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
    model.add(tf.keras.layers.Dense(output_shape, activation='sigmoid'))  # Salida
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[tf.keras.metrics.MeanSquaredError()])
    return model


# Funcion para aplicar un filtro gaussiano a las columnas de las distancias para suavizar el trazado de los datos de entrenamiento
def gauss(data, sigma):
    # Aplicar filtro gaussiano con cv2 a las columnas de las distancias (32)
    for i in range(32):
        data[:, i] = cv2.GaussianBlur(data[:, i], (1, sigma), 0).flatten()
    return data

# Funcion PROVISIONAL PARA MOSTRAR GRAFICAS EN LA MEMORIA
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


def entrenar(arquitectura, learning_rate, epochs, input, output):
    # Crear el modelo
    model = crear_ann(arquitectura[1:-1], learning_rate, arquitectura[0], arquitectura[-1])

    # Entrenar el modelo
    model.fit(input, output, epochs=epochs, verbose=1)

    return model
    


def eval():
    modelo = tf.keras.models.load_model('./anns/ann_conj3_71_norm.keras')

    # Datos de test
    input_test = np.loadtxt('./txts/input2_n.txt', delimiter=',')
    output_test = np.loadtxt('./txts/output2.txt', delimiter=',')

    # Normalizar los datos
    input_test_t = Conjuntos.conjunto_3(input_test)

    modelo.evaluate(input_test_t, output_test)



def entrenar_bucle(conjuntos, arquitecturas, learning_rates, epochs):
    # Cargar los datos
    input = np.loadtxt('./txts/input.txt', delimiter=',')
    output = np.loadtxt('./txts/output.txt', delimiter=',')

    #Datos de test
    input_test = np.loadtxt('./txts/input2.txt', delimiter=',')
    output_test = np.loadtxt('./txts/output2.txt', delimiter=',')

    total_anns = len(conjuntos) * len(arquitecturas) * len(learning_rates) * len(epochs)
    current_ann = 0

    for conjunto in conjuntos:
        # Normalizar los datos
        normalizar_funcion = getattr(Conjuntos, f'conjunto_{conjunto}')
        input_norm = normalizar_funcion(input)
        input_test_norm = normalizar_funcion(input_test)

        for i, arquitectura in enumerate(arquitecturas[conjunto-1]):
            for j, lr in enumerate(learning_rates):
                for k, ep in enumerate(epochs):
                    current_ann += 1
                    print(f"Entrenando conjunto:{conjunto} arquitectura {arquitectura} lr:{lr} ep:{ep} ({current_ann}/{total_anns})")
                    model = entrenar(arquitectura, lr, ep, input_norm, output)
                    # Guardar el modelo
                    model_name = f"conj{conjunto}_{'_'.join(map(str, arquitectura[1:-1]))}_Lr{lr}_Epo{ep}"
                    model.save(f'./anns/{model_name}.keras')
                    # Evaluar el modelo e ir añadiendo los resultados a un archivo de texto
                    results = model.evaluate(input_test_norm, output_test)
                    with open('./anns/results.txt', 'a') as f:
                        f.write(f"{model_name}: {results}\n")

def main():
    # Definimos las arquitecturas que vamos a probar
    conjuntos = [3]
    arquitecturas= [#Las del conjunto 1
                    []
                    #Las del conjunto 2
                    ,[]
                    #Las del conjunto 3
                    ,[[22, 20, 15, 4, 2], [22, 15, 10, 2],  [22, 10, 6, 2], [22, 12, 2], [22, 20, 2]]]
    
    learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001]
    epochs = [100, 200, 500, 1000]

    # Entrenar las arquitecturas
    entrenar_bucle(conjuntos, arquitecturas, learning_rates, epochs)





def pruebas():
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


if __name__ == "__main__":
    main()