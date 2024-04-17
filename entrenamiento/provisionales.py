#Funciones provisionales para la creacion de la memoria del proyecto
import numpy as np
import matplotlib.pyplot as plt
from Conjuntos import *
from entrenamiento import cargar_datos
import tensorflow as tf


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



#----------------------------------------------------------------------------
# No es una buena manera de evaluar ya que los puntos que estan en el borde entre dos zonas logicamente no los va a acertar 
# bien siempre, entonces va a dar una media mas baja de la que realmente tendría, hay que evaluar con metricas como las de loss, error medio, etc.
#----------------------------------------------------------------------------


# Funcion para evaluar por zonas en pantalla la precision de un modelo
def evaluar_zonas(conjunto, model, hor_div, ver_div):
    # Cargar los datos
    inputs, outputs = cargar_datos()

    # Definir los límites de las zonas
    hor_limits = np.linspace(0, 1, hor_div + 1)
    ver_limits = np.linspace(0, 1, ver_div + 1)

    # Crear una matriz para almacenar los resultados
    results = np.empty((ver_div, hor_div), dtype=object)

    # Iterar sobre las zonas
    for i in range(ver_div):
        for j in range(hor_div):
            # Encontrar los índices de los outputs que caen dentro de esta zona
            indices = np.where((outputs[:, 0] >= hor_limits[j]) & (outputs[:, 0] < hor_limits[j + 1]) & 
                               (outputs[:, 1] >= ver_limits[i]) & (outputs[:, 1] < ver_limits[i + 1]))

            # Extraer los inputs y outputs correspondientes
            zone_inputs = inputs[indices]

            # Normalizar los datos al conjunto 
            normalizar_funcion = getattr(Conjuntos, f'conjunto_{conjunto}')
            zone_input_norm = normalizar_funcion(zone_inputs)

            # Hacer la predicción con el modelo
            zone_predictions = model.predict(zone_input_norm)

            # Verificar si las predicciones caen dentro de la zona
            correct_predictions = np.where((zone_predictions[:, 0] >= hor_limits[j]) & (zone_predictions[:, 0] < hor_limits[j + 1]) & 
                                           (zone_predictions[:, 1] >= ver_limits[i]) & (zone_predictions[:, 1] < ver_limits[i + 1]))

            # Calcular el porcentaje de aciertos
            accuracy = len(correct_predictions[0]) / len(zone_predictions) * 100

            # Almacenar el resultado
            results[i, j] = round(accuracy,2)

    return results


def evaluar_todos_los_modelos(conjunto, hor_div, ver_div):
    # Inicializar el mejor resultado
    mejor_media = -np.inf
    mejor_modelo = None

    # Recorrer todos los archivos en la carpeta 'anns/'
    for filename in os.listdir('./anns/'):
        if filename.endswith('.keras'):
            # Cargar el modelo
            model = tf.keras.models.load_model(f'./anns/{filename}')

            # Ejecutar la función de evaluación
            results = evaluar_zonas(conjunto, model, hor_div, ver_div)

            # Calcular la media de los resultados
            media = np.mean(results)

            print(f'Conjunto: {filename} - Precision: {media}')

            # Si esta media es la mejor hasta ahora, guardar este modelo
            if media > mejor_media:
                print("Nueva mejor media!!")
                mejor_media = media
                mejor_modelo = filename

    # Devolver el nombre del mejor modelo
    return mejor_modelo



# Haz el main
if __name__ == '__main__':
    # Comprueba que tf reconoce la grafica
    print(tf.config.list_physical_devices('GPU'))
