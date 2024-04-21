#Funciones provisionales para la creacion de la memoria del proyecto
import numpy as np
import matplotlib.pyplot as plt
from Conjuntos import *
import tensorflow as tf
from scipy.ndimage import gaussian_filter1d



#------------------ FUNCIONES PARA LOS DATOS ------------------
#--------------------------------------------------------------

# Funcion para cargar los datos de entrenamiento
def cargar_datos():
    # Cargar los datos
    input = np.loadtxt('./txts/input.txt', delimiter=',')
    output = np.loadtxt('./txts/output.txt', delimiter=',')

    return input, output

# Funcion para cargar los datos de test
def cargar_datos_test():
    # Cargar los datos
    input = np.loadtxt('./txts/input2.txt', delimiter=',')
    output = np.loadtxt('./txts/output2.txt', delimiter=',')

    return input, output


def suavizar_datos(data, sigma):
    # Aplicar filtro gaussiano a cada columna
    for i in range(data.shape[1]):
        data[:, i] = gaussian_filter1d(data[:, i], sigma)
    return data


#------------------ FUNCIONES PARA LAS GRAFICAS ------------------
#---------------------------------------------------------------

# Funcion para mostrar las gráficas de los datos suaizados
def mostrar_graficas_suavizado_datos():
    input, output = cargar_datos()
    # Puntos a graficar
    puntos = [0, 3]

    # Sigmas para suavizar los datos
    sigmas = [5, 11, 15, 21]

    # Crear una figura para las gráficas
    fig, axs = plt.subplots(len(sigmas), len(puntos), figsize=(15, 15))

    for i, sigma in enumerate(sigmas):
        # Suavizar los datos
        datos_suavizados = suavizar_datos(input.copy()[:], sigma)

        for j, punto in enumerate(puntos):
            # Crear la gráfica
            axs[i, j].plot(input[:100, punto], label='Real')
            axs[i, j].plot(datos_suavizados[:100, punto], label='Suavizado')
            axs[i, j].legend()

            # Establecer el título de la gráfica
            axs[i, j].set_title(f'Sigma: {sigma}, Punto: {punto}')

    # Mostrar las gráficas
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


# Haz el main
if __name__ == '__main__':
    # Comprueba que tf reconoce la grafica
    print(tf.config.list_physical_devices('GPU'))
