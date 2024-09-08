"""
Fichero que contiene funciones para la visualización de datos y modelos, 
Solamente es utilizado para el desarrollo y la creación de la memoria

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys
import matplotlib.colors as mcolors
import torch
import optuna
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, Subset
import time
import os
import shutil 
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor
import inspect
from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd

#Importamos el detector
sys_copy = sys.path.copy()
from EditorFrames import EditorFrames
from DatasetEntero import DatasetEntero
from Conjuntos import *
sys.path.append('././')
from Servicios.Detector import Detector

sys.path = sys_copy

# Distancia euclídea
def euclidean_loss(y_true, y_pred):
    return torch.sqrt(torch.sum((y_pred - y_true) ** 2, dim=-1)).mean()

# Funcion para cargar los datos de entrenamiento
def cargar_datos(carpeta):
    # Cargar los datos
    input = np.loadtxt(f'./entrenamiento/datos/txts/{carpeta}/input.txt', delimiter=',')
    output = np.loadtxt(f'./entrenamiento/datos/txts/{carpeta}/output.txt', delimiter=',')

    return input, output

def suavizar_datos(data, sigma):
    # Aplicar filtro gaussiano a cada columna
    for i in range(data.shape[1]):
        data[:, i] = gaussian_filter1d(data[:, i], sigma)
    return data

#---------------------------------------------------------------
#------------------ FUNCIONES PARA LAS GRAFICAS ------------------
#---------------------------------------------------------------

# Funcion para mostrar las gráficas de los datos suaizados
def mostrar_graficas_suavizado_datos():
    input, output = cargar_datos("memoria")
    # Punto a graficar
    punto = 0

    # Sigma para suavizar los datos
    sigma = 5

    # Crear una figura para la gráfica
    fig, ax = plt.subplots(figsize=(15, 5))

    # Suavizar los datos
    datos_suavizados = suavizar_datos(input.copy()[:], sigma)

    # Crear la gráfica
    ax.plot(input[:300, punto], label='Real')
    ax.plot(datos_suavizados[:300, punto], label='Suavizado')
    ax.legend()

    # Establecer el título de la gráfica
    ax.set_title(f'Sigma: {sigma}, Índice: {punto}')

    # Mostrar la gráfica
    plt.tight_layout()
    plt.show()


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




#---------------- FUNCIONES PARA LA PONDERACION ----------------
#Funcion de ponderar por cuadrantes
def ponderar_graficas():
    def ponderar1d(x, LimiteBajo, LimiteAlto, Centro):
        # Calculamos los límites de la zona no afectada
        ComienzoZonaNoAfectada = LimiteBajo + (Centro - LimiteBajo) / 2
        FinZonaNoAfectada = LimiteAlto - (LimiteAlto - Centro) / 2

        # Calculamos los puntos para la función polinómica
        puntos = np.array([LimiteBajo, ComienzoZonaNoAfectada, Centro, FinZonaNoAfectada, LimiteAlto])
        valores = np.array([0, ComienzoZonaNoAfectada, 0.5, FinZonaNoAfectada, 1])

        # Crear la función polinómica
        polinomio = np.poly1d(np.polyfit(puntos, valores, 4))

        # Calcular el valor ponderado
        return np.clip(polinomio(x), 0, 1)

    # Generar valores de mirada desde 0 hasta 1
    x = np.linspace(0, 1, 100)

    # Calcular los valores ponderados
    LimiteBajo = 0.2
    LimiteAlto = 0.8
    Centro = 0.5
    ComienzoZonaNoAfectada = LimiteBajo + (Centro - LimiteBajo) / 2
    FinZonaNoAfectada = LimiteAlto - (LimiteAlto - Centro) / 2
    miradas_ponderadas = np.array([ponderar1d(mirad, LimiteBajo, LimiteAlto, Centro) for mirad in x])

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(x, label='Valor original')
    plt.plot(miradas_ponderadas, label='Valor ponderado')
    plt.ylabel('Valor')
    plt.title('Función de ponderación 1D')
    # Añadir etiquetas verticales específicas
    etiquetas_posiciones = [LimiteBajo * 100, ComienzoZonaNoAfectada * 100, Centro * 100, FinZonaNoAfectada * 100, LimiteAlto * 100]
    etiquetas_nombres = [f'LB: {LimiteBajo}', f'CZA: {ComienzoZonaNoAfectada:.2f}', f'C: {Centro}', f'FZA: {FinZonaNoAfectada:.2f}', f'LA: {LimiteAlto}']
    plt.xticks(etiquetas_posiciones, etiquetas_nombres)
    plt.legend()
    plt.grid(True)
    plt.show()


def ponderar(mirada, limiteAbajoIzq, limiteAbajoDer, limiteArribaIzq, limiteArribaDer, Desplazamiento):
    def calcular_limites_esquina(cuadrante):
        if cuadrante == 0:
            return limiteAbajoIzq[0], limiteAbajoIzq[1], limiteAbajoDer[0], limiteArribaIzq[1]
        elif cuadrante == 1:
            return limiteAbajoIzq[0], limiteAbajoDer[1], limiteAbajoDer[0], limiteArribaDer[1]
        elif cuadrante == 2:
            return limiteArribaIzq[0], limiteAbajoIzq[1], limiteArribaDer[0], limiteArribaIzq[1]
        elif cuadrante == 3:
            return limiteArribaIzq[0], limiteAbajoDer[1], limiteArribaDer[0], limiteArribaDer[1]

    def ponderar_esquina(mirada, esquina_limites):
        LimiteBajoX, LimiteBajoY, LimiteAltoX, LimiteAltoY = esquina_limites

        # Calculamos los límites de la zona no afectada
        ComienzoZonaNoAfectadaX = LimiteBajoX + (Desplazamiento[0] - LimiteBajoX) / 2
        FinZonaNoAfectadaX = LimiteAltoX - (LimiteAltoX - Desplazamiento[0]) / 2
        ComienzoZonaNoAfectadaY = LimiteBajoY + (Desplazamiento[1] - LimiteBajoY) / 2
        FinZonaNoAfectadaY = LimiteAltoY - (LimiteAltoY - Desplazamiento[1]) / 2

        # Calculamos las x y las y de las Xs
        Xx = np.array([LimiteBajoX, ComienzoZonaNoAfectadaX, Desplazamiento[0], FinZonaNoAfectadaX, LimiteAltoX])
        Xy = np.array([0, ComienzoZonaNoAfectadaX, 0.5, FinZonaNoAfectadaX, 1])
        Yx = np.array([LimiteBajoY, ComienzoZonaNoAfectadaY, Desplazamiento[1], FinZonaNoAfectadaY, LimiteAltoY])
        Yy = np.array([0, ComienzoZonaNoAfectadaY, 0.5, FinZonaNoAfectadaY, 1])

        # Crear la función polinómica
        polinomioX = np.poly1d(np.polyfit(Xx, Xy, 4))
        polinomioY = np.poly1d(np.polyfit(Yx, Yy, 4))

        # Calcular el valor ponderado
        return np.array([np.clip(polinomioX(mirada[0]), 0, 1), np.clip(polinomioY(mirada[1]), 0, 1)])

    def calcular_distancia(mirada, esquina):
        return np.sqrt((mirada[0] - esquina[0])**2 + (mirada[1] - esquina[1])**2)

    # Definir las cuatro esquinas
    esquinas = [limiteAbajoIzq, limiteAbajoDer, limiteArribaIzq, limiteArribaDer]
    ponderaciones = []

    # Calcular la ponderación para cada esquina
    for esquina_limites in esquinas:
        ponderacion_esquina = ponderar_esquina(mirada, calcular_limites_esquina(esquinas.index(esquina_limites)))
        ponderaciones.append(ponderacion_esquina)


    # Calcular la distancia de la mirada a cada esquina
    distancias = [calcular_distancia(mirada, esquina) for esquina in esquinas]

    # Normalizar las distancias para obtener pesos de ponderación
    pesos = np.array([1 / (distancia*2 + 1) for distancia in distancias])  # Cambio aquí

    # Realizar normalizacion min-max de los pesos
    pesos = (pesos - np.min(pesos)) / (np.max(pesos) - np.min(pesos))

    # Sumar los pesos
    suma_pesos = np.sum(pesos)

    # Ponderar las ponderaciones de acuerdo a las distancias
    ponderacion_final = np.zeros_like(ponderaciones[0])
    for i, ponderacion_esquina in enumerate(ponderaciones):
        ponderacion_final += ponderacion_esquina * (pesos[i] / suma_pesos)

    return ponderacion_final



def ponderar_graficas2():
    limiteAbajoIzq = [0.1, 0.1]
    limiteAbajoDer = [0.9, 0.1]
    limiteArribaIzq = [0.1, 0.9]
    limiteArribaDer = [0.9, 0.9]
    Desplazamiento = [0.5, 0.5]

    # Generar valores de mirada desde 0 hasta 1
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    mirada = np.array([[i, j] for i in x for j in y])

    # Calcular los valores ponderados
    miradas_ponderadas = np.array([ponderar(mirad, limiteAbajoIzq, limiteAbajoDer, limiteArribaIzq, limiteArribaDer, Desplazamiento) for mirad in mirada])

    # Crear la gráfica
    fig, ax = plt.subplots()
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["green", "yellow", "orange", "red"])

    for mirada, ponderada in zip(mirada, miradas_ponderadas):
        # Calcular la longitud de la flecha
        longitud = np.sqrt((ponderada[0] - mirada[0])**2 + (ponderada[1] - mirada[1])**2)

        # Calcular el color de la flecha
        color = cmap(longitud/0.17)

        # Dibujar la flecha
        ax.arrow(mirada[0], mirada[1], ponderada[0] - mirada[0], ponderada[1] - mirada[1], color=color)

    plt.show()





def optimizar_ponderacion(modelo, dataset, ann): 
    # Cargar el modelo y los datos
    dataloader = DataLoader(dataset, batch_size=10000)

    num_args = len(inspect.signature(modelo.forward).parameters)

    modelo = modelo.to('cuda')

    def objective(trial):
        # Definir los límites de las esquinas y el desplazamiento
        limiteAbajoIzqX = trial.suggest_float('limiteAbajoIzqX', 0.00, 0.05)
        limiteAbajoIzqY = trial.suggest_float('limiteAbajoIzqY', 0.00, 0.05)
        limiteAbajoDerX = trial.suggest_float('limiteAbajoDerX', 0.95, 1.00)
        limiteAbajoDerY = trial.suggest_float('limiteAbajoDerY', 0.00, 0.05)
        limiteArribaIzqX = trial.suggest_float('limiteArribaIzqX', 0.00, 0.05)
        limiteArribaIzqY = trial.suggest_float('limiteArribaIzqY', 0.95, 1.00)
        limiteArribaDerX = trial.suggest_float('limiteArribaDerX', 0.95, 1.00)
        limiteArribaDerY = trial.suggest_float('limiteArribaDerY', 0.95, 1.00)
        DesplazamientoX = trial.suggest_float('DesplazamientoX', 0.47, 0.53)
        DesplazamientoY = trial.suggest_float('DesplazamientoY', 0.47, 0.53)

        limiteAbajoIzq = [limiteAbajoIzqX, limiteAbajoIzqY]
        limiteAbajoDer = [limiteAbajoDerX, limiteAbajoDerY]
        limiteArribaIzq = [limiteArribaIzqX, limiteArribaIzqY]
        limiteArribaDer = [limiteArribaDerX, limiteArribaDerY]
        Desplazamiento = [DesplazamientoX, DesplazamientoY]

        # Calcular las predicciones del modelo
        modelo.eval()
        predicciones_totales = []
        posiciones_totales = []
        for data in dataloader:
            if num_args == 2:
                predicciones = modelo(data[0][0], data[0][1])
                posiciones = data[-1]
        
            elif ann:
                predicciones = modelo(data[0])
                posiciones = data[-1]
            else:
                predicciones = modelo(data[1])
                posiciones = data[-1]

            predicciones_totales.extend(predicciones.cpu().detach().numpy())
            posiciones_totales.extend(posiciones.cpu().detach().numpy())

        #for i, prediccion in enumerate(predicciones_totales): con tqdm
        for i, prediccion in enumerate(predicciones_totales):
            predicciones_totales[i] = ponderar(prediccion, limiteAbajoIzq, limiteAbajoDer, limiteArribaIzq, limiteArribaDer, Desplazamiento)

        # Convertir de nuevo a tensor de PyTorch
        predicciones_totales = torch.tensor(predicciones_totales).float().to('cuda')
        posiciones_totales = torch.tensor(posiciones_totales).float().to('cuda')

        # Calcular el error
        error = mse_loss(posiciones_totales, predicciones_totales).item()

        return error

    initial_params = {
        'limiteAbajoIzqX': 0.00,
        'limiteAbajoIzqY': 0.00,
        'limiteAbajoDerX': 1.00,
        'limiteAbajoDerY': 0.00,
        'limiteArribaIzqX': 0.00,
        'limiteArribaIzqY': 1.00,
        'limiteArribaDerX': 1.00,
        'limiteArribaDerY': 1.00,
        'DesplazamientoX': 0.50,
        'DesplazamientoY': 0.50
    }
    study = optuna.create_study(direction='minimize')
    study.enqueue_trial(initial_params)
    study.optimize(objective, n_trials=100, n_jobs=5)  

    error_antes = 0
    for data in dataloader:
        if num_args == 2:
            predicciones = modelo(data[0][0], data[0][1])
            posiciones = data[-1]
    
        elif ann:
            predicciones = modelo(data[0])
            posiciones = data[-1]
        else:
            predicciones = modelo(data[1])
            posiciones = data[-1]
        error_antes += mse_loss(posiciones, predicciones).item() 
        #error_antes += euclidean_loss(posiciones, predicciones).item()
    
    print('Los mejores parámetros son:', study.best_params)
    print('Error antes de optimizar:', error_antes/len(dataloader))
    print('El error después de optimizar es:', study.best_value)


def imprimir_estructura_modelo(model):
    for i, layer in enumerate(model.modules()):
        print(i, layer)


#Comparar los modelos de texto con las bd conjuntas:
def comparar_modelos_1_aprox(models_conjs):
    for i, model_conj in enumerate(models_conjs):
        model = model_conj[0]
        conjunto = model_conj[1]

        # Cargar los datos
        dataset = DatasetEntero("unidos")

        #Crear el dataloader
        data_loader = DataLoader(dataset, batch_size=1000, shuffle=True, num_workers=2, pin_memory=True)

        #Evaluar el modelo
        model.eval()

        error = 0
        for inputs, outputs in data_loader:
            outputs_m = model(inputs)
            error += euclidean_loss(outputs_m, outputs)

        error /= len(data_loader)
        print(f'Error del modelo {i}: {error}')

def crear_tensores_database():
    img_dir = './entrenamiento/datos/frames/recortados/15-35-15'
    img_dir_destino = './entrenamiento/datos/frames/byn/15-35-15'
    imagenes = os.listdir(img_dir)
    imagenes_ordenadas = sorted(imagenes, key=lambda x: int(x.split('_')[1].split('.')[0]))
    list_dir_destino = os.listdir(img_dir_destino)

    # Inicializamos una lista para almacenar todas las imágenes
    images = []

    for file_name in tqdm(imagenes_ordenadas):
        if file_name.endswith('.jpg'):
            if file_name not in list_dir_destino:
                # Cargamos la imagen, la convertimos a escala de grises y la normalizamos
                image = Image.open(os.path.join(img_dir, file_name)).convert('L')
                tensor_image = ToTensor()(image) / 255
                images.append(tensor_image)

    # Guardamos todas las imágenes en un solo archivo tensor
    torch.save(images, os.path.join(img_dir_destino, 'imagenes.pt'))



#Probar el tts de microsoft
def graficas_precision_modelos(modelo, dataset, ann, parametros_opt=None):
    
    dataloader = DataLoader(dataset, batch_size=10000)

    num_args = len(inspect.signature(modelo.forward).parameters)

    modelo = modelo.to('cuda')

    # Evaluar el modelo
    miradas_t = []
    posiciones_t = []
    modelo.eval()
    for data in dataloader:
        if num_args == 2:
            miradas = modelo(data[0][0], data[0][1])
            posiciones = data[-1]
    
        elif ann:
            miradas = modelo(data[0])
            posiciones = data[-1]
        else:
            miradas = modelo(data[1])
            posiciones = data[-1]

        
        miradas = miradas.cpu().detach().numpy()    
        posiciones = posiciones.cpu().detach().numpy()

        miradas_t.extend(miradas)
        posiciones_t.extend(posiciones)

    # Crear la gráfica
    if parametros_opt is None:
        grafica_calor(posiciones_t, miradas_t, 0.05, 0.3)
    else:
        miradas_p = []
        #for mirada in miradas_t: con tqdm
        for mirada in tqdm(miradas_t):
            mirada_p = ponderar(mirada, parametros_opt[0], parametros_opt[1], parametros_opt[2], parametros_opt[3], parametros_opt[4])
            miradas_p.append(mirada_p)
        print('Error euclidean antes de ponderar:', mse_loss(torch.tensor(miradas_t).float(), torch.tensor(posiciones_t).float()).item())
        print('Error euclidean después de ponderar:', mse_loss(torch.tensor(miradas_p).float(), torch.tensor(posiciones_t).float()).item())
        grafica_calor(posiciones_t, miradas_p, 0.08, 0.3)






def grafica_flechas(posiciones_t, miradas_t, lower_limit, upper_limit):
    # Crear la gráfica
    fig, ax = plt.subplots()
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["green", "yellow", "orange", "red"])

    for mirada, ponderada in zip(miradas_t, posiciones_t):

        longitud_normalized = (np.sqrt((ponderada - mirada)**2) - lower_limit) / (upper_limit - lower_limit)

        # Dibujar la flecha
        color = cmap(longitud_normalized)
        ax.arrow(mirada[0], mirada[1], ponderada[0] - mirada[0], ponderada[1] - mirada[1], color=color)

    # Crear una barra de colores
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=lower_limit, vmax=upper_limit))
    fig.colorbar(sm, ax=ax)

    plt.show()

def grafica_calor(posiciones_t, miradas_t, lower_limit, upper_limit):
    # Inicializar una matriz 2D para los errores
    filas, columnas = 45, 80
    errores = np.zeros((filas, columnas))
    contador = np.zeros((filas, columnas))    

    for mirada, posicion in zip(miradas_t, posiciones_t):
        mirada = np.clip(mirada, 0, 1)
        
        # Calcular el error euclidiano normalizado
        error_euc = np.sqrt((posicion - mirada)**2).mean()
        
        # Convertir la posición a un índice en el rango
        index = [int(posicion[1]*filas), int(posicion[0]*columnas)]

        # Asignar el error a la posición correspondiente en la matriz de errores
        errores[index[0], index[1]] += error_euc
        contador[index[0], index[1]] += 1

    for i in range(filas):
        for j in range(columnas):
            if contador[i, j] != 0:
                errores[i, j] /= contador[i, j]
            else:
                errores[i, j] = np.inf  # Set the error to infinity for positions with a count of 0


    # Aplicar el filtro gaussiano a los errores, ignorando los valores infinitos
    indices_inf = np.isinf(errores)
    errores[indices_inf] = np.nan

    # Crear el mapa de calor
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["green", "yellow", "orange", "red"])
    plt.imshow(errores, cmap=cmap, interpolation='nearest', vmin=lower_limit, vmax=upper_limit, origin='lower')
    
    # Add a colorbar
    plt.colorbar()  
    plt.xticks([0, columnas], [0, 1])
    plt.yticks([0, filas], [0, 1])
    plt.show()


def grafico_ojos():
    input, output = cargar_datos("memoria")
    for i in range(input.shape[1]):
        # Suavizar los datos para esta persona y esta columna
        input[:, i] = gaussian_filter1d(input[:, i], 21)  

    for idx in range(574, 700):  # Crear gráficos para los índices 638 a 644
        fig, ax = plt.subplots(figsize=(12, 8))
        
        inpt_izq = input[idx]
        media_ojos_mirando_izquierda = []
        for i, medida in enumerate(inpt_izq[:16]):
            media_ojos_mirando_izquierda.append(np.mean([inpt_izq[i], inpt_izq[i+16]]))
        
        inpt_der = input[idx +65]  # Ajustar el índice para el ojo derecho
        media_ojos_mirando_derecha = []
        for i, medida in enumerate(inpt_der[:16]):
            media_ojos_mirando_derecha.append(np.mean([inpt_der[i], inpt_der[i+16]]))
        
        indices = np.arange(16)
        
        # Gráfico de dispersión con líneas conectadas
        ax.plot(indices, media_ojos_mirando_izquierda, marker='o', linestyle='-', color='skyblue', label='Ojos mirando a la izquierda')
        ax.plot(indices, media_ojos_mirando_derecha, marker='o', linestyle='-', color='salmon', label='Ojos mirando a la derecha')

        # Añadir títulos y etiquetas
        ax.set_title(f'Comparación de la Media de Ambos Ojos')
        ax.set_xlabel('Índice')
        ax.set_ylabel('Media')
        ax.set_xticks(indices)
        ax.set_xticklabels(indices)
        ax.legend()

        # Añadir cuadrícula
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()





def reentrenar_grafica():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    #Cargar los datos 
    dataset = DatasetEntero("unidos")

    dataset2 = DatasetEntero("memoria")

    #Cargar el modelo
    modelo = torch.load('./entrenamiento/modelos/aprox1_92.pt').to('cuda')
    print("Error con el dataset1:", mse_loss(modelo(dataset[:][0]), dataset[:][1]).item())
    print("Error con el dataset2:", mse_loss(modelo(dataset2[:][0]), dataset2[:][1]).item())

    #Entrenar el modelo
    optimizer = optim.Adam(modelo.parameters(), lr=0.00001)

    train_losses = []
    test_losses = []
    loss = nn.MSELoss()    

    #COMPROBAR EL NUMERO DE EPOCHS CREO QUE SON DEMASIADOS FAVORECE AL OVERFITTING
    for epoch in range(100):
        optimizer.zero_grad()

        # Entrenamiento y cálculo de la pérdida
        input, output = dataset2[:]
        input_test, output_test = dataset[:]

        train_predictions = modelo(input)
        train_loss = loss(train_predictions, output)
        train_losses.append(train_loss.item())

        test_predictions = modelo(input_test)
        test_loss = loss(test_predictions, output_test)
        test_losses.append(test_loss.item())

        # Actualizar el modelo
        train_loss.backward()
        optimizer.step()

    #Crear la grafica con los losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Loss base de datos')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train loss y Loss base de datos')
    plt.legend()
    plt.grid(True)
    plt.show()




# Graficar máximo, mínimo, media y desviación de los indices [32-36] de la base de datos
def graficas_posicion_orientacion():
    #Cargar datos 
    input, _ = cargar_datos("con_imagenes")

    #Suavizar los datos
    for i in range(input.shape[1]):
        input[:, i] = gaussian_filter1d(input[:, i], 5)
    

    # S     das
    selected_columns = input[:, [32, 33, 34, 35, 36]]

    # Calcular máximos y mínimos
    min_vals = np.min(selected_columns, axis=0)
    max_vals = np.max(selected_columns, axis=0)

    # Normalizar los datos
    normalized_columns = np.clip((selected_columns - 0.3) / (0.4), 0, 1)

    # Crear las gráficas
    fig, ax = plt.subplots()

    # Dibujar las barras originales
    for i in range(selected_columns.shape[1]):
        ax.bar(i, max_vals[i] - 0.5, bottom=0.5, color='blue', alpha=0.5, width=0.4)
        ax.bar(i, 0.5 - min_vals[i], bottom=min_vals[i], color='blue', alpha=0.5, width=0.4)

    # Dibujar las barras normalizadas
    for i in range(normalized_columns.shape[1]):
        ax.bar(i, np.max(normalized_columns[:, i]) - 0.5, bottom=0.5, color='orange', alpha=0.5, width=0.2)
        ax.bar(i, 0.5 - np.min(normalized_columns[:, i]), bottom=np.min(normalized_columns[:, i]), color='orange', alpha=0.5, width=0.2)


    # Dibujar la línea horizontal en el eje 0.5
    ax.axhline(y=0.5, color='black', linestyle='--')

    # Activar la cuadrícula horizontal
    ax.yaxis.grid(True)


    # Configurar el gráfico
    ax.set_xticks(range(selected_columns.shape[1]))
    ax.set_xticklabels(['Orientación X', 'Orientación Y', 'Orientación Z', 'Posición X', 'Posición Y'])
    ax.set_ylabel('Valores')
    ax.set_title('Gráficas de Posición y Orientación')
    # Crear las barras para la leyenda
    original_patch = plt.Line2D([0], [0], color='blue', alpha=0.5, lw=5)
    normalized_patch = plt.Line2D([0], [0], color='orange', alpha=0.5, lw=5)

    # Editar la leyenda
    ax.legend([original_patch, normalized_patch], ['Rango valores originales', 'Rango valores normalizados'], loc='upper right')
    plt.show()


def obtener_rango_casilla(indice, filas, columnas):
    # Calcular el tamaño de cada casilla
    ancho_casilla = 1 / columnas
    alto_casilla = 0.8 / filas
    
    # Calcular la posición de la casilla
    fila = filas - 1 - (indice // columnas)
    columna = indice % columnas
    
    # Calcular los rangos de x e y
    x_min = columna * ancho_casilla
    x_max = (columna + 1) * ancho_casilla
    y_min = 0.2 + fila * alto_casilla
    y_max = 0.2 + (fila + 1) * alto_casilla
    
    return [(x_min, x_max), (y_min, y_max)]


def evaluar_precision_fila(casilla, puntos):
    rango = obtener_rango_casilla(casilla, 3, 4)
    (xmin, xmax), (ymin, ymax) = rango
    
    # Calculamos el centro de la casilla
    centro_x = (xmin + xmax) / 2
    centro_y = (ymin + ymax) / 2
    
    distancias = []
    for punto in puntos:
        x, y = punto
        distancia = np.sqrt((x - centro_x) ** 2 + (y - centro_y) ** 2)
        distancias.append(distancia)
    
    media_distancias = np.mean(distancias)
    return media_distancias





def normalizar_puntos(casilla, puntos):
    rango = obtener_rango_casilla(casilla, 3, 4)
    (xmin, xmax), (ymin, ymax) = rango
    
    puntos_normalizados = []
    for punto in puntos:
        x, y = punto
        x_norm = (x - xmin) / (xmax - xmin)
        y_norm = (y - ymin) / (ymax - ymin)
        puntos_normalizados.append((x_norm, y_norm))
    
    return puntos_normalizados




def calcular_precisiones():
    df = pd.read_excel("./pruebas/analizar.xlsx")
    precisiones = {}
    puntos_norm = []
    puntos_dibujar = []
    
    for index, row in df.iterrows():
        # Obtener los datos
        casilla = row[2]
        posiciones = eval(row[3])

        # Calcular la precisión
        precision = evaluar_precision_fila(casilla, posiciones)        
        if casilla not in precisiones:
            precisiones[casilla] = []
        precisiones[casilla].append(precision)

        # Normalizar los puntos
        p = normalizar_puntos(casilla, posiciones)
        puntos_norm.extend(p)

        # Grafica de los puntos unidos por líneas
        if len(p) > 20:
            puntos_dibujar.append(p[-20:])


    # Imprimir las precisiones
    for casilla, valores in precisiones.items():
        media_precision = np.mean(valores)
        print(f"Precisión índice {casilla}: {media_precision}")

    print(f"Precisión media: {np.mean([np.mean(valores) for valores in precisiones.values()])}")

    # Grafica de dispersión
    puntos_norm = np.array(puntos_norm)
    plt.scatter(puntos_norm[:, 0], puntos_norm[:, 1], alpha=0.5)
    plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-')  
    plt.show()

    # Hacer la media de los puntos dibujar, cada columna con la suya, y entre las x y entre las ys
    puntos_dibujar = np.array(puntos_dibujar)
    media_puntos = np.mean(puntos_dibujar, axis=0)

    #hacer grafico de las y
    plt.plot(media_puntos[:, 1])
    plt.show()
    




# ---------------------------------------------------------------------------------------------------------





# Haz el main
if __name__ == '__main__':
    #------------ PARA GRAFICAS DE LOS DATOS SUAVIZADOS ----------------
    #mostrar_graficas_suavizado_datos()

    #------------ PARA EDITAR LOS FRAMES Y RECORTARLOS ----------------
    #Para recortar solo los ojos
    #EditorFrames((200,50), ancho=15, altoArriba=15, altoAbajo=15).editar_frames()

    #Para recortar por enicma de las cejas y encima de la nariz
    #EditorFrames((200,70), ancho=15, altoArriba=35, altoAbajo=15).editar_frames()

    #Para editarlos por encima de las cejas y debajo de la nariz
    #EditorFrames((210,120), ancho=20, altoArriba=35, altoAbajo=55).editar_frames()

    #------------ PARA EVALUAR POR ZONAS ----------------
    # Hay que revisar como hace porque tengo q mirar ahora como va con imagenes tbn
    #model = torch.load('./anns/pytorch/modeloRELU.pth')
    #results = evaluar_zonas(2, model, 4, 4)

    #------------ PARA LAS GRAFICAS DE PONDERAR ----------------
    #Para la grafica de minimo, empezar, medio, fin, maximo
    #ponderar_graficas()

    #Para la grafica de las flechas de colores
    #ponderar_graficas2()

    #Para optimizar la ponderacion hay que ajustar las cosas dentro de la funcion con el modelo a probar etc
    # modelo = torch.load('./entrenamiento/modelos/aprox1_9Final.pt')
    # dataset = DatasetEntero("unidos")
    # optimizar_ponderacion(modelo, dataset, ann=True)

    #------------ PARA VERIFICAR LA ESTRUCTURA DE UN MODELO GUARDADO ----------------
    # model = torch.load('./entrenamiento/modelos/aprox1_9Final.pt')
    # imprimir_estructura_modelo(model)


    #------------ PARA COMPROBAR NEURONAS MUERTAS ----------------
    #Lo comentado al final

    #------------ PARA COMPARAR LOS MODELOS DE TEXTO ----------------
    # model1 = torch.load('./entrenamiento/modelos/9-10000b0005lr.pt')
    # model2 = torch.load('./entrenamiento/modelos/aprox1_9.pt')
    # model3 = torch.load('./entrenamiento/modelos/aprox1_92.pt')
    # models_conjs  = [(model1, 1), (model2, 1), (model3, 1)]
    # comparar_modelos_1_aprox(models_conjs)

    #------------ PARA LIMPIAR LOS INDICES DE LA BASE DE DATOS ----------------
    #Sirve para borrar a una persona de la base de datos cogiendo el numero de la primera foto "frame_XXXX.jpg" y el de la ultima +1
    #Asi borramos datos que esten mal o que no queramos tener en la BD
    #limpiar_indices_database(range(26519, 28717+1))

    #------------ PARA CONCATENAR LOS DATOS DE LOS TXTS Y LOS FRAMES ----------------  
    #renombrar_frames()
    # concat_db()

    #------------ PARA CREAR TENSORES DE LAS BASES DE DATOS ----------------
    #crear_tensores_database()

    #------------ PARA MAPA DE CALOR DE PRECISIÓN DEL MODELO ----------------
    #modelo = torch.load('./entrenamiento/modelos/aprox1_9Final.pt')
    #dataset = DatasetEntero('memoria')
    #sacar los datso de las personas 14 y 23
    #Coger 1 indice de cada 10 del dataset
    #dataset = torch.utils.data.Subset(dataset, range(0, len(dataset), 10))
    #graficas_precision_modelos(modelo,  dataset, ann=True)
   
    # #Para el mapa de calor pero optimizado limiteAbajoIzq, limiteAbajoDer, limiteArribaIzq, limiteArribaDer, Desplazamiento):
    # parametros = [[0.1, 0.3],
    #             [0.9, 0.1],
    #             [0.05, 0.9],
    #             [0.9, 0.9],
    #             [0.5, 0.5]]
    
    # modelo = torch.load('./entrenamiento/modelos/aprox1_9.pt')
    # dataset = DatasetEntero("memoria")
    # graficas_precision_modelos(modelo, dataset, ann=True, parametros_opt=parametros)

    #-----------------------GRAFICO DE BARRAS QUE REPRESENTA AL OJO ---------------------------
    #grafico_ojos()

    #-----------------------GRAFICA REENTRENAR EL MODELO ---------------------------
    #reentrenar_grafica()

    #-----------------------GRAFICAS DE POSICION Y ORIENTACION ---------------------------
    #graficas_posicion_orientacion()

    #cuantas epoch se entreno el modelo
    # model = torch.load('./entrenamiento/modelos/9-10000b0005lr.pt')
    # print(model.history)

    #-------------------EVALUAR PRUEBAS---------------------
    #calcular_precisiones()

    pass