#Funciones provisionales para la creacion de la memoria del proyecto
import numpy as np
import matplotlib.pyplot as plt
from Conjuntos import *
from scipy.ndimage import gaussian_filter1d
import sys
import matplotlib.colors as mcolors
import torch
import optuna
from torch.nn.functional import mse_loss
from EditorFrames import EditorFrames
from DatasetEntero import DatasetEntero
from torch.utils.data import DataLoader
import time
import os
import shutil 
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor

#Importamos el detector
sys_copy = sys.path.copy()
sys.path.append('./')
from Detector import Detector
sys.path = sys_copy

# Funcion para cargar los datos de entrenamiento
def cargar_datos():
    # Cargar los datos
    input = np.loadtxt('./entrenamiento/datos/txts/input.txt', delimiter=',')
    output = np.loadtxt('./entrenamiento/datos/txts/output.txt', delimiter=',')

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




#---------------- FUNCIONES PARA LA PONDERACION ----------------
#--------------------------------------------------------------

#Funcion de ponderar por cuadrantes
def ponderar_graficas():
    def _ponderar(mirada):
        # Definir los límites
        arriba_izq = np.array([0.05, 1])
        arriba_der = np.array([1, 1])
        abajo_izq = np.array([0, 0])
        abajo_der = np.array([1, 0])

        # Definir los límites de las zonas
        LimiteBajoX = 0
        LimiteAltoX = 0
        LimiteBajoY = 0
        LimiteAltoY = 0

        # Calcular el cuadrante de la mirada
        cuadrante = 0
        if mirada[0] >= 0.5:
            cuadrante += 1
        if mirada[1] >= 0.5:
            cuadrante += 2

        # Ponemos los limites de la esquina
        if cuadrante == 0:
            LimiteBajoX = abajo_izq[0]
            LimiteBajoY = abajo_izq[1]
            LimiteAltoX = abajo_der[0]
            LimiteAltoY = arriba_izq[1]
        if cuadrante == 1:
            LimiteAltoX = abajo_der[0]
            LimiteBajoY = abajo_der[1]
            LimiteAltoY = arriba_der[1]
            LimiteBajoX = abajo_izq[0]
        if cuadrante == 2:
            LimiteBajoX = arriba_izq[0]
            LimiteAltoY = arriba_izq[1]
            LimiteBajoY = abajo_izq[1]
            LimiteAltoX = arriba_der[0]
        if cuadrante == 3:
            LimiteAltoX = arriba_der[0]
            LimiteAltoY = arriba_der[1]
            LimiteBajoY = abajo_der[1]
            LimiteBajoX = arriba_izq[0]

        # Calculamos los limites de la zona no afectada
        ComienzoZonaNoAfectadaX = LimiteBajoX + (0.5-LimiteBajoX)/2
        FinZonaNoAfectadaX = LimiteAltoX - (LimiteAltoX-0.5)/2
        ComienzoZonaNoAfectadaY = LimiteBajoY + (0.5-LimiteBajoY)/2
        FinZonaNoAfectadaY = LimiteAltoY - (LimiteAltoY-0.5)/2

        # Calculamos las x y las y de las Xs
        Xx = np.array([LimiteBajoX, ComienzoZonaNoAfectadaX, 0.5, FinZonaNoAfectadaX, LimiteAltoX])
        Xy = np.array([0, ComienzoZonaNoAfectadaX, 0.5, FinZonaNoAfectadaX, 1])
        Yx = np.array([LimiteBajoY, ComienzoZonaNoAfectadaY, 0.5, FinZonaNoAfectadaY, LimiteAltoY])
        Yy = np.array([0, ComienzoZonaNoAfectadaY, 0.5, FinZonaNoAfectadaY, 1])

        # Crear la función polinómica
        polinomioX = np.poly1d(np.polyfit(Xx, Xy, 4))
        polinomioY = np.poly1d(np.polyfit(Yx, Yy, 4))

        # Calcular el valor ponderado
        return np.array([np.clip(polinomioX(mirada[0]),0,1), np.clip(polinomioY(mirada[1]),0,1)])


    # Generar valores de mirada desde 0 hasta 1
    x = np.linspace(0, 1, 100)
    mirada = np.array([[i, i] for i in x])

    # Calcular los valores ponderados
    miradas_ponderadas = np.array([_ponderar(mirad) for mirad in mirada])

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(mirada[:, 0], label='Mirada original')
    plt.plot(miradas_ponderadas[:, 0], label='Mirada ponderada')
    plt.xlabel('Fotograma')
    plt.ylabel('Valor de X de la mirada')
    plt.title('Función de ponderación')
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

    # Imprimir el peso de cada esquina
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





def optimizar_ponderacion():
    # Cargar el modelo y los datos
    model = torch.load('./entrenamiento/modelos/modelo_ann_1_11.pth')
    input, output = cargar_datos()
    input = Conjuntos.conjunto_1(input)

    # Limpiar los datos con el ojo cerrado
    index = np.where(input[:, -2] < input[:, -1])
    input = np.delete(input, index, axis=0)
    output = np.delete(output, index, axis=0)

    # Pasar los datos y el modelo a la GPU
    input = torch.tensor(input).float().to('cuda')
    output = torch.tensor(output).float().to('cuda')
    model = model.to('cuda')

    def objective(trial):
        # Definir los límites de las esquinas y el desplazamiento
        limiteAbajoIzqX = trial.suggest_float('limiteAbajoIzqX', 0.00, 0.25)
        limiteAbajoIzqY = trial.suggest_float('limiteAbajoIzqY', 0.00, 0.25)
        limiteAbajoDerX = trial.suggest_float('limiteAbajoDerX', 0.75, 1.00)
        limiteAbajoDerY = trial.suggest_float('limiteAbajoDerY', 0.00, 0.25)
        limiteArribaIzqX = trial.suggest_float('limiteArribaIzqX', 0.00, 0.25)
        limiteArribaIzqY = trial.suggest_float('limiteArribaIzqY', 0.75, 1.00)
        limiteArribaDerX = trial.suggest_float('limiteArribaDerX', 0.75, 1.00)
        limiteArribaDerY = trial.suggest_float('limiteArribaDerY', 0.75, 1.00)
        DesplazamientoX = trial.suggest_float('DesplazamientoX', 0.45, 0.55)
        DesplazamientoY = trial.suggest_float('DesplazamientoY', 0.45, 0.55)

        limiteAbajoIzq = [limiteAbajoIzqX, limiteAbajoIzqY]
        limiteAbajoDer = [limiteAbajoDerX, limiteAbajoDerY]
        limiteArribaIzq = [limiteArribaIzqX, limiteArribaIzqY]
        limiteArribaDer = [limiteArribaDerX, limiteArribaDerY]
        Desplazamiento = [DesplazamientoX, DesplazamientoY]

        # Calcular las predicciones del modelo
        predicciones = model(input).cpu().detach().numpy()  # Mover las predicciones a la CPU

        for i, prediccion in enumerate(predicciones):
            predicciones[i] = ponderar(prediccion, limiteAbajoIzq, limiteAbajoDer, limiteArribaIzq, limiteArribaDer, Desplazamiento)

        # Convertir de nuevo a tensor de PyTorch
        predicciones = torch.tensor(predicciones).float().to('cuda')

        # Calcular el error
        error = mse_loss(output, predicciones).item()

        return error

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)  # Puedes ajustar el número de pruebas según tus necesidades

    print('Los mejores parámetros son:', study.best_params)


def imprimir_estructura_modelo(model):
    for i, layer in enumerate(model.modules()):
        print(i, layer)


#Comparar los modelos de texto con las bd conjuntas:
def comparar_modelos_1_aprox(models_conjs):
    for i, model_conj in enumerate(models_conjs):
        model = model_conj[0]
        conjunto = model_conj[1]

        # Cargar los datos
        DatasetText1 = DatasetEntero('./entrenamiento/datos/txts/input0.txt', './entrenamiento/datos/txts/output0.txt', 21, conjunto=conjunto, imagenes=False)
        DatasetText2 = DatasetEntero('./entrenamiento/datos/txts/input1.txt', './entrenamiento/datos/txts/output1.txt', 21, conjunto=conjunto, imagenes=False)

        #Unir los datasets
        input = np.concatenate((DatasetText1.txt_input_data, DatasetText2.txt_input_data))
        output = np.concatenate((DatasetText1.txt_output_data, DatasetText2.txt_output_data))
        dataset = torch.utils.data.TensorDataset(torch.tensor(input).float(), torch.tensor(output).float())

        #Crear el dataloader
        data_loader = DataLoader(dataset, batch_size=5000, shuffle=True)

        #Evaluar el modelo
        model.eval()

        error = 0
        for inputs, outputs in data_loader:
            outputs_m = model(inputs)
            error += mse_loss(outputs_m, outputs)

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


# ----------------------------       COMPROBAR NEURONAS MUERTAS      -------------------------------------

# def check_dead_neurons(model, data_loader):
#     dead_neurons = []

#     for i, layer in enumerate(model.modules()):
#         if isinstance(layer, nn.ReLU):
#             inputs = next(iter(data_loader))[0]
#             outputs = layer(inputs)
#             num_dead_neurons = (outputs == 0).sum().item()
#             dead_neurons.append((i, num_dead_neurons))

#     return dead_neurons

#COMPROBAR NEURNAS MUERTAS-------------------------------------

    # model = torch.load('./anns/pytorch/modeloRELU.pth')
    # input, output = cargar_datos()
    # input = Conjuntos.conjunto_2(input)

    # # cREATE A DATALOADER
    # input = torch.from_numpy(input).float()
    # output = torch.from_numpy(output).float()
    # dataset = torch.utils.data.TensorDataset(input, output)
    # data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # dead_neurons = check_dead_neurons(model, data_loader)
    # print(dead_neurons)

# Funcion para renombrar archivos frame añadiendole al nombre original la persona a la que pertenece sabiendo los rangos de las personas


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
    #optimizar_ponderacion()

    #------------ PARA VERIFICAR LA ESTRUCTURA DE UN MODELO GUARDADO ----------------
    # model = torch.load('./entrenamiento/modelos/modelo_ajustado.pth')
    # imprimir_estructura_modelo(model)


    #------------ PARA COMPROBAR NEURONAS MUERTAS ----------------
    #Lo comentado al final

    #------------ PARA COMPARAR LOS MODELOS DE TEXTO ----------------
    #model1 = torch.load('./entrenamiento/modelos/modelo_ann_1_11_inpt0.pth')
    #model2 = torch.load('./entrenamiento/modelos/modelo_ann_1_11_inpt1.pth')
    # model3 = torch.load('./entrenamiento/modelos/modelo_ajustado.pth')
    # models_conjs  = [(model1, 1), (model2, 1), (model3, 2)]
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



    pass