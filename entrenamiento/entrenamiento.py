# Importamos las librerías necesarias
import torch
from torch import nn
import torch.optim as optim
import numpy as np
from scipy.ndimage import gaussian_filter1d
from Conjuntos import Conjuntos
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import models
import keyboard
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
import matplotlib.colors as mcolors
import numpy as np
from tqdm import tqdm
import optuna
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import sys
from FusionNet import FusionNet
from CNNs import CNNs
from ANNs import ANNs

############################################################################################################################
###############################################    FUNCIONES DE LOSS    ##############################################
############################################################################################################################

# Error cuadrático media
mse_loss = nn.MSELoss()

# Distancia euclídea
def euclidean_loss(y_true, y_pred):
    return torch.sqrt(torch.sum((y_pred - y_true) ** 2, dim=-1))


############################################################################################################################
###############################################    GRAFICAR PERDIDAS    ########################################################
############################################################################################################################

def graficar_perdidas_vt(train_losses, val_losses, test_losses):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Entrenamiento', color='blue')
    plt.plot(val_losses, label='Validación', color='green')
    plt.plot(test_losses, label='Prueba', color='red')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()

def graficar_perdidas(train_losses, test_losses):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Entrenamiento', color='blue')
    plt.plot(test_losses, label='Prueba', color='red')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()



############################################################################################################################
################################### FUNCIONES PARA LOS DATOS ###############################################################
############################################################################################################################

# Funcion para cargar los datos de entrenamiento
def cargar_datos():
    # Cargar los datos
    input = np.loadtxt('./entrenamiento/datos/txts/input0.txt', delimiter=',')
    output = np.loadtxt('./entrenamiento/datos/txts/output0.txt', delimiter=',')
    return input, output

def cargar_datos1():
    # Cargar los datos
    input = np.loadtxt('./entrenamiento/datos/txts/input1.txt', delimiter=',')
    output = np.loadtxt('./entrenamiento/datos/txts/output1.txt', delimiter=',')
    return input, output

# Carga las imagenes de frames/1 y los outputs de txts/output1.txt
def cargar_datos_cnn():
    # Cargar los inputs
    inputs = []
    archivos = os.listdir('./entrenamiento/datos/frames/1')
    #Transformamos a escala de grises ya que el ojo es en blanco y la pupila en negro, no deberia afectar
    for i, nombre_archivo in enumerate(archivos):
        img = cv2.imread(os.path.join('./entrenamiento/datos/frames/1', nombre_archivo), cv2.IMREAD_GRAYSCALE)/ 255.0        
        inputs.append(img)
        print(f'\rCargando.. {(i+1)/len(archivos)}%')
        #Imprimir siempre en la
    inputs = np.array(inputs)    
    inputs = np.expand_dims(inputs, axis=1)  # Añade una dimensión para los canales

    # Cargar los outputs
    output = np.loadtxt('./entrenamiento/datos/txts/output1.txt', delimiter=',')
    return inputs, output

def suavizar_datos(data, sigma):
    # Aplicar filtro gaussiano a cada columna
    for i in range(data.shape[1]):
        data[:, i] = gaussian_filter1d(data[:, i], sigma)
    return data

# Divide los datos en conjuntos de entrenamiento y prueba
def preparar_test(input, output, porcentaje):
    # Calcula el tamaño del conjunto de prueba
    test_size = porcentaje / 100.0
    
    # Divide los datos en conjuntos de entrenamiento y prueba
    #42 es la semilla para que sean los experimentos repetibles
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=test_size, random_state=42)
    
    return input_train, input_test, output_train, output_test





#------------------ FUNCIONES PARA CREAR LAS REDES ------------------
#----------------------------------------------------------



#Entrena la red y devuelve los errores de entrenamiento, validación y test
def entrenar_validacion(model, optimizer, loss_function, input_train, output_train, input_val, output_val, input_test, output_test, epochs):
    train_losses = []
    val_losses = []
    test_losses = []
    for epoch in range(epochs):  # número de épocas
        optimizer.zero_grad()  # reinicia los gradientes
        train_predictions = model(input_train)  # pasa los datos de entrenamiento a través de la red
        train_loss = loss_function(train_predictions, output_train)  # calcula la pérdida de entrenamiento
        train_loss.backward()  # retropropaga los errores
        optimizer.step()  # actualiza los pesos
        train_losses.append(train_loss.item())

        # Calcular la pérdida de validación
        val_predictions = model(input_val)
        val_loss = loss_function(val_predictions, output_val)
        val_losses.append(val_loss.item())

        # Calcular la pérdida de prueba
        test_predictions = model(input_test)
        test_loss = loss_function(test_predictions, output_test)
        test_losses.append(test_loss.item())

        print(f'Epoch {epoch+1}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}, Test Loss: {test_loss.item()}')

    return model, train_losses, val_losses, test_losses
    # Lo mismo que entrenar pero sin validacion


def entrenar(model, optimizer, loss_function, input_train, output_train, input_test, output_test, epochs, batch_size):
    train_losses = []
    test_losses = []
    models = []

    plt.ion()  # Activa el modo interactivo de matplotlib
    fig, ax = plt.subplots()

    # Calcular el numero de lotes
    num_batches = len(input_train) // batch_size

    for epoch in range(epochs):
        train_loss_total = 0
        test_loss_total = 0

        for i in range(num_batches):
            # Obtiene el lote actual
            input_batch = input_train[i*batch_size:(i+1)*batch_size]
            output_batch = output_train[i*batch_size:(i+1)*batch_size]
            
            # Entrenamiento y cálculo de la pérdida EN CADA LOTE YA QUE NO TENGO VRAM SUFICIENTE PARA TODAS LAS IMG A LA VEZ
            train_predictions = model(input_batch)
            train_loss = loss_function(train_predictions, output_batch)
            train_loss_total += train_loss.item()

            # Cálculo de la pérdida
            test_predictions = model(input_test)
            test_loss = loss_function(test_predictions, output_test)
            test_loss_total += test_loss.item()

            # Actualizar el modelo
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # Guardar el modelo actual
        models.append(copy.deepcopy(model))
        
        # Calcula las pérdidas promedio para la época
        train_loss_avg = train_loss_total / num_batches
        test_loss_avg = test_loss_total / num_batches

        # Guarda las pérdidas promedio
        train_losses.append(train_loss_avg)
        test_losses.append(test_loss_avg)

        # Graficar las pérdidas en tiempo real
        ax.clear()
        ax.plot(train_losses, label='Train Loss')
        ax.plot(test_losses, label='Test Loss')
        ax.legend()
        plt.draw()
        plt.pause(0.001)
        print(f'Epoch {epoch}, Train Loss: {train_loss_avg}, Test Loss: {test_loss_avg}', end='\r')
        
        # Detener el entrenamiento si se presiona la tecla 'p'
        if keyboard.is_pressed('p'):
            print("Entrenamiento detenido por el usuario.")
            break


    print("Epoch mejor modelo: ", test_losses.index(min(test_losses)))
    print("Perdida test mejor modelo: ", min(test_losses), "Perdida train mejor modelo: ", train_losses[test_losses.index(min(test_losses))])
    print("Que modelo guardar?")
    guardar = int(input())
    model = models[guardar]

    plt.ioff()  # Desactiva el modo interactivo
    return model, train_losses, test_losses



def entrenar_fusion(model, optimizer, loss_function, input_train_ann, input_train_cnn, output_train, input_test_ann, input_test_cnn, output_test, epochs, batch_size):
    train_losses = []
    test_losses = []
    models = []

    plt.ion()  # Activa el modo interactivo de matplotlib
    fig, ax = plt.subplots()

    # Calcular el numero de lotes
    num_samples = len(input_train_ann)
    if batch_size > num_samples:
        batch_size = num_samples
    num_batches = num_samples // batch_size

    for epoch in range(epochs):
        train_loss_total = 0
        test_loss_total = 0

        for i in range(num_batches):
            # Obtiene el lote actual
            input_batch_ann = input_train_ann[i*batch_size:(i+1)*batch_size]
            output_batch = output_train[i*batch_size:(i+1)*batch_size]
            input_batch_cnn = input_train_cnn[i*batch_size:(i+1)*batch_size]
            
            # Entrenamiento y cálculo de la pérdida EN CADA LOTE YA QUE NO TENGO VRAM SUFICIENTE PARA TODAS LAS IMG A LA VEZ
            train_predictions = model(input_batch_ann, input_batch_cnn)
            train_loss = loss_function(train_predictions, output_batch)
            train_loss_total += train_loss.item()

            # Cálculo de la pérdida
            test_predictions = model(input_test_ann, input_test_cnn)
            test_loss = loss_function(test_predictions, output_test)
            test_loss_total += test_loss.item()

            # Actualizar el modelo
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # Guardar el modelo actual
        models.append(copy.deepcopy(model))
        
        # Calcula las pérdidas promedio para la época
        train_loss_avg = train_loss_total / num_batches
        test_loss_avg = test_loss_total / num_batches

        # Guarda las pérdidas promedio
        train_losses.append(train_loss_avg)
        test_losses.append(test_loss_avg)

        # Graficar las pérdidas en tiempo real
        ax.clear()
        ax.plot(train_losses, label='Train Loss')
        ax.plot(test_losses, label='Test Loss')
        ax.legend()
        plt.draw()
        plt.pause(0.001)
        print(f'Epoch {epoch}, Train Loss: {train_loss_avg}, Test Loss: {test_loss_avg}', end='\r')

        # Liberamos memoria
        #torch.cuda.empty_cache()
        
        # Detener el entrenamiento si se presiona la tecla 'p'
        if keyboard.is_pressed('p'):
            print("Entrenamiento detenido por el usuario.")
            break


    print("Epoch mejor modelo: ", test_losses.index(min(test_losses)))
    print("Perdida test mejor modelo: ", min(test_losses), "Perdida train mejor modelo: ", train_losses[test_losses.index(min(test_losses))])
    print("Que modelo guardar?")
    guardar = int(input())
    model = models[guardar]

    plt.ioff()  # Desactiva el modo interactivo
    return model, train_losses, test_losses


# # Para el conjunto 1
# # Definir la red neuronal
# entradas = 14
# topology = [17, 10, 5]

# # Cargar los datos, procesarlos y moverlos a la GPU
# input, output = cargar_datos()
# input_final = Conjuntos.conjunto_1(input)

# # Convertir los datos a tensores de PyTorch y moverlos a la GPU
# input_final = torch.from_numpy(input_final).float().to("cuda")
# output = torch.from_numpy(output).float().to("cuda")

# # Definir el número de folds
# n_folds = 5

# # Crear el objeto KFold
# kf = KFold(n_splits=n_folds)

# # Listas para guardar las pérdidas de cada fold
# train_losses = []
# val_losses = []

# for train_index, val_index in kf.split(input_final):
#     # Crear un nuevo modelo para cada fold
#     model = crear_ann(entradas, topology)
#     model = model.to("cuda")

#     # Definir el optimizador
#     optimizer = optim.Adam(model.parameters(), lr=0.01)

#     # Dividir los datos en entrenamiento y validación
#     input_train, input_val = input_final[train_index], input_final[val_index]
#     output_train, output_val = output[train_index], output[val_index]

#     # Entrenar la red
#     model, train_loss, val_loss = entrenar_k(model, optimizer, mse_loss, input_train, output_train, input_val, output_val, 1000)

#     # Guardar las pérdidas de entrenamiento y validación
#     train_losses = train_losses + train_loss
#     val_losses = val_losses + val_loss

# # Calcular el error medio de todos los folds
# mean_train_loss = sum(train_losses) / len(train_losses)
# mean_val_loss = sum(val_losses) / len(val_losses)

# print(f'Error medio de entrenamiento: {mean_train_loss}')
# print(f'Error medio de validación: {mean_val_loss}')

def entrenar_ann():
    # Definir la red neuronal
    model = ANNs.crear_ann_1_1()
    model = model.to("cuda")  

    # Cargar los datos, procesarlos y moverlos a la GPU
    print("Cargando datos")
    input, output = cargar_datos()    

    #Limpiamos los datos con el ojo cerrado
    index = np.where(input[:, -2] < input[:, -1])
    input = np.delete(input, index, axis=0)
    output = np.delete(output, index, axis=0)

    # Crear el conjunto de test
    input, input_test, output, output_test = preparar_test(input, output, 10)

    #Suaavizamos los datos
    print("Suavizando datos")
    input = suavizar_datos(input, 5)

    # Convertir los datos a conjunto
    print("Procesando datos a conjunto")
    input_train = Conjuntos.conjunto_1(input)    
    input_test = Conjuntos.conjunto_1(input_test)

    # Dividir los datos en entrenamiento y validación
    #input_train, input_val, output_train, output_val = train_test_split(input_final, output, test_size=0.1)

    # Convertir los datos a tensores de PyTorch y moverlos a la GPU
    print("Moviendo datos a GPU")
    input_train = torch.from_numpy(input_train).float().to("cuda")
    output_train = torch.from_numpy(output).float().to("cuda")
    # input_val = torch.from_numpy(input_val).float().to("cuda")
    # output_val = torch.from_numpy(output_val).float().to("cuda")
    input_test = torch.from_numpy(input_test).float().to("cuda")
    output_test = torch.from_numpy(output_test).float().to("cuda")

    # Definir el optimizador
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenar la red
    print("Entrenando")
    model, train_losses, test_losses = entrenar(model, optimizer, mse_loss, input_train, output_train, input_test, output_test, 1500, 10000)

    # Mover el modelo a la CPU
    model = model.to("cpu")  

    # Guardar el modelo
    torch.save(model, './anns/pytorch/modelo.pth')

    # Graficar las pérdidas
    graficar_perdidas(train_losses, test_losses)




def entrenar_cnn():
    # Definir la red neuronal
    model = CNNs.crear_cnn_2()
    model = model.to("cuda")  

    # Cargar los datos, procesarlos y moverlos a la GPU
    print("Cargando datos")
    input, output = cargar_datos_cnn()  # Asegúrate de que tus datos estén en el formato correcto para la CNN
    input, input_test, output, output_test = preparar_test(input, output, 10)

    #FALTA LIMPIAR LOS DATOS CON EL OJO CERRADO

    # Convertir los datos a tensores de PyTorch y moverlos a la GPU
    print("Moviendo datos a GPU")
    input_train = torch.from_numpy(input).float().to("cuda")
    output_train = torch.from_numpy(output).float().to("cuda")
    input_test = torch.from_numpy(input_test).float().to("cuda")
    output_test = torch.from_numpy(output_test).float().to("cuda")

    # Definir el optimizador
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenar la red
    print("Entrenando")
    model, train_losses, test_losses = entrenar(model, optimizer, mse_loss, input_train, output_train, input_test, output_test, 1500, 1000)

    # Mover el modelo a la CPU
    model = model.to("cpu")  

    # Guardar el modelo
    torch.save(model, './cnns/modeloCNN.pth')

    # Graficar las pérdidas
    graficar_perdidas(train_losses, test_losses)




def entrenar_resnet_ann(epochs):
    # Cargar los datos
    input_train, output_train = cargar_datos()
    input_train, input_test, output_train, output_test = preparar_test(input_train, output_train, 10)

    # Convertir a conjunto2
    input_train = Conjuntos.conjunto_2(input_train)
    input_test = Conjuntos.conjunto_2(input_test)

    input_train = np.reshape(input_train, (-1, 1, 23))
    input_test = np.reshape(input_test, (-1, 1, 23))

    input_train = np.repeat(input_train[:, :, np.newaxis], 3, axis=1)
    input_test = np.repeat(input_test[:, :, np.newaxis], 3, axis=1)

    # Cargar los datos y moverlos a la GPU
    input_train = torch.from_numpy(input_train).float().to("cuda")
    output_train = torch.from_numpy(output_train).float().to("cuda")
    input_test = torch.from_numpy(input_test).float().to("cuda")
    output_test = torch.from_numpy(output_test).float().to("cuda")

    # Cargar el modelo preentrenado normal no el custom
    model = models.resnet50(pretrained=True)

    # Congelar los pesos del modelo
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazar la última capa del modelo para adaptarlo a tu problema
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Asume que tu problema es de regresión con dos valores de salida

    # Definir el optimizador y la función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    loss_function = mse_loss  # Usa la pérdida cuadrática media para la regresión

    model = model.to("cuda")

    # Entrenar el modelo
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        train_predictions = model(input_train)
        train_loss = loss_function(train_predictions, output_train)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

        # Calcular la pérdida de prueba
        test_predictions = model(input_test)
        test_loss = loss_function(test_predictions, output_test)
        test_losses.append(test_loss.item())

        print(f'Epoch {epoch+1}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}')

    # Guardar el modelo en la cpu
    model = model.to("cpu")
    torch.save(model, './anns/pytorch/modelo.pth')
    graficar_perdidas(train_losses, test_losses)


def entrenar_combinado():
    print("Creando redes")
    # Crear las redes ANN y CNN
    ann = ANNs.crear_ann_2_1()
    cnn = CNNs.crear_cnn_1()

    # Crear la red de fusión
    model = FusionNet(ann, cnn)
    model = model.to("cuda")

    # AL USAR LA SEMILLA PARA LOS DATOS REPETIBLES, EN LAS DOS REDES USA LOS MISMOS
    # Cargar los datos
    print("Cargando datos")
    input_ann, output_ORG = cargar_datos1()
    input_cnn, _ = cargar_datos_cnn()

    #Limpiamos los datos con el ojo cerrado
    index = np.where(input_ann [:, -2] < input_ann[:, -1])

    #Limpiamos 
    input_ann = np.delete(input_ann, index, axis=0)
    output_ORG = np.delete(output_ORG, index, axis=0)
    input_cnn = np.delete(input_cnn, index, axis=0)

    # Crear el conjunto de test 
    input_ann_train, input_ann_test, output, output_test = preparar_test(input_ann, output_ORG, 10)
    input_cnn_train, input_cnn_test, _ , _ = preparar_test(input_cnn, output_ORG, 10)

    #Suaavizamos los datos de la ann
    print("Suavizando datos")
    input_ann_train = suavizar_datos(input_ann_train, 5)

    #Convertir a conjunto los datos para la ann
    print("Procesando datos a conjunto")
    input_ann_train = Conjuntos.conjunto_2(input_ann_train)
    input_ann_test = Conjuntos.conjunto_2(input_ann_test)

    # Convertir a tensores de PyTorch y moverlos a la GPU
    print("Moviendo datos a GPU")
    input_ann_train = torch.from_numpy(input_ann_train).float().to("cuda")
    input_ann_test = torch.from_numpy(input_ann_test).float().to("cuda")

    input_cnn_train = torch.from_numpy(input_cnn_train).float().to("cuda")
    input_cnn_test = torch.from_numpy(input_cnn_test).float().to("cuda")

    output_train = torch.from_numpy(output).float().to("cuda")
    output_test = torch.from_numpy(output_test).float().to("cuda")

    # Definir el optimizador
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # Entrenar la red
    print("Entrenando")
    model, train_losses, test_losses = entrenar_fusion(model, optimizer, mse_loss, input_ann_train, input_cnn_train, output_train, input_ann_test, input_cnn_test, output_test, 1500, 1000)

    # Mover el modelo a la CPU
    model = model.to("cpu")

    # Guardar el modelo
    torch.save(model, './fusiones/pytorch/modeloFusion.pth')

    # Graficar las pérdidas
    graficar_perdidas(train_losses, test_losses)




if __name__ == '__main__':


#     # entrenar_resnet(1500)
#     entrenar1()
    #ponderar_graficas()
    #entrenar1()
    #optimizar_ponderacion()
    #entrenar_combinado()
    #entrenar_ann()
    #editar_frames(ratio=200/50, ancho=15, altoArriba=15, altoAbajo=15)
    #analizar_datos_conjunto2()
    entrenar_cnn()