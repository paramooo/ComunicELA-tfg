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
from torch.utils.data import DataLoader
from Dataset import Dataset
from sklearn.model_selection import KFold
import inspect
from torch.utils.data import SubsetRandomSampler




##################################### FUNCION DE LOSS EUCLIDEA ############################################ (sin usar)
# Distancia euclídea
def euclidean_loss(y_true, y_pred):
    return torch.sqrt(torch.sum((y_pred - y_true) ** 2, dim=-1))



###############################################    GRAFICAR PERDIDAS    ########################################################

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



###############################################    ENTRENAR    ########################################################


def entrenar(model, optimizer, loss_function, train_dataloader, val_dataloader, test_dataloader, epochs, ann=None, graficas=False):
    train_losses = []
    val_losses = []
    test_losses = []
    models = []
    if graficas:
        plt.ion()  # Activa el modo interactivo de matplotlib
        fig, ax = plt.subplots()

    # Comprobar el número de argumentos que requiere la función de predicción del modelo
    num_args = len(inspect.signature(model.forward).parameters)

    for epoch in range(epochs):
        train_loss_total = 0
        val_loss_total = 0
        test_loss_total = 0

        # Entrenamiento
        model.train()
        for i, data in enumerate(train_dataloader):
            # Mover los datos a la GPU
            data = [item.to("cuda") for item in data]

            # Entrenamiento y cálculo de la pérdida
            if num_args == 2:
                train_predictions = model(data[0], data[1])
            elif ann:
                train_predictions = model(data[0])
            else:
                train_predictions = model(data[1])
            train_loss = loss_function(train_predictions, data[-1].float())
            train_loss_total += train_loss.item()

            # Actualizar el modelo
            optimizer.zero_grad()
            print("Train loss: ", train_loss, "It: ", i)
            train_loss.backward()
            optimizer.step()

        # Validación
        model.eval()
        print("Hey")
        with torch.no_grad():
            for data in val_dataloader:
                print("Hey2")
                # Mover los datos a la GPU
                data = [item.to("cuda") for item in data]
                print("Data: ", data.type())

                # Cálculo de la pérdida
                if num_args == 2:
                    val_predictions = model(data[0], data[1])
                elif ann:
                    val_predictions = model(data[0])
                else:
                    val_predictions = model(data[1])

                val_loss = loss_function(val_predictions, data[-1])
                val_loss_total += val_loss.item()

        # Test
        model.eval()
        with torch.no_grad():
            for data in test_dataloader:
                # Mover los datos a la GPU
                data = [item.to("cuda") for item in data]

                # Cálculo de la pérdida
                if num_args == 2:
                    test_predictions = model(data[0], data[1])
                elif ann:
                    test_predictions = model(data[0])
                else:
                    test_predictions = model(data[1])

                test_loss = loss_function(test_predictions, data[-1])
                test_loss_total += test_loss.item()

        # Guardar el modelo actual
        models.append(copy.deepcopy(model))
        
        # Calcula las pérdidas promedio para la época
        train_loss_avg = train_loss_total / len(train_dataloader)
        val_loss_avg = val_loss_total / len(val_dataloader)
        test_loss_avg = test_loss_total / len(test_dataloader)

        # Guarda las pérdidas promedio
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        test_losses.append(test_loss_avg)

        if graficas:
            # Graficar las pérdidas en tiempo real
            ax.clear()
            ax.plot(train_losses, label='Train Loss')
            ax.plot(val_losses, label='Validation Loss')
            ax.plot(test_losses, label='Test Loss')
            ax.legend()
            plt.draw()
            plt.pause(0.001)
        print(f'Epoch {epoch}, Train Loss: {train_loss_avg}, Validation Loss: {val_loss_avg}, Test Loss: {test_loss_avg}', end='\r')
        
        # Detener el entrenamiento si se presiona la tecla 'p'
        if keyboard.is_pressed('p'):
            print("Entrenamiento detenido por el usuario.")
            break

    print("Epoch mejor modelo: ", val_losses.index(min(val_losses)))
    print("Perdida val mejor modelo: ", min(val_losses), "Perdida train mejor modelo: ", train_losses[val_losses.index(min(val_losses))])
    print("Que modelo guardar?")
    guardar = int(input())
    model = models[guardar]

    plt.ioff()  # Desactiva el modo interactivo
    return model, train_losses, val_losses, test_losses




def entrenar_con_kfold(model, optimizer, loss_function, dataloader, epochs, n_splits, ann=None, graficas=False):
    if not isinstance(n_splits, int) or n_splits <= 1:
        raise ValueError("n_splits debe ser un entero mayor que 1")

    kfold = KFold(n_splits=n_splits, shuffle=True)
    losses = []

    for fold, (train_index, val_index) in enumerate(kfold.split(dataloader.dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_index)

        train_dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, sampler=train_subsampler)
        val_dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, sampler=val_subsampler)

        model, train_losses, val_losses, _ = entrenar(model, optimizer, loss_function, train_dataloader, val_dataloader, None, epochs, ann, graficas=False)

        losses.append((train_losses, val_losses))

        print(f'Fold {fold+1}, Train Loss: {losses[-1].item()}', end='\r')
    return model, losses


if __name__ == '__main__':
    # Crear un Dataset
    print("Iniciando Dataset")
    conjunto = 2
    dataset = Dataset('./entrenamiento/datos/frames/recortados/15-15-15', './entrenamiento/datos/txts/input1.txt', './entrenamiento/datos/txts/output1.txt', 21, conjunto=conjunto)

    # Dividir el Dataset en entrenamiento y validacion
    print("Dividiendo el Dataset")
    # train_dataset, temp_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    # val_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=42)

    #Coger los indices
    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    # Crear los samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Crear los DataLoaders
    total_dataloader = DataLoader(dataset, batch_size=50, num_workers=4)
    train_dataloader = DataLoader(dataset, batch_size=50, sampler=train_sampler, num_workers=10)
    val_dataloader = DataLoader(dataset, batch_size=50, sampler=val_sampler, num_workers=10)
    test_dataloader = DataLoader(dataset, batch_size=50, sampler=test_sampler, num_workers=10)
    
    # Funcion de loss
    loss = nn.MSELoss()
    #loss = euclidean_loss()

    # Crear la red
    #model = ANNs()
    model = CNNs().crear_cnn_1()
    #model = FusionNet()

    # Mover el modelo a la GPU
    model = model.to("cuda")

    # Optimizador
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenar el modelo
    print("Entrenando el modelo")
    model, train_losses, val_losses, test_losses = entrenar(model, optimizer, loss, train_dataloader, val_dataloader, test_dataloader, 300, ann=False, graficas=False)
    #model, losses = entrenar_con_kfold(model, optimizer, loss, total_dataloader, epochs=1000, n_splits=5, ann=False, graficas=False)
    
    #  Mover el modelo a la CPU
    model = model.to("cpu")

    # Guardar el modelo con el nombre adecuado y el error 
    torch.save(model, f'./entrenamiento/modelos/modelo_cnn_entera.pth')

    # Graficar las pérdidas
    graficar_perdidas(train_losses, val_losses, test_losses)


