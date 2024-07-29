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
from DatasetImg import DatasetImg
from DatasetText import DatasetText
from sklearn.model_selection import KFold
import inspect
from torch.utils.data import SubsetRandomSampler




##################################### FUNCION DE LOSS EUCLIDEA ############################################ (sin usar)
# Distancia euclídea
def euclidean_loss(y_true, y_pred):
    return torch.sqrt(torch.sum((y_pred - y_true) ** 2, dim=-1)).mean()



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

def graficar_perdidas(train_losses, val_losses, test_losses):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Entrenamiento', color='blue')
    plt.plot(val_losses, label='Validación', color='green')
    if test_losses is not None:
        plt.plot(test_losses, label='Prueba', color='red')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()



###############################################    ENTRENAR    ########################################################


def entrenar(model, train_dataloader, val_dataloader, test_dataloader, epochs, lr, ann=None, graficas=False):
    model = model.to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    test_losses = []
    euc_losses = []
    models = []

    loss_function = nn.MSELoss()
    loss_euclidean = euclidean_loss

    early_stopping_rounds = 20

    best_val_loss = float('inf')
    rounds_without_improvement = 0

    if graficas:
        plt.ion()  # Activa el modo interactivo de matplotlib
        fig, ax = plt.subplots()

    # Comprobar el número de argumentos que requiere la función de predicción del modelo
    num_args = len(inspect.signature(model.forward).parameters)

    for epoch in range(epochs):
        train_loss_total = 0
        val_loss_total = 0
        test_loss_total = 0
        euc_loss_total = 0

        # Entrenamiento
        model.train()
        for data in train_dataloader:
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
            train_loss.backward()
            optimizer.step()

        # Validación
        model.eval()
        with torch.no_grad():
            for data in val_dataloader:
                # Mover los datos a la GPU
                data = [item.to("cuda") for item in data]

                # Cálculo de la pérdida
                if num_args == 2:
                    val_predictions = model(data[0], data[1])
                elif ann:
                    val_predictions = model(data[0])
                else:
                    val_predictions = model(data[1])

                val_loss_total += loss_function(val_predictions, data[-1]).item()
                euc_loss_total += loss_euclidean(val_predictions, data[-1]).item()

        val_loss_avg = val_loss_total / len(val_dataloader)
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

        if rounds_without_improvement >= early_stopping_rounds:
            # print("Entrenamiento detenido por falta de mejora en el error de validación.")
            break

        # Test
        if test_dataloader is not None:
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

                    test_loss_total += loss_function(test_predictions, data[-1]).item()

        # Guardar el modelo actual
        models.append(copy.deepcopy(model))
        
        # Calcula las pérdidas 
        train_loss_avg = train_loss_total / len(train_dataloader)
        train_losses.append(train_loss_avg)

        val_loss_avg = val_loss_total / len(val_dataloader)
        val_losses.append(val_loss_avg)

        if test_dataloader is not None:
            test_loss_avg = test_loss_total / len(test_dataloader)
            test_losses.append(test_loss_avg)

        euc_loss_avg = euc_loss_total / len(val_dataloader)
        euc_losses.append(euc_loss_avg)

        


        if graficas:
            # Graficar las pérdidas en tiempo real
            ax.clear()
            ax.plot(train_losses, label='Train Loss')
            ax.plot(val_losses, label='Validation Loss')
            if test_dataloader is not None:
                ax.plot(test_losses, label='Test Loss')
            ax.plot(euc_losses, label='Euc Loss')
            ax.legend()
            plt.draw()
            plt.pause(0.001)

        # if test_dataloader is not None:
        #     print(f'Epoch {epoch}, Train Loss: {train_loss_avg}, Validation Loss: {val_loss_avg}, Test Loss: {test_loss_avg}, Sin mejorar: {rounds_without_improvement}/{early_stopping_rounds}', end='\r')
        # else:
        #     print(f'Epoch {epoch}, Train Loss: {train_loss_avg}, Validation Loss: {val_loss_avg}, Sin mejora: {rounds_without_improvement}/{early_stopping_rounds}' , end='\r')
        
        # Detener el entrenamiento si se presiona la tecla 'p'
        # if keyboard.is_pressed('p'):
        #     print("Entrenamiento detenido por el usuario.")
        #     break

    # Seleccionar el modelo con menor error de validación
    model = models[val_losses.index(min(val_losses))]

    if graficas:
        plt.ioff()
        plt.close(fig)

    # if test_dataloader is not None:
    #     print(f'Train loss: {train_losses[val_losses.index(min(val_losses))]}, Val loss: {min(val_losses)}, Test loss: {test_losses[val_losses.index(min(val_losses))]}')
    # else:
    #     print(f'Train loss: {train_losses[val_losses.index(min(val_losses))]}, Val loss: {min(val_losses)}')
    
    # Mover el modelo a la CPU
    model = model.to("cpu")

    return model, train_losses, val_losses, test_losses, euc_losses, val_losses.index(min(val_losses))




def entrenar_con_kfold(modelo, dataloader, cambios_de_persona, epochs, lr, ejecuciones_fold, ann=None, graficas=False):
    losses_train = []
    losses_val = []
    losses_euc = []

    for i in range(len(cambios_de_persona)):
        # Los índices de entrenamiento serán todos los datos excepto la persona i
        if i == 0:
            indices_entrenamiento = list(range(cambios_de_persona[i], cambios_de_persona[-1]))
            indices_prueba = list(range(0, cambios_de_persona[i]))
        else:
            indices_entrenamiento = list(range(0, cambios_de_persona[i-1])) + list(range(cambios_de_persona[i], cambios_de_persona[-1]))
            indices_prueba = list(range(cambios_de_persona[i-1], cambios_de_persona[i]))

        # Inicializar las pérdidas de cada fold
        fold_losses_train = []
        fold_losses_val = []
        fold_losses_euc = []

        for k in range(ejecuciones_fold):
            # Modelo nuevo para cada fold
            model = copy.deepcopy(modelo)

            train_subsampler = torch.utils.data.SubsetRandomSampler(indices_entrenamiento)
            val_subsampler = torch.utils.data.SubsetRandomSampler(indices_prueba)

            train_dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, sampler=train_subsampler)
            val_dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, sampler=val_subsampler)

            model, train_losses, val_losses, _, euc_losses, j = entrenar(model, train_dataloader, val_dataloader, None, epochs, lr, ann, graficas)

            print(f"\rFold: {i}, Ejecucion: {k}, ErrorMSEActual: {val_losses[j]} EucLoss: {euc_losses[j]}", end='')

            fold_losses_train.append(train_losses[j])
            fold_losses_val.append(val_losses[j])
            fold_losses_euc.append(euc_losses[j])
        
        # Calcular la media de las pérdidas para este fold
        losses_train.append(np.mean(fold_losses_train))
        losses_val.append(np.mean(fold_losses_val))
        losses_euc.append(np.mean(fold_losses_euc))

    return losses_train , losses_val, losses_euc
 
def aproximacion1():
    # Crear un dataframe
    df = pd.DataFrame(columns=['Modelo', 'Mean EMC Val', 'Std EMC Val', 'Mean EUC Loss', 'Std EUC Loss'])

    #Conjuntos
    conjuntos = [1]
    models = {
        1 : [ANNs().crear_ann_1_13()],
        2 : [ANNs().crear_ann_2_6(), ANNs().crear_ann_2_7()],
        3 : [ANNs().crear_ann_3_6(), ANNs().crear_ann_3_7()],
        4 : [ANNs().crear_ann_4_6(), ANNs().crear_ann_4_7()]}

    #Indices de cambio de persona para el k-fold con personas diferentes
    indices_cambio_persona = [2174, 4280, 6318, 8596, 13000, 15278, 19834, 22112, 24390, 24759, 26958, 29086, 31285, 33487]

    # Bucle principal
    for conjunto in conjuntos:
        #Crear un Dataset
        dataset = DatasetText('./entrenamiento/datos/txts/input1.txt', './entrenamiento/datos/txts/output1.txt', 21, conjunto=conjunto)

        #Crear un DataLoader
        total_dataloader = DataLoader(dataset, batch_size=50000, num_workers=4)  

        #Cargar los modelos
        models = models[conjunto]    

        for j, model in enumerate(models):
            train_losses, val_losses, euc_losses = entrenar_con_kfold(model, total_dataloader, indices_cambio_persona, epochs=800, lr=0.05, ejecuciones_fold=10, ann=True, graficas=False)

            # Añadir los resultados al DataFrame
            linea = pd.Series({'Modelo': j+13, 'Mean EMC Val': np.mean(val_losses), 'Std EMC Val': np.std(val_losses), 'Mean EUC Loss': np.mean(euc_losses), 'Std EUC Loss': np.std(euc_losses)})
            df = pd.concat([df, linea.to_frame().T])

            # Imprimirlos por pantalla
            print("Modelo: ", j+13, "Conjunto: ", conjunto, 
                "\nMean EMC Val: ", np.mean(val_losses), 
                "\nStd EMC Val: ", np.std(val_losses),
                "\nMean EUC Loss: ", np.mean(euc_losses),
                "\nStd EUC Loss: ", np.std(euc_losses))
            
    df_existente = pd.read_excel('./entrenamiento/resultados/Aproximacion1.xlsx')
    df = pd.concat([df_existente, df])
    df.to_excel('./entrenamiento/resultados/Aproximacion1.xlsx', index=False)



def aproximacion2():
    #model = CNNs().crear_cnn_1()

    
    #Coger los indices
    # indices = list(range(len(dataset)))
    # train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
    # val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    # Crear un DataLoader
    # train_dataloader = DataLoader(dataset, batch_size=5000, sampler=SubsetRandomSampler(train_indices), num_workers=4)
    # val_dataloader = DataLoader(dataset, batch_size=5000, sampler=SubsetRandomSampler(val_indices), num_workers=4)
    # test_dataloader = DataLoader(dataset, batch_size=5000, sampler=SubsetRandomSampler(test_indices), num_workers=4)
    pass
def aproximacion3():
    #model = FusionNet()

    
    
    pass
def entrenar_final():
    #Coger los indices
    # indices = list(range(len(dataset)))
    # train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
    # val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    # Crear un DataLoader
    # train_dataloader = DataLoader(dataset, batch_size=5000, sampler=SubsetRandomSampler(train_indices), num_workers=4)
    # val_dataloader = DataLoader(dataset, batch_size=5000, sampler=SubsetRandomSampler(val_indices), num_workers=4)
    # test_dataloader = DataLoader(dataset, batch_size=5000, sampler=SubsetRandomSampler(test_indices), num_workers=4)

    #  Mover el modelo a la CPU
    #model = model.to("cpu")

    # Guardar el modelo con el nombre adecuado y el error 
    #torch.save(model, f'./entrenamiento/modelos/modelo_cnn_entera.pth')

    # Graficar las pérdidas
    #graficar_perdidas(train_losses, val_losses, None)
    pass


if __name__ == '__main__':

    #Semilla aleatoriedad
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Aproximacion 1
    aproximacion1()




