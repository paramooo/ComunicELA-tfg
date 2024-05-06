# Importamos las librerías necesarias
import torch
from torch import nn
import torch.optim as optim
import numpy as np
from scipy.ndimage import gaussian_filter1d
from Conjuntos import Conjuntos
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd



#------------------ FUNCIONES PARA LOS DATOS ------------------
#--------------------------------------------------------------

# Funcion para cargar los datos de entrenamiento
def cargar_datos():
    # Cargar los datos
    input = np.loadtxt('./txts/input0.txt', delimiter=',')
    output = np.loadtxt('./txts/output0.txt', delimiter=',')

    return input, output

# Funcion para cargar los datos de test
def cargar_datos_test():
    # Cargar los datos
    input = np.loadtxt('./txts/input_test0.txt', delimiter=',')
    output = np.loadtxt('./txts/output_test0.txt', delimiter=',')

    return input, output


def suavizar_datos(data, sigma):
    # Aplicar filtro gaussiano a cada columna
    for i in range(data.shape[1]):
        data[:, i] = gaussian_filter1d(data[:, i], sigma)
    return data



#------------------ FUNCIONES DE LOSS ------------------
#-------------------------------------------------------

# Distancia euclídea
def euclidean_loss(y_true, y_pred):
    return torch.sqrt(torch.sum((y_pred - y_true) ** 2, dim=-1))

# Error medio cuadrático (MSE)
mse_loss = nn.MSELoss()



#------------------ FUNCIONES PARA LA ANN ------------------
#----------------------------------------------------------

# Función para crear la ANN
def crear_ann(entradas, topology):
    model = nn.Sequential()
    model.add_module("dense_in", nn.Linear(entradas, topology[0]))  # Entrada
    model.add_module("relu_in", nn.ReLU())
    for i in range(len(topology)-1):  # Capas ocultas
        model.add_module("dense"+str(i+1), nn.Linear(topology[i], topology[i+1]))
        model.add_module("relu"+str(i+1), nn.ReLU())
    model.add_module("dense_out", nn.Linear(topology[-1], 2))  # Salida
    model.add_module("sigmoid_out", nn.Sigmoid())
    return model

#Entrena la red y devuelve los errores de entrenamiento, validación y test
def entrenar(model, optimizer, loss_function, input_train, output_train, input_val, output_val, input_test, output_test, epochs):
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



def entrenar_k(model, optimizer, loss_function, input_train, output_train, input_val, output_val, epochs):
    train_losses = []
    val_losses = []
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

        print(f'Epoch {epoch+1}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}')

    return model, train_losses, val_losses


def graficar_perdidas(train_losses, val_losses, test_losses):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Entrenamiento', color='blue')
    plt.plot(val_losses, label='Validación', color='green')
    plt.plot(test_losses, label='Prueba', color='red')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()

# # Para el conjunto 1
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Definir la red neuronal
# entradas = 14
# topology = [17, 10, 5]

# # Cargar los datos, procesarlos y moverlos a la GPU
# input, output = cargar_datos()
# input_final = Conjuntos.conjunto_1(input)

# # Convertir los datos a tensores de PyTorch y moverlos a la GPU
# input_final = torch.from_numpy(input_final).float().to(device)
# output = torch.from_numpy(output).float().to(device)

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
#     model = model.to(device)

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

def entrenar1():
    # Para el conjunto 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Definir la red neuronal
    entradas = 23
    topology = [20, 15, 5]
    model = crear_ann(entradas, topology)
    model = model.to(device)  

    # Cargar los datos, procesarlos y moverlos a la GPU
    input, output = cargar_datos()    
    print(input.shape)
    input_final = Conjuntos.conjunto_2(input)

    # Dividir los datos en entrenamiento y validación
    input_train, input_val, output_train, output_val = train_test_split(input_final, output, test_size=0.1)

    # Convertir los datos a tensores de PyTorch y moverlos a la GPU
    input_train = torch.from_numpy(suavizar_datos(input_train,5)).float().to(device)
    output_train = torch.from_numpy(output_train).float().to(device)
    input_val = torch.from_numpy(input_val).float().to(device)
    output_val = torch.from_numpy(output_val).float().to(device)

    # Cargar los datos de prueba, procesarlos y moverlos a la GPU
    input_test, output_test = cargar_datos_test()
    input_test = Conjuntos.conjunto_2(input_test)
    input_test = torch.from_numpy(input_test).float().to(device)
    output_test = torch.from_numpy(output_test).float().to(device)

    # Definir el optimizador
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # Entrenar la red
    model, train_losses, val_losses, test_losses = entrenar(model, optimizer, mse_loss, input_train, output_train, input_val, output_val, input_test, output_test, 1000)

    # Graficar las pérdidas
    graficar_perdidas(train_losses, val_losses, test_losses)

    # Mover el modelo a la CPU
    model = model.to("cpu")  

    # Guardar el modelo
    torch.save(model, './anns/pytorch/modelo.pth')




import matplotlib.pyplot as plt

def analizar_datos_conjunto2():
    # Cargar los datos
    input, output = cargar_datos()
    input_c = input
    input = Conjuntos.conjunto_2(input)

    # Crear un DataFrame vacío para almacenar los resultados
    results = pd.DataFrame()

    # Calcular y almacenar las estadísticas para cada columna
    for i in enumerate({16,17,18,19,20,21,22}):
        io = i[1] + 16
        i = i[1]
        stats = {
            'Columna': i,
            'Desviación (Transformado)': round(np.std(input[:,i]),4),
            'Desviación (Original)': round(np.std(input_c[:,io]),4),
            'Media (Transformado)': round(np.mean(input[:,i]),4),
            'Media (Original)': round(np.mean(input_c[:,io]),4),
            'Máximo (Transformado)': round(np.max(input[:,i]),4),
            'Máximo (Original)': round(np.max(input_c[:,io]),4),
            'Mínimo (Transformado)': round(np.min(input[:,i]),4),
            'Mínimo (Original)': round(np.min(input_c[:,io]),4)
        }
        # Añadir las estadísticas al DataFrame con pd
        results = pd.concat([results, pd.DataFrame(stats, index=[0])], ignore_index=True)

    # Establecer la columna 'Columna' como el índice del DataFrame
    results.set_index('Columna', inplace=True)

    # Imprimir la tabla en un excel
    results.to_excel('./estadisticas.xlsx')
    print(results)

    # Crear gráficos de barras para cada estadística
    fig, axs = plt.subplots(4, figsize=(10,20))
    fig.suptitle('Comparación de estadísticas')
    axs[0].bar(results.index, results['Desviación (Transformado)'], label='Transformado')
    axs[0].bar(results.index, results['Desviación (Original)'], label='Original', alpha=0.5)
    axs[0].set_ylabel('Desviación')
    axs[0].legend()

    axs[1].bar(results.index, results['Media (Transformado)'], label='Transformado')
    axs[1].bar(results.index, results['Media (Original)'], label='Original', alpha=0.5)
    axs[1].set_ylabel('Media')
    axs[1].legend()

    axs[2].bar(results.index, results['Máximo (Transformado)'], label='Transformado')
    axs[2].bar(results.index, results['Máximo (Original)'], label='Original', alpha=0.5)
    axs[2].set_ylabel('Máximo')
    axs[2].legend()

    axs[3].bar(results.index, results['Mínimo (Transformado)'], label='Transformado')
    axs[3].bar(results.index, results['Mínimo (Original)'], label='Original', alpha=0.5)
    axs[3].set_ylabel('Mínimo')
    axs[3].legend()

    plt.show()



if __name__ == '__main__':
    entrenar1()