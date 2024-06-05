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


##############################################  ANALIS DE DATOS ########################################################

# ----------------------------       ANALIZAR DATOS CONJUNTO 2      -------------------------------------

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







# ----------------------------       COMPROBAR NEURONAS MUERTAS      -------------------------------------

def check_dead_neurons(model, data_loader):
    dead_neurons = []

    for i, layer in enumerate(model.modules()):
        if isinstance(layer, nn.ReLU):
            inputs = next(iter(data_loader))[0]
            outputs = layer(inputs)
            num_dead_neurons = (outputs == 0).sum().item()
            dead_neurons.append((i, num_dead_neurons))

    return dead_neurons

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





###############################################    GRAFICAR DATOS    ########################################################

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






def ponderar_graficas():
    def ponderar(mirada):
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
    miradas_ponderadas = np.array([ponderar(mirad) for mirad in mirada])

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




def ponderar_graficas2():
    limiteAbajoIzq = [0.1, 0.1]
    limiteAbajoDer = [0.9, 0.1]
    limiteArribaIzq = [0.1, 0.9]
    limiteArribaDer = [0.9, 0.9]
    Desplazamiento = [0.5, 0.5]
    def ponderar(mirada):
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

    # Generar valores de mirada desde 0 hasta 1
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    mirada = np.array([[i, j] for i in x for j in y])

    # Calcular los valores ponderados
    miradas_ponderadas = np.array([ponderar(mirad) for mirad in mirada])

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

############################################################################################################################
############################################################################################################################
############################################################################################################################


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
    # Limita salida a rango 0-1
    model.add_module("sigmoid_out", nn.Sigmoid())
    return model

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

def entrenar(model, optimizer, loss_function, input_train, output_train, input_test, output_test, epochs):
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    models = []

    plt.ion()  # Activa el modo interactivo de matplotlib
    fig, ax = plt.subplots()

    for epoch in range(epochs):
        # Entrenamiento y cálculo de la pérdida
        train_predictions = model(input_train)
        train_loss = loss_function(train_predictions, output_train)
        train_losses.append(train_loss.item())

        # Validación y cálculo de la pérdida
        test_predictions = model(input_test)
        test_loss = loss_function(test_predictions, output_test)
        test_losses.append(test_loss.item())

        # Guardar el mejor modelo
        if test_loss.item() < best_loss:
            best_loss = test_loss.item()
        models.append(model)

        # Actualizar el modelo
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Graficar las pérdidas en tiempo real
        ax.clear()
        ax.plot(train_losses, label='Train Loss')
        ax.plot(test_losses, label='Test Loss')
        ax.legend()
        plt.draw()
        plt.pause(0.01)

        # Detener el entrenamiento si se presiona la tecla 'p'
        if keyboard.is_pressed('p'):
            print("Entrenamiento detenido por el usuario.")
            break
    print("Epoch mejor modelo: ", test_losses.index(min(test_losses)))
    print("Perdida test mejor modelo: ", min(test_losses), "Perdida train mejor modelo: ", train_losses[test_losses.index(min(test_losses))])
    print("Que modelo guardar?")
    guardar = int(input())
    #Evaluar el modelo elegido
    model = models[guardar]
    test_predictions = model(input_test)
    test_loss = loss_function(test_predictions, output_test)
    train_predictions = model(input_train)
    train_loss = loss_function(train_predictions, output_train)
    print("Modelo ", guardar, " -> Perdida test: ", test_loss.item(), "Perdida train: ", train_loss.item())
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

def entrenar1():
    
    # Definir la red neuronal
    entradas = 23
    topology = [100,300,500,400,100]
    model = crear_ann(entradas, topology)
    model = model.to("cuda")  

    # Cargar los datos, procesarlos y moverlos a la GPU
    print("Cargando datos")
    input, output = cargar_datos()    
    input_test, output_test = cargar_datos_test()

    #Limpiamos los datos con el ojo cerrado
    index = np.where(input[:, -2] < input[:, -1])
    input = np.delete(input, index, axis=0)
    output = np.delete(output, index, axis=0)

    #Suaavizamos los datos
    print("Suavizando datos")
    input = suavizar_datos(input, 5)

    # Convertir los datos a conjunto
    print("Procesando datos a conjunto")
    input_train = Conjuntos.conjunto_2(input)    
    input_test = Conjuntos.conjunto_2(input_test)

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
    model, train_losses, test_losses = entrenar(model, optimizer, mse_loss, input_train, output_train, input_test, output_test, 1500)



    # Mover el modelo a la CPU
    model = model.to("cpu")  

    # Guardar el modelo
    torch.save(model, './anns/pytorch/modelo.pth')

    # Graficar las pérdidas
    graficar_perdidas(train_losses, test_losses)




import matplotlib.pyplot as plt

def entrenar_resnet(epochs):
    # Cargar los datos
    input_train, output_train = cargar_datos()
    input_test, output_test = cargar_datos_test()

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
    loss_function = nn.MSELoss()  # Usa la pérdida cuadrática media para la regresión

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







if __name__ == '__main__':
#     # entrenar_resnet(1500)
#     entrenar1()
    ponderar_graficas()
    #entrenar1()

