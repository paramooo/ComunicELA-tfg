# Importamos las librerías necesarias
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import copy
import inspect



# Semilla
# seed = 42
# torch.cuda.manual_seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)

#Activar optimizacion cudnn
torch.backends.cudnn.benchmark = True


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

def graficar_perdidas(train_losses, val_losses, test_losses, euc_losses, indice):
  plt.figure(figsize=(10,5))
  plt.plot(train_losses, label='Entrenamiento', color='green')
  plt.plot(val_losses, label='Validación', color='orange')
  if test_losses is not None:
    plt.plot(test_losses, label='Test', color='red')
  if euc_losses is not None:
    plt.plot(euc_losses, label='Euc Loss', color='purple')
  plt.axvline(x=indice, color='purple', linestyle='--', label='Indice')
  plt.title('Pérdida durante el entrenamiento')
  plt.xlabel('Época')
  plt.ylabel('Pérdida')
  plt.legend()
  plt.show()



###############################################    ENTRENAR    ########################################################


def entrenar(model, train_dataloader, val_dataloader, test_dataloader, epochs, lr, ann=None, graficas=False, final = False):
    model = model.to("cuda")
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    train_losses = []
    val_losses = []
    test_losses = []
    euc_losses_val = []
    ecu_losses_test = []
    models = []

    loss_function = nn.MSELoss()
    loss_euclidean = euclidean_loss

    early_stopping_rounds = 30

    best_val_loss = float('inf')
    rounds_without_improvement = 0

    if graficas:
        plt.ion()  # Activa el modo interactivo de matplotlib
        fig, ax = plt.subplots()

    # Comprobar el número de argumentos que requiere la función de predicción del modelo
    num_args = len(inspect.signature(model.forward).parameters)

    # Ajuste del learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, threshold=0.001) 

    def calculate_loss(data, model, loss_function, num_args, ann):
        if num_args == 2:
            predictions = model(data[0], data[1])
        elif num_args == 3:
            predictions = model(data[0], data[1], data[2])
        elif ann:
            predictions = model(data[0])
        else:
            predictions = model(data[1])
        return loss_function(predictions, data[-1])

    for epoch in range(epochs):
        train_loss_total = 0
        val_loss_total = 0
        test_loss_total = 0
        euc_loss_total_val = 0
        euc_loss_total_test = 0

        # # Entrenamiento
        model.train()
        for data in train_dataloader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
              train_loss = calculate_loss(data, model, loss_function, num_args, ann)
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_total += train_loss.item()
        train_loss_avg = train_loss_total / len(train_dataloader)
        train_losses.append(train_loss_avg)


        # Validación
        model.eval()
        with torch.no_grad():
            for data in val_dataloader:
                with torch.cuda.amp.autocast():
                    val_loss = calculate_loss(data, model, loss_function, num_args, ann)
                    euc_loss_val = calculate_loss(data, model, loss_euclidean, num_args, ann)
                val_loss_total += val_loss.item()
                euc_loss_total_val += euc_loss_val.item()
        val_loss_avg = val_loss_total / len(val_dataloader)
        euc_loss_avg_val = euc_loss_total_val / len(val_dataloader)
        val_losses.append(val_loss_avg)
        euc_losses_val.append(euc_loss_avg_val)


        # Test
        if test_dataloader is not None:
            model.eval()
            with torch.no_grad():
                for data in test_dataloader:
                    with torch.cuda.amp.autocast():
                        test_loss = calculate_loss(data, model, loss_function, num_args, ann)
                        euc_loss_test = calculate_loss(data, model, loss_euclidean, num_args, ann)
                    test_loss_total += test_loss.item() 
                    euc_loss_total_test += euc_loss_test.item() 
            test_loss_avg = test_loss_total / len(test_dataloader)
            test_losses.append(test_loss_avg)
            euc_loss_avg_test = euc_loss_total_test / len(test_dataloader)
            ecu_losses_test.append(euc_loss_avg_test)
        
        

        # Guardar el modelo
        if final:
            models.append(copy.deepcopy(model))

        # Ajuste del learning rate
        scheduler.step(train_loss_avg)

        # Early stopping
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

        if  rounds_without_improvement >= early_stopping_rounds:
            print("Entrenamiento detenido por falta de mejora en el error de validación.")
            break


        if graficas :
            # Graficar las pérdidas en tiempo real
            ax.clear()
            ax.plot(np.clip(train_losses[:], 0, 0.2), label='Train Loss', color='blue')
            ax.plot(np.clip(val_losses[:], 0, 0.2), label='Validation Loss', color='orange')
            #ax.plot(np.clip(euc_losses_val, 0, 0.2), label='Euc Loss Val', color='purple')
            if test_dataloader is not None:
                ax.plot(np.clip(test_losses[:], 0, 1), label='Test Loss', color='red')
                #ax.plot(np.clip(ecu_losses_test, 0, 1), label='Euc Loss Test', color='pink')
            ax.legend()
            plt.draw()
            plt.pause(0.001)

        # if test_dataloader is not None:
        #     print(f"Epoch: {epoch}, Train Loss: {train_loss_avg}, Val Loss: {val_loss_avg}, Test Loss: {test_loss_avg}, Euc Loss Val: {euc_loss_avg_val}, Euc Loss Test: {euc_loss_avg_test}")
        # else:
        #     print(f"Epoch: {epoch}, Train Loss: {train_loss_avg}, Val Loss: {val_loss_avg}, Euc Loss Val: {euc_loss_avg_val}")
    
    indice = val_losses.index(best_val_loss)

    if graficas:
        plt.ioff()
        plt.close(fig)

    if final:
        models = [model.to("cpu") for model in models]

    return models, train_losses, val_losses, test_losses, euc_losses_val , indice

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def entrenar_con_kfold(modelo, dataloader, epochs, lr, ejecuciones_fold, ann=None, graficas=False):
    losses_train = []
    losses_val = []
    losses_euc = []
    cambios_de_persona = dataloader.dataset.get_indices_persona()
    numero_de_persona = len(np.unique(cambios_de_persona))
    for i in range(numero_de_persona):
        # Obtener los índices de entrenamiento y prueba para esta persona
        train_indices = [idx for idx, persona in enumerate(cambios_de_persona) if persona != i+1]
        val_indices = [idx for idx, persona in enumerate(cambios_de_persona) if persona == i+1]

        # Inicializar las pérdidas de cada fold
        fold_losses_train = []
        fold_losses_val = []
        fold_losses_euc = []

        for k in range(ejecuciones_fold):
            # Modelo nuevo para cada fold
            model = modelo()

            if not ann:
              model.apply(weights_init)

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_indices)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_indices)

            train_dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, sampler=train_subsampler)
            val_dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, sampler=val_subsampler)

            _, train_losses, val_losses, _, euc_losses, j = entrenar(model, train_dataloader, val_dataloader, None, epochs, lr, ann, graficas)

            print(f"Fold: {i}, Ejecucion: {k}, ErrorMSEActual: {val_losses[j]} EucLoss: {euc_losses[j]} Epochs:{len(euc_losses)} EpochModelo: {j}")

            fold_losses_train.append(train_losses[j])
            fold_losses_val.append(val_losses[j])
            fold_losses_euc.append(euc_losses[j])

        # Calcular la media de las pérdidas para este fold
        losses_train.append(np.mean(fold_losses_train))
        losses_val.append(np.mean(fold_losses_val))
        losses_euc.append(np.mean(fold_losses_euc))

    return losses_train , losses_val, losses_euc

