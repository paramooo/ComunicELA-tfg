from entrenamiento import entrenar, graficar_perdidas
from ANNs import ANNs
import pandas as pd
import numpy as np
from DatasetEntero import DatasetEntero
from torch.utils.data import DataLoader
import torch

if __name__ == '__main__':
    
    modelo = ANNs().crear_ann_1_3()

    # Crear un dataframe
    df = pd.DataFrame(columns=['Modelo', 'Mean EMC Val', 'Std EMC Val', 'Mean EUC Loss', 'Std EUC Loss'])

    #Crear los Dataset
    dataset = DatasetEntero("texto_solo")
    dataset_test = DatasetEntero("memoria")

    #Dividir el de train en train y val
    total_dataloader = DataLoader(dataset, batch_size=6000, num_workers=2)
    #dataset_val = DatasetEntero("memoria")
    #val_dataloader = DataLoader(dataset_val, batch_size=10000, num_workers=2)

    #Indices
    indices = dataset.get_indices_persona()
    unique_persons = np.unique(indices)

    # Seleccionar aleatoriamente dos personas para validación y 1 diferente para test
    personas_val = np.random.choice(unique_persons, 2, replace=False)
    unique_persons = unique_persons[~np.isin(unique_persons, personas_val)]  # Eliminar personas_val de unique_persons
    #personas_test = np.random.choice(unique_persons, 1, replace=False)

    # Obtener los índices de entrenamiento, validación
    indices_train = np.where(~np.isin(indices, personas_val))[0]
    indices_val = np.where(np.isin(indices, personas_val))[0]


    # persona_val = 7
    # persona_test = 24
    # Obtener los índices de entrenamiento, validación y test
    # indices_train = np.where((indices != persona_val) & (indices != persona_test))[0]
    # indices_val = np.where(indices == persona_val)[0]
    # indices_test = np.where(indices == persona_test)[0]

    #Dataloaders
    train_dataloader = DataLoader(total_dataloader.dataset, batch_size=total_dataloader.batch_size,  sampler=torch.utils.data.SubsetRandomSampler(indices_train))
    
    val_dataloader = DataLoader(total_dataloader.dataset, batch_size=total_dataloader.batch_size, sampler=torch.utils.data.SubsetRandomSampler(indices_val))
    #test_dataloader = DataLoader(total_dataloader.dataset, batch_size=total_dataloader.batch_size, sampler=torch.utils.data.SubsetRandomSampler(indices_test))
    test_dataloader = DataLoader(dataset_test, batch_size=6000)
    

    # print("Persona de validación: ", persona_val, "Persona de test: ", persona_test)
    print("Personas de validación: ", personas_val)
    models, train_losses, val_losses, test_losses , euc_losses_val, indice = entrenar(modelo, train_dataloader, val_dataloader, test_dataloader=test_dataloader, epochs=250, lr=0.0002, graficas=True, ann = True, final = True)
    
    graficar_perdidas(train_losses, val_losses, test_losses=None, euc_losses = None, indice=indice)
    #graficar_perdidas(train_losses, val_losses, test_losses, euc_losses_val, indice)

    print(f"Mejor {indice}, Modelo a guardar?")
    i = int(input())
    torch.save(models[i], './entrenamiento/modelos/aprox1_finaal9.pt')
    graficar_perdidas(train_losses, val_losses, test_losses=None, euc_losses = None, indice=i)