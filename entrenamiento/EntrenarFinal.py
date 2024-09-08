from entrenamiento import entrenar, graficar_perdidas
from ANNs import ANNs
import pandas as pd
import numpy as np
from DatasetEntero import DatasetEntero
from torch.utils.data import DataLoader
import torch

"""
Fichero que se encarga de entrenar el modelo final con el dataset entero

"""


if __name__ == '__main__':
    
    modelo = ANNs().crear_ann_1_9()

    # Crear un dataframe
    df = pd.DataFrame(columns=['Modelo', 'Mean EMC Val', 'Std EMC Val', 'Mean EUC Loss', 'Std EUC Loss'])

    #Crear los Dataset
    dataset = DatasetEntero("unidos")

    #Dividir el de train en train y val
    total_dataloader = DataLoader(dataset, batch_size=6000, num_workers=2)
    indices = dataset.get_indices_persona()
    unique_persons = np.unique(indices)
    personas_val = np.random.choice(unique_persons, 2, replace=False)

    # Obtener los índices de entrenamiento, validación
    indices_train = np.where(~np.isin(indices, personas_val))[0]
    indices_val = np.where(np.isin(indices, personas_val))[0]

    #Dataloaders
    train_dataloader = DataLoader(total_dataloader.dataset, batch_size=total_dataloader.batch_size, sampler=torch.utils.data.SubsetRandomSampler(indices_train))
    val_dataloader = DataLoader(total_dataloader.dataset, batch_size=total_dataloader.batch_size, sampler=torch.utils.data.SubsetRandomSampler(indices_val))

    models, train_losses, val_losses, test_losses , euc_losses_val, indice = entrenar(modelo, train_dataloader, val_dataloader, test_dataloader=None, epochs=250, lr=0.0002, graficas=True, ann = True, final = True)
    
    graficar_perdidas(train_losses, val_losses, test_losses=None, euc_losses = None, indice=indice)

    print(f"Mejor {indice}, Modelo a guardar?")
    i = int(input())
    torch.save(models[i], './entrenamiento/modelos/aprox1_9Final.pt')
    graficar_perdidas(train_losses, val_losses, test_losses=None, euc_losses = None, indice=i)