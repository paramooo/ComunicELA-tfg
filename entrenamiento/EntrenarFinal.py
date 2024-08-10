from entrenamiento import entrenar, graficar_perdidas
from ANNs import ANNs
import pandas as pd
import numpy as np
from DatasetEntero import DatasetEntero
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    
    modelo = ANNs().crear_ann_1_9()

    # Crear un dataframe
    df = pd.DataFrame(columns=['Modelo', 'Mean EMC Val', 'Std EMC Val', 'Mean EUC Loss', 'Std EUC Loss'])

    #Crear los Dataset
    dataset_train = DatasetEntero('./entrenamiento/datos/frames/byn/15-15-15', './entrenamiento/datos/txts/input.txt', './entrenamiento/datos/txts/output.txt', 21, conjunto=1, imagenes=False)
    dataset_test = DatasetEntero('./entrenamiento/datos/frames/byn/15-15-15', './entrenamiento/datos/txts/texto_solo/input.txt', './entrenamiento/datos/txts/texto_solo/output.txt', 21, conjunto=1, imagenes=False)

    #Dividir el de train en train y val
    total_dataloader = DataLoader(dataset_train, batch_size=10000, num_workers=2)

    #Indices
    indices = dataset_train.get_indices_persona()
    unique_persons = np.unique(indices)
    indices_train = np.where(indices != unique_persons[1])[0]
    # indces_test = np.where(indices == unique_persons[-2])[0]    
    indices_val = np.where(indices == unique_persons[1])[0]
    print("Indices val:", min(indices_val), max(indices_val))

    #Dataloaders
    train_dataloader = DataLoader(total_dataloader.dataset, batch_size=total_dataloader.batch_size,  sampler=torch.utils.data.SubsetRandomSampler(indices_train))
    val_dataloader = DataLoader(total_dataloader.dataset, batch_size=total_dataloader.batch_size, sampler=torch.utils.data.SubsetRandomSampler(indices_val))
    # test_dataloader = DataLoader(total_dataloader.dataset, batch_size=total_dataloader.batch_size, sampler=torch.utils.data.SubsetRandomSampler(indces_test))
    test_dataloader = DataLoader(dataset_test, batch_size=10000)

    
    print("Empezando con el modelo: ", 1, " del conjunto: ", 1)
    models, train_losses, val_losses, test_losses , euc_losses, indice = entrenar(modelo, train_dataloader, val_dataloader, test_dataloader, epochs=300, lr=0.001, graficas=True, ann = True, final = True)
    
    graficar_perdidas(train_losses, val_losses, test_losses, euc_losses, indice)

    print("Modelo a guardar:")
    i = int(input())
    torch.save(models[i], './entrenamiento/modelos/aprox1_9.pt')