from CNNs import CNNs
from DatasetEntero import DatasetEntero
from torch.utils.data import DataLoader
from entrenamiento import entrenar_con_kfold
import pandas as pd
import numpy as np


def aproximacion2(i, model, carpeta, dataset):
    # Crear un dataframe
    df = pd.DataFrame(columns=['Modelo', 'Mean EMC Val', 'Std EMC Val', 'Mean EUC Loss', 'Std EUC Loss'])

    #Crear un DataLoader
    total_dataloader = DataLoader(dataset, batch_size=100, num_workers=2, pin_memory=True)

    print("Empezando con el modelo: ", i, " de la carpeta: ", carpeta)

    _, val_losses, euc_losses = entrenar_con_kfold(model, total_dataloader, epochs=300, lr=0.002, ejecuciones_fold=5, graficas=False)

    # Añadir los resultados al DataFrame
    linea = pd.Series({'Modelo': f"{i}-{carpeta}", 'Mean EMC Val': np.mean(val_losses), 'Std EMC Val': np.std(val_losses), 'Mean EUC Loss': np.mean(euc_losses), 'Std EUC Loss': np.std(euc_losses)})
    df = pd.concat([df, linea.to_frame().T])

    # Imprimirlos por pantalla
    print("Modelo: ", i, "Carpeta: ", carpeta,
        "\nMean EMC Val: ", np.mean(val_losses),
        "\nStd EMC Val: ", np.std(val_losses),
        "\nMean EUC Loss: ", np.mean(euc_losses),
        "\nStd EUC Loss: ", np.std(euc_losses))

    df_existente = pd.read_excel('./entrenamiento/resultados/Aproximacion2.xlsx')
    df = pd.concat([df_existente, df])
    df.to_excel('./entrenamiento/resultados/Aproximacion2.xlsx', index=False)


#Modelo
models = {
          #"20-35-55":[CNNs().crear_cnn_2_1, CNNs().crear_cnn_2, CNNs().crear_cnn_3, CNNs().crear_cnn_4],
          "15-15-15":[CNNs().crear_cnn_3_4],
          "15-35-15":[CNNs().crear_cnn_4_3, CNNs().crear_cnn_4_4]
          }
            

datasets = {
            "15-15-15":DatasetEntero('./entrenamiento/datos/frames/byn/15-15-15', './entrenamiento/datos/txts/input.txt', './entrenamiento/datos/txts/output.txt', 21, conjunto=1, imagenes=True),
            "15-35-15":DatasetEntero('./entrenamiento/datos/frames/byn/15-35-15', './entrenamiento/datos/txts/input.txt', './entrenamiento/datos/txts/output.txt', 21, conjunto=1, imagenes=True)}
            #"20-35-55":DatasetEntero('./entrenamiento/datos/frames/byn/20-35-55', './entrenamiento/datos/txts/input.txt', './entrenamiento/datos/txts/output.txt', 21, conjunto=1, imagenes=True)}




if __name__ == '__main__':
    contador = 2
    for (carpeta, modelos) in models.items():
        for model in modelos:
            contador += 1
            aproximacion2(contador, model, carpeta, datasets[carpeta])

