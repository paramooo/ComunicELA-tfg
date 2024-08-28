from CNNs import CNNs
from DatasetEntero import DatasetEntero
from torch.utils.data import DataLoader
from entrenamiento import entrenar_con_kfold
import pandas as pd
import numpy as np


def aproximacion2(i, model, lr, carpeta = "15-15-15"):
    # Crear un dataframe
    df = pd.DataFrame(columns=['Modelo', 'Mean EMC Val', 'Std EMC Val', 'Mean EUC Loss', 'Std EUC Loss'])

    dataset = DatasetEntero("con_imagenes", img_dir=f'./entrenamiento/datos/frames/byn/{carpeta}')

    total_dataloader = DataLoader(dataset, batch_size=100, num_workers=2, pin_memory=True)

    print("Empezando con el modelo: ", i, " de la carpeta: ", carpeta)

    _, val_losses, euc_losses = entrenar_con_kfold(model, total_dataloader, epochs=250, lr=lr, ejecuciones_fold=5, graficas=True, ann=False)

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

if __name__ == "__main__":
    #Modelo
    models_t = [{
            "20-35-55":[CNNs().crear_cnn_2_1, CNNs().crear_cnn_2, CNNs().crear_cnn_3, CNNs().crear_cnn_4],
            "15-15-15":[CNNs().crear_cnn_3_3, CNNs().crear_cnn_3_4],
            "15-35-15":[CNNs().crear_cnn_4_3, CNNs().crear_cnn_4_4],
            }
        ,
        {
            "15-15-15":[CNNs().crear_resnet18(), CNNs().crear_resnet34],
            }
    ]


    if __name__ == '__main__':
        contador = 0
        for i, models in enumerate(models_t):
            for (carpeta, modelos) in models.items():
                for model in modelos:
                    contador += 1
                    if i == 0:
                        aproximacion2(contador, model, 0.002, carpeta)
                    else:
                        aproximacion2(contador, model, 0.0005, carpeta)
