from entrenamiento import entrenar_con_kfold
import pandas as pd
import numpy as np
from DatasetEntero import DatasetEntero
from torch.utils.data import DataLoader
from FusionNet import FusionNet1, FusionNet2, FusionNet3



def aproximacion4(i, model, dataset):
    # Crear un dataframe
    df = pd.DataFrame(columns=['Modelo', 'Mean EMC Val', 'Std EMC Val', 'Mean EUC Loss', 'Std EUC Loss'])

    total_dataloader = DataLoader(dataset, batch_size=300, num_workers=2, pin_memory=True)

    print("Empezando con el modelo: ", i)

    _, val_losses, euc_losses = entrenar_con_kfold(model, total_dataloader, epochs=300, lr=0.001, ejecuciones_fold=5, graficas=False, ann=False)

    # Añadir los resultados al DataFrame
    linea = pd.Series({'Modelo': i, 'Mean EMC Val': np.mean(val_losses), 'Std EMC Val': np.std(val_losses), 'Mean EUC Loss': np.mean(euc_losses), 'Std EUC Loss': np.std(euc_losses)})
    df = pd.concat([df, linea.to_frame().T])

    # Imprimirlos por pantalla
    print("Modelo FusionNet: ", i,
        "\nMean EMC Val: ", np.mean(val_losses),
        "\nStd EMC Val: ", np.std(val_losses),
        "\nMean EUC Loss: ", np.mean(euc_losses),
        "\nStd EUC Loss: ", np.std(euc_losses))
    
    path = './entrenamiento/resultados/Aproximacion4.xlsx'
    df_existente = pd.read_excel(path)
    df = pd.concat([df_existente, df])
    df.to_excel(path, index=False)



if __name__ == "__main__":
  models = [FusionNet1().crear, FusionNet2().crear, FusionNet3().crear]

  dataset = DatasetEntero('./entrenamiento/datos/frames/byn/15-15-15', './entrenamiento/datos/txts/input.txt', './entrenamiento/datos/txts/output.txt', imagenes=True, )

  contador = 0
  for model in models:
    contador += 1
    aproximacion4(contador, model, dataset)