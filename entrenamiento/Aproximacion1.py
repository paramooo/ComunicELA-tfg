import pandas as pd
import numpy as np
from DatasetEntero import DatasetEntero
from torch.utils.data import DataLoader
from entrenamiento import entrenar_con_kfold
from ANNs import ANNs

def aproximacion1(i, model, conjunto):  # Crear un dataframe

  df = pd.DataFrame(columns=['Modelo', 'Mean EMC Val', 'Std EMC Val', 'Mean EUC Loss', 'Std EUC Loss'])

  #Crear un Dataset
  dataset = DatasetEntero('./entrenamiento/datos/txts/input.txt', './entrenamiento/datos/txts/output.txt', 21, conjunto=conjunto)

  #Crear un DataLoader
  total_dataloader = DataLoader(dataset, batch_size=10000, num_workers=2, pin_memory=True)

  print("Empezando con el modelo: ", i, " del conjunto: ", conjunto)
  _, val_losses, euc_losses = entrenar_con_kfold(model, total_dataloader, epochs=300, lr=0.01, ejecuciones_fold=5, ann=True, graficas=False)

  # Añadir los resultados al DataFrame
  linea = pd.Series({'Modelo': f"{i}-{conjunto}", 'Mean EMC Val': np.mean(val_losses), 'Std EMC Val': np.std(val_losses), 'Mean EUC Loss': np.mean(euc_losses), 'Std EUC Loss': np.std(euc_losses)})
  df = pd.concat([df, linea.to_frame().T])

  # Imprimirlos por pantalla
  print("Modelo: ", i, "Conjunto: ", conjunto,
      "\nMean EMC Val: ", np.mean(val_losses),
      "\nStd EMC Val: ", np.std(val_losses),
      "\nMean EUC Loss: ", np.mean(euc_losses),
      "\nStd EUC Loss: ", np.std(euc_losses))

  #Si no existe el archivo, lo crea
  path = './entrenamiento/resultados/Aproximacion1.xlsx'
  df_existente = pd.read_excel(path)
  df = pd.concat([df_existente, df])
  df.to_excel(path, index=False)



if __name__ == "__main__":
  modelos = {
      1 : [ANNs().crear_ann_1_1, ANNs().crear_ann_1_2, ANNs().crear_ann_1_3, ANNs().crear_ann_1_4, ANNs().crear_ann_1_5, ANNs().crear_ann_1_6, ANNs().crear_ann_1_7, ANNs().crear_ann_1_8, ANNs().crear_ann_1_9, ANNs().crear_ann_1_10],
      2 : [ANNs().crear_ann_2_9],
      3 : [ANNs().crear_ann_3_9],
      4 : [ANNs().crear_ann_4_9],
      1 : [ANNs().crear_ann_9_sinsigmoid],
  }

  contador = 0
  for (conj, modelos) in modelos.items():
    for model in modelos:
      aproximacion1(contador, model, conj)
