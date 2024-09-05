from pandas import DataFrame, Series, concat, read_excel
from numpy import mean as np_mean, std as np_std
from DatasetEntero import DatasetEntero
from torch.utils.data import DataLoader
from entrenamiento import entrenar_con_kfold
from ANNs import ANNs

def aproximacion1(i, model, conjunto):  # Crear un dataframe

  df = DataFrame(columns=['Modelo', 'Mean EMC Val', 'Std EMC Val', 'Mean EUC Loss', 'Std EUC Loss'])

  #Crear un Dataset
  dataset = DatasetEntero("texto_solo")

  #Crear un DataLoader
  total_dataloader = DataLoader(dataset, batch_size=6000, num_workers=2, pin_memory=True)

  print("Empezando con el modelo: ", i, " del conjunto: ", conjunto)
  _, val_losses, euc_losses = entrenar_con_kfold(model, total_dataloader, epochs=250, lr=0.0002, ejecuciones_fold=5, ann=True, graficas=True)

  # Añadir los resultados al DataFrame
  linea = Series({'Modelo': f"{i}-{conjunto}", 'Mean EMC Val': np_mean(val_losses), 'Std EMC Val': np_std(val_losses), 'Mean EUC Loss': np_mean(euc_losses), 'Std EUC Loss': np_std(euc_losses)})
  df = concat([df, linea.to_frame().T])

  # Imprimirlos por pantalla
  print("Modelo: ", i, "Conjunto: ", conjunto,
      "\nMean EMC Val: ", np_mean(val_losses),
      "\nStd EMC Val: ", np_std(val_losses),
      "\nMean EUC Loss: ", np_mean(euc_losses),
      "\nStd EUC Loss: ", np_std(euc_losses))

  #Si no existe el archivo, lo crea
  path = './entrenamiento/resultados/Aproximacion1.xlsx'
  df_existente = read_excel(path)
  df = concat([df_existente, df])
  df.to_excel(path, index=False)



if __name__ == "__main__":
  modelos_inicial = {
      1 : [ANNs().crear_ann_1_1, ANNs().crear_ann_1_2, ANNs().crear_ann_1_3, ANNs().crear_ann_1_4, ANNs().crear_ann_1_5],
  }

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
      contador += 1
      aproximacion1(contador, model, conj)
