def aproximacion1(i, model, conjunto):  # Crear un dataframe

  df = pd.DataFrame(columns=['Modelo', 'Mean EMC Val', 'Std EMC Val', 'Mean EUC Loss', 'Std EUC Loss'])

  #Crear un Dataset
  dataset = DatasetEntero('/content/drive/MyDrive/DatosComunicELA/datos/datos/frames/byn/byn/15-15-15', '/content/drive/MyDrive/DatosComunicELA/datos/datos/txts/input.txt', '/content/drive/MyDrive/DatosComunicELA/datos/datos/txts/output.txt', 21, conjunto=conjunto, imagenes=False)

  #Crear un DataLoader
  total_dataloader = DataLoader(dataset, batch_size=40000, num_workers=2, pin_memory=True)

  print("Empezando con el modelo: ", i, " del conjunto: ", conjunto)
  _, val_losses, euc_losses = entrenar_con_kfold(model, total_dataloader, epochs=800, lr=0.01, ejecuciones_fold=10, ann=True, graficas=True)

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
  df_existente = pd.read_excel('/content/drive/MyDrive/DatosComunicELA/resultados/Aproximacion1.xlsx')
  df = pd.concat([df_existente, df])
  df.to_excel('/content/drive/MyDrive/DatosComunicELA/resultados/Aproximacion1.xlsx', index=False)


modelos = {
    1 : [ANNs().crear_ann_1_1]
}

contador = 0
for (conj, modelos) in modelos.items():
  for model in modelos:
    contador += 1
    aproximacion1(contador, model, conj)
