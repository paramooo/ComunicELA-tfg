from entrenamiento import entrenar_con_kfold
from pandas import DataFrame, Series, concat, read_excel
from numpy import mean as np_mean, std as np_std
from DatasetEntero import DatasetEntero
from torch.utils.data import DataLoader
from FusionNet import FusionNet1, FusionNet2, FusionNet3, FusionNet4, FusionNet5, FusionNet6, FusionNet7, FusionNet8, FusionNet9, FusionNet10



def aproximacion3(i, model, dataset):
    
    # Crear un dataframe
    df = DataFrame(columns=['Modelo', 'Mean EMC Val', 'Std EMC Val', 'Mean EUC Loss', 'Std EUC Loss'])
    total_dataloader = DataLoader(dataset, batch_size=250, num_workers=2, pin_memory=True)

    print("Empezando con el modelo: ", i)
    _, val_losses, euc_losses = entrenar_con_kfold(model, total_dataloader, epochs=300, lr=0.0001, ejecuciones_fold=5, graficas=False, ann=False)

    # Añadir los resultados al DataFrame
    linea = Series({'Modelo': i, 'Mean EMC Val': np_mean(val_losses), 'Std EMC Val': np_std(val_losses), 'Mean EUC Loss': np_mean(euc_losses), 'Std EUC Loss': np_std(euc_losses)})
    df = concat([df, linea.to_frame().T])

    # Imprimirlos por pantalla
    print("Modelo FusionNet: ", i,
        "\nMean EMC Val: ", np_mean(val_losses),
        "\nStd EMC Val: ", np_std(val_losses),
        "\nMean EUC Loss: ", np_mean(euc_losses),
        "\nStd EUC Loss: ", np_std(euc_losses))
    
    # Guardar los resultados en un archivo Excel
    path = './entrenamiento/resultados/Aproximacion3.xlsx'
    df_existente = read_excel(path)
    df = concat([df_existente, df])
    df.to_excel(path, index=False)


if __name__ == "__main__":
  models = [FusionNet1().crear(), FusionNet2().crear(), FusionNet3().crear(), FusionNet4().crear(),
           FusionNet5().crear(), FusionNet6().crear(), FusionNet7().crear(), FusionNet8().crear(),
           FusionNet9().crear(), FusionNet10().crear()]

  contador = 0
  dataset = DatasetEntero("con_imagenes", img_dir='./entrenamiento/datos/frames/byn/15-15-15')
  for model in models:
    contador += 1
    aproximacion3(contador, model, dataset)