import torch.nn as nn

class ANNs:
    def __init__(self):
        pass

    def crear_ann(self, entradas, topology, salidas=2):
        model = nn.Sequential()
        model.add_module("dense_in", nn.Linear(entradas, topology[0]))  # Entrada
        model.add_module("relu_in", nn.ReLU())
        for i in range(len(topology)-1):  # Capas ocultas
            model.add_module("dense"+str(i+1), nn.Linear(topology[i], topology[i+1]))
            model.add_module("relu"+str(i+1), nn.ReLU())

        model.add_module("dense_out", nn.Linear(topology[-1], salidas))  # Salida
        # Limita salida a rango 0-1
        model.add_module("sigmoid_out", nn.Sigmoid())

        return model


    # Primera sub-aproximacion con todos los datos como conjunto de entrenamiento PARA SABER QUE ARQUITECTURA ES LA MEJOR
    def crear_ann_1_1(self):
        return self.crear_ann(39, [50])

    def crear_ann_1_2(self):
        return self.crear_ann(39, [90])

    def crear_ann_1_3(self):
        return self.crear_ann(39, [50, 100])

    def crear_ann_1_4(self):
        return self.crear_ann(39, [300, 400, 500, 400])

    def crear_ann_1_5(self):
        return self.crear_ann(39, [100, 300, 400, 500, 100])

    # Al ser el mejor el modelo 3, probamos estructuras semejantes

    def crear_ann_1_6(self):
        return self.crear_ann(39, [70, 100])

    def crear_ann_1_7(self):
        return self.crear_ann(39, [100, 200])

    def crear_ann_1_8(self):
        return self.crear_ann(39, [100, 100, 100])

    def crear_ann_1_9(self):
        return self.crear_ann(39, [80, 100, 80])

    def crear_ann_1_10(self):
        return self.crear_ann(39, [150])




    # La mejor aproximacion es la del modelo 9 ya que ha sido el mejor de todos
    def crear_ann_2_9(self):
        return self.crear_ann(23, [80, 100, 80])


    # Como los resultados han empeorado, mantenemos las caracteristicas del conjunto1 pero recortando a 0.3-0.7 los ultimos (por probar)
    def crear_ann_3_9(self):
        return self.crear_ann(39, [80, 100, 80])


    # Como la mejor es la 3_7 pues probamos ahora con el conjunto 4
    def crear_ann_4_9(self):
        return self.crear_ann(37, [80, 100, 80])
    
    def crear_ann_9_sinsigmoid(self):
        model = nn.Sequential()
        model.add_module("dense_in", nn.Linear(39, 80))
        model.add_module("relu_in", nn.ReLU())
        model.add_module("dense1", nn.Linear(80, 100))
        model.add_module("relu1", nn.ReLU())
        model.add_module("dense_out", nn.Linear(100, 2))
        return model





    #ANNS PARA LA FUSION NET
    def crear_ann_f_1(self):
            return self.crear_ann(39, [80, 100], salidas=80)
    