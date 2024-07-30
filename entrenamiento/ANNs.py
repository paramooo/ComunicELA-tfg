import torch.nn as nn

class ANNs:
    def __init__(self):
        pass

    def crear_ann(self, entradas, topology):
        model = nn.Sequential()
        model.add_module("dense_in", nn.Linear(entradas, topology[0]))  # Entrada
        model.add_module("relu_in", nn.ReLU())
        for i in range(len(topology)-1):  # Capas ocultas
            model.add_module("dense"+str(i+1), nn.Linear(topology[i], topology[i+1]))
            model.add_module("relu"+str(i+1), nn.ReLU())
        
        model.add_module("dense_out", nn.Linear(topology[-1], 2))  # Salida
        # Limita salida a rango 0-1
        model.add_module("sigmoid_out", nn.Sigmoid())

        return model
    
    
    # Primera sub-aproximacion con todos los datos como conjunto de entrenamiento PARA SABER QUE ARQUITECTURA ES LA MEJOR
    def crear_ann_1_1(self):
        return self.crear_ann(39, [50])
    
    def crear_ann_1_2(self):
        return self.crear_ann(39, [50, 100])
    
    def crear_ann_1_3(self):
        return self.crear_ann(39, [50, 80, 50])
    
    def crear_ann_1_4(self):
        return self.crear_ann(39, [50, 100, 100])
    
    # Las mejores han sido la 2 y la 3
    def crear_ann_1_5(self):
        return self.crear_ann(39, [20])
    
    def crear_ann_1_6(self):
        return self.crear_ann(39, [90])
    
    def crear_ann_1_7(self):
        return self.crear_ann(39, [20, 20])
    
    def crear_ann_1_8(self):
        return self.crear_ann(39, [90, 90])
    
    def crear_ann_1_9(self):
        return self.crear_ann(39, [20, 50, 20])
    
    def crear_ann_1_10(self):
        return self.crear_ann(39, [90, 50])
    
    def crear_ann_1_11(self):
        return self.crear_ann(39, [150])
    
    def crear_ann_1_12(self):
        return self.crear_ann(39, [300])
    
    def crear_ann_1_13(self):
        return self.crear_ann(39, [150, 150])
    

    # La mejor aproximacion es la del modelo 11
    # Ahora probamos con diferentes conjuntos de datos normalizados de cierta manera (limitando la orientacion de la cabeza, etc)
    def crear_ann_2_11(self):
        return self.crear_ann(23, [150])
    
    
    # Como los resultados han empeorado, mantenemos las caracteristicas del conjunto1 pero recortando a 0.3-0.7 los ultimos (por probar)
    def crear_ann_3_11(self):
        return self.crear_ann(39, [150])
    

    # Como la mejor es la 3_7 pues probamos ahora con el conjunto 4
    def crear_ann_4_11(self):
        return self.crear_ann(37, [150])
