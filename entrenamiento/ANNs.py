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
    
    # Las anns para el conjunto 1 ( los datos en crudo)
    def crear_ann_1_1(self):
        return self.crear_ann(39, [50, 80, 20])
    
    # La ann para el conjunto2
    def crear_ann_2_1(self):
        return self.crear_ann(23, [100, 50, 10])
