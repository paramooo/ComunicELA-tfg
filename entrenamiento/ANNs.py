import torch.nn as nn

class ANNs:
    """
    Modelos de redes neuronales artificiales (ANNs/RNAs)
    
    """

    def __init__(self):
        pass

    def crear_ann(self, entradas, topology, salidas=2):
        """
        Función para crear un modelo de red neuronal artificial (ANN) con una topología dada.

        Args:
            entradas (int): Número de entradas de la red.
            topology (list): Lista con la cantidad de neuronas en cada capa oculta.
            salidas (int): Número de salidas de la red.

        Returns:
            nn.Sequential: Modelo de red neuronal artificial.
        """
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


    # Modelos aproximación inicial
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

    
    
    
    # Modelos aproximación 1
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




    # Pe-procesado A
    def crear_ann_2_9(self):
        return self.crear_ann(23, [80, 100, 80])


    # Pre-procesado B
    def crear_ann_3_9(self):
        return self.crear_ann(39, [80, 100, 80])


    # Pre-procesado C
    def crear_ann_4_9(self):
        return self.crear_ann(37, [80, 100, 80])
    
    # Probar si empeora los resultados la sigmoidal, pero no
    def crear_ann_9_sinsigmoid(self):
        model = nn.Sequential()
        model.add_module("dense_in", nn.Linear(39, 80))
        model.add_module("relu_in", nn.ReLU())
        model.add_module("dense1", nn.Linear(80, 100))
        model.add_module("relu1", nn.ReLU())
        model.add_module("dense_out", nn.Linear(100, 2))
        return model





    #Modelos para la aproximación 3 (hibridas)
    def crear_ann_f_1(self):
            return self.crear_ann(39, [80, 100], salidas=80)
    
    def crear_ann_f_2(self):
            return self.crear_ann(39, [80, 100], salidas=25)
    
    def crear_ann_f_3(self):
            return self.crear_ann(39, [80, 100], salidas=5)
