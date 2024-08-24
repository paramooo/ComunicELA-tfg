import torch
import torch.nn as nn
from ANNs import ANNs
from CNNs import CNNs

class FusionNet(nn.Module):
    def __init__(self, ann, cnn):
        super(FusionNet, self).__init__()
        self.ann = ann
        self.cnn = cnn
        self.fusion_layer = None

    def forward(self, x_ann, x_cnn):
        out_ann = self.ann(x_ann)
        out_cnn = self.cnn(x_cnn)
        # Concatena las salidas de las dos redes
        fusion = torch.cat((out_ann, out_cnn), dim=1)
        # Pasa la concatenación a través de la capa de fusión
        out = self.fusion_layer(fusion)
        return out

    def crear(self):
        return self

class FusionNet1(FusionNet):
    def __init__(self):
        super(FusionNet1, self).__init__(ANNs().crear_ann_f_1(), CNNs().crear_cnn_f_1())
        self.fusion_layer = nn.Sequential(
            nn.Linear(160, 80),
            nn.ReLU(),
            nn.Linear(80, 2),
            nn.Sigmoid()
        )

class FusionNet2(FusionNet):
    def __init__(self):
        super(FusionNet2, self).__init__(ANNs().crear_ann_f_1(), CNNs().crear_cnn_f_1())
        self.fusion_layer = nn.Sequential(
            nn.Linear(160, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Sigmoid()
       )

class FusionNet3(FusionNet):
    def __init__(self):
        super(FusionNet3, self).__init__(ANNs().crear_ann_f_1(), CNNs().crear_cnn_f_2())
        self.fusion_layer = nn.Sequential(
            nn.Linear(130, 80),
            nn.ReLU(),
            nn.Linear(80, 2),
            nn.Sigmoid()
       )

class FusionNet4(FusionNet):
    def __init__(self):
        super(FusionNet4, self).__init__(ANNs().crear_ann_f_2(), CNNs().crear_cnn_f_3())
        self.fusion_layer = nn.Sequential(
            nn.Linear(40, 2),
            nn.Sigmoid()
       )
        
class FusionNet5(FusionNet):
    def __init__(self):
        super(FusionNet5, self).__init__(ANNs().crear_ann_f_2(), CNNs().crear_cnn_f_3())
        self.fusion_layer = nn.Sequential(
            nn.Linear(40, 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Linear(30, 2),
            nn.Sigmoid()
       )
        
class FusionNet6(FusionNet):
    def __init__(self):
        super(FusionNet6, self).__init__(ANNs().crear_ann_f_2(), CNNs().crear_cnn_f_3())
        self.fusion_layer = nn.Sequential(
            nn.Linear(40, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
            nn.Sigmoid()
       )

        
class FusionNet7(FusionNet):
    def __init__(self):
        super(FusionNet7, self).__init__(ANNs().crear_ann_f_2(), CNNs().crear_cnn_f_3())
        self.fusion_layer = nn.Sequential(
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Linear(30, 2),
            nn.Sigmoid()
       )
        
        
class FusionNet8(FusionNet):
    def __init__(self):
        super(FusionNet8, self).__init__(ANNs().crear_ann_f_2(), CNNs().crear_cnn_f_3())
        self.fusion_layer = nn.Sequential(
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 2),
            nn.Sigmoid()
       )
        
class FusionNet9(FusionNet):
    def __init__(self):
        super(FusionNet9, self).__init__(ANNs().crear_ann_f_2(), CNNs().crear_cnn_f_3())
        self.fusion_layer = nn.Sequential(
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.Sigmoid()
       )
        
class FusionNet10(FusionNet):
    def __init__(self):
        super(FusionNet10, self).__init__(ANNs().crear_ann_f_2(), CNNs().crear_cnn_f_3())
        self.fusion_layer = nn.Sequential(
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
            nn.Sigmoid()
       )
        




