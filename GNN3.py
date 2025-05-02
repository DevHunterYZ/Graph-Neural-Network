import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Örnek biyolojik sinir ağı grafiği (4 beyin bölgesi, 2 bağlantı)
edge_index = torch.tensor([[0, 1, 2, 3, 0, 1],  
                           [1, 0, 3, 2, 2, 3]], dtype=torch.long)

# Her beyin bölgesine ait özellikler (örneğin, nörolojik aktivite)
x = torch.tensor([[1.0, 0.5],   # Bölge 0
                  [0.5, 1.0],   # Bölge 1
                  [1.0, 0.2],   # Bölge 2
                  [0.8, 0.7]],  # Bölge 3
                  dtype=torch.float)

# Grafik veri nesnesi
data = Data(x=x, edge_index=edge_index)

# GCN modeli
class GNNModel(nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(2, 4)  # input dim: 2, output dim: 4
        self.conv2 = GCNConv(4, 2)  # input dim: 4, output dim: 2 (her bölge için tahmin)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Modeli oluştur ve veriye uygula
model = GNNModel()
out = model(data)

print("Beyin bölgelerinin çıkışları:\n", out)
