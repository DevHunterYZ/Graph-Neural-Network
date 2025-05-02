!pip install torch-geometric
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Örnek grafik verisi (4 düğüm, kenarlarla bağlanmış)
edge_index = torch.tensor([[0, 1, 2, 3, 0, 1],
                           [1, 0, 3, 2, 2, 3]], dtype=torch.long)

# Her düğüm için 3 özellikten oluşan vektör
x = torch.tensor([[1, 0, 1],
                  [0, 1, 0],
                  [1, 1, 0],
                  [0, 0, 1]], dtype=torch.float)

# Grafik datası oluştur
data = Data(x=x, edge_index=edge_index)

# Basit bir GCN modeli
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 4)  # input dim: 3, hidden dim: 4
        self.conv2 = GCNConv(4, 2)  # hidden dim: 4, output dim: 2

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Modeli çalıştır
model = GCN()
out = model(data)
print("Çıktılar:\n", out)
