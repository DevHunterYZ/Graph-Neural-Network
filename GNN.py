import torch

# 4 düğüm, her biri 3 özellikli
x = torch.tensor([[1.0, 0.0, 1.0],   # Node 0
                  [0.0, 1.0, 0.0],   # Node 1
                  [1.0, 1.0, 0.0],   # Node 2
                  [0.0, 0.0, 1.0]])  # Node 3

# Kenar bağlantıları (source → target)
edge_index = torch.tensor([[0, 1, 2, 3, 0, 1],  # kaynak düğüm
                           [1, 0, 3, 2, 2, 3]]) # hedef düğüm

num_nodes = x.size(0)
feature_dim = x.size(1)

# 1. Aşama: Mesaj geçirme (source → target)
messages = torch.zeros_like(x)

for i in range(edge_index.size(1)):
    src = edge_index[0, i]
    tgt = edge_index[1, i]
    messages[tgt] += x[src]

# 2. Aşama: Agregasyon (ortalama alma)
# Her düğümün kaç komşusu var?
degree = torch.zeros(num_nodes)
for i in range(edge_index.size(1)):
    tgt = edge_index[1, i]
    degree[tgt] += 1

# Komşu ortalamalarını al.
for i in range(num_nodes):
    if degree[i] > 0:
        messages[i] /= degree[i]

# 3. Aşama: Güncelleme (basit bir lineer katman gibi davranalım)
W = torch.randn(feature_dim, feature_dim)
updated_features = torch.matmul(messages, W)

print("Yeni düğüm özellikleri:\n", updated_features)
