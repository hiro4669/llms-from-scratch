import torch
import torch.nn as nn


torch.manual_seed(123)
tok = nn.Embedding(4, 3)

print("embed", tok.weight)
x1 = torch.tensor([1, 3])
x2 = torch.tensor([0, 2])

y1 = tok(x1)
y2 = tok(x2)
print(y1)
print(y2)

batch = []
batch.append(x1)
batch.append(x2)

print(batch)
batch = torch.stack(batch, dim=0)
print(batch)

y = tok(batch)
print(y)
