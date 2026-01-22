import torch


"""
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
out = torch.stack([a, b], dim=0)
print(out)
print(out.shape)

out2 = torch.stack([out, out], dim=0)
print(out2)
print(out2.shape)

print(">>>>>>>>>>>>>>>>>>>")
out3 = torch.stack([out, out], dim=2)
print(out3)
print(out3.shape)
"""

A = torch.tensor([[1, 2], [11, 22]])
B = torch.tensor([[3, 4], [33, 44]])

C = torch.stack([A, B], dim=0)
print(A.shape)
print(C.shape)
print(C)

D = torch.stack([A, B], dim=1)
print(D.shape)
print(D)

E = torch.stack([A, B], dim=2)
print(E.shape)
print(E)
