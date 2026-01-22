import torch
import torch.nn.functional as F

"""
x = torch.rand(5, 3)
print(x)

tensor0d = torch.tensor(1)
tensor1d = torch.tensor([1,2,3])
tensor2d = torch.tensor([[1,2],
                                        [3,4]])

print(tensor2d)
"""

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2])
b = torch.tensor([0.0])

z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)
print(z)
print(loss)


# x = torch.tensor([2.0], requires_grad=True)  # 勾配を追跡
# x = torch.tensor([2.0])  # 勾配を追跡
# y = x**2 + 3 * x + 1  # y = x^2 + 3x + 1
# y.backward()  # yをxで微分

# print(x.grad)  # tensor([7.])
