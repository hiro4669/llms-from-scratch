import torch


print("-- headに分けずにアテンションを計算することを模擬する -- ")

x = torch.arange(6).view(3, 2)
print(x.shape)
print(x)

y = x.T

print("アテンション")
z = x @ y
print(z)

print("-- headに分けてアテンションを計算する--")

keys = x.view(3, 2, 1)
query = x.view(3, 2, 1)

print(keys.shape)
print(keys)

print(">>>>>>>>>>>>>>>")

keys = keys.transpose(0, 1)
query = query.transpose(0, 1)

print("keys:")
print(keys.shape)
print(keys)

print("query:")
print(query.shape)
print(query)

print("keys-----------")
print(keys.transpose(1, 2))


attn = query @ keys.transpose(1, 2)
print(attn.shape)
print(attn)
