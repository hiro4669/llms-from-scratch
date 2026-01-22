import torch

ary = [2, 3, 5, 1]
input_ids = torch.tensor(ary)

print("input:", input_ids)
print(input_ids.shape)

vocab_size = 6
output_dim = 3

torch.manual_seed(123)

embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)

print("-------------------------")
print(embedding_layer(torch.tensor([3])))
print("-------------------------")
print(embedding_layer(input_ids))
