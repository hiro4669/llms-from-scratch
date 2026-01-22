from dataloader import create_dataloader_v1
import torch

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# print(raw_text)

vocab_size = 50257
output_dim = 256

# 50257 x 256の行列を作る．Tokenの埋め込み表現
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)

data_iter = iter(dataloader)
inputs, target = next(data_iter)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("Input Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

print("Target Token IDs:\n", target)
print("\nTarget shape:\n", target.shape)

# print("\n")
# print("Target Token IDs:\n", target)

print("\n")
print("Embedding layer >> ", token_embedding_layer.weight.shape)
# print(token_embedding_layer.weight)

# dataloaderから出力されたトークンの埋め込みを取得．
# token_embedding_layerの行はトークンIDのインデックス，列が埋め込み表現
token_embeddings = token_embedding_layer(inputs)
print("--token_embeddings.shape--\n", token_embeddings.shape)
print()

context_length = max_length
# トークンの位置情報の埋め込み表現
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# [0,1,2,3]のベクトルを作る
place = torch.arange(context_length)
# print(place)
# 位置情報の埋め込み表現を作る
pos_embeddings = pos_embedding_layer(place)
print("--pos_embedding.shape--")
print(pos_embeddings.shape)

# トークンの埋め込み表現と，位置情報の埋め込み表現を加算する
# 行列の形が違う(8x4x256と4x256) が，pytorchのブロードキャストという仕組みで
# 計算できる
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
