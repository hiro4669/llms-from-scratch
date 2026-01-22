import torch




inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
           [0.55, 0.87, 0.66], # journey  (x^2)
           [0.57, 0.85, 0.64], # starts   (x^3)
           [0.22, 0.58, 0.33], # with     (x^4)
           [0.77, 0.25, 0.10], # one      (x^5)
           [0.05, 0.80, 0.55]] # step     (x^6)
)

# Journeyを対象にする
query = inputs[1]
print(query)

atten_score_2 = torch.empty(inputs.shape[0])

print(inputs.shape[0]) # 6
print(atten_score_2.shape)

for i, x_i in enumerate(inputs):
    atten_score_2[i] = torch.dot(x_i, query) # queryとのドット積(内積)を計算

print("attention score: ", atten_score_2)

# attensionを正規化し，確率分布にする
'''
atten_weight_2_tmp = atten_score_2 / atten_score_2.sum()
print("original sum:", atten_score_2.sum())
print("Attension weights:", atten_weight_2_tmp)
print("Sum:", atten_weight_2_tmp.sum())


def softmax_native(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_native = softmax_native(atten_score_2)
print("Attention weights:", attn_weights_2_native)
print("SUM:", attn_weights_2_native.sum())
'''

# attention scoreを確率分布にするため，softmaxを適用
print("-----------------------")
attn_weights2 = torch.softmax(atten_score_2, dim=0)
print("Attension weights:", attn_weights2)
print()

# Attension weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])

# コンテキストベクトルを求める
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
#print(query.shape) # 3
#print(context_vec_2) # tensor([0,0,0])

for i, x_i in enumerate(inputs):
    print(attn_weights2[i])
    print(x_i)
    context_vec_2 += attn_weights2[i] * x_i
    print("vec_2:", context_vec_2)

#こんな感じになる
'''
tensor(0.1385) # Token1 とToken1のアテンションを計算
tensor([0.4300, 0.1500, 0.8900])
vec_2: tensor([0.0596, 0.0208, 0.1233])
tensor(0.2379) # Token2 とToken2のアテンションを計算
tensor([0.5500, 0.8700, 0.6600])
vec_2: tensor([0.1904, 0.2277, 0.2803]) #Token1の結果に加算，を繰り返す
tensor(0.2333)
tensor([0.5700, 0.8500, 0.6400])
vec_2: tensor([0.3234, 0.4260, 0.4296])
tensor(0.1240)
tensor([0.2200, 0.5800, 0.3300])
vec_2: tensor([0.3507, 0.4979, 0.4705])
tensor(0.1082)
tensor([0.7700, 0.2500, 0.1000])
vec_2: tensor([0.4340, 0.5250, 0.4813])
tensor(0.1581)
tensor([0.0500, 0.8000, 0.5500])
vec_2: tensor([0.4419, 0.6515, 0.5683]) # queryの文脈での意味が集約されている
'''
