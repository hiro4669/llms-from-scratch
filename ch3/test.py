import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],
    ]  # step     (x^6)
)

x_2 = inputs[1]
d_in = inputs.shape[1]  # 3
d_out = 2

print(x_2)
# 内積
print(x_2 @ x_2)
print(x_2.shape)


torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print("W_query shape: {}\n {}".format(W_query.data.shape, W_query))

query_2 = x_2 @ W_query  # 1x3 @ 3x2 = 1x2
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print("query_2 shape: {}\n {}".format(query_2.data.shape, query_2))

print()
keys = inputs @ W_key  # 6x3 @ 3x2 = 6x2
values = inputs @ W_value  # 6x3 @ 3x2 = 6x2
print("keys {}".format(keys))

keys_2 = keys[1]

print()
print("query_2 = {}".format(query_2))
print("key_s   = {}".format(keys_2))
atten_score_22 = query_2.dot(keys_2)
print("atten_score22")
print(atten_score_22)

# query2とkeysのドット積(内積を計算している)
attn_score_2 = query_2 @ keys.T  # 1x2 @ 2x6 = 1x6
print(attn_score_2)

d_k = keys.shape[-1]  # 2

attn_weights_2 = torch.softmax(attn_score_2 / d_k**0.5, dim=-1)
print(attn_weights_2)  # 1x6

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
