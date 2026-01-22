import torch
import torch.nn as nn

torch.manual_seed(0)

# サンプル入力: batch=1, seq_len=3, d_model=4
x = torch.tensor([[ 
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 3.0, 4.0, 5.0],
    [3.0, 4.0, 5.0, 6.0]
]])
batch, seq_len, d_model = x.shape

# --- Single-head ---
W_q_single = nn.Linear(d_model, d_model, bias=False)
W_k_single = nn.Linear(d_model, d_model, bias=False)

Q_single = W_q_single(x)
K_single = W_k_single(x)

# QK^T
attn_single = Q_single @ K_single.transpose(1, 2)
print("Single-head QK^T:\n", attn_single)

# --- Multi-head ---
num_heads = 2
head_dim = d_model // num_heads

W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)

Q = W_q(x)
K = W_k(x)

# reshape to (batch, seq_len, num_heads, head_dim)
Q = Q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
K = K.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

# QK^T per head
attn_multi = torch.matmul(Q, K.transpose(-2, -1))
print("\nMulti-head QK^T (per head):")
for h in range(num_heads):
    print(f"Head {h}:\n", attn_multi[0,h])
