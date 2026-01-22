
import torch
import torch.nn as nn

torch.manual_seed(0)

# 簡単のため
batch, seq_len, d_model = 1, 3, 4
num_heads = 2
d_k = d_model // num_heads  # 2

# 入力 (batch, seq_len, d_model)
x = torch.randn(batch, seq_len, d_model)

# W_q, W_k, W_v をランダム初期化
W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)

# ===== Single-head =====
Q = W_q(x)  # (1, 3, 4)
K = W_k(x)  # (1, 3, 4)
V = W_v(x)  # (1, 3, 4)

attn_single = torch.softmax(Q @ K.transpose(-2, -1) / (d_model**0.5), dim=-1)
context_single = attn_single @ V

print("Single-head attention matrix:\n", attn_single[0])
print("Single-head context vector:\n", context_single[0])

# ===== Multi-head =====
Qh = W_q(x).view(batch, seq_len, num_heads, d_k).transpose(1, 2)  # (1, 2, 3, 2)
Kh = W_k(x).view(batch, seq_len, num_heads, d_k).transpose(1, 2)
Vh = W_v(x).view(batch, seq_len, num_heads, d_k).transpose(1, 2)

attn_multi = torch.softmax(Qh @ Kh.transpose(-2, -1) / (d_k**0.5), dim=-1)  # 各ヘッドの attention
context_multi = attn_multi @ Vh  # (1, 2, 3, 2)

# ヘッドごとの結果を表示
for h in range(num_heads):
    print(f"\nMulti-head attention matrix (head {h}):\n", attn_multi[0, h])
    print(f"Multi-head context vector (head {h}):\n", context_multi[0, h])

# 最後にヘッドを結合
context_concat = context_multi.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
print("\nMulti-head concatenated context vector:\n", context_concat[0])
