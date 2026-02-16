import torch
import tiktoken
import sys
import os


sys.path.append(os.path.abspath("../ch2"))
sys.path.append(os.path.abspath("../ch4"))

from gptmodel import GPTModel
from dataloader import create_dataloader_v1
from utils import generate_text_simple
from utils import generate


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


# 5-4.pyで保存したモデルをロードして生成する

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPTModel(GPT_CONFIG_124M)
model.to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

## from here 5.3.3
print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4,
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
