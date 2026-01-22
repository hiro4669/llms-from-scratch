import torch
import torch.nn as nn
import tiktoken

from gptmodel import GPTModel


def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        print("logits.shape1: ", logits.shape)
        logits = logits[:, -1, :]
        print("logits.shape2: ", logits.shape)
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


if __name__ == "__main__":
    print("4-8")
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)

    model = GPTModel(GPT_CONFIG_124M)

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    # encoded_tensor = torch.tensor(encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)
    model.eval()

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)
