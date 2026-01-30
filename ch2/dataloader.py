import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


"""
tokenizer = tiktoken.get_encoding("gpt2")

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1 : context_size + 1]

print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)
    # print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

"""

""" for test
a = [1,2,3,4,5]
print(a[0:])
print(a[1:])
print(a[:5])

for i in range(0, len(a), 1):
    print(i)

for i in range(0, len(a) - 3, 2):
    v = a[i : i + 3]
    t = a[i+1 : i + 3 + 1]
    print(v, t)
    print(torch.tensor(v), torch.tensor(t))
"""  # end of for test


""" for test
print(enc_text[:20])
max_length = 5
stride = 2
for i in range(0, len(enc_text) - max_length, stride):
    input_chunk = enc_text[i : i + max_length]
    target_chunk = enc_text[i + 1 : i + max_length + 1]
    # print(i)
    print(input_chunk, target_chunk)
    print(torch.tensor(input_chunk).shape)
"""


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        # Tokenize the entire text
        # token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        # token_ids = tokenizer.encode(txt)
        # assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

        print("total input_ids len = ", len(self.input_ids))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


class GPTDatasetV2(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        # token_ids = tokenizer.encode(txt)
        assert (
            len(token_ids) > max_length
        ), "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    enc_text = tokenizer.encode(raw_text)

    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    print("---------- [ batch = 1 ]--------------------------")
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    print("\n>>>>>>>>> original tokens >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(enc_text[:50])
    print(len(enc_text))
    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    print("---------- [ batch = 8 ]--------------------------")

    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)
