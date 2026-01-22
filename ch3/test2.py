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

dummy = torch.ones(6, 6) * 2
print(dummy)


mask = torch.triu(torch.ones(6, 6), diagonal=1)
print(mask)

print()
print(mask.bool())

dummy.masked_fill_(mask.bool()[:6, :6], -torch.inf)

print(">>>>>>>>>>>>>>>>>>>>>>>>>")
print(dummy)
