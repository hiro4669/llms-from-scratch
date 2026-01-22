import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # first layers
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


if __name__ == "__main__":
    model = NeuralNetwork(50, 3)
    print(model)

    print("---------------------------------------------------")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable model parameters: ", num_params)
    print("---------------------------------------------------")

    print(">>>>>>>>>   weight")
    print(model.layers[0].weight.shape)
    print(model.layers[0].weight)

    print("\n>>>>>>>>>   bias")
    print(model.layers[0].bias.shape)
    print(model.layers[0].bias)

    torch.manual_seed(123)
    model = NeuralNetwork(50, 3)
    print(model.layers[0].weight)

    torch.manual_seed(123)
    X = torch.rand(1, 50)
    out = model(X)
    print(out)
