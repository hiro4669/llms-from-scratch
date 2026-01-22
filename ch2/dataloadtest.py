import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from nntest import NeuralNetwork
import torch.nn.functional as F


X_train = torch.tensor(
    [[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]]
)

y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor(
    [
        [-0.8, 2.8],
        [2.6, -1.6],
    ]
)

y_test = torch.tensor([0, 1])


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

print(len(train_ds))


torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True
)

test_loader = DataLoader(dataset=test_ds, batch_size=2, shuffle=False, num_workers=0)

print(len(train_loader))

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}", x, y)


print("-------------------------------------")
torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for name, param in model.named_parameters():
    print(name, param.shape)

print("Total parameters:", sum(p.numel() for p in model.parameters()))

num_epochs = 3
for epoch in range(num_epochs):
    model.train()

    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### LOGGING
        print(
            f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
            f" | Train/Val Loss: {loss:.2f}"
        )

    model.eval()

with torch.no_grad():
    # outputs = model(X_train)
    outputs = model(X_test)
print(outputs)

torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(probas)

# print("------------")
# data_iter = iter(train_loader)
# hoge = next(data_iter)
# print(hoge)
