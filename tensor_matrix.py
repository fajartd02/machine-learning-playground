import torch

X = torch.tensor([
    [10],
    [38],
    [100],
    [150],
])


print(X.size(0))
print(X.shape[0])
print(X[:, 0])
print(X[0, :])