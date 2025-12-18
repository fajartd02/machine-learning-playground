import torch

b = 32
w1 = 1.8

X1 = torch.tensor(10)

y_pred = 1 * b + X1 * w1
print(y_pred)