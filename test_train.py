import torch
from torch import nn

X1 = torch.tensor([[10.0]], dtype=torch.float32) # Input: Temperature in °C
y1 = torch.tensor([[50.0]], dtype=torch.float32) # Actual value: Temperature °F

X2 = torch.tensor([[37.78]], dtype=torch.float32) # Input: Temperature in °C
y2 = torch.tensor([[100.0]], dtype=torch.float32) # Actual value: Temperature °F

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


print("BEFORE")
print(model.bias)
print(model.weight)
print("======")

for i in range(0, 100000):
    if i % 10000 == 0:
        print(f"Progress: {i}/1000 {i/100000*100}%")
    model.zero_grad()
    outputs = model(X1)
    loss = loss_fn(outputs, y1)
    loss.backward()
    optimizer.step()

print("AFTER")
print(model.bias)
print(model.weight)
print("======")

y_pred = model(X1)
print(y_pred)
