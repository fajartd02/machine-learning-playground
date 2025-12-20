import torch
from torch import nn


X1 = torch.tensor([[40.0]], dtype=torch.float32)
y1 = torch.tensor([[104.0]], dtype=torch.float32)

# Input: Temperature in °C
X2 = torch.tensor([[10]], dtype=torch.float32) 
# Actual value: Temperature °F
y2 = torch.tensor([[50]], dtype=torch.float32) 

# Input: Temperature in °C
X3 = torch.tensor([[37.78]], dtype=torch.float32) 
# Actual value: Temperature °F
y3 = torch.tensor([[100.0]], dtype=torch.float32)

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

print("START")
print(model.bias)
print(model.weight)
print("=============")

for i in range(0, 150000):
    if i % 1000 == 0:
        print(f"Progress is {i}/100000 - {i/150000*100}%")
    
    optimizer.zero_grad()
    outputs = model(X1)
    loss = loss_fn(outputs, y1)
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    outputs = model(X2)
    loss = loss_fn(outputs, y2)
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    outputs = model(X3)
    loss = loss_fn(outputs, y3)
    loss.backward()
    optimizer.step()

print("AFTER")
print(model.bias)
print(model.weight)
print("=============")

y_pred1 = model(X1)
y_pred2 = model(X2)
y_pred3 = model(X3)

print("PRED")
print("=============")
print(y_pred1)
print(y_pred2)
print(y_pred3)