import torch
from torch import nn
# FAJAR:
# For inference, you must create a model with the same architecture as training 
# load the trained weights, and switch the model to inference mode.

model = nn.Linear(1, 1)
model.load_state_dict(torch.load("converter_celcius_to_farenheit.pth"))
model.eval()

input_from_user = [10, 38, 100, 150, 200]
X = torch.tensor(input_from_user, dtype=torch.float32).unsqueeze(1)
# print(X)

with torch.no_grad():
    y = model(X)

print(y)