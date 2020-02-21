import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

x_train = np.random.rand(150).reshape(-1, 1).astype('float32')
noise = (np.random.normal(size=150) / 10).reshape(-1, 1).astype('float32')
y_train = 3 * x_train - 1 + noise

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    outputs = model(inputs)
    optim.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optim.step()
    if(epoch % 100 == 0):
        print(loss.item())

outputs = model(torch.from_numpy(x_train)).detach().numpy()
w, b = model.parameters()
print(w.item(), b.item())
plt.plot(x_train, y_train, 'go', label='data')
plt.plot(x_train, outputs, label='predict')
plt.legend()
plt.show()
