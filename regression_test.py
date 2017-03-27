import matplotlib as mpl
import numpy as np

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as fn


def generate_xy(n: int = 10, beta: float = 0.5, epsilon: float = 0.5, threshold: float = None):
    x = np.random.uniform(0, 10, size=(n, 1)).astype('float32')
    y = beta * x + np.random.normal(0, epsilon, (n, 1))
    if not threshold:
        return x, y
    else:
        yy = np.array([[0.0, 1.0] if value <= threshold else [1.0, 0.0] for value in y]).astype('float32')
        return x, yy


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1, bias=True)

    def forward(self, x):
        x = self.linear(x)
        return x


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=2, bias=True)

    def forward(self, x):
        x = self.linear(x)
        x = fn.softmax(x)
        return x


def create_linear_model():
    trainx, trainy = generate_xy(n=30, beta=1.5, epsilon=2)
    model = LinearRegression()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    for epoch in range(500):
        inputs = Variable(torch.from_numpy(trainx))
        targets = Variable(torch.from_numpy(trainy))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    plt.plot(trainx, trainy, 'k.', alpha=0.5)
    predicted = model(Variable(torch.from_numpy(trainx))).data.numpy()
    plt.plot(trainx, predicted, 'b-', alpha=0.5)
    plt.show()
    params = list(model.parameters())
    print(params)


def create_logistic_model():
    trainx, trainy = generate_xy(n=30, beta=1.0, epsilon=1.0, threshold=3.0)
    model = LogisticRegression()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(1000):
        inputs = Variable(torch.from_numpy(trainx))
        targets = Variable(torch.from_numpy(trainy))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    plt.plot(trainx, trainy[:, 1], 'ko', alpha=0.25)
    _, predicted = torch.max(outputs.data, 1)
    plt.plot(trainx, predicted.numpy(), 'bs', alpha=0.25)
    plt.show()
    params = list(model.parameters())
    print(params)


if __name__ == "__main__":
    create_logistic_model()
    # create_linear_model()
