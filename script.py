import numpy as np
import requests
import gzip
import os
import hashlib
import matplotlib.pyplot as plt

# fetch function
def fetch(url):
    fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()


# fetch some data
X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1,28,28))
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1,28,28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

# backpropagation with library

import torch
import torch.nn as nn
from tqdm import trange

# define model
class MarNet(torch.nn.Module):
    def __init__(self):
        super(MarNet, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.act =nn.ReLU()
        self.l2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x

# create a model
model = MarNet()

# training loop
BS = 128
loss_function = nn.CrossEntropyLoss()
# choose optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)
losses, accuracies = [], []

for i in (t := trange(1000)):
    samp = np.random.randint(0, X_train.shape[0], size=(BS))
    X = torch.tensor(X_train[samp].reshape((-1, 28*28))).float()
    Y = torch.tensor(Y_train[samp]).long()
    optimizer.zero_grad()
    out = model(X)
    cat = torch.argmax(out, dim=1)
    accuracy = (cat == Y).float().mean()
    loss = loss_function(out,Y)
    loss.backward()
    optimizer.step()
    loss, accuracy = loss.item(), accuracy.item()
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description("loss %.2f accuracy %.2f" %  (loss, accuracy))

# plot result of training (accuracy and loss)
plt.ylim(-0.5,1.5)
plt.plot(losses)
plt.plot(accuracies)
plt.show()

# evaluation of model on test data
Y_test_preds = torch.argmax(model(torch.tensor(X_test.reshape((-1, 28*28))).float()), dim=1).numpy()
res = (Y_test == Y_test_preds).mean()
print("Percentage correctly classified in test set: ", res*100, "%")
