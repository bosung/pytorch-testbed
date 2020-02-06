import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, Sigmoid, KLDivLoss, Softmax, LogSoftmax, BCEWithLogitsLoss, SoftMarginLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.optim import Adam


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(2, 3)
        self.linear2 = nn.Linear(3, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # return self.linear2(self.sigmoid(self.linear1(inputs)))
        return self.linear2(self.relu(self.linear1(inputs)))
        # return (self.linear1(inputs)


np.random.seed(0)
torch.manual_seed(0)

# generate 2d classification dataset
N = 50
x1 = np.random.normal(0, 0.6, N)
y1 = np.random.normal(0.2, 0.6, N)
positives = torch.cat([torch.tensor(x1).unsqueeze(1), torch.tensor(y1).unsqueeze(1)], dim=1)
p_label = positives.new_ones(N, dtype=torch.long)

x2 = np.random.normal(1, 0.5, N * 10)
y2 = np.random.normal(1, 0.5, N * 10)
negatives = torch.cat([torch.tensor(x2).unsqueeze(1), torch.tensor(y2).unsqueeze(1)], dim=1)
# n_label = negatives.new_zeros(N * 5, dtype=torch.long)
n_label = positives.new_ones(N * 10, dtype=torch.long) * -1

all_dataset = torch.cat([positives, negatives], dim=0).float()
all_labels = torch.cat([p_label, n_label], dim=0)
train_data = TensorDataset(all_dataset, all_labels)
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=10)

model = Classifier()
optimizer = Adam(model.parameters())

t_step = 0
for ep in range(1, 1000 + 1):
    model.train()
    ep_loss = 0
    for step, batch in enumerate(train_dataloader):
        inputs, labels = tuple(t for t in batch)
        logits = model(inputs)
        # loss_fct = CrossEntropyLoss(reduction='none')
        # _loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        # loss = _loss.mean()
        loss_fct = SoftMarginLoss()
        loss = loss_fct(logits[:, 1], labels.float())

        # if (step + 1) % 4 == 0:
        #     print("loss: %8.4f" % loss)
        ep_loss = ep_loss + loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        t_step = t_step + 1
    print("[ep %d] loss: %8.4f" % (ep, ep_loss/step))
    if t_step > 2000:
        break


fig, ax = plt.subplots()

xx = np.linspace(-1.6, 2.4, 50)
yy = np.linspace(-1.6, 2.4, 50)
xv, yv = np.meshgrid(xx, yy)

test_array = np.zeros((2500, 2))
k = 0
for i in range(50):
    for j in range(50):
        test_array[k] = [xv[i, j], yv[i, j]]
        k = k+1

test_dataset = torch.tensor(test_array).float()
model.eval()
outs = model(test_dataset)
outs = nn.Softmax(dim=1)(outs)
outs = outs[:, 1].view(50, 50)

out_array = np.zeros((50, 50))

# t = ax.contourf(xv, yv, outs.detach().numpy(), alpha=0.4, levels=[0.1 * e for e in range(0, 11)], cmap="terrain")
t = ax.contourf(xv, yv, outs.detach().numpy(), levels=[0, 0.5, 1], cmap="Greys_r")
# ax.clabel(t, fmt='%2.1f', colors='r', fontsize=10)


ax.scatter(x1, y1, c='r', marker='^', s=40)
# set bound
m_size = min(x2[x2 < 2.3].size, y2[y2 < 2.3].size)
ax.scatter(x2[x2 < 2.3][:m_size], y2[y2 < 2.3][:m_size], c='b')

# plt.plot(xx, bound, 'r--', lw=3)

fig.colorbar(t, ax=ax)
plt.show()
