from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import dataloader
from cross_attention_loader import LoadData
from cross_attention_model import model
from constants import MainPath

# training parameters
batch_size = 80
lr = 0.0001
num_epoch = 100



# model initialization
generator = model()

if torch.cuda.is_available():
    generator.cuda()


# L2 loss
loss_function = nn.MSELoss()

# adam
g_optim = torch.optim.Adagrad(generator.parameters(), lr=lr)



# data loader
train_dataset = LoadData()
train_loader = dataloader.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=8
    )

num_batch = train_dataset.size/batch_size


def to_variable(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad)


for current_epoch in tqdm(range(1, num_epoch + 1)):

    g_cost_avg = 0
    for batch_erp, batch_cmp1, batch_cmp2, batch_cmp3, batch_cmp4, batch_cmp5, batch_cmp6, label in train_loader:
        # load data
        batch_erp = to_variable(batch_erp, requires_grad=False)
        batch_cmp1 = to_variable(batch_cmp1, requires_grad=False)
        batch_cmp2 = to_variable(batch_cmp2, requires_grad=False)
        batch_cmp3 = to_variable(batch_cmp3, requires_grad=False)
        batch_cmp4 = to_variable(batch_cmp4, requires_grad=False)
        batch_cmp5 = to_variable(batch_cmp5, requires_grad=False)
        batch_cmp6 = to_variable(batch_cmp6, requires_grad=False)
        label = to_variable(label, requires_grad=False)

        g_optim.zero_grad()

        # obtain ERP and 6 CMP faces
        out = generator(batch_erp, batch_cmp1, batch_cmp2, batch_cmp3, batch_cmp4, batch_cmp5, batch_cmp6)

        hard_loss = loss_function(out, label)
        g_cost_avg += hard_loss
        hard_loss.backward()
        g_optim.step()

    g_cost_avg /= num_batch
    print('Epoch:', current_epoch, ' train_loss->', g_cost_avg)
    if (current_epoch + 1) % 5 == 0:
        torch.save(generator.state_dict(),
                   MainPath + str(current_epoch+1) + '.pkl')
torch.save(generator.state_dict(), './cross_attention_100.pkl')

print('Done')
