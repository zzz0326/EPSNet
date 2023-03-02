import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import time
from kld_cc_nss_loss import Loss
from tqdm import tqdm
from torch.autograd import Variable
from supervised_loader import DataLoader
from supervised_model import Model_rethink
from torch.optim import Adam
import torch.nn as nn

# training parameters
batch_size = 16
num_epoch = 20
lr = 0.0001


generator = Model_rethink()

# load the FindCMP trained encoder weight
print("initalizing with Salgan encoder weights.....")
weight = torch.load(
    "./cross_attention_100.pkl")
generator.load_state_dict(weight, strict=False)
del weight


if torch.cuda.is_available():
    generator.cuda()

loss = Loss()
bce = nn.BCELoss()

g_optim = Adam(generator.parameters(), lr=lr)

dataloader = DataLoader(batch_size)
num_batch = dataloader.num_batches
print(num_batch, num_batch)



def to_variable(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad)


counter = 0
DIR_TO_SAVE = "./generator_output/"
if not os.path.exists(DIR_TO_SAVE):
    os.makedirs(DIR_TO_SAVE)



for current_epoch in tqdm(range(1, num_epoch + 1)):
    n_updates = 1
    kld_avg = 0
    cc_avg = 0
    nss_avg = 0

    for idx in range(int(num_batch)):
        # load data
        (batch_img, batch_map, batch_fix) = dataloader.get_4000batch()
        batch_img = to_variable(batch_img, requires_grad=False)
        batch_map = to_variable(batch_map, requires_grad=False)
        batch_fix = to_variable(batch_fix, requires_grad=False)

        g_optim.zero_grad()

        fake_map = generator(batch_img)
        kld, cc, nss = loss(fake_map, batch_map, batch_fix)
        t_loss = kld
        g_loss = torch.sum(t_loss)
        kld_avg += kld.item()

        g_loss.backward()
        g_optim.step()

        n_updates += 1
        counter += 1

    kld_avg /= num_batch



    print('Epoch:', current_epoch, ' train_loss->', kld_avg)


torch.save(generator.state_dict(), './saliency_model.pkl')

print('Done')
