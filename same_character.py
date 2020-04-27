import random
import multiprocessing

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from attn_network import *
from sampler import *

writer = SummaryWriter()

net = clfr2_wAux( res_strides=(2,2,2,2,2),
            res_depth = 64,
            model_features=128, 
            num_attnblocks=12,
            num_heads = 8,
            classes = 1000,
            in_features = 3,
            out_classes = 2,
            input_size = 256,
            dropout_prob = 0.0)

#net = torch.load("run1.pth")

net = net.cuda()

batch_size = 4

learn_rate = 0.0003
opt_params = [{'params': list(net.parameters()), 'lr': learn_rate, 'weight_decay':0.00003}]
optimizer = optim.Adam(opt_params)

loss_func = nn.NLLLoss()

net = net.cuda()

classification_weight = 1

folder = "d1k/d1k/"

val_folder = "d1k/d1k_validation/"
val_batch_size = 128
val_step = 10

save_name = "run2.pth"

bs = batch_sampler(folder = folder, batch_size = batch_size, num_procs = 6)
val_bs = batch_sampler(folder = val_folder, batch_size = val_batch_size, num_procs = 6)

max_steps = 10000
for i in range(max_steps):
    batch = bs.next()
    
    out = net(batch)
    
    optimizer.zero_grad()
 
    is_same = out['out'].squeeze()
    im1_class = out['im1_class'].squeeze()
    im2_class = out['im2_class'].squeeze()

    same_l = loss_func(is_same, batch['y'].cuda())

    print(batch['img_1_class'])
    print(im1_class)

    im1_c_l = loss_func(im1_class, batch['img_1_class'].cuda())
    im2_c_l = loss_func(im2_class, batch['img_2_class'].cuda())

    class_l = (im1_c_l + im2_c_l) * 0.5
    
    l = same_l + classification_weight * class_l

    l.backward()
    optimizer.step()

    writer.add_scalar('Loss/train_same', float(same_l), i)
    writer.add_scalar('Loss/train_class', float(class_l), i)
    writer.add_scalar('Loss/train', float(l), i)
    print("\n",out['out'].squeeze(), batch['y'])
    if i % val_step == 0:
        with torch.no_grad():
            batch = val_bs.next()
            
            out = net(batch)
            out = out['out'].squeeze()

            pred = torch.argmax(out, dim = 1)
            
            apr = (pred-(batch['y'].cuda())).abs().sum()
            val_pct_correct = 1.0 - (apr / float(batch_size))

            val_loss = loss_func(out, batch['y'].cuda())
            writer.add_scalar('Loss/val_class', float(val_loss), i)
            writer.add_scalar('Loss/val_pct_correct', val_pct_correct, i)

torch.save(net, save_name)

    
    