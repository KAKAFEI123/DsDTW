# -*- coding: UTF-8 -*-
"""
DsDTW: Deep soft-DTW method for dynamic signature verification.
"""
import os, pickle
import numpy 
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nutils
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import dataset.datasetTrainAll_SF as dataset
from dsdtw import DSDTW as Model

parser = argparse.ArgumentParser(description='Dynamic signature verification')
parser.add_argument('--train-shot-g', type=int, default=5, metavar='TRSG', 
                    help='number of genuine samples per class per training batch(default: 5)') 
parser.add_argument('--train-shot-f', type=int, default=10, metavar='TRSG',
                    help='number of forgery samples per class per training batch(default: 10)')
parser.add_argument('--train-tasks', type=int, default=4, 
                    help='number of tasks per batch')
parser.add_argument('--epochs', type=int, default=20, 
                    help='number of epochs to train (default: 20)')
parser.add_argument('--seed', type=int, default=111, metavar='S',
                    help='numpy random seed (default: 111)')
parser.add_argument('--save-interval', type=int, default=3, 
                    help='how many epochs to wait before saving the model.')
parser.add_argument('--lr', type=float, default=0.01, 
                    help='learning rate')

args = parser.parse_args()
n_task = args.train_tasks
n_shot_g = args.train_shot_g
n_shot_f = args.train_shot_f

numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

sigDict = pickle.load(open("../Data/MCYT_dev.pkl", "rb"), encoding='iso-8859-1')
dset = dataset.dataset(
                    sigDict=sigDict,
                    taskSize=n_task, 
                    taskNumGen=n_shot_g, 
                    taskNumNeg=n_shot_f,
                    finger_scene=False
                )
del sigDict
sigDict = pickle.load(open("../Data/BSID_dev.pkl", "rb"), encoding='iso-8859-1')
dset.addDatabase(sigDict, finger_scene=False)
del sigDict
sigDict = pickle.load(open("../Data/EBio1_dev.pkl", "rb"), encoding='iso-8859-1')
dset.addDatabase(sigDict, finger_scene=False)
del sigDict
# sigDict = pickle.load(open("../Unsupervised_sigVer/Data/EBio2_dev.pkl", "rb"), encoding='iso-8859-1')
# dset.addDatabase(sigDict, finger_scene=False)
# del sigDict
# sigDict = pickle.load(open("../Unsupervised_sigVer/Data/EBio1_dev_finger.pkl", "rb"), encoding='iso-8859-1')
# dset.addDatabase(sigDict, finger_scene=True)
# del sigDict
# sigDict = pickle.load(open("../Unsupervised_sigVer/Data/EBio2_dev_finger.pkl", "rb"), encoding='iso-8859-1')
# dset.addDatabase(sigDict, finger_scene=True)
# del sigDict

sampler = dataset.batchSampler(dset, loop=False)
dataLoader = DataLoader(dset, batch_sampler=sampler, collate_fn=dataset.collate_fn)

model = Model(
            n_in=dset.featDim, 
            n_layers=2,
            n_hidden=128,
            n_out=64,
            n_task=n_task,
            n_shot_g=n_shot_g,
            n_shot_f=n_shot_f,
        )
model.train(mode=True)
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) 
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) 

if not os.path.exists("./models/%d"%args.seed):
    os.makedirs("./models/%d"%args.seed)
    
for epoch in range(0, args.epochs):
    print("lr at epoch %d：%f" % (epoch, optimizer.param_groups[0]['lr'])) 
    TriLoss_std = 0
    TriLoss_hard = 0
    Var = 0

    for idx, batch in enumerate(dataLoader):
        sig, lens, label = batch
        mask = model.getOutputMask(lens)

        sig = Variable(torch.from_numpy(sig)).cuda()
        mask = Variable(torch.from_numpy(mask)).cuda()
        label = Variable(torch.from_numpy(label)).cuda()

        optimizer.zero_grad()
        output, length, _ = model(sig, mask) #(N,T,D)
        triLoss_std, triLoss_hard, var = model.tripletLoss(output, length)

        (triLoss_hard+0.01*var).backward() 
        optimizer.step()
        
        TriLoss_std += triLoss_std.item()
        TriLoss_hard += triLoss_hard.item()
        Var += var.item()

        if (idx + 1) % 50 == 0:
            print ("epoch:",epoch, "idx:",idx)
            print("TriLoss_std:", format(TriLoss_std/50,'.6f'), "TriLoss_hard:", \
                format(TriLoss_hard/50,'.6f'), "Var:", format(Var/50,'.6f'))
            TriLoss_std = 0
            TriLoss_hard = 0
            Var = 0

    lr_scheduler.step()
    
    if epoch % args.save_interval == 0:
       torch.save(model.state_dict(), "models/%d/epoch%d"%(args.seed, epoch))

torch.save(model.state_dict(), "models/%d/epochEnd"%args.seed)




