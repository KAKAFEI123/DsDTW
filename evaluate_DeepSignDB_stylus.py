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
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader

import dataset.datasetTest_SF as dataset
from dsdtw import DSDTW as Model
from dist import dist_seq, dist_seq_rf

parser = argparse.ArgumentParser(description='Online signature verification')
parser.add_argument('--train-shot-g', type=int, default=4, metavar='TRSG', 
                    help='number of genuine samples per class per training batch(default: 4)') #Genuine samples only
parser.add_argument('--train-shot-f', type=int, default=0, metavar='TRSG', 
                    help='number of forgery samples per class per training batch(default: 3)')
parser.add_argument('--train-tasks', type=int, default=1, 
                    help='number of tasks per batch')
parser.add_argument('--seed', type=int, default=111, metavar='S',
                    help='numpy random seed (default: 111)')
parser.add_argument('--epoch', type=str, default="End", 
                    help='model from the i-th epoch for testing')
parser.add_argument('--rf', action='store_true',
                    help='test random forgery or not')

args = parser.parse_args()
n_task = args.train_tasks
n_shot_g = args.train_shot_g
n_shot_f = args.train_shot_f
print("Random Forgery Scenario:",args.rf)

numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

sigDict = pickle.load(open("../Data/MCYT_eva.pkl", "rb"), encoding='iso-8859-1')
num_g = 25; num_f = 25
print("For MCYT:")

dset = dataset.dataset(sigDict=sigDict, finger_scene=False)
sampler = dataset.batchSampler(dset)
dataLoader = DataLoader(dset, batch_sampler=sampler, collate_fn=dataset.collate_fn)

model = Model(
          n_in=dset.featDim, 
          n_layers=2,
          n_hidden=128, 
          n_out=64,
          n_task=n_task,
          n_shot_g=n_shot_g, 
          n_shot_f=n_shot_f, 
          batchsize=num_g+num_f)
model.load_state_dict(torch.load("models/%d/epoch%s"%(args.seed, args.epoch)))

model.cuda()
model.train(mode=False)
model.eval()

feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length, ht = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)

if not os.path.exists("log/seed%d"%args.seed):
    os.makedirs("log/seed%d"%args.seed)

if not os.path.exists("log/seed%d/mcyt"%args.seed):
    os.makedirs("log/seed%d/mcyt"%args.seed)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)
numpy.save("log/seed%d/mcyt/dtw_dist_p%s.npy"%(args.seed, args.epoch), DIST_P)
numpy.save("log/seed%d/mcyt/dtw_dist_n%s.npy"%(args.seed, args.epoch), DIST_N)
numpy.save("log/seed%d/mcyt/dtw_dist_temp%s.npy"%(args.seed, args.epoch), DIST_TEMP)

sigDict = pickle.load(open("../Data/BSID_eva.pkl", "rb"), encoding='iso-8859-1')
num_g = 16; num_f = 12
print("For BiosecurID:")

dset = dataset.dataset(sigDict=sigDict, finger_scene=False)
sampler = dataset.batchSampler(dset)
dataLoader = DataLoader(dset, batch_sampler=sampler, collate_fn=dataset.collate_fn)

model.h0 = Variable(torch.zeros(2, num_g+num_f, 128).cuda(), requires_grad=False)

feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length, ht = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)

if not os.path.exists("log/seed%d/bio"%args.seed):
    os.makedirs("log/seed%d/bio"%args.seed)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)
numpy.save("log/seed%d/bio/dtw_dist_p%s.npy"%(args.seed, args.epoch), DIST_P)
numpy.save("log/seed%d/bio/dtw_dist_n%s.npy"%(args.seed, args.epoch), DIST_N)
numpy.save("log/seed%d/bio/dtw_dist_temp%s.npy"%(args.seed, args.epoch), DIST_TEMP)


sigDict = pickle.load(open("../Data/BSDS2_eva.pkl", "rb"), encoding='iso-8859-1')
num_g = 19; num_f = 20
print("For Biosecure DS2:")

dset = dataset.dataset(sigDict=sigDict, finger_scene=False)
sampler = dataset.batchSampler(dset)
dataLoader = DataLoader(dset, batch_sampler=sampler, collate_fn=dataset.collate_fn)

model.h0 = Variable(torch.zeros(2, num_g+num_f, 128).cuda(), requires_grad=False)

feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    # For BSDB2, 4 templates in session 1 + session2
    idxs = numpy.concatenate([numpy.array([0,1,2,3]), numpy.arange(15, 50)])
    sig = sig[idxs]; lens = lens[idxs]; label = label[idxs]

    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length, ht = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)

if not os.path.exists("log/seed%d/bsds2"%args.seed):
    os.makedirs("log/seed%d/bsds2"%args.seed)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)
numpy.save("log/seed%d/bsds2/dtw_dist_p%s.npy"%(args.seed, args.epoch), DIST_P)
numpy.save("log/seed%d/bsds2/dtw_dist_n%s.npy"%(args.seed, args.epoch), DIST_N)
numpy.save("log/seed%d/bsds2/dtw_dist_temp%s.npy"%(args.seed, args.epoch), DIST_TEMP)

sigDict = pickle.load(open("../Data/EBio2_eva.pkl", "rb"), encoding='iso-8859-1')
num_g = 8; num_f = 6
print("For eBS DS2 w2:")

dset = dataset.dataset(sigDict=sigDict, finger_scene=False)
sampler = dataset.batchSampler(dset)
dataLoader = DataLoader(dset, batch_sampler=sampler, collate_fn=dataset.collate_fn)

model.h0 = Variable(torch.zeros(2, num_g+num_f, 128).cuda(), requires_grad=False)

feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    # For EBio2
    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length, ht = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)

if not os.path.exists("log/seed%d/ebio2"%args.seed):
    os.makedirs("log/seed%d/ebio2"%args.seed)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)
numpy.save("log/seed%d/ebio2/dtw_dist_p%s.npy"%(args.seed, args.epoch), DIST_P)
numpy.save("log/seed%d/ebio2/dtw_dist_n%s.npy"%(args.seed, args.epoch), DIST_N)
numpy.save("log/seed%d/ebio2/dtw_dist_temp%s.npy"%(args.seed, args.epoch), DIST_TEMP)


sigDict = pickle.load(open("../Data/EBio1_eva.pkl", "rb"), encoding='iso-8859-1')
num_g = 8; num_f = 6

dset = dataset.dataset(sigDict=sigDict, finger_scene=False)
sampler = dataset.batchSampler(dset)
dataLoader = DataLoader(dset, batch_sampler=sampler, collate_fn=dataset.collate_fn)

model.h0 = Variable(torch.zeros(2, num_g+num_f, 128).cuda(), requires_grad=False)

print("For eBS DS1 w1:")
feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    # For EBio1
    device = 0
    idxs = numpy.concatenate([numpy.array([0,1,10,11,20,21,30,31]) + device * 2,
                              numpy.array([40,41,42,55,56,57]) + device * 3])
    sig = sig[idxs]; lens = lens[idxs]; label = label[idxs]
    sig = sig[:,0:int(numpy.max(lens)),:]

    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length, ht = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)

if not os.path.exists("log/seed%d/ebio1"%args.seed):
    os.makedirs("log/seed%d/ebio1"%args.seed)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)
numpy.save("log/seed%d/ebio1/dtw_dist_p%s_d0.npy"%(args.seed, args.epoch), DIST_P)
numpy.save("log/seed%d/ebio1/dtw_dist_n%s_d0.npy"%(args.seed, args.epoch), DIST_N)
numpy.save("log/seed%d/ebio1/dtw_dist_temp%s_d0.npy"%(args.seed, args.epoch), DIST_TEMP)

print("For eBS DS1 w2:")
feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    # For EBio1
    device = 1
    idxs = numpy.concatenate([numpy.array([0,1,10,11,20,21,30,31]) + device * 2,
                              numpy.array([40,41,42,55,56,57]) + device * 3])
    sig = sig[idxs]; lens = lens[idxs]; label = label[idxs]
    sig = sig[:,0:int(numpy.max(lens)),:]

    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length, ht = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)

if not os.path.exists("log/seed%d/ebio1"%args.seed):
    os.makedirs("log/seed%d/ebio1"%args.seed)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)
numpy.save("log/seed%d/ebio1/dtw_dist_p%s_d1.npy"%(args.seed, args.epoch), DIST_P)
numpy.save("log/seed%d/ebio1/dtw_dist_n%s_d1.npy"%(args.seed, args.epoch), DIST_N)
numpy.save("log/seed%d/ebio1/dtw_dist_temp%s_d1.npy"%(args.seed, args.epoch), DIST_TEMP)

print("For eBS DS1 w3:")
feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    # For EBio1
    device = 2
    idxs = numpy.concatenate([numpy.array([0,1,10,11,20,21,30,31]) + device * 2,
                              numpy.array([40,41,42,55,56,57]) + device * 3])
    sig = sig[idxs]; lens = lens[idxs]; label = label[idxs]
    sig = sig[:,0:int(numpy.max(lens)),:]

    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length, ht = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)

if not os.path.exists("log/seed%d/ebio1"%args.seed):
    os.makedirs("log/seed%d/ebio1"%args.seed)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)
numpy.save("log/seed%d/ebio1/dtw_dist_p%s_d2.npy"%(args.seed, args.epoch), DIST_P)
numpy.save("log/seed%d/ebio1/dtw_dist_n%s_d2.npy"%(args.seed, args.epoch), DIST_N)
numpy.save("log/seed%d/ebio1/dtw_dist_temp%s_d2.npy"%(args.seed, args.epoch), DIST_TEMP)

print("For eBS DS1 w4:")
feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    # For EBio1
    device = 3
    idxs = numpy.concatenate([numpy.array([0,1,10,11,20,21,30,31]) + device * 2,
                              numpy.array([40,41,42,55,56,57]) + device * 3])
    sig = sig[idxs]; lens = lens[idxs]; label = label[idxs]
    sig = sig[:,0:int(numpy.max(lens)),:]

    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length, ht = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)

if not os.path.exists("log/seed%d/ebio1"%args.seed):
    os.makedirs("log/seed%d/ebio1"%args.seed)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)
numpy.save("log/seed%d/ebio1/dtw_dist_p%s_d3.npy"%(args.seed, args.epoch), DIST_P)
numpy.save("log/seed%d/ebio1/dtw_dist_n%s_d3.npy"%(args.seed, args.epoch), DIST_N)
numpy.save("log/seed%d/ebio1/dtw_dist_temp%s_d3.npy"%(args.seed, args.epoch), DIST_TEMP)

print("For eBS DS1 w5:")
feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    # For EBio1
    device = 4
    idxs = numpy.concatenate([numpy.array([0,1,10,11,20,21,30,31]) + device * 2,
                              numpy.array([40,41,42,55,56,57]) + device * 3])
    sig = sig[idxs]; lens = lens[idxs]; label = label[idxs]
    sig = sig[:,0:int(numpy.max(lens)),:]

    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length, ht = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)

if not os.path.exists("log/seed%d/ebio1"%args.seed):
    os.makedirs("log/seed%d/ebio1"%args.seed)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)
numpy.save("log/seed%d/ebio1/dtw_dist_p%s_d4.npy"%(args.seed, args.epoch), DIST_P)
numpy.save("log/seed%d/ebio1/dtw_dist_n%s_d4.npy"%(args.seed, args.epoch), DIST_N)
numpy.save("log/seed%d/ebio1/dtw_dist_temp%s_d4.npy"%(args.seed, args.epoch), DIST_TEMP)


