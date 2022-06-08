# -*- coding:utf-8 -*-
import numpy
import pickle 
from matplotlib import pyplot as plt
import sys
from .utils import *

class dataset(object):
    """docstring for dataset"""
    def __init__(self, sigDict, finger_scene=False):
        super(dataset, self).__init__()
        self.testKeys = sorted(sigDict.keys())
        self.numGen = len(sigDict[self.testKeys[0]][True])
        self.accumNumNeg = len(sigDict[self.testKeys[0]][True]) + len(sigDict[self.testKeys[0]][False])
        
        self.feats = []
        print (">>>>> Extracting features... <<<<<")
        for key in self.testKeys:
            sys.stdout.write(">>>>> User key: %d <<<<<\r"%key)
            sys.stdout.flush()
            featExt(sigDict[key][True]+sigDict[key][False], self.feats, finger_scene=finger_scene)
        print (">>>>> Done <<<<<")
        self.featDim = self.feats[0].shape[1]
        self.lens = numpy.zeros(len(self.feats), dtype=numpy.float32)
        for i, f in enumerate(self.feats):
            self.lens[i] = f.shape[0]

    def __getitem__(self, index):
        sig = self.feats[index]
        # sig = pathDrop(sig)
        sigLen = sig.shape[0]
        sigLabel = int(index / self.accumNumNeg) # Note that keys start from 1, while labels start from 0. 

        return sig, sigLen, sigLabel

    def __len__(self):
        return len(self.testKeys) 

class batchSampler(object):
    """docstring for sampler"""
    def __init__(self, dataset):
        super(batchSampler, self).__init__()
        self.numIters = len(dataset)
        self.index = numpy.arange(0, self.numIters, dtype=numpy.int32)
        self.accumNumNeg = dataset.accumNumNeg

    def __iter__(self):
        for uidx in self.index:
            # One batch one user
            batch = numpy.arange(uidx * self.accumNumNeg, (uidx + 1) * self.accumNumNeg, dtype=numpy.int32)
            yield batch

    def __len__(self):
        return self.numIters

def collate_fn(batch):
    ''' `batch` is a list of tuple where 1-st element is the signature, 2-nd element is its length and 3-rd element is the label.
    '''
    batchSize = len(batch)
    sig = [item[0] for item in batch]
    sigLen = numpy.array([item[1] for item in batch], dtype=numpy.float32)
    sigLabel = numpy.array([item[2] for item in batch], dtype=numpy.int64)
    maxLen = int(numpy.max(sigLen))

    # for i in range(0, 40):
    #     plt.plot(sig[i][:, -2] + (i / 5) * 1.0, sig[i][:, -1] + (i % 5) * 1.0)
    # plt.show()
    
    sigPadded = numpy.zeros((batchSize, maxLen, sig[0].shape[1]), dtype=numpy.float32)
    for idx, s in enumerate(sig):
        sigPadded[idx,:s.shape[0]] = s

    return sigPadded, sigLen, sigLabel


if __name__ == '__main__':
    dset = dataset(sigPath="/home/lai/Documents/online-signature-verification/Unsupervised_sigVeri/Data/MCYT100_pad_ori_LP10Hz.pkl")
    sampler = batchSampler(dset)


