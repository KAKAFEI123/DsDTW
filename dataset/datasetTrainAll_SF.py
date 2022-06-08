# -*- coding:utf-8 -*-
import numpy
import pickle 
from matplotlib import pyplot as plt
import sys
from .utils import *

class dataset(object):
    """docstring for dataset"""
    def __init__(self, sigDict, taskSize=1, taskNumGen=5, taskNumNeg=5, saveKey=True, finger_scene=False):
        super(dataset, self).__init__()
        self.trainKeys = list(sigDict.keys())
        self.taskSize = taskSize
        self.taskNumGen = taskNumGen
        self.taskNumNeg = taskNumNeg
        
        self.feats = []
        self.numGen = numpy.zeros(len(self.trainKeys), dtype=numpy.int32)
        self.numNeg = numpy.zeros(len(self.trainKeys), dtype=numpy.int32)
        # The dataset is user-by-user arranged in the following format: anchors, genuines, forgeries, anchors, ...
        print (">>>>> Extracting features... <<<<<")
        for idx, key in enumerate(self.trainKeys):
            sys.stdout.write(">>>>> User key: %d <<<<<\r"%key)
            sys.stdout.flush()
            featExt(sigDict[key][True], self.feats, finger_scene=finger_scene)
            featExt(sigDict[key][False], self.feats, finger_scene=finger_scene)
            self.numGen[idx] = len(sigDict[key][True])
            self.numNeg[idx] = len(sigDict[key][False])
        print (">>>>> Done <<<<<")
        self.accumNum2 = numpy.cumsum(self.numGen + self.numNeg) 
        self.accumNum = numpy.roll(self.accumNum2, 1); self.accumNum[0] = 0 
        
        self.featDim = self.feats[0].shape[1]
        self.lens = numpy.zeros(len(self.feats), dtype=numpy.float32)
        for i, f in enumerate(self.feats):
            self.lens[i] = f.shape[0]

    def __getitem__(self, index):
        sig = self.feats[index]
        sigLen = self.lens[index]
        sigLabel = numpy.sum(index>=self.accumNum2) # Note that keys start from 1, while labels start from 0. 

        return sig, sigLen, sigLabel

    def __len__(self):
        return len(self.trainKeys) 

    def addDatabase(self, sigDict, finger_scene=False):
        newKeys = list(sigDict.keys())
        self.trainKeys = numpy.concatenate((self.trainKeys, newKeys))
        N = len(self.feats)

        numGen = numpy.zeros(len(newKeys), dtype=numpy.int32)
        numNeg = numpy.zeros(len(newKeys), dtype=numpy.int32)
        print (">>>>> Extracting features... <<<<<")
        for i, key in enumerate(newKeys):
            sys.stdout.write(">>>>> User key: %d <<<<<\r"%key)
            sys.stdout.flush()
            featExt(sigDict[key][True], self.feats, finger_scene=finger_scene)
            featExt(sigDict[key][False], self.feats, finger_scene=finger_scene)
            numGen[i] = len(sigDict[key][True])
            numNeg[i] = len(sigDict[key][False])
        print (">>>>> Done <<<<<")
        self.numGen = numpy.concatenate((self.numGen, numGen))
        self.numNeg = numpy.concatenate((self.numNeg, numNeg))
        self.accumNum2 = numpy.cumsum(self.numGen + self.numNeg) 
        self.accumNum = numpy.roll(self.accumNum2, 1); self.accumNum[0] = 0 

        lens = numpy.zeros(len(self.feats)-N, dtype=numpy.float32)
        for i in range(N, len(self.feats)):
            lens[i-N] = self.feats[i].shape[0]
        self.lens = numpy.concatenate((self.lens, lens))

    # def histLens(self):
    #     lens = numpy.zeros(len(self.feats))
    #     for idx, f in enumerate(self.feats):
    #         lens[idx] = f.shape[0]
    #     plt.hist(lens, bins=30)
    #     plt.show()
        
class batchSampler(object):
    """docstring for sampler"""
    def __init__(self, dataset, loop=False):
        super(batchSampler, self).__init__()
        self.taskSize = dataset.taskSize
        self.index = numpy.arange(0, len(dataset.trainKeys), dtype=numpy.int32)
        # self.index = numpy.repeat(self.index, self.taskSize, axis=0)
        self.taskNumGen = dataset.taskNumGen
        self.taskNumNeg = dataset.taskNumNeg
        self.numGen = dataset.numGen
        self.numNeg = dataset.numNeg
        self.accumNum = dataset.accumNum
        self.numIters = len(dataset)
        self.loop = loop

    def __iter__(self):
        batch = []
        numpy.random.shuffle(self.index)
        for i in range(self.numIters):
            if self.loop:
                idxs = self.index[numpy.arange(i, i+self.taskSize)%len(self.index)]
            else:
                idxs = numpy.random.choice(self.index, size=self.taskSize, replace=False)
            for idx in idxs:
                gen = numpy.random.choice(self.numGen[idx], size=1+self.taskNumGen, replace=False) 
                ## SF
                # neg = numpy.random.choice(self.numNeg[idx], size=self.taskNumNeg, replace=False) + self.numGen[idx]
                # batch.append(gen + self.accumNum[idx])
                # batch.append(neg + self.accumNum[idx])
                ## SF + RF
                neg = numpy.random.choice(self.numNeg[idx], size=self.taskNumNeg//2, replace=False) + self.numGen[idx]
                batch.append(gen + self.accumNum[idx])
                batch.append(neg + self.accumNum[idx])
                idxs_RF = (idx+numpy.random.randint(1, len(self.index), size=self.taskNumNeg//2))%len(self.index)
                for idx in idxs_RF:
                    neg = numpy.random.choice(self.numGen[idx], size=1, replace=False) 
                    batch.append(neg + self.accumNum[idx])
                ## RF
                # batch.append(gen + self.accumNum[idx])
                # idxs_RF = (idx+numpy.random.randint(1, len(self.index), size=self.taskNumNeg))%len(self.index)
                # for idx in idxs_RF:
                #     neg = numpy.random.choice(self.numGen[idx], size=1, replace=False) 
                #     batch.append(neg + self.accumNum[idx])
            batch = numpy.concatenate(batch, axis=0).astype(numpy.int32)
            yield batch
            batch = []

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

    # print (sigLabel)
    # plt.plot(sig[0][:, -2], sig[0][:, -1])
    # for i in range(1, 6):
    #     plt.plot(sig[i][:, -2] + (i - 1) * 1.0, sig[i][:, -1] + 1.0)
    # for i in range(6, 11):
    #     plt.plot(sig[i][:, -2] + (i - 6) * 1.0, sig[i][:, -1] + 2.0)
    # plt.show()

    sigPadded = numpy.zeros((batchSize, maxLen, sig[0].shape[1]), dtype=numpy.float32)
    for idx, s in enumerate(sig):
        sigPadded[idx,:s.shape[0]] = s

    return sigPadded, sigLen, sigLabel