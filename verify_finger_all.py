# !/usr/bin/env python
# -*- coding:utf-8 -*-
import os, pickle
import timeit
import numpy 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

import argparse

opj = os.path.join

numpy.set_printoptions(threshold=1e4)

parser = argparse.ArgumentParser(description='Online signature verification')
parser.add_argument('--train-shot-g', type=int, default=4, metavar='TRSG', #Training set in meta-training (meta-batch)
                    help='number of genuine samples per class per training batch(default: 4)') #Genuine samples only
parser.add_argument('--epoch', type=str, default="End", 
                    help='model from the i-th epoch for testing')
args = parser.parse_args()
n_train_g = args.train_shot_g
epoch = args.epoch

def selectTemplate(distMatrix):
    refNum = distMatrix.shape[0]
    if refNum == 1:
        return None, 1, 1, 1, 1
    distMatrix = (distMatrix + distMatrix.transpose())
    distAve = numpy.sum(distMatrix, axis=1) / (refNum - 1)
    idx = numpy.argmin(distAve)
    dvar = numpy.sqrt((numpy.sum(distMatrix**2) / refNum / (refNum - 1)- (numpy.sum(distMatrix) / refNum / (refNum - 1))**2))
    # '''Arithmetic mean
    # '''   
    dmean = numpy.sum(distMatrix) / refNum / (refNum - 1) 
    '''distance of reference signatures to the template signature dtmp'''
    dtmp = numpy.sum(distMatrix[:, idx]) / (refNum - 1)
    '''distance of reference signatures to their farthest neighbor dmax'''
    dmax = numpy.mean(numpy.max(distMatrix, axis=1))
    '''distance of reference signatures to their nearest neighbor dmin'''
    distMatrix[range(refNum), range(refNum)] = float("inf")
    distMatrix[distMatrix==0] = float("inf")
    dmin = numpy.mean(numpy.min(distMatrix, axis=1))
    
    return idx, dtmp**0.5, dmax**0.5, dmin**0.5, dmean**0.5

def getEER(FAR, FRR):
    a = FRR <= FAR
    s = numpy.sum(a)
    a[-s-1] = 1
    a[-s+1:] = 0
    FRR = FRR[a]
    FAR = FAR[a] 
    a = [[FRR[1]-FRR[0], FAR[0]-FAR[1]], [-1, 1]]
    b = [(FRR[1]-FRR[0])*FAR[0]-(FAR[1]-FAR[0])*FRR[0], 0]
    return numpy.linalg.solve(a, b)

def scoreScatter(gen, forg):
    ax = plt.subplot()
    ax.scatter(gen[:,0],gen[:,1], color='r')
    ax.scatter(forg[:,0],forg[:,1], color='k', marker="*")
    # ax.set_xlim((0.0, 2.5))
    # ax.set_ylim((0.0, 2.5))
    ax.set_xlabel("score$_{min}$", fontsize=20)
    ax.set_ylabel("score$_{ave}$", fontsize=20)
    k = (numpy.sum(gen[:,1] / gen[:,0]) + numpy.sum(forg[:,1] / forg[:,0])) / (forg.shape[0] + gen.shape[0])
    x = numpy.linspace(0, 0.5, 500)  
    y = -x / k + 0.5
    plt.plot(x, y)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("RAN$_r$ verification scores", fontsize=20)
    plt.show()

if n_train_g == 4:
    LOOP = False
elif n_train_g == 1:
    LOOP = True
else:
    raise ValueError
print("LOOP:", LOOP)

EER_all = []

n_users = 35
n_test_g = 8 - 4
n_test_f = 6
# n_test_f = n_users - 1
N_TEST_G = n_users * n_test_g
N_TEST_F = n_users * n_test_f

ROC_FAR = 0
ROC_FRR = 0
TOTAL_P = 0
TOTAL_N = 0

EER_G = []; EER_L = []
print("For eBS DS1 w4, finger inputs:")
for seed in [111,222,333,444,555]:
    DIST_P = numpy.load("log/seed%d/ebio1_finger/dtw_dist_p%s_d0.npy"%(seed, epoch))
    DIST_N = numpy.load("log/seed%d/ebio1_finger/dtw_dist_n%s_d0.npy"%(seed, epoch))
    DIST_TEMP = numpy.load("log/seed%d/ebio1_finger/dtw_dist_temp%s_d0.npy"%(seed, epoch))
    if LOOP:
        for i in range(4):
            datum_p = []
            datum_n = []
            EERs = []
            for ii in range(n_users):   
                dmax_p = numpy.max(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
                dmin_p = numpy.min(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
                dmean_p = numpy.mean(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 

                dmax_n = numpy.max(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 
                dmin_n = numpy.min(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 
                dmean_n = numpy.mean(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1)

                datum_p.append(numpy.concatenate((dmax_p[:,None], dmin_p[:,None], dmean_p[:,None]), axis=1) / 50.) 
                datum_n.append(numpy.concatenate((dmax_n[:,None], dmin_n[:,None], dmean_n[:,None]), axis=1) / 50.) 

            datum_p = numpy.concatenate(datum_p, axis=0)
            datum_n = numpy.concatenate(datum_n, axis=0)

            for ii in range(n_users):    
                k = 1 #Simply set to 1.
                c = numpy.arange(0, 50, 0.01)[None,:]
                FRR = 1. - numpy.sum(numpy.sum(datum_p[ii*n_test_g:(ii+1)*n_test_g,1:] * [1, 1/k], axis=1)[:,None] - c <= 0, axis=0) / float(n_test_g)
                FAR = 1. - numpy.sum(numpy.sum(datum_n[ii*n_test_f:(ii+1)*n_test_f,1:] * [1, 1/k], axis=1)[:,None] - c >= 0, axis=0) / float(n_test_f)
                EERs.append(getEER(FAR, FRR)[0] * 100)
            EER_L.append(numpy.mean(EERs))

            k = 1 #Simply set to 1.
            c = numpy.arange(0, 50, 0.01)[None,:]
            FRR = 1. - numpy.sum(numpy.sum(datum_p[:, 1:] * [1, 1/k], axis=1)[:,None] - c <= 0, axis=0) / float(datum_p.shape[0])
            FAR = 1. - numpy.sum(numpy.sum(datum_n[:, 1:] * [1, 1/k], axis=1)[:,None] - c >= 0, axis=0) / float(datum_n.shape[0])
            EER_G.append(getEER(FAR, FRR)[0] * 100)

            ROC_FAR += FAR * 0.2 * 0.25 * datum_n.shape[0]
            ROC_FRR += FRR * 0.2 * 0.25 * datum_p.shape[0]
    else:
        datum_p = []
        datum_n = []
        EERs = []
        for ii in range(n_users):   
            idx, dtmp, dmax, dmin, dmean = selectTemplate(DIST_TEMP[ii*n_train_g:(ii+1)*n_train_g,0:n_train_g])
            
            dmax_p = numpy.max(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:n_train_g], axis=1) / dmax 
            dmin_p = numpy.min(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:n_train_g], axis=1) / dmin 
            dmean_p = numpy.mean(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:n_train_g], axis=1) / dmean 

            dmax_n = numpy.max(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:n_train_g], axis=1) / dmax 
            dmin_n = numpy.min(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:n_train_g], axis=1) / dmin 
            dmean_n = numpy.mean(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:n_train_g], axis=1) / dmean 

            datum_p.append(numpy.concatenate((dmax_p[:,None], dmin_p[:,None], dmean_p[:,None]), axis=1) / 10.) 
            datum_n.append(numpy.concatenate((dmax_n[:,None], dmin_n[:,None], dmean_n[:,None]), axis=1) / 10.) 

        datum_p = numpy.concatenate(datum_p, axis=0)
        datum_n = numpy.concatenate(datum_n, axis=0)

        for ii in range(n_users):    
            k = 1 #Simply set to 1.
            c = numpy.arange(0, 50, 0.01)[None,:]
            FRR = 1. - numpy.sum(numpy.sum(datum_p[ii*n_test_g:(ii+1)*n_test_g,1:] * [1, 1/k], axis=1)[:,None] - c <= 0, axis=0) / float(n_test_g)
            FAR = 1. - numpy.sum(numpy.sum(datum_n[ii*n_test_f:(ii+1)*n_test_f,1:] * [1, 1/k], axis=1)[:,None] - c >= 0, axis=0) / float(n_test_f)
            EERs.append(getEER(FAR, FRR)[0] * 100)
        EER_L.append(numpy.mean(EERs))

        # scoreScatter(datum_p, datum_n)
        k = 1. #Simply set to 1.
        c = numpy.arange(0, 50, 0.01)[None,:]
        FRR = 1. - numpy.sum(numpy.sum(datum_p[:, 1:] * [1, 1/k], axis=1)[:,None] - c <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - numpy.sum(numpy.sum(datum_n[:, 1:] * [1, 1/k], axis=1)[:,None] - c >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 0.2 * datum_n.shape[0]
        ROC_FRR += FRR * 0.2 * datum_p.shape[0]

print ("Global threshold:", numpy.mean(EER_G), "Local threshold:", numpy.mean(EER_L))
TOTAL_P += datum_p.shape[0] 
TOTAL_N += datum_n.shape[0] 

EER_G = []; EER_L = []
print("For eBS DS1 w5, finger inputs:")
for seed in [111,222,333,444,555]:
    DIST_P = numpy.load("log/seed%d/ebio1_finger/dtw_dist_p%s_d1.npy"%(seed, epoch))
    DIST_N = numpy.load("log/seed%d/ebio1_finger/dtw_dist_n%s_d1.npy"%(seed, epoch))
    DIST_TEMP = numpy.load("log/seed%d/ebio1_finger/dtw_dist_temp%s_d1.npy"%(seed, epoch))
    if LOOP:
        for i in range(4):
            datum_p = []
            datum_n = []
            EERs = []
            for ii in range(n_users):   
                dmax_p = numpy.max(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
                dmin_p = numpy.min(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
                dmean_p = numpy.mean(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 

                dmax_n = numpy.max(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 
                dmin_n = numpy.min(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 
                dmean_n = numpy.mean(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 

                datum_p.append(numpy.concatenate((dmax_p[:,None], dmin_p[:,None], dmean_p[:,None]), axis=1) / 50.) 
                datum_n.append(numpy.concatenate((dmax_n[:,None], dmin_n[:,None], dmean_n[:,None]), axis=1) / 50.) 

            datum_p = numpy.concatenate(datum_p, axis=0)
            datum_n = numpy.concatenate(datum_n, axis=0)

            for ii in range(n_users):    
                k = 1 #Simply set to 1.
                c = numpy.arange(0, 50, 0.01)[None,:]
                FRR = 1. - numpy.sum(numpy.sum(datum_p[ii*n_test_g:(ii+1)*n_test_g,1:] * [1, 1/k], axis=1)[:,None] - c <= 0, axis=0) / float(n_test_g)
                FAR = 1. - numpy.sum(numpy.sum(datum_n[ii*n_test_f:(ii+1)*n_test_f,1:] * [1, 1/k], axis=1)[:,None] - c >= 0, axis=0) / float(n_test_f)
                EERs.append(getEER(FAR, FRR)[0] * 100)
            EER_L.append(numpy.mean(EERs))

            k = 1 #Simply set to 1.
            c = numpy.arange(0, 50, 0.01)[None,:]
            FRR = 1. - numpy.sum(numpy.sum(datum_p[:, 1:] * [1, 1/k], axis=1)[:,None] - c <= 0, axis=0) / float(datum_p.shape[0])
            FAR = 1. - numpy.sum(numpy.sum(datum_n[:, 1:] * [1, 1/k], axis=1)[:,None] - c >= 0, axis=0) / float(datum_n.shape[0])
            EER_G.append(getEER(FAR, FRR)[0] * 100)

            ROC_FAR += FAR * 0.2 * 0.25 * datum_n.shape[0]
            ROC_FRR += FRR * 0.2 * 0.25 * datum_p.shape[0]
    else:
        datum_p = []
        datum_n = []
        EERs = []
        for ii in range(n_users):   
            idx, dtmp, dmax, dmin, dmean = selectTemplate(DIST_TEMP[ii*n_train_g:(ii+1)*n_train_g,0:n_train_g])
            
            dmax_p = numpy.max(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:n_train_g], axis=1) / dmax 
            dmin_p = numpy.min(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:n_train_g], axis=1) / dmin 
            dmean_p = numpy.mean(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:n_train_g], axis=1) / dmean 
           
            dmax_n = numpy.max(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:n_train_g], axis=1) / dmax 
            dmin_n = numpy.min(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:n_train_g], axis=1) / dmin 
            dmean_n = numpy.mean(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:n_train_g], axis=1) / dmean

            datum_p.append(numpy.concatenate((dmax_p[:,None], dmin_p[:,None], dmean_p[:,None]), axis=1) / 10.) 
            datum_n.append(numpy.concatenate((dmax_n[:,None], dmin_n[:,None], dmean_n[:,None]), axis=1) / 10.) 

        datum_p = numpy.concatenate(datum_p, axis=0)
        datum_n = numpy.concatenate(datum_n, axis=0)

        for ii in range(n_users):    
            k = 1 #Simply set to 1.
            c = numpy.arange(0, 50, 0.01)[None,:]
            FRR = 1. - numpy.sum(numpy.sum(datum_p[ii*n_test_g:(ii+1)*n_test_g,1:] * [1, 1/k], axis=1)[:,None] - c <= 0, axis=0) / float(n_test_g)
            FAR = 1. - numpy.sum(numpy.sum(datum_n[ii*n_test_f:(ii+1)*n_test_f,1:] * [1, 1/k], axis=1)[:,None] - c >= 0, axis=0) / float(n_test_f)
            EERs.append(getEER(FAR, FRR)[0] * 100)
        EER_L.append(numpy.mean(EERs))

        # scoreScatter(datum_p, datum_n)
        k = 1 #Simply set to 1.
        c = numpy.arange(0, 50, 0.01)[None,:]
        FRR = 1. - numpy.sum(numpy.sum(datum_p[:, 1:] * [1, 1/k], axis=1)[:,None] - c <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - numpy.sum(numpy.sum(datum_n[:, 1:] * [1, 1/k], axis=1)[:,None] - c >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 0.2 * datum_n.shape[0]
        ROC_FRR += FRR * 0.2 * datum_p.shape[0]

print ("Global threshold:", numpy.mean(EER_G), "Local threshold:", numpy.mean(EER_L))
TOTAL_P += datum_p.shape[0] 
TOTAL_N += datum_n.shape[0] 

EER_G = []; EER_L = []
print("For eBS DS2 w5, finger inputs:")
for seed in [111,222,333,444,555]:
    DIST_P = numpy.load("log/seed%d/ebio2_finger/dtw_dist_p%s_d0.npy"%(seed, epoch))
    DIST_N = numpy.load("log/seed%d/ebio2_finger/dtw_dist_n%s_d0.npy"%(seed, epoch))
    DIST_TEMP = numpy.load("log/seed%d/ebio2_finger/dtw_dist_temp%s_d0.npy"%(seed, epoch))
    if LOOP:
        for i in range(4):
            datum_p = []
            datum_n = []
            EERs = []
            for ii in range(n_users):   
                dmax_p = numpy.max(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
                dmin_p = numpy.min(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
                dmean_p = numpy.mean(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 

                dmax_n = numpy.max(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 
                dmin_n = numpy.min(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 
                dmean_n = numpy.mean(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 

                datum_p.append(numpy.concatenate((dmax_p[:,None], dmin_p[:,None], dmean_p[:,None]), axis=1) / 50.) 
                datum_n.append(numpy.concatenate((dmax_n[:,None], dmin_n[:,None], dmean_n[:,None]), axis=1) / 50.) 

            datum_p = numpy.concatenate(datum_p, axis=0)
            datum_n = numpy.concatenate(datum_n, axis=0)

            for ii in range(n_users):    
                k = 1 #Simply set to 1.
                c = numpy.arange(0, 50, 0.01)[None,:]
                FRR = 1. - numpy.sum(numpy.sum(datum_p[ii*n_test_g:(ii+1)*n_test_g,1:] * [1, 1/k], axis=1)[:,None] - c <= 0, axis=0) / float(n_test_g)
                FAR = 1. - numpy.sum(numpy.sum(datum_n[ii*n_test_f:(ii+1)*n_test_f,1:] * [1, 1/k], axis=1)[:,None] - c >= 0, axis=0) / float(n_test_f)
                EERs.append(getEER(FAR, FRR)[0] * 100)
            EER_L.append(numpy.mean(EERs))

            k = 1 #Simply set to 1.
            c = numpy.arange(0, 50, 0.01)[None,:]
            FRR = 1. - numpy.sum(numpy.sum(datum_p[:, 1:] * [1, 1/k], axis=1)[:,None] - c <= 0, axis=0) / float(datum_p.shape[0])
            FAR = 1. - numpy.sum(numpy.sum(datum_n[:, 1:] * [1, 1/k], axis=1)[:,None] - c >= 0, axis=0) / float(datum_n.shape[0])
            EER_G.append(getEER(FAR, FRR)[0] * 100)

            ROC_FAR += FAR * 0.2 * 0.25 * datum_n.shape[0]
            ROC_FRR += FRR * 0.2 * 0.25 * datum_p.shape[0]
    else:
        datum_p = []
        datum_n = []
        EERs = []
        for ii in range(n_users):   
            idx, dtmp, dmax, dmin, dmean = selectTemplate(DIST_TEMP[ii*n_train_g:(ii+1)*n_train_g,0:n_train_g])
            
            dmax_p = numpy.max(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:n_train_g], axis=1) / dmax 
            dmin_p = numpy.min(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:n_train_g], axis=1) / dmin 
            dmean_p = numpy.mean(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:n_train_g], axis=1) / dmean 

            dmax_n = numpy.max(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:n_train_g], axis=1) / dmax 
            dmin_n = numpy.min(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:n_train_g], axis=1) / dmin 
            dmean_n = numpy.mean(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:n_train_g], axis=1) / dmean 

            datum_p.append(numpy.concatenate((dmax_p[:,None], dmin_p[:,None], dmean_p[:,None]), axis=1) / 10.) 
            datum_n.append(numpy.concatenate((dmax_n[:,None], dmin_n[:,None], dmean_n[:,None]), axis=1) / 10.) 

        datum_p = numpy.concatenate(datum_p, axis=0)
        datum_n = numpy.concatenate(datum_n, axis=0)

        for ii in range(n_users):    
            k = 1 #Simply set to 1.
            c = numpy.arange(0, 50, 0.01)[None,:]
            FRR = 1. - numpy.sum(numpy.sum(datum_p[ii*n_test_g:(ii+1)*n_test_g,1:] * [1, 1/k], axis=1)[:,None] - c <= 0, axis=0) / float(n_test_g)
            FAR = 1. - numpy.sum(numpy.sum(datum_n[ii*n_test_f:(ii+1)*n_test_f,1:] * [1, 1/k], axis=1)[:,None] - c >= 0, axis=0) / float(n_test_f)
            EERs.append(getEER(FAR, FRR)[0] * 100)
        EER_L.append(numpy.mean(EERs))

        # scoreScatter(datum_p, datum_n)
        k = 1 #Simply set to 1.
        c = numpy.arange(0, 50, 0.01)[None,:]
        FRR = 1. - numpy.sum(numpy.sum(datum_p[:, 1:] * [1, 1/k], axis=1)[:,None] - c <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - numpy.sum(numpy.sum(datum_n[:, 1:] * [1, 1/k], axis=1)[:,None] - c >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 0.2 * datum_n.shape[0]
        ROC_FRR += FRR * 0.2 * datum_p.shape[0]

print ("Global threshold:", numpy.mean(EER_G), "Local threshold:", numpy.mean(EER_L))
TOTAL_P += datum_p.shape[0] 
TOTAL_N += datum_n.shape[0] 

EER_G = []; EER_L = []
print("For eBS DS2 w6, finger inputs:")
for seed in [111,222,333,444,555]:
    DIST_P = numpy.load("log/seed%d/ebio2_finger/dtw_dist_p%s_d1.npy"%(seed, epoch))
    DIST_N = numpy.load("log/seed%d/ebio2_finger/dtw_dist_n%s_d1.npy"%(seed, epoch))
    DIST_TEMP = numpy.load("log/seed%d/ebio2_finger/dtw_dist_temp%s_d1.npy"%(seed, epoch))
    if LOOP:
        for i in range(4):
            datum_p = []
            datum_n = []
            EERs = []
            for ii in range(n_users):   
                dmax_p = numpy.max(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
                dmin_p = numpy.min(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
                dmean_p = numpy.mean(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 

                dmax_n = numpy.max(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 
                dmin_n = numpy.min(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 
                dmean_n = numpy.mean(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 

                datum_p.append(numpy.concatenate((dmax_p[:,None], dmin_p[:,None], dmean_p[:,None]), axis=1) / 50.) 
                datum_n.append(numpy.concatenate((dmax_n[:,None], dmin_n[:,None], dmean_n[:,None]), axis=1) / 50.) 

            datum_p = numpy.concatenate(datum_p, axis=0)
            datum_n = numpy.concatenate(datum_n, axis=0)

            for ii in range(n_users):    
                k = 1 #Simply set to 1.
                c = numpy.arange(0, 50, 0.01)[None,:]
                FRR = 1. - numpy.sum(numpy.sum(datum_p[ii*n_test_g:(ii+1)*n_test_g,1:] * [1, 1/k], axis=1)[:,None] - c <= 0, axis=0) / float(n_test_g)
                FAR = 1. - numpy.sum(numpy.sum(datum_n[ii*n_test_f:(ii+1)*n_test_f,1:] * [1, 1/k], axis=1)[:,None] - c >= 0, axis=0) / float(n_test_f)
                EERs.append(getEER(FAR, FRR)[0] * 100)
            EER_L.append(numpy.mean(EERs))

            k = 1 #Simply set to 1.
            c = numpy.arange(0, 50, 0.01)[None,:]
            FRR = 1. - numpy.sum(numpy.sum(datum_p[:, 1:] * [1, 1/k], axis=1)[:,None] - c <= 0, axis=0) / float(datum_p.shape[0])
            FAR = 1. - numpy.sum(numpy.sum(datum_n[:, 1:] * [1, 1/k], axis=1)[:,None] - c >= 0, axis=0) / float(datum_n.shape[0])
            EER_G.append(getEER(FAR, FRR)[0] * 100)

            ROC_FAR += FAR * 0.2 * 0.25 * datum_n.shape[0]
            ROC_FRR += FRR * 0.2 * 0.25 * datum_p.shape[0]
    else:
        datum_p = []
        datum_n = []
        EERs = []
        for ii in range(n_users):   
            idx, dtmp, dmax, dmin, dmean = selectTemplate(DIST_TEMP[ii*n_train_g:(ii+1)*n_train_g,0:n_train_g])
            
            dmax_p = numpy.max(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:n_train_g], axis=1) / dmax 
            dmin_p = numpy.min(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:n_train_g], axis=1) / dmin 
            dmean_p = numpy.mean(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:n_train_g], axis=1) / dmean 

            dmax_n = numpy.max(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:n_train_g], axis=1) / dmax 
            dmin_n = numpy.min(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:n_train_g], axis=1) / dmin 
            dmean_n = numpy.mean(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:n_train_g], axis=1) / dmean 

            datum_p.append(numpy.concatenate((dmax_p[:,None], dmin_p[:,None], dmean_p[:,None]), axis=1) / 10.) 
            datum_n.append(numpy.concatenate((dmax_n[:,None], dmin_n[:,None], dmean_n[:,None]), axis=1) / 10.) 

        datum_p = numpy.concatenate(datum_p, axis=0)
        datum_n = numpy.concatenate(datum_n, axis=0)

        for ii in range(n_users):    
            k = 1 #Simply set to 1.
            c = numpy.arange(0, 50, 0.01)[None,:]
            FRR = 1. - numpy.sum(numpy.sum(datum_p[ii*n_test_g:(ii+1)*n_test_g,1:] * [1, 1/k], axis=1)[:,None] - c <= 0, axis=0) / float(n_test_g)
            FAR = 1. - numpy.sum(numpy.sum(datum_n[ii*n_test_f:(ii+1)*n_test_f,1:] * [1, 1/k], axis=1)[:,None] - c >= 0, axis=0) / float(n_test_f)
            EERs.append(getEER(FAR, FRR)[0] * 100)
        EER_L.append(numpy.mean(EERs))

        # scoreScatter(datum_p, datum_n)
        k = 1 #Simply set to 1.
        c = numpy.arange(0, 50, 0.01)[None,:]
        FRR = 1. - numpy.sum(numpy.sum(datum_p[:, 1:] * [1, 1/k], axis=1)[:,None] - c <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - numpy.sum(numpy.sum(datum_n[:, 1:] * [1, 1/k], axis=1)[:,None] - c >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 0.2 * datum_n.shape[0]
        ROC_FRR += FRR * 0.2 * datum_p.shape[0]

print ("Global threshold:", numpy.mean(EER_G), "Local threshold:", numpy.mean(EER_L))
TOTAL_P += datum_p.shape[0]
TOTAL_N += datum_n.shape[0]

final_global = getEER(ROC_FAR*1.0/TOTAL_N, ROC_FRR*1.0/TOTAL_P)[0] * 100
print("Overall EER under global threshold:", final_global)
