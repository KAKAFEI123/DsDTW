#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy, pdb
from fastdtw import fastdtw as dtw #https://github.com/slaypni/fastdtw/issues
from matplotlib import pyplot as plt
from soft_dtw_cuda import SoftDTW
import torch

def dist_seq(FEAT_SEQ, ng, nf, num_g, num_f):
    DIST_P = []
    DIST_N = []
    DIST_TEMP = []
    print("Calculating DTW distance...")
    for idx, feat_seq in enumerate(FEAT_SEQ):
        # print ("DTW %d"%(idx))
        feat_a = feat_seq[0:ng]
        feat_p = feat_seq[(ng+nf):(num_g+nf)]
        feat_n = feat_seq[(num_g+nf):]
            
        dist_p = numpy.zeros((num_g-ng, ng))
        dist_n = numpy.zeros((num_f, ng))
        dist_temp = numpy.zeros((ng, ng))
        
        for i in range(ng):
            fp = feat_a[i]
            fps = numpy.sum(fp, axis=1)
            fp = numpy.delete(fp, numpy.where(fps == 0)[0], axis=0) 
            for j in range(i+1, ng): 
                fa = feat_a[j]
                fas = numpy.sum(fa, axis=1)
                fa = numpy.delete(fa, numpy.where(fas == 0)[0], axis=0) 
                dist, path = dtw(fp, fa, radius=2, dist=1)
                dist = dist / (fp.shape[0] + fa.shape[0])
                dist_temp[i, j] = dist
        for i in range(num_g-ng):
            fp = feat_p[i]
            fps = numpy.sum(fp, axis=1)
            fp = numpy.delete(fp, numpy.where(fps == 0)[0], axis=0) 
            for j in range(ng):
                fa = feat_a[j]
                fas = numpy.sum(fa, axis=1)
                fa = numpy.delete(fa, numpy.where(fas == 0)[0], axis=0) 
                dist, path = dtw(fp, fa, radius=2, dist=1)
                dist = dist / (fp.shape[0] + fa.shape[0])
                dist_p[i, j] = dist
        for i in range(num_f):
            fn = feat_n[i]
            fns = numpy.sum(fn, axis=1)
            fn = numpy.delete(fn, numpy.where(fns == 0)[0], axis=0) 
            for j in range(ng):
                fa = feat_a[j]
                fas = numpy.sum(fa, axis=1)
                fa = numpy.delete(fa, numpy.where(fas == 0)[0], axis=0) 
                dist, path = dtw(fn, fa, radius=2, dist=1)
                dist = dist / (fn.shape[0] + fa.shape[0])
                dist_n[i, j] = dist
        DIST_P.append(dist_p)
        DIST_N.append(dist_n)
        DIST_TEMP.append(dist_temp)
    DIST_P = numpy.concatenate(DIST_P, axis=0)
    DIST_N = numpy.concatenate(DIST_N, axis=0)
    DIST_TEMP = numpy.concatenate(DIST_TEMP, axis=0)

    return DIST_P, DIST_N, DIST_TEMP

def dist_seq_rf(FEAT_SEQ, ng, nf, num_g, num_f):
    DIST_P = []
    DIST_N = []
    DIST_TEMP = []
    FEAT_A = []
    FEAT_P = []
    for idx, feat_seq in enumerate(FEAT_SEQ):
        feat_a = feat_seq[0:ng]
        feat_p = feat_seq[(ng+nf):(num_g+nf)]
        FEAT_A.append(feat_a)
        FEAT_P.append(feat_p)
    del FEAT_SEQ
    print("Calculating DTW distance...")
    for idx, feat_a in enumerate(FEAT_A):
        # print ("DTW %d"%(idx))
        feat_p = FEAT_P[idx]
        feat_n = []
        for i in range(len(FEAT_A)):
            if i!=idx:
                feat_n.append(FEAT_P[i][2])
                # feat_n.append(FEAT_A[i][0])
        dist_p = numpy.zeros((feat_p.shape[0], ng))
        dist_n = numpy.zeros((len(feat_n), ng))
        dist_temp = numpy.zeros((ng, ng))
        for i in range(ng):
            fp = feat_a[i]
            fps = numpy.sum(fp, axis=1)
            fp = numpy.delete(fp, numpy.where(fps == 0)[0], axis=0) 
            for j in range(i+1, ng): 
                fa = feat_a[j]
                fas = numpy.sum(fa, axis=1)
                fa = numpy.delete(fa, numpy.where(fas == 0)[0], axis=0) 
                dist, path = dtw(fp, fa, radius=2, dist=1) 
                dist = dist / (fp.shape[0] + fa.shape[0])
                dist_temp[i, j] = dist
        for i in range(feat_p.shape[0]):
            fp = feat_p[i]
            fps = numpy.sum(fp, axis=1)
            fp = numpy.delete(fp, numpy.where(fps == 0)[0], axis=0) 
            for j in range(ng):
                fa = feat_a[j]
                fas = numpy.sum(fa, axis=1)
                fa = numpy.delete(fa, numpy.where(fas == 0)[0], axis=0) 
                dist, path = dtw(fp, fa, radius=2, dist=1) 
                dist = dist / (fp.shape[0] + fa.shape[0])
                dist_p[i, j] = dist
        for i in range(len(feat_n)):
            fn = feat_n[i]
            fns = numpy.sum(fn, axis=1)
            fn = numpy.delete(fn, numpy.where(fns == 0)[0], axis=0) 
            for j in range(ng):
                fa = feat_a[j]
                fas = numpy.sum(fa, axis=1)
                fa = numpy.delete(fa, numpy.where(fas == 0)[0], axis=0) 
                dist, path = dtw(fn, fa, radius=2, dist=1) 
                dist = dist / (fn.shape[0] + fa.shape[0])
                dist_n[i, j] = dist        
        DIST_P.append(dist_p)
        DIST_N.append(dist_n)
        DIST_TEMP.append(dist_temp)
    DIST_P = numpy.concatenate(DIST_P, axis=0)
    DIST_N = numpy.concatenate(DIST_N, axis=0)
    DIST_TEMP = numpy.concatenate(DIST_TEMP, axis=0)
    return DIST_P, DIST_N, DIST_TEMP
