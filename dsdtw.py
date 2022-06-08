#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy, pdb
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nutils
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from soft_dtw_cuda import SoftDTW
from dtw_cuda import DTW

class DSDTW(nn.Module):
    def __init__(self,
                n_in,
                n_layers=2,
                n_hidden=128, 
                n_out=64, 
                n_shot_g=5, 
                n_shot_f=5, 
                n_task=1,
                batchsize=None,
                alpha=None):
        super(DSDTW, self).__init__() 
        ''' Define the network and the training loss. '''
        self.n_shot_g = n_shot_g 
        self.n_shot_f = n_shot_f
        self.n_task = n_task
        self.smoothCElossMask = torch.zeros(n_task * (1 + n_shot_g + n_shot_f)).cuda()
        for i in range(n_task):
            self.smoothCElossMask[i*(1+n_shot_g+n_shot_f):i*(1+n_shot_g+n_shot_f)+1+n_shot_g]=(1.0+n_shot_g+n_shot_f)/(1.0+n_shot_g)
        if batchsize is None:
            batchsize = (n_shot_g + n_shot_f + 1) * n_task
        self.conv = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=7, stride=1, padding=3, bias=True),
                    nn.MaxPool1d(2, 2, ceil_mode=True),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(n_out, n_hidden, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.1),
                )

        self.rnn = nn.GRU(n_hidden, n_hidden, n_layers, dropout=0.1, batch_first=True, bidirectional=False) #(input_size,hidden_size,num_layers)
        self.h0 = Variable(torch.zeros(n_layers, batchsize, n_hidden).cuda(), requires_grad=False)
        ## close update gate
        for i in range(n_layers):
            eval("self.rnn.bias_hh_l%d"%i)[n_hidden:2*n_hidden].data.fill_(-1e10) #Initial update gate bias
            eval("self.rnn.bias_ih_l%d"%i)[n_hidden:2*n_hidden].data.fill_(-1e10) #Initial update gate bias
        
        self.linear = nn.Linear(n_hidden, n_out, bias=False)
        
        nn.init.kaiming_normal_(self.linear.weight, a=1) 
        nn.init.kaiming_normal_(self.conv[0].weight, a=0)
        nn.init.kaiming_normal_(self.conv[3].weight, a=0)
        nn.init.zeros_(self.conv[0].bias)
        nn.init.zeros_(self.conv[3].bias)

        ''' soft-DTW and DTW: When gamma=0, soft-DTW becomes DTW.'''
        self.dtw = SoftDTW(True, gamma=5, normalize=False, bandwidth=0.1)
        # self.dtw = DTW(True, normalize=False, bandwidth=0.1)

    def getOutputMask(self, lens):    
        lens = numpy.array(lens, dtype=numpy.int32)
        lens = (lens + 1) // 2
        N = len(lens); D = numpy.max(lens)
        mask = numpy.zeros((N, D), dtype=numpy.float32)
        for i in range(N):
            mask[i, 0:lens[i]] = 1.0
        return mask

    def forward(self, x, mask):
        length = torch.sum(mask, dim=1)
        length, indices = torch.sort(length, descending=True)
        x = torch.index_select(x, 0, indices)
        mask = torch.index_select(mask, 0, indices)

        '''CNN'''
        output = x.transpose(1,2) #(N,D,T)
        output = self.conv(output)
        output = output.transpose(1,2) #(N,T,D)
        output = output * mask.unsqueeze(2)

        '''RANs'''
        output = nutils.rnn.pack_padded_sequence(output, list(length.cpu().numpy()), batch_first=True)
        output, hidden = self.rnn(output, self.h0)
        output, length = nutils.rnn.pad_packed_sequence(output, batch_first=True) 
        length = Variable(length).cuda()

        '''Recover the original order'''
        _, indices = torch.sort(indices, descending=False)
        output = torch.index_select(output, 0, indices)
        length = torch.index_select(length, 0, indices)
        mask = torch.index_select(mask, 0, indices)


        if self.training:            
            length = (length//2).float()
            output = self.linear(output)
            '''Average Pooling'''
            output = F.avg_pool1d(output.permute(0,2,1),2,2,ceil_mode=False).permute(0,2,1)
            '''Max Pooling'''
            # output = F.max_pool1d(output.permute(0,2,1),2,2,ceil_mode=False).permute(0,2,1)
        else:   
            length = length.float()
            output = self.linear(output)
            output = output * mask.unsqueeze(2)

        return output, length, hidden

    def EuclideanDistances(self,a,b):
        sq_a = a**2
        sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)
        sq_b = b**2
        sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)
        bt = b.t()
        return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))

    def tripletLoss(self, x, length, margin=1.):
        Ng = self.n_shot_g
        Nf = self.n_shot_f
        Nt = self.n_task
        step = 1 + Ng + Nf
        var = triLoss_std = triLoss_hard = 0
        for i in range(Nt):
            anchor = x[i*step]
            pos = x[i*step+1:i*step+1+Ng]
            neg = x[i*step+1+Ng:(i+1)*step]
            len_a = length[i*step]
            len_p = length[i*step+1:i*step+1+Ng]
            len_n = length[i*step+1+Ng:(i+1)*step]
            dist_g = torch.zeros((len(pos)), dtype=x.dtype, device=x.device)
            dist_n = torch.zeros((len(neg)), dtype=x.dtype, device=x.device)

            '''Average_Pooling_2,4,6'''
            for i in range(len(pos)):
                dist_g[i] = self.dtw(anchor[None, :int(len_a)], pos[i:i+1, :int(len_p[i])]) / (len_a + len_p[i])
            for i in range(len(neg)):
                dist_n[i] = self.dtw(anchor[None, :int(len_a)], neg[i:i+1, :int(len_n[i])]) / (len_a + len_n[i])
            
            '''Average_Pooling_infinite:
            The soft-DTW distance degenerates to the squareEuclidean distance.'''
            # for i in range(len(pos)):
            #     dist_g[i] = self.EuclideanDistances(torch.mean(anchor[None, :int(len_a)],dim=1), torch.mean(pos[i:i+1, :int(len_p[i])],dim=1))/2 #/ (len_a + len_p[i])
            # for i in range(len(neg)):
            #     dist_n[i] = self.EuclideanDistances(torch.mean(anchor[None, :int(len_a)],dim=1), torch.mean(neg[i:i+1, :int(len_n[i])],dim=1))/2 #/ (len_a + len_n[i])

            '''Inner class variation'''
            var += torch.sum(dist_g) / Ng
            '''Triplet loss, Ng * Nf triplets in total'''
            triLoss = F.relu(dist_g.unsqueeze(1) - dist_n.unsqueeze(0) + margin) #(Ng, Nf)
            triLoss_std += torch.mean(triLoss) 
            triLoss_hard += torch.sum(triLoss) / (triLoss.data.nonzero(as_tuple=False).size(0) + 1) 
        var = var / Nt
        triLoss_std = triLoss_std / Nt
        triLoss_hard = triLoss_hard / Nt
        return [triLoss_std, triLoss_hard, var]

    