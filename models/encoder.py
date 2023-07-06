# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN(nn.Module):
    def __init__(self, filters_num, k1, k2, padding=True):
        super(CNN, self).__init__()

        if padding:
            self.cnn = nn.Conv2d(1, filters_num, (k1, k2), padding=(int(k1 / 2), 0))
        else:
            self.cnn = nn.Conv2d(1, filters_num, (k1, k2))

    def multi_attention_pooling(self, x, qv):
        att_weight = torch.matmul(qv.unsqueeze(1), x.permute(0,1,3,2))
        att_score = F.softmax(att_weight, dim=3)

        x = torch.matmul(att_score, x)
        x = torch.sum(x,dim=1)
        return x

    def attention_pooling(self, x, qv):
        att_weight = torch.matmul(x, qv)
        att_score = F.softmax(att_weight, dim=1)
        x = x * att_score

        return x.sum(1)

    def forward(self, x, max_num, review_len):
        x = x.view(-1, review_len, self.cnn.kernel_size[1])
        x = x.unsqueeze(1)
        x = F.relu(self.cnn(x)).squeeze(3)

        if max_num == 1:
            x = x.permute(0, 2, 1)
        else:
            x = x.view(-1, max_num, self.cnn.out_channels, review_len)
            x = x.permute(0, 1, 3, 2)
        return x
