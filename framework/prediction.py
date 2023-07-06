# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionLayer(nn.Module):
    def __init__(self, opt):
        super(PredictionLayer, self).__init__()
        self.output = opt.output
        if opt.output == "lfm":
            self.model = LFM(opt.feature_dim, opt.user_num, opt.item_num)

    def forward(self, feature, uid, iid):
        if self.output == "lfm":
            return self.model(feature, uid, iid)


class LFM(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(LFM, self).__init__()


        self.fc = nn.Linear(dim, 1)

        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc.bias, a=0.5, b=1.5)
        nn.init.uniform_(self.b_users, a=0.5, b=1.5)

    def rescale_sigmoid(self, score, a, b):
        return a + torch.sigmoid(score) * (b - a)

    def forward(self, feature, user_id, item_id):
        return self.fc(feature) + self.b_users[user_id] + self.b_items[item_id]