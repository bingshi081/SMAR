# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math

class FusionLayer(nn.Module):
    def __init__(self, opt):
        super(FusionLayer, self).__init__()
        if opt.self_att:
            self.attn = SelfAtt(opt.id_emb_size, opt.num_heads)
        self.opt = opt
        self.linear = nn.Linear(opt.feature_dim, opt.feature_dim)
        self.drop_out = nn.Dropout(0.5)
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.constant_(self.linear.bias, 0.1)

    def forward(self, u_out, i_out):
        if self.opt.self_att:
            out = self.attn(u_out, i_out)
            return out
        if self.opt.r_id_merge == 'cat':
            u_out = u_out.view(u_out.size(0), -1)
            i_out = i_out.view(i_out.size(0), -1)
        else:
            u_out = u_out.sum(1)
            i_out = i_out.sum(1)

        if self.opt.ui_merge == 'cat':
            out = torch.cat([u_out, i_out], 1)
        elif self.opt.ui_merge == 'add':
            out = u_out + i_out
        elif self.opt.ui_merge == 'element dot':
            out = u_out * i_out
        elif self.opt.ui_merge == 'matmul':
            out = torch.bmm(u_out.unsqueeze(1) , i_out.unsqueeze(2)).squeeze(1)

        return out


class SelfAtt(nn.Module):
    def __init__(self, dim, num_heads):
        super(SelfAtt, self).__init__()
        self.dim = dim
        self.Q_linear = nn.Linear(6 * dim, dim)
        self.K_linear = nn.Linear(6 * dim, dim)
        self.V_linear = nn.Linear(6 * dim, dim)

        self.Q1_linear = nn.Linear(6 * dim, dim)
        self.K1_linear = nn.Linear(6 * dim, dim)
        self.V1_linear = nn.Linear(6 * dim, dim)

    def forward(self, user_fea, item_fea):
        fea = torch.cat([user_fea, item_fea], 1).unsqueeze(1)
        Q = self.Q_linear(fea)
        K = self.K_linear(fea)
        V = self.V_linear(fea)

        Q1 = self.Q1_linear(fea)
        K1 = self.K1_linear(fea)
        V1 = self.V1_linear(fea)

        fea0 = 1/math.sqrt(self.dim)*torch.matmul(Q,(K.permute(0,2,1)))*V
        fea1 = 1/math.sqrt(self.dim)*torch.matmul(Q1,(K1.permute(0,2,1)))*V1
        fea_final = fea0.squeeze()+fea1.squeeze()
        return fea_final
