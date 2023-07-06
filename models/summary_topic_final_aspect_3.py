# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .encoder import CNN


class summary_topic_final_aspect_3(nn.Module):
    def __init__(self, opt, pointer_num=3):
        super(summary_topic_final_aspect_3, self).__init__()

        self.opt = opt
        self.num_fea = 2
        self.pointer_num = pointer_num
        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.u_max_r = opt.setNum_rev_of_user
        self.i_max_r = opt.setNum_rev_of_item
        self.review_len = opt.setNum_word_of_rev
        self.summary_len = opt.setNum_word_of_summary
        self.fc_g1 = nn.Linear(opt.word_dim, opt.word_dim)
        self.fc_g2 = nn.Linear(opt.word_dim, opt.word_dim)
        self.review_coatt = nn.ModuleList([Co_Attention(opt.word_dim, gumbel=True, pooling='max') for _ in range(pointer_num)])
        self.drop_out = nn.Dropout(opt.summary_model_drop_out)
        user_num = self.opt.user_num
        item_num = self.opt.item_num
        id_emb_size = self.opt.id_emb_size
        att_id_emb_size = self.opt.att_id_emb_size
        self.item_aspect_embs = nn.Embedding(item_num, self.opt.num_aspect * opt.word_dim)
        self.uid_emb = nn.Embedding(user_num, id_emb_size)
        self.iid_emb = nn.Embedding(item_num, id_emb_size)
        self.user_rev_encoder = CNN(self.opt.r_filters_num, self.opt.kernel_size, self.opt.word_dim)
        self.item_rev_encoder = CNN(self.opt.r_filters_num, self.opt.kernel_size, self.opt.word_dim)
        self.u_fc_layer = nn.Linear(self.opt.num_aspect*self.opt.word_dim, self.opt.id_emb_size)
        self.i_fc_layer = nn.Linear(self.opt.num_aspect*self.opt.word_dim, self.opt.id_emb_size)
        self.u_qv_fc_layer = nn.Linear(2 * self.opt.word_dim, self.opt.word_dim)
        self.i_qv_fc_layer = nn.Linear(2 * self.opt.word_dim, self.opt.word_dim)
        self.reset_para()


    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc, user_summaries, item_summaries = datas
        u_rev_word_embs = self.user_word_embs(user_reviews)
        i_rev_word_embs = self.item_word_embs(item_reviews)
        u_rev_word_embs = self.drop_out(u_rev_word_embs)
        i_rev_word_embs = self.drop_out(i_rev_word_embs)
        u_summary_word_embs = self.user_word_embs(user_summaries)
        i_summary_word_embs = self.item_word_embs(item_summaries)
        u_summary_word_embs = self.drop_out(u_summary_word_embs)
        i_summary_word_embs = self.drop_out(i_summary_word_embs)
        u_rev_cnn_word_embs = self.user_rev_encoder(u_rev_word_embs, self.u_max_r, self.review_len)
        i_rev_cnn_word_embs = self.item_rev_encoder(i_rev_word_embs, self.i_max_r, self.review_len)
        u_rev_cnn_word_embs = self.drop_out(u_rev_cnn_word_embs)
        i_rev_cnn_word_embs = self.drop_out(i_rev_cnn_word_embs)
        u_summary_cnn_word_embs = self.user_rev_encoder(u_summary_word_embs, self.u_max_r, self.summary_len)
        i_summary_cnn_word_embs = self.item_rev_encoder(i_summary_word_embs, self.i_max_r, self.summary_len)
        u_summary_cnn_word_embs = self.drop_out(u_summary_cnn_word_embs)
        i_summary_cnn_word_embs = self.drop_out(i_summary_cnn_word_embs)
        u_summary_embs = u_summary_cnn_word_embs.mean(2)
        i_summary_embs = i_summary_cnn_word_embs.mean(2)
        u_rev_cnn_word_embs_view = u_rev_cnn_word_embs.reshape(len(uids), self.u_max_r, self.review_len*self.opt.word_dim)
        i_rev_cnn_word_embs_view = i_rev_cnn_word_embs.reshape(len(iids), self.i_max_r, self.review_len*self.opt.word_dim)
        item_w_aspect = self.item_aspect_embs(iids).view(-1, self.opt.num_aspect, self.opt.word_dim)
        item_w_aspect = self.drop_out(item_w_aspect)

        u_r_fea = []
        i_r_fea = []
        flag_first_review = 0

        for i in range(self.pointer_num):
            r_coatt = self.review_coatt[i]
            p_u, p_i = r_coatt(u_summary_embs, i_summary_embs)
            u_rev_words = p_u.float().bmm(u_rev_cnn_word_embs_view)
            u_words = u_rev_words.view(len(uids), self.review_len, self.opt.word_dim)
            i_rev_words = p_i.float().bmm(i_rev_cnn_word_embs_view)
            i_words = i_rev_words.view(len(iids), self.review_len, self.opt.word_dim)

            u_summary_embedding_one = p_u.float().bmm(u_summary_embs)
            u_summary_embedding_vec = u_summary_embedding_one.expand(len(uids), self.opt.num_aspect, \
                                                                     self.opt.word_dim)
            u_summary_embedding_vec = self.drop_out(u_summary_embedding_vec)
            user_w_aspect_plus_summary = torch.cat([item_w_aspect, u_summary_embedding_vec], dim=2)

            i_summary_embedding_one = p_i.float().bmm(i_summary_embs)
            i_summary_embedding_vec = i_summary_embedding_one.expand(len(iids), self.opt.num_aspect, \
                                                                         self.opt.word_dim)
            i_summary_embedding_vec = self.drop_out(i_summary_embedding_vec)
            item_w_aspect_plus_summary = torch.cat([item_w_aspect, i_summary_embedding_vec], dim=2)

            u_r_fea_one = self.multi_attention_pooling_user(u_words, user_w_aspect_plus_summary)

            i_r_fea_one = self.multi_attention_pooling_item(i_words, item_w_aspect_plus_summary)

            if flag_first_review == 0:
                flag_first_review = 1
                u_r_fea = u_r_fea_one
                i_r_fea = i_r_fea_one
            else:
                u_r_fea = u_r_fea + u_r_fea_one
                i_r_fea = i_r_fea + i_r_fea_one

        u_fea = u_r_fea.view(-1, self.opt.num_aspect*self.opt.word_dim)
        i_fea = i_r_fea.view(-1, self.opt.num_aspect*self.opt.word_dim)

        u_fea_final = torch.stack([self.uid_emb(uids), self.u_fc_layer(u_fea)], 1)
        i_fea_final = torch.stack([self.iid_emb(iids), self.i_fc_layer(i_fea)], 1)

        return u_fea_final, i_fea_final

    def multi_attention_pooling_user(self, x, qv):
        qv_now = self.u_qv_fc_layer(qv)
        att_weight = torch.matmul(qv_now, x.permute(0, 2, 1))
        att_score = F.softmax(att_weight/0.5, dim=2)
        x = torch.matmul(att_score, x)
        return x

    def multi_attention_pooling_item(self, x, qv):
        qv_now = self.i_qv_fc_layer(qv)
        att_weight = torch.matmul(qv_now, x.permute(0, 2, 1))
        att_score = F.softmax(att_weight/0.5, dim=2)
        x = torch.matmul(att_score, x)
        return x

    def reset_para(self):
        for fc in [self.fc_g1, self.fc_g2, self.u_fc_layer, self.i_fc_layer, self.u_qv_fc_layer, self.i_qv_fc_layer]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.uniform_(fc.bias, -0.1, 0.1)

        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.user_word_embs.weight.data.copy_(w2v.cuda())
                self.item_word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.user_word_embs.weight.data.copy_(w2v)
                self.item_word_embs.weight.data.copy_(w2v)
        else:
            nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)
        nn.init.xavier_normal_(self.uid_emb.weight)
        nn.init.xavier_normal_(self.iid_emb.weight)

class Co_Attention(nn.Module):
    def __init__(self, dim, gumbel, pooling):
        super(Co_Attention, self).__init__()
        self.gumbel = gumbel
        self.pooling = pooling
        self.M = nn.Parameter(torch.randn(dim, dim))
        self.fc_u = nn.Linear(dim, dim)
        self.fc_i = nn.Linear(dim, dim)

        self.reset_para()

    def reset_para(self):
        nn.init.xavier_uniform_(self.M, gain=1)
        nn.init.uniform_(self.fc_u.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_u.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc_i.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_i.bias, -0.1, 0.1)

    def forward(self, u_fea, i_fea):
        S = u_fea.matmul(self.M).bmm(i_fea.permute(0, 2, 1))
        if self.pooling == 'max':
            u_score = S.max(2)[0]
            i_score = S.max(1)[0]
        else:
            u_score = S.mean(2)
            i_score = S.mean(1)
        if self.gumbel:
            p_u = F.gumbel_softmax(u_score, hard=True, dim=1)
            p_i = F.gumbel_softmax(i_score, hard=True, dim=1)
        else:
            p_u = F.softmax(u_score, dim=1)
            p_i = F.softmax(i_score, dim=1)
        return p_u.unsqueeze(1), p_i.unsqueeze(1)


