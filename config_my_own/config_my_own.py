# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn

class DefaultConfig:

    model = 'DeepCoNN'
    dataset = 'Digital_Music_data'
    use_gpu = True
    gpu_id = 0
    multi_gpu = False
    gpu_ids = []

    seed = 2019
    num_epochs = 40
    num_workers = 0

    optimizer = 'Adam'
    weight_decay = 1e-3
    lr = 2e-3
    loss_method = 'mse'
    drop_out = 0.85
    summary_model_drop_out = 0.6

    use_word_embedding = True

    num_aspect = 5
    id_emb_size = 32
    att_id_emb_size = 100
    query_mlp_size = 128
    fc_dim = 32

    biaoji = 100
    doc_len = 500
    filters_num = 100
    kernel_size = 3

    num_fea = 1
    use_review = True
    use_doc = True
    self_att = False
    num_heads = 1

    r_filters_num = 100
    kernel_size = 3
    attention_size = 32
    att_method = 'matrix'
    review_weight = 'softmax'


    r_id_merge = 'cat'
    ui_merge = 'cat'
    output = 'fm'

    fine_step = True
    save_path=""
    print_opt = 'default'

    batch_size = 128
    print_step = 100
    have_saved_model = 0
    best_valid_mse = 2.00
    setNum_rev_of_user = -1
    setNum_rev_of_item = -1
    setNum_word_of_rev = -1
    setNum_word_of_summary = -1
    train_data_size = -1
    early_stop_epoch = -1



    def set_path(self, name):
        self.data_root = f'./dataset/{name}'
        prefix = f'{self.data_root}/train'

        self.user_list_path = f'{prefix}/userReview2Index.npy'
        self.item_list_path = f'{prefix}/itemReview2Index.npy'

        self.user2itemid_path = f'{prefix}/user_item2id.npy'
        self.item2userid_path = f'{prefix}/item_user2id.npy'

        self.user_doc_path = f'{prefix}/userDoc2Index.npy'
        self.item_doc_path = f'{prefix}/itemDoc2Index.npy'

        self.user_summary_list_path = f'{prefix}/userSummary2Index.npy'
        self.item_summary_list_path = f'{prefix}/itemSummary2Index.npy'

        self.w2v_path = f'{prefix}/w2v.npy'

    def parse(self, para):
        self.users_review_list = np.load(self.user_list_path, encoding='bytes')
        self.items_review_list = np.load(self.item_list_path, encoding='bytes')
        self.user2itemid_list = np.load(self.user2itemid_path, encoding='bytes')
        self.item2userid_list = np.load(self.item2userid_path, encoding='bytes')
        self.user_doc = np.load(self.user_doc_path, encoding='bytes')
        self.item_doc = np.load(self.item_doc_path, encoding='bytes')

        self.users_summary_list = np.load(self.user_summary_list_path, encoding='bytes')
        self.items_summary_list = np.load(self.item_summary_list_path, encoding='bytes')

        for k, v in para.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)


class Office_Products_data_71_Config(DefaultConfig):
    def __init__(self):
        self.set_path('Office_Products_data/data71')

    vocab_size = 14549
    word_dim = 100

    setNum_word_of_rev = 229
    setNum_rev_of_user = 14
    setNum_rev_of_item = 35
    setNum_word_of_summary = 8

    user_num = 4905 + 2
    item_num = 2420 + 2

    train_data_size = 42611