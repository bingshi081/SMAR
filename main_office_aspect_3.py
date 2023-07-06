# -*- encoding: utf-8 -*-
import time
import random
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

import config_my_own
from dataset import ReviewData
from framework import Model
import models


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def unpack_input(opt, x):
    uids, iids = list(zip(*x))
    uids = list(uids)
    iids = list(iids)

    user_reviews = opt.users_review_list[uids]
    user_summaries = opt.users_summary_list[uids]
    user_item2id = opt.user2itemid_list[uids]
    user_doc = opt.user_doc[uids]

    item_reviews = opt.items_review_list[iids]
    item_summaries = opt.items_summary_list[iids]
    item_user2id = opt.item2userid_list[iids]
    item_doc = opt.item_doc[iids]

    data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc, user_summaries, item_summaries]

    if opt.use_gpu:
        data = list(map(lambda x: torch.LongTensor(x).cuda(), data))
    else:
        data = list(map(lambda x: torch.LongTensor(x), data))
    return data


def train():
    para = {}
    para["biaoji"] = 2

    para['model'] = "summary_topic_final_aspect_3"

    para["dataset"] = "Office_Products_data_71"


    para["output"] = "lfm"
    para["use_gpu"] = True
    para["num_aspect"] = 3

    para["num_fea"] = 2
    para["batch_size"] = 128
    para["print_step"] = 110

    para["have_saved_model"] = 0
    para["best_valid_mse"] = 0
    para["lr"] = 0.002
    para["num_epochs"] = 80


    opt = getattr(config_my_own, para['dataset'] + '_Config')()

    opt.parse(para)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if (len(opt.gpu_ids) == 0) and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)



    Model_obj = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        Model_obj.cuda()
        if len(opt.gpu_ids) > 0:
            Model_obj = nn.DataParallel(Model_obj, device_ids=opt.gpu_ids)

    if Model_obj.model.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    train_data = ReviewData(opt.data_root, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)

    val_data = ReviewData(opt.data_root, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = optim.Adam(Model_obj.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    min_loss = 1e+10
    mse_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    smooth_mae_func = nn.SmoothL1Loss()

    opt.save_path = 'checkpoints/' + opt.model + '_' + str(opt.dataset) + '_' + str(opt.print_opt) + '.pth'

    if opt.have_saved_model == 1:
        best_res = opt.best_valid_mse
        Model_obj.load(opt.save_path)
    else:
        best_res = 1e+10

    start_time = time.time()
    time_flag = 1


    for epoch in tqdm(range(opt.num_epochs)):
        mse_train=0
        mae_train=0
        mse_epoch=0
        mae_epoch=0

        Model_obj.train()
        for idx, (train_datas, scores) in enumerate(tqdm(train_data_loader)):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            train_datas = unpack_input(opt, train_datas)
            optimizer.zero_grad()
            output = Model_obj(train_datas)
            mse_loss = mse_func(output, scores)
            mse_train += mse_loss.item()
            mse_epoch += mse_loss.item()*len(scores)

            mae_loss = mae_func(output, scores)
            mae_train += mae_loss.item()
            mae_epoch += mae_loss.item()*len(scores)

            if opt.loss_method == 'mse':
                loss = mse_loss
            if opt.loss_method == 'rmse':
                loss = torch.sqrt(mse_loss) / 2.0
            if opt.loss_method == 'mae':
                loss = mae_loss
            if opt.loss_method == 'smooth_mae':
                loss = smooth_mae_loss

            loss.backward()
            optimizer.step()

            if idx % opt.print_step == 0 and idx > 0:
                if time_flag == 1:
                    time_flag = 0
                    end_time = time.time()

                mse_train=0
                mae_train=0

                mse_val, mae_val = predict(Model_obj, val_data_loader, opt)

                if mse_val < best_res:
                    Model_obj.save(name=opt.save_path)
                    best_res = mse_val

        scheduler.step()
        mse_epoch = mse_epoch * 1.0 / len(train_data)
        mae_epoch = mae_epoch * 1.0 / len(train_data)


def predict(Model_obj, data_loader, opt):
    total_mseloss = 0.0
    total_maeloss = 0.0
    Model_obj.eval()
    with torch.no_grad():
        for idx, (test_data, scores) in enumerate(tqdm(data_loader)):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            test_data = unpack_input(opt, test_data)

            output = Model_obj(test_data)
            mse_loss = torch.sum((output-scores)**2)
            total_mseloss += mse_loss.item()

            mae_loss = torch.sum(abs(output-scores))
            total_maeloss += mae_loss.item()

    data_len = len(data_loader.dataset)
    mse = total_mseloss * 1.0 / data_len
    mae = total_maeloss * 1.0 / data_len
    Model_obj.train()
    return mse, mae


if __name__ == "__main__":
    train()
