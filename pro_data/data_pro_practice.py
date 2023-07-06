# -*- coding: utf-8 -*-
import json
import pandas as pd
import re
import sys
import os
import gzip
import numpy as np
import time
from sklearn.model_selection import train_test_split
from operator import itemgetter
import gensim
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from tqdm import tqdm

P_REVIEW = 0.85
MAX_DF = 1.0
MIN_DF = 5
MAX_VOCAB = 50000
DOC_LEN = 500
PRE_W2V_BIN_PATH = "../glove.6B.100d.txt"


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

def parse(path):
    g = gzip.open(path, 'r')
    for l in tqdm(g):
        yield eval(l)


def get_count(data, id):
    idList = data[[id, 'ratings']].groupby(id, as_index=False)
    idListCount = idList.size()
    return idListCount


def get_number_of_users_in_data(data):
    userCount, itemCount = get_count(data, 'user_id'), get_count(data, 'item_id')
    userNum_all = userCount.shape[0]
    itemNum_all = itemCount.shape[0]
    return userNum_all, itemNum_all


def filter_triplets(tp, min_uc=20, min_sc=20):
    for _ in range(10):
        usercount = get_count(tp, 'user_id')
        tp = tp[tp['user_id'].isin(usercount.index[usercount >= min_uc])]

        songcount = get_count(tp, 'item_id')
        tp = tp[tp['item_id'].isin(songcount.index[songcount >= min_sc])]

        usercount, songcount = get_count(tp, 'user_id'), get_count(tp, 'item_id')

    return tp


def numerize(data):
    userCount, itemCount = get_count(data, 'user_id'), get_count(data, 'item_id')
    uidList = userCount.index
    iidList = itemCount.index
    user2id = dict((uid, i) for (i, uid) in enumerate(uidList))
    item2id = dict((iid, i) for (i, iid) in enumerate(iidList))

    uid = list(map(lambda x: user2id[x], data['user_id']))
    iid = list(map(lambda x: item2id[x], data['item_id']))
    data['user_id'] = uid
    data['item_id'] = iid
    return data


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)
    return string.strip().lower()


def construct_reviews_dict_and_iid_dict_from_train_data(data_train, filename):
    user_reviews_dict = {}
    item_reviews_dict = {}
    user_iid_dict = {}
    item_uid_dict = {}
    user_summaries_dict = {}
    item_summaries_dict = {}
    user_reviews_and_summaries_dict = {}


    for i in data_train.values:
        str_review = clean_str(i[3].encode('ascii', 'ignore').decode('ascii'))
        str_summary = clean_str(i[4].encode('ascii', 'ignore').decode('ascii'))
        str_review_and_summary = clean_str(i[3].encode('ascii', 'ignore').decode('ascii') + ' ' \
                                           + i[4].encode('ascii', 'ignore').decode('ascii'))

        if filename == "Yelp2013":
            str_review = clean_str(str_review)

        if len(str_review.strip()) == 0:
            str_review = "<unk>"
        if len(str_summary.strip()) == 0:
            str_summary = "<unk>"
        if len(str_review_and_summary.strip()) == 0:
            str_review_and_summary = "<unk>"

        if i[0] in user_reviews_dict:
            user_reviews_dict[i[0]].append(str_review)
            user_summaries_dict[i[0]].append(str_summary)
            user_iid_dict[i[0]].append(i[1])
            user_reviews_and_summaries_dict[i[0]].append(str_review_and_summary)
        else:
            user_reviews_dict[i[0]] = [str_review]
            user_summaries_dict[i[0]] = [str_summary]
            user_iid_dict[i[0]] = [i[1]]
            user_reviews_and_summaries_dict[i[0]] = [str_review_and_summary]

        if i[1] in item_reviews_dict:
            item_reviews_dict[i[1]].append(str_review)
            item_summaries_dict[i[1]].append(str_summary)
            item_uid_dict[i[1]].append(i[0])
        else:
            item_reviews_dict[i[1]] = [str_review]
            item_summaries_dict[i[1]] = [str_summary]
            item_uid_dict[i[1]] = [i[0]]

    user_reviews = []
    for ind in range(len(user_reviews_dict)):
        user_reviews.append(' <SEP> '.join(user_reviews_dict[ind]))

    item_reviews = []
    for ind in range(len(item_reviews_dict)):
        item_reviews.append(' <SEP> '.join(item_reviews_dict[ind]))

    user_reviews_and_summaries = []
    for ind in range(len(user_reviews_and_summaries_dict)):
        user_reviews_and_summaries.append(' <SEP> '.join(user_reviews_and_summaries_dict[ind]))


    vectorizer = TfidfVectorizer(min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_VOCAB)
    vectorizer.fit(user_reviews_and_summaries)
    vocab = vectorizer.vocabulary_
    vocab['<SEP>'] = MAX_VOCAB

    def clean_review(user_reviews_dict):
        new_dict = {}
        for k, text in user_reviews_dict.items():
            new_reviews = []
            for r in text:
                words = ' '.join([w for w in r.split() if w in vocab])
                new_reviews.append(words)
            new_dict[k] = new_reviews
        return new_dict

    def calculate_doc_len(user_reviews):
        len_sum_list = []

        for line in user_reviews:
            review = [word for word in line.split() if word in vocab]
            len_sum_list.append(len(review))

        def percent_len_sum(rlist, percent):
            x = np.sort(rlist)
            xLen = len(x)
            return x[int(percent * xLen) - 1]

        return percent_len_sum(len_sum_list, P_REVIEW)

    def clean_doc(user_reviews, set_word_num_doc):
        new_raw = []
        for line in user_reviews:
            review = [word for word in line.split() if word in vocab]
            if len(review) > set_word_num_doc:
                review = review[:set_word_num_doc]
            new_raw.append(review)
        return new_raw

    user_reviews_dict = clean_review(user_reviews_dict)
    item_reviews_dict = clean_review(item_reviews_dict)

    user_summaries_dict = clean_review(user_summaries_dict)
    item_summaries_dict = clean_review(item_summaries_dict)

    calculate_doc_len(user_reviews)
    user_review2doc = clean_doc(user_reviews, DOC_LEN)

    calculate_doc_len(item_reviews)
    item_review2doc = clean_doc(item_reviews, DOC_LEN)

    word_index = {}
    word_index['<unk>'] = 0

    index_num_map_word_index = {}
    index_num_map_word_index[0] = '<unk>'

    for i, w in enumerate(vocab.keys(), 1):
        word_index[w] = i
        index_num_map_word_index[i] = w

    return word_index, user_review2doc, item_review2doc, user_reviews_dict, item_reviews_dict, user_summaries_dict, item_summaries_dict, user_iid_dict, item_uid_dict, index_num_map_word_index

def construct_revid_dict_from_train_data(data_train, filename, setNum_rev_of_user, setNum_rev_of_item, userNum_all, itemNum_all):
    user_reviews_dict = {}
    item_reviews_dict = {}
    user_iid_dict = {}
    item_uid_dict = {}

    user_revid_dict = {}
    item_revid_dict = {}

    user_rt_dict = {}
    item_rt_dict = {}

    num_idx = 0

    for i in data_train.values:
        str_review = clean_str(i[3].encode('ascii', 'ignore').decode('ascii'))
        if filename == "Yelp2013":
            str_review = clean_str(str_review)

        if len(str_review.strip()) == 0:
            str_review = "<unk>"


        if i[0] in user_reviews_dict:
            user_reviews_dict[i[0]].append(str_review)
            user_iid_dict[i[0]].append(i[1])

            user_revid_dict[i[0]].append(num_idx)
            user_rt_dict[i[0]].append(float(i[2]))
        else:
            user_reviews_dict[i[0]] = [str_review]
            user_iid_dict[i[0]] = [i[1]]

            user_revid_dict[i[0]] = [num_idx]
            user_rt_dict[i[0]] = [float(i[2])]

        if i[1] in item_reviews_dict:
            item_reviews_dict[i[1]].append(str_review)
            item_uid_dict[i[1]].append(i[0])

            item_revid_dict[i[1]].append(num_idx)
            item_rt_dict[i[1]].append(float(i[2]))
        else:
            item_reviews_dict[i[1]] = [str_review]
            item_uid_dict[i[1]] = [i[0]]

            item_revid_dict[i[1]] = [num_idx]
            item_rt_dict[i[1]] = [float(i[2])]

        num_idx = num_idx + 1

    def padding_rev_or_rt_ids(iids, num, pad_id):
        if len(iids) >= num:
            new_iids = iids[:num]
        else:
            new_iids = iids + [pad_id] * (num - len(iids))
        return new_iids

    user_revid_list = []
    user_rt_list = []
    for i in range(userNum_all):
        user_revid_list.append(padding_rev_or_rt_ids(user_revid_dict[i], setNum_rev_of_user, -1))
        user_rt_list.append(padding_rev_or_rt_ids(user_rt_dict[i], setNum_rev_of_user, -1))

    item_revid_list = []
    item_rt_list = []
    for i in range(itemNum_all):
        item_revid_list.append(padding_rev_or_rt_ids(item_revid_dict[i], setNum_rev_of_item, -1))
        item_rt_list.append(padding_rev_or_rt_ids(item_rt_dict[i], setNum_rev_of_item, -1))


    return user_revid_list, item_revid_list, user_rt_list, item_rt_list



def countNum(user_reviews_dict):

    minNum_rev_of_user = 100
    maxNum_rev_of_user = 0

    minNum_word_of_rev = 3000
    maxNum_word_of_rev = 0

    numList_rev_of_user = []
    numList_word_of_rev = []

    for i, rev_list in user_reviews_dict.items():
        if len(rev_list) < minNum_rev_of_user:
            minNum_rev_of_user = len(rev_list)
        if len(rev_list) > maxNum_rev_of_user:
            maxNum_rev_of_user = len(rev_list)
        numList_rev_of_user.append(len(rev_list))
        for rev in rev_list:
            if rev != "":
                wordTokens = rev.split()
            if len(wordTokens) < minNum_word_of_rev:
                minNum_word_of_rev = len(wordTokens)
            if len(wordTokens) > maxNum_word_of_rev:
                maxNum_word_of_rev = len(wordTokens)
            numList_word_of_rev.append(len(wordTokens))

    def percent_num(rlist):
        x = np.sort(rlist)
        xLen = len(x)
        return x[int(P_REVIEW * xLen) - 1]
    setNum_rev_of_user = percent_num(numList_rev_of_user)
    setNum_word_of_rev = percent_num(numList_word_of_rev)

    return setNum_rev_of_user, setNum_word_of_rev



def construct_pandas_data_frame_from_file(filename):
    users_id = []
    items_id = []
    ratings = []
    reviews = []
    summaries = []
    file = open(filename, errors='ignore')
    for line in file:
        js = json.loads(line)
        if str(js['reviewerID']) == 'unknown':
            print("unknown user id")
            continue
        if str(js['asin']) == 'unknown':
            print("unkown item id")
            continue
        reviews.append(js['reviewText'])
        users_id.append(str(js['reviewerID']))
        items_id.append(str(js['asin']))
        ratings.append(str(js['overall']))
        summaries.append(js['summary'])

    data_frame = {'user_id': pd.Series(users_id), 'item_id': pd.Series(items_id),
                  'ratings': pd.Series(ratings), 'reviews': pd.Series(reviews), 'summaries': pd.Series(summaries)}
    data = pd.DataFrame(data_frame)

    del users_id, items_id, ratings, reviews, summaries
    return data



def check_users_in_train_data(data_train, data_test, userNum_all, itemNum_all):
    userCount, itemCount = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
    uids_train = userCount.index
    iids_train = itemCount.index
    userNum_in_train = userCount.shape[0]
    itemNum_in_train = itemCount.shape[0]

    uidMiss = []
    iidMiss = []
    if userNum_in_train != userNum_all or itemNum_in_train != itemNum_all:
        for uid in range(userNum_all):
            if uid not in uids_train:
                uidMiss.append(uid)
        for iid in range(itemNum_all):
            if iid not in iids_train:
                iidMiss.append(iid)

    if len(uidMiss):
        for uid in uidMiss:
            df_temp = data_test[data_test['user_id'] == uid]
            data_test = data_test[data_test['user_id'] != uid]
            data_train = pd.concat([data_train, df_temp])

    if len(iidMiss):
        for iid in iidMiss:
            df_temp = data_test[data_test['item_id'] == iid]
            data_test = data_test[data_test['item_id'] != iid]
            data_train = pd.concat([data_train, df_temp])

    userCount, itemCount = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
    uids_train = userCount.index
    iids_train = itemCount.index
    userNum_in_train = userCount.shape[0]
    itemNum_in_train = itemCount.shape[0]

    return data_train, data_test

def extract_rating_idx_directed_array(data_dict):
    y = []
    for i in data_dict.values:
        y.append(int(float(i[2])))
    edge_rating_idx_directed = np.array(y)
    return edge_rating_idx_directed

def extract(data_dict):
    x = []
    y = []
    for i in data_dict.values:
        uid = i[0]
        iid = i[1]
        x.append([uid, iid])
        y.append(float(i[2]))
    return x, y


def construct_user_item_pair_rating(data_train, data_val, data_test):
    x_train, y_train = extract(data_train)
    x_val, y_val = extract(data_val)
    x_test, y_test = extract(data_test)
    return x_train, y_train, x_val, y_val, x_test, y_test


def construct_ui_graph_from_train_data(data_train, userNum_all, itemNum_all):
    ui_src = []
    ui_dst = []
    for i in data_train.values:
        ui_src.append(i[0])
        ui_dst.append(i[1] + userNum_all)
    ui_node_map_index = []
    for i in range(userNum_all + itemNum_all):
        ui_node_map_index.append(i)
    return ui_src, ui_dst, ui_node_map_index


def save_npy_to_save_folder(save_folder, x_train, y_train, x_val, y_val, x_test, y_test):
    np.save(f"{save_folder}/train/Train.npy", x_train)
    np.save(f"{save_folder}/train/Train_Score.npy", y_train)
    np.save(f"{save_folder}/val/Val.npy", x_val)
    np.save(f"{save_folder}/val/Val_Score.npy", y_val)
    np.save(f"{save_folder}/test/Test.npy", x_test)
    np.save(f"{save_folder}/test/Test_Score.npy", y_test)


def make_save_folder(filename, number_of_fold):
    idx_str = str(number_of_fold)
    save_folder = '../dataset/' + filename[:-7] + "_data" + "/data" + idx_str

    if not os.path.exists(save_folder + '/train'):
        os.makedirs(save_folder + '/train')
    if not os.path.exists(save_folder + '/val'):
        os.makedirs(save_folder + '/val')
    if not os.path.exists(save_folder + '/test'):
        os.makedirs(save_folder + '/test')

    return save_folder


def padding_text(textList, num):
    new_textList = []
    if len(textList) >= num:
        new_textList = textList[:num]
    else:
        padding = [[0] * len(textList[0]) for _ in range(num - len(textList))]
        new_textList = textList + padding
    return new_textList


def padding_ids(iids, num, pad_id):
    if len(iids) >= num:
        new_iids = iids[:num]
    else:
        new_iids = iids + [pad_id] * (num - len(iids))
    return new_iids


def padding_doc(doc, word_length):
    new_doc = []
    for d in doc:
        if len(d) < word_length:
            d = d + [0] * (word_length - len(d))
        else:
            d = d[:word_length]
        new_doc.append(d)
    return new_doc


def padding_description_text_idx(itemNum_all, item_description2doc, word_index, setNum_word_of_description):
    itemDescribe2Index = []
    for i in range(itemNum_all):
        doc2index = [word_index[w] for w in item_description2doc[i]]
        itemDescribe2Index.append(doc2index)

    itemDescribe2Index = padding_doc(itemDescribe2Index, setNum_word_of_description)
    return itemDescribe2Index

def npy_to_save_folder(save_folder, userReview2Index, user_iid_list, userDoc2Index, itemReview2Index, item_uid_list, itemDoc2Index, word_index, index_num_map_word_index, userSummary2Index, itemSummary2Index):
    np.save(f"{save_folder}/train/userReview2Index.npy", userReview2Index)
    np.save(f"{save_folder}/train/user_item2id.npy", user_iid_list)
    np.save(f"{save_folder}/train/userDoc2Index.npy", userDoc2Index)
    np.save(f"{save_folder}/train/itemReview2Index.npy", itemReview2Index)
    np.save(f"{save_folder}/train/item_user2id.npy", item_uid_list)
    np.save(f"{save_folder}/train/itemDoc2Index.npy", itemDoc2Index)
    np.save(f"{save_folder}/train/word_index.npy", word_index)
    np.save(f"{save_folder}/train/index_num_map_word_index.npy", index_num_map_word_index)
    np.save(f"{save_folder}/train/userSummary2Index.npy", userSummary2Index)
    np.save(f"{save_folder}/train/itemSummary2Index.npy", itemSummary2Index)




def construct_word_emb_from_glove_txt(PRE_W2V_BIN_PATH, word_index):
    vocab_item = sorted(word_index.items(), key=itemgetter(1))
    w2v = []
    out = 0

    pre_word2v = {}
    with open(PRE_W2V_BIN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            l = line.split()
            word = l[0]
            pre_word2v[word] = np.array(l[1:], dtype=np.float32)

    for word, key in vocab_item:
        if word in pre_word2v:
            w2v.append(pre_word2v[word])
        else:
            out += 1
            w2v.append(np.random.uniform(-1.0, 1.0, (100,)))
    w2vArray = np.array(w2v)
    return w2v

def extract_review_idx_directed_array(data_train, word_index, setNum_word_of_rev):
    edge_review_idx_list = []
    for i in tqdm(data_train.values):
        str_review = clean_str(i[3].encode('ascii', 'ignore').decode('ascii'))
        if filename == "Yelp2013":
            str_review = clean_str(str_review)

        if len(str_review.strip()) == 0:
            str_review = "<unk>"

        review_word_idx = [word_index[w] for w in str_review.split() if w in word_index]
        if len(review_word_idx) < setNum_word_of_rev:
            review_word_idx = review_word_idx + [0] * (setNum_word_of_rev - len(review_word_idx))
        else:
            review_word_idx = review_word_idx[:setNum_word_of_rev]

        edge_review_idx_list.append(review_word_idx)

    edge_review_idx_directed = np.array(edge_review_idx_list)
    return edge_review_idx_directed


if __name__ == '__main__':
    start_time = time.time()
    filename = "Office_Products_5.json"
    number_of_fold = 81

    save_folder = make_save_folder(filename, number_of_fold)
    data = construct_pandas_data_frame_from_file(filename)
    userNum_all, itemNum_all = get_number_of_users_in_data(data)
    data = numerize(data)
    data_train, data_test = train_test_split(data, test_size=0.2,random_state=1234)
    data_train, data_test = check_users_in_train_data(data_train, data_test, userNum_all, itemNum_all)
    data_test, data_val = train_test_split(data_test, test_size=0.5, random_state=1234)

    x_train, y_train, x_val, y_val, x_test, y_test = construct_user_item_pair_rating(data_train, data_val,
                                                                                     data_test)
    save_npy_to_save_folder(save_folder, x_train, y_train, x_val, y_val, x_test, y_test)
    word_index, user_review2doc, item_review2doc, user_reviews_dict, \
                     item_reviews_dict, user_summaries_dict, item_summaries_dict, user_iid_dict, item_uid_dict, index_num_map_word_index \
        = construct_reviews_dict_and_iid_dict_from_train_data(data_train,
                                                              filename)
    setNum_rev_of_user, setNum_word_of_rev_user = countNum(user_reviews_dict)
    setNum_rev_of_item, setNum_word_of_rev_item = countNum(item_reviews_dict)
    setNum_word_of_rev = max(setNum_word_of_rev_user, setNum_word_of_rev_item)
    setNum_summary_of_user, setNum_word_of_summary_user = countNum(user_summaries_dict)
    setNum_summary_of_item, setNum_word_of_summary_item = countNum(item_summaries_dict)
    setNum_word_of_summary = max(setNum_word_of_summary_user, setNum_word_of_summary_item)
    userReview2Index = []
    userSummary2Index = []
    userDoc2Index = []
    user_iid_list = []
    for i in range(userNum_all):
        count_user = 0
        dataList = []
        textList = user_reviews_dict[i]
        u_iids = user_iid_dict[i]
        u_reviewList = []
        user_iid_list.append(padding_ids(u_iids, setNum_rev_of_user, itemNum_all + 1))
        doc2index = [word_index[w] for w in user_review2doc[i]]
        for text in textList:
            text2index = []
            wordTokens = text.strip().split()
            if len(wordTokens) == 0:
                wordTokens = ['<unk>']
            text2index = [word_index[w] for w in wordTokens]
            if len(text2index) < setNum_word_of_rev:
                text2index = text2index + [0] * (setNum_word_of_rev - len(text2index))
            else:
                text2index = text2index[:setNum_word_of_rev]
            u_reviewList.append(text2index)
        userReview2Index.append(padding_text(u_reviewList, setNum_rev_of_user))
        summaryList = user_summaries_dict[i]
        u_summaryList = []
        for text in summaryList:
            text2index = []
            wordTokens = text.strip().split()
            if len(wordTokens) == 0:
                wordTokens = ['<unk>']
            text2index = [word_index[w] for w in wordTokens]
            if len(text2index) < setNum_word_of_summary:
                text2index = text2index + [0] * (setNum_word_of_summary - len(text2index))
            else:
                text2index = text2index[:setNum_word_of_summary]
            u_summaryList.append(text2index)
        userSummary2Index.append(padding_text(u_summaryList, setNum_rev_of_user))
        userDoc2Index.append(doc2index)
    userDoc2Index = padding_doc(userDoc2Index, DOC_LEN)
    itemReview2Index = []
    itemSummary2Index = []
    itemDoc2Index = []
    item_uid_list = []
    for i in range(itemNum_all):
        count_item = 0
        dataList = []
        textList = item_reviews_dict[i]
        i_uids = item_uid_dict[i]
        i_reviewList = []
        item_uid_list.append(padding_ids(i_uids, setNum_rev_of_item, userNum_all + 1))
        doc2index = [word_index[w] for w in item_review2doc[i]]
        for text in textList:
            text2index = []
            wordTokens = text.strip().split()
            if len(wordTokens) == 0:
                wordTokens = ['<unk>']
            text2index = [word_index[w] for w in wordTokens]
            if len(text2index) < setNum_word_of_rev:
                text2index = text2index + [0] * (setNum_word_of_rev - len(text2index))
            else:
                text2index = text2index[:setNum_word_of_rev]
            i_reviewList.append(text2index)
        itemReview2Index.append(padding_text(i_reviewList, setNum_rev_of_item))
        summaryList = item_summaries_dict[i]
        i_summaryList = []
        for text in summaryList:
            text2index = []
            wordTokens = text.strip().split()
            if len(wordTokens) == 0:
                wordTokens = ['<unk>']
            text2index = [word_index[w] for w in wordTokens]
            if len(text2index) < setNum_word_of_summary:
                text2index = text2index + [0] * (setNum_word_of_summary - len(text2index))
            else:
                text2index = text2index[:setNum_word_of_summary]
            i_summaryList.append(text2index)
        itemSummary2Index.append(padding_text(i_summaryList, setNum_rev_of_item))
        itemDoc2Index.append(doc2index)
    itemDoc2Index = padding_doc(itemDoc2Index, DOC_LEN)
    npy_to_save_folder(save_folder, userReview2Index, user_iid_list, userDoc2Index,
                       itemReview2Index, item_uid_list, itemDoc2Index, word_index, index_num_map_word_index, userSummary2Index, itemSummary2Index)
    w2v = construct_word_emb_from_glove_txt(PRE_W2V_BIN_PATH, word_index)
    np.save(f"{save_folder}/train/w2v.npy", w2v)
