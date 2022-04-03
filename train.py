# coding=utf8
import torch
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm
from CBOW import CBOW
import jieba
import pickle as pkl
import time
from functools import wraps


# def fn_timer(function):
#     @wraps(function)
#     def function_timer(*args, **kwargs):
#         t0 = time.time()
#         result = function(*args, **kwargs)
#         t1 = time.time()
#         print("Total time running %s: %s seconds" %
#               (function.func_name, str(t1 - t0))
#               )
#         return result
#
#     return function_timer


def pre_process():
    # 加载数据
    with open('data/msr_training.utf8', 'r', encoding='utf-8') as f:
        tmp = f.readlines()
    t = []
    for i in tmp:
        t1 = i.replace('“  ', '')
        t2 = t1.replace('\n', '')
        t.append(t2)
    sum_list2 = []
    for i in t:
        sum_2 = i.replace('  ', '')
        sum_list2.append(sum_2)
    with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    # 去除停用词 构词表
    words_ = []
    vocab = set()
    for sen in sum_list2:
        words = jieba.lcut(sen)
        temp = []
        for wd in words:
            if wd not in stopwords:
                temp.append(wd)
                vocab.add(wd)
        words_.append(temp)
    vocab_dist = {}
    for i, wd in enumerate(list(vocab)):
        vocab_dist[wd] = i
    return words_, vocab_dist


def train(words, vocab):
    torch.manual_seed(1)
    vocab_size = len(vocab)
    train_cbow = []
    for line in words:
        for i in range(2, len(line) - 2):
            temp = ([[line[i - 2], line[i - 1]], [line[i + 1], line[i + 2]]], line[i])
            train_cbow.append(temp)
    hidden_dim = 32
    embedding_dim = 64
    losses = []
    model = CBOW(vocab_size, embedding_dim, hidden_dim)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = opt.SGD(model.parameters(), lr=3e-2)
    for i in tqdm(range(10)):
        total_loss = 0
        for context, target in tqdm(train_cbow):
            context_indexes = []
            for t in context:
                context_indexes.extend([vocab[t[0]], vocab[t[1]]])
            context_indexes = torch.tensor(context_indexes, dtype=torch.long)
            model.zero_grad()
            prob = model(context_indexes, vocab[target])
            true_label = torch.tensor([vocab[target]], dtype=torch.long)
            loss = loss_fun(prob, true_label)  # 预测值 真实值
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        torch.save(model, 'data/CBOW.bin')
    # [print(i / len(train_cbow)) for i in losses]


if __name__ == '__main__':
    # _words, _vocab = pre_process()
    # with open('data/words.pkl', 'wb') as f:
    #     pkl.dump(_words, f)
    # with open('data/vocab.pkl', 'wb') as f:
    #     pkl.dump(_vocab, f)
    with open('data/words.pkl', 'rb') as f:
        _words = pkl.load(f)
    with open('data/vocab.pkl', 'rb') as f:
        _vocab = pkl.load(f)
    train(_words, _vocab)
