# coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, target):
        # embed = F.one_hot(inputs, self.vocab_size).float()
        embed = self.embedding(inputs).sum(0) / 4
        hidden_o = F.relu(self.layer1(embed))  # 非线性变换
        output_ = self.layer2(hidden_o)  # 仿射变换
        Nec_list = self.__random__(target, 20)
        temp = torch.ones_like(output_, dtype=torch.bool)
        for i in Nec_list:
            temp[i] = False
        masked_out = output_.masked_fill(temp, -np.inf)
        out = F.softmax(masked_out, dim=0).view(1, -1)
        # out = F.softmax(output_, dim=0).view(1, -1)
        return out

    def __random__(self, target, num_random):
        rand_list = np.random.randint(self.vocab_size, size=num_random)
        if target in rand_list:
            while True:
                temp = np.random.randint(self.vocab_size, size=1)
                if temp != target:
                    rand_list = np.append(rand_list, temp)
                    break
        else:
            rand_list = np.append(rand_list, target)
        return rand_list

