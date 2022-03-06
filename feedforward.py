import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from numpy import mean


class Config(object):
    def __init__(self):
        self.vocab = pkl.load(open('./vocab.pkl', 'rb'))# 读取词表
        self.train_data = pkl.load(open('./train_data.pkl', 'rb'))# 读取训练数据
        self.target = pkl.load(open('./target.pkl', 'rb'))# 读取标签

        self.learing_rate = 0.000015# 学习率
        self.epoch = 2# epoch次数

        self.output_size = 4
        self.embed_dim = 16


class Model(nn.Module):
    def __init__(self, output_size, voc_size, embed_dim):
        super(Model, self).__init__()
        self.hidlayer = nn.Linear(embed_dim, 8)
        self.outlayer = nn.Linear(8, output_size)
        self.embedding = nn.Embedding(voc_size, embed_dim)

    def forward(self, inlayer):
        emd = self.embedding(inlayer)
        h_out = self.hidlayer(emd)
        h_out = F.relu(h_out)
        out = self.outlayer(h_out)
        out = F.softmax(out)
        return out


def maskNLLLoss(inp, target):
    crossEntropy = -torch.log(torch.gather(inp, 1, target))
    loss = crossEntropy.mean()
    return loss


def test(model):
    text = '这首先是个民族问题，民族的感情问题。'
    hang = []
    for word in text:
        hang.append(Config().vocab[word])
    test_tensor = torch.tensor(hang, dtype=torch.long)
    res = model(test_tensor)
    res = res.detach().numpy()
    [print(np.argmax(r)) for r in res]



if __name__ =='__main__':
    torch.manual_seed(1)
    # torch.cuda.is_available()   # 判断是否存在GPU可用，存在返回True
    # device = torch.device('cuda:0')      # 将设备改为0号GPU
    # model.to(device)                  # 将模型使用GPU训练
    config = Config()
    voc_size = len(config.vocab)

    train_data_list = []
    for lin in config.train_data:
        hang = []
        for word in lin:
            hang.append(config.vocab[word])
        train_data_list.append(torch.tensor(hang, dtype=torch.long))

    target_dict = {'B': [1, 0, 0, 0],
                   'M': [0, 1, 0, 0],
                   'E': [0, 0, 1, 0],
                   'S': [0, 0, 0, 1]}
    target_list = []
    for lin in config.target:
        hang = []
        for tag in lin:
            hang.append(target_dict[tag])
        target_list.append(torch.tensor(hang, dtype=torch.long))

    model = Model(config.output_size, voc_size, config.embed_dim)

    optimizer = torch.optim.SGD(model.parameters(), lr=config.learing_rate)
    for i in range(config.epoch):
        for j, k in enumerate(train_data_list):
            optimizer.zero_grad()
            out = model(k)
            loss = maskNLLLoss(out, target_list[j])
            loss.backward()
            optimizer.step()
            print(loss.item())
        torch.save(model, './cut.bin')
    test(model)

