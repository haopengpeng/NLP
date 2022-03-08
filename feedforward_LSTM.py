import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from tqdm import tqdm


class Config(object):
    def __init__(self):
        self.vocab = pkl.load(open('./vocab.pkl', 'rb'))# 读取词表
        self.train_data = pkl.load(open('./train_data.pkl', 'rb'))# 读取训练数据
        self.target = pkl.load(open('./target.pkl', 'rb'))# 读取标签

        self.learing_rate = 0.0001# 学习率
        self.epoch = 10# epoch次数

        self.output_size = 4# 输出维度
        self.embed_dim = 64# 字向量维度


class Model(nn.Module):
    def __init__(self, output_size, voc_size, embed_dim):
        super(Model, self).__init__()

        '''全连接网络参数'''
        # self.hidlayer = nn.Linear(embed_dim, 32)
        # self.hidlayer2 = nn.Linear(32, 16)
        # self.hidlayer3 = nn.Linear(16, 8)

        '''lstm网络参数'''
        self.hidden_size = 16
        self.linear = nn.Linear(64, 4)

        self.outlayer = nn.Linear(8, output_size)
        self.embedding = nn.Embedding(voc_size, embed_dim)

    def forward(self, inlayer):
        emd = self.embedding(inlayer)

        '''全连接网络'''
        # h_out = self.hidlayer(emd)
        # h_out = F.relu(h_out)
        #
        # h2_out = self.hidlayer2(h_out)
        # h2_out = F.relu(h2_out)
        #
        # h3_out = self.hidlayer3(h2_out)
        # h3_out = F.relu(h3_out)
        #
        # out = self.outlayer(h3_out)
        # out = F.softmax(out)
        # return out

        '''lstm网络'''
        lstm = nn.LSTM(64, 64)# 构造‘lstm神经元’格式
        inputs = torch.unsqueeze(emd, 1)# 构造输入‘lstm网络’的句子
        hidden = (torch.randn(1, 1, 64), torch.randn(1, 1, 64))# 初始化隐藏状态

        out, hidden = lstm(inputs, hidden)
        out = torch.squeeze(out, 1)
        out = F.softmax(out)
        out = self.linear(out)
        return out


def Loss(inp, target):
    loss = F.cross_entropy(inp, target)
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
    config = Config()
    # device = torch.device("cuda:0")
    voc_size = len(config.vocab)

    train_data_list = []
    for lin in config.train_data:
        hang = []
        for word in lin:
            hang.append(config.vocab[word])
        train_data_list.append(torch.tensor(hang, dtype=torch.long))

    target_dict = {'B': 0,
                   'M': 1,
                   'E': 2,
                   'S': 3}
    target_list = []
    for lin in config.target:
        hang = []
        for tag in lin:
            hang.append(target_dict[tag])
        target_list.append(torch.tensor(hang, dtype=torch.long))

    model = Model(config.output_size, voc_size, config.embed_dim)

    optimizer = torch.optim.SGD(model.parameters(), lr=config.learing_rate)

    '''全连接网络训练'''
    # for i in tqdm(range(config.epoch)):
    #     for j, k in enumerate(train_data_list):
    #         optimizer.zero_grad()
    #         out = model(k)
    #         loss = Loss(out, target_list[j])
    #         loss.backward()
    #         optimizer.step()
    #         print(loss.item())
    #     torch.save(model, './cut.bin')
    # test(model)

    '''双向lstm训练'''
    for i in tqdm(range(config.epoch)):
        for j,k in enumerate(train_data_list):
            optimizer.zero_grad()
            out =model(k)
            loss = Loss(out, target_list[j])
            loss.backward()
            optimizer.step()
            print(loss.item())
        torch.save(model,'./cut_lstm.bin')
    test(model)
