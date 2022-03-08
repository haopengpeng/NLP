import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from tqdm import tqdm


class Config(object):
    def __init__(self):
        self.vocab = pkl.load(open('./vocab.pkl', 'rb'))  # 读取词表
        self.train_data = pkl.load(open('./train_data.pkl', 'rb'))  # 读取训练数据
        self.target = pkl.load(open('./target.pkl', 'rb'))  # 读取标签

        self.learning_rate = 0.000015  # 学习率
        self.epoch = 2  # epoch次数

        self.output_size = 4
        self.embed_dim = 32


class Model(nn.Module):
    def __init__(self, output_size, vocab_size, embed_dim):
        super(Model, self).__init__()
        self.hid_layer = nn.Linear(embed_dim, 64)
        self.out_layer = nn.Linear(64, output_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, in_layer):
        emd = self.embedding(in_layer)
        h_out = self.hid_layer(emd)
        h_out = F.relu(h_out)
        out_ = self.out_layer(h_out)
        out_ = F.softmax(out_, dim=1)
        return out_


# def maskNLLLoss(inp, target):
#     crossEntropy = -torch.log(torch.gather(inp, 1, target))
#     loss = crossEntropy.mean()
#     return loss


def model_eval(model_out, true_label):
    confusion_matrix = torch.zeros([2, 2], dtype=torch.long)
    predict_label = torch.argmax(model_out, 1)
    accuracy = []
    precision = []
    recall = []
    f_1 = []
    for l in range(4):
        tp_num, fp_num, fn_num, tn_num = 0, 0, 0, 0
        for p, t in zip(predict_label, true_label):
            if p == t and t == l:
                tp_num += 1
            if p == l and t != l:
                fp_num += 1
            if p != l and p != t:
                fn_num += 1
            if p != l and p == t:
                tn_num += 1
        accuracy.append((tp_num + tn_num) / (tp_num + tn_num + fp_num + fn_num))
        try:
            prec = tp_num / (tp_num + fp_num)
        except:
            prec = 0.0
        try:
            rec = tp_num / (tp_num + fn_num)
        except:
            rec = 0
        precision.append(prec)
        recall.append(rec)
        if prec == 0 and rec == 0:
            f_1.append(0)
        else:
            f_1.append((2 * prec * rec) / (prec + rec))
    ave_acc = torch.tensor(accuracy, dtype=torch.float).mean()
    ave_prec = torch.tensor(precision, dtype=torch.float).mean()
    ave_rec = torch.tensor(recall, dtype=torch.float).mean()
    ave_f1 = torch.tensor(f_1, dtype=torch.float).mean()
    return ave_acc, ave_prec, ave_rec, ave_f1




def test(model_):
    text = '这首先是个民族问题，民族的感情问题。'
    hang_ = []
    for wd in text:
        hang_.append(Config().vocab[wd])
    test_tensor = torch.tensor(hang_, dtype=torch.long)
    res = model_(test_tensor)
    res = res.detach().numpy()
    # [print(np.argmax(r)) for r in res]
    print(res)


if __name__ == '__main__':
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
    losses = []
    acc = []
    rec = []
    prec = []
    f1 = []
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    loss_f = nn.CrossEntropyLoss()
    for i in tqdm(range(config.epoch)):
        for j, k in enumerate(train_data_list):
            optimizer.zero_grad()
            out = model(k)
            # loss = maskNLLLoss(out, target_list[j])
            loss = loss_f(out, target_list[j])
            loss.backward()
            optimizer.step()
            acc_, prec_, rec_, f1_ = model_eval(out, target_list[j])
            acc.append(acc_.item())
            prec.append(prec_.item())
            rec.append(rec_.item())
            f1.append(f1_.item())
            # print('acc: ' + str(acc_.item()) + '\tprec: ' + str(prec_.item()) +'\trec: ' + str(rec_.item()) + '\tf1: ' + str(f1_.item()))
            losses.append(loss.item())
        torch.save(model, './cut.bin')
    print('acc: ' + str(torch.tensor(acc).mean().item()) + '\tprec: ' + str(torch.tensor(prec).mean().item())
          +'\trec: ' + str(torch.tensor(rec).mean().item()) + '\tf1: ' + str(torch.tensor(f1).mean().item()))
    # test(model)
