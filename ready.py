import pickle as pkl
import time

def build_train_data(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        tmp = f.readlines()
    t = []
    for i in tmp:
        t1 = i.replace('“  ','')
        t2 = t1.replace('\n','')
        t.append(t2)
    sum_list2 = []
    for i in t:
        sum_2 = i.replace('  ','')

        sum_list2.append(sum_2)

    with open('./train_data.pkl','wb') as f:
        pkl.dump(sum_list2, f)


def build_target(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        tmp = f.readlines()

    t = []
    for i in tmp:
        t1 = i.replace('“  ','')
        t2 = t1.replace('\n','')
        t.append(t2)


    sum_list = []
    for i in t:
        sum_ = ''
        for j in i.split():
            if len(j) == 1:
                sum_ += 'S'
                continue
            else:
                sum_ += 'B'
                for k in range(1, len(j)):
                    if k == len(j) - 1:
                        sum_ += 'E'
                    else:
                        sum_ += 'M'
        sum_list.append(sum_)

    with open('./target.pkl', 'wb') as f:
        pkl.dump(sum_list, f)


def build_vocab_dict(file_path):
    vocab_dic = {}
    with open(file_path, 'rb') as f:
        z = pkl.load(f)
        for line in z:
            for hang in line:
                vocab_dic[hang] = vocab_dic.get(hang, 0) + 1
        vocab_dic_sorted = sorted(vocab_dic.items(), key=lambda x: x[1], reverse=True)

    vocab_dic2 = {word_count[0]: idx for idx, word_count in enumerate(vocab_dic_sorted)}
    with open('./vocab.pkl', 'wb') as f:
        pkl.dump(vocab_dic2, f)


if __name__ == '__main__':
    build_train_data('./train.txt')
    # time.sleep(5)
    build_target('./train.txt')
    build_vocab_dict('./train_data.pkl')