# coding:utf-8
from keras.preprocessing import sequence
# 获得n-gram特征
n_gram = 2

def dm_get_nGram(input_list):
    # 列表推倒式
    new_list = [input_list[i:] for i in range(n_gram)]
    print(new_list)
    results = set(zip(*new_list))
    return results
# 句子长短补齐或者截断
cut_len = 10
def dm_norm_length(data):
    #  post默认是后比如补齐或者截断，pre:默认在前面补齐或者截断
    return sequence.pad_sequences(data, cut_len, padding="post", truncating='post')


def get_norm_length(data):
    list1 = []
    for value in data:
        if len(value) > cut_len:
            list1.append(value[:cut_len])
        else:
            zero_list = [0] * (cut_len - len(value))
            new_list = value + zero_list
            list1.append(new_list)
    return list1

if __name__ == '__main__':
    # input_list = [1, 3, 2, 1, 5, 3]
    # results = dm_get_nGram(input_list)
    # print(results)
    data = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
            [2, 32, 1, 23, 1]]
    # results = dm_norm_length(data)
    results = get_norm_length(data)
    print(results)



