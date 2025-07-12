# 导入torch工具
import torch
# 导入nn准备构建模型
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 导入torch的数据源 数据迭代器工具包
from torch.utils.data import Dataset, DataLoader
# 用于获得常见字母及字符规范化
import string
# 导入时间工具包
import time
# 引入制图工具包
import matplotlib.pyplot as plt

# 获取常见的字符
letters = string.ascii_letters + " ,.;'"
n_letters = len(letters)

# 国家名 种类数
categorys = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese',
             'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']
# 国家名 个数
categorynum = len(categorys)
print('categorys--->', categorys)

# 读取数据到内存

def read_data(filepath):
    # 定义两个空列表，一个存储x,一个存储y
    my_list_x, my_list_y = [], []
    with open(filepath, mode='r', encoding='utf-8')as f:
        lines = f.readlines()

    for line in lines:
        if len(line) <= 5:
            continue
        x, y = line.strip().split('\t')
        my_list_x.append(x)
        my_list_y.append(y)

    return my_list_x, my_list_y

# 构建Dataset

class NameClassDataset(Dataset):
    def __init__(self, my_list_x, my_list_y):
        super().__init__()
        # 样本x
        self.my_list_x = my_list_x
        # 标签y
        self.my_list_y = my_list_y
        # 获取样本长度
        self.sample_len = len(my_list_x)

    # 定义魔法方法
    def __len__(self):
        return self.sample_len

    # 定义getitem方法
    def __getitem__(self, index):
        # index防止索引溢出
        index =min(max(0, index), self.sample_len-1)

        # 根据index取出对应的样本和标签
        x = self.my_list_x[index]
        y = self.my_list_y[index]
        # print(f'x--->{x}')
        # print(f'y--->{y}')
        # 将x转换为one-hot编码张量形式
        tensor_x = torch.zeros(len(x), n_letters)
        # print(f'tensor_x-->{tensor_x}')
        for ix, letter in enumerate(x):
            tensor_x[ix][letters.find(letter)] = 1
        # print(f'tensor_x-->{tensor_x}')
        tensor_y = torch.tensor(categorys.index(y), dtype=torch.long)
        return tensor_x, tensor_y
# 测试一下dataset
def test_dataset():
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    # print(len(my_dataset))
    # print(my_dataset.__len__())
    tensor_x, tensor_y = my_dataset[0]
    # tensor_x, tensor_y = my_dataset.__getitem__(0)
    print(f'tensor_x--》{tensor_x}')
    print(f'tensor_y--》{tensor_y}')

# 定义迭代器
def get_dataloader():
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    print(f'len(my_dataloader)-->{len(my_dataloader)}')
    for tensor_x, tensor_y in my_dataloader:
        print(f'tensor_x-->{tensor_x.shape}')
        print(f'tensor_y-->{tensor_y}')
        break



if __name__ == '__main__':
   # test_dataset()
    get_dataloader()















