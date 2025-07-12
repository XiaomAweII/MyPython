# 导入torch工具
import json

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
from tqdm import tqdm
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

# 定义RNN模型
class My_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        # input_size-->【词嵌入维度】
        self.input_size = input_size
        # hidden_size-->【RNN模型输出的隐藏层维度】
        self.hidden_size = hidden_size
        # output_size-->【最终输出层单元个数：18】
        self.output_size = output_size
        # num_layers-->【RNN隐藏层个数】
        self.num_layers = num_layers
        # 定义rnn层
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers)

        # 定义全连接层（输出层）：
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        # input输入的是二维的：shape:【6, 57】，代表6个单词，每个单词用57个数字表示
        # 但是rnn模型接受的输入是三维的，因此需要升维
        input = input.unsqueeze(1) # [6, 1, 57]
        # 将input和hidden送入模型
        output, hn = self.rnn(input, hidden)
        # print(f'output--》{output.shape}')
        # print('output', output)
        # print(f'hn--》{hn.shape}')
        # print(f'output[-1]--》{output[-1].shape}')
        # print('output[-1]', output[-1])
        result = self.linear(output[-1])
        # print(f'result-->{result.shape}')
        # 经过log_softmax
        return self.softmax(result), hn
    # 初始化hidden
    def inithidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)



# 定义LSTM模型
class My_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        # input_size-->【词嵌入维度】
        self.input_size = input_size
        # hidden_size-->【RNN模型输出的隐藏层维度】
        self.hidden_size = hidden_size
        # output_size-->【最终输出层单元个数：18】
        self.output_size = output_size
        # num_layers-->【RNN隐藏层个数】
        self.num_layers = num_layers
        # 定义lstm层
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,
                            self.num_layers)

        # 定义全连接层（输出层）：
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, c):
        input = input.unsqueeze(dim=1)
        # 经过lstm模型
        output, (hn, cn) = self.lstm(input, (hidden, c))

        # 取出最后一个单词的隐藏层张量表示
        temp = output[-1]
        result = self.linear(temp)
        return self.softmax(result), hn, cn

    def inithidden(self):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        return h0, c0


# 定义GRU模型
class My_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        # input_size-->【词嵌入维度】
        self.input_size = input_size
        # hidden_size-->【RNN模型输出的隐藏层维度】
        self.hidden_size = hidden_size
        # output_size-->【最终输出层单元个数：18】
        self.output_size = output_size
        # num_layers-->【RNN隐藏层个数】
        self.num_layers = num_layers
        # gru
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers)

        # 定义全连接层（输出层）：
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        # input输入的是二维的：shape:【6, 57】，代表6个单词，每个单词用57个数字表示
        # 但是rnn模型接受的输入是三维的，因此需要升维
        input = input.unsqueeze(1) # [6, 1, 57]
        # 将input和hidden送入模型
        output, hn = self.gru(input, hidden)
        # print(f'output--》{output.shape}')
        # print('output', output)
        # print(f'hn--》{hn.shape}')
        # print(f'output[-1]--》{output[-1].shape}')
        # print('output[-1]', output[-1])
        result = self.linear(output[-1])
        # print(f'result-->{result.shape}')
        # 经过log_softmax
        return self.softmax(result), hn
    # 初始化hidden
    def inithidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


# 测试一下模型
def test_model():
    # 实例化模型
    # my_rnn = My_RNN(input_size=57,hidden_size=128, output_size=18)
    # my_lstm = My_LSTM(input_size=57,hidden_size=128, output_size=18)
    my_gru = My_GRU(input_size=57,hidden_size=128, output_size=18)
    # 准备数据
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    print(f'len(my_dataloader)-->{len(my_dataloader)}')
    for tensor_x, tensor_y in my_dataloader:
        # 从my_dataloader里面迭代出来tensor_x-->shape-->【1, 5, 57】-->batch_size,seq_len, input_size
        input = tensor_x[0]
        hidden = my_gru.inithidden()
        output, hn = my_gru(input, hidden)
        print(f'output--》{output.shape}')
        break

# 定义全局变量参数：
my_lr = 1e-3
epochs = 1
# 训练rnn模型
def train_rnn():
    # 读取数据
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    # 实例化dataset数据源对象
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    # 实例化模型
    # n_letters=57, hidden_size=128,类别总数output_size=18
    my_rnn = My_RNN(input_size=57, hidden_size=128, output_size=18)
    # 实例化损失函数对象
    my_nll_loss = nn.NLLLoss()
    # 实例化优化器对象
    my_optim = optim.Adam(my_rnn.parameters(), lr=my_lr)
    # 定义打印日志的参数
    start_time = time.time()
    total_iter_num = 0 # 当前已经训练的样本总数
    total_loss = 0  # 已经训练的损失值
    total_loss_list = [] # 每隔n个样本，保存平均损失值
    total_acc_num = 0 # 预测正确的样本个数
    total_acc_list = [] # 每隔n个样本，保存平均准确率
    # 开始训练
    for epoch_idx in range(epochs):
        # 实例化dataloader
        my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
        # 开始内部迭代数据，送入模型
        for i, (x, y) in enumerate(tqdm(my_dataloader)):
            # print(f'x--》{x.shape}')
            # print(f'y--》{y}')
            output, hn = my_rnn(input=x[0], hidden=my_rnn.inithidden())
            # print(f'output--》{output}') # [1, 18]
            # 计算损失
            my_loss = my_nll_loss(output, y)
            # print(f'my_loss--》{my_loss}')
            # print(f'my_loss--》{type(my_loss)}')

            # 梯度清零
            my_optim.zero_grad()
            # 反向传播
            my_loss.backward()
            # 梯度更新
            my_optim.step()

            # 统计一下已经训练样本的总个数
            total_iter_num = total_iter_num + 1

            # 统计一下已经训练样本的总损失
            total_loss = total_loss + my_loss.item()

            # 统计已经训练的样本中预测正确的个数
            i_predict_num = 1 if torch.argmax(output).item() == y.item() else 0
            total_acc_num = total_acc_num + i_predict_num
            # 每隔100次训练保存一下平均损失和准确率
            if total_iter_num % 100 == 0:
                avg_loss = total_loss / total_iter_num
                total_loss_list.append(avg_loss)

                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)
            # 每隔2000次训练打印一下日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_acc_num / total_iter_num
                temp_time = time.time() - start_time
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' %(epoch_idx+1, temp_loss, temp_time, temp_acc))
        torch.save(my_rnn.state_dict(), './save_model/ai20_rnn_%d.bin'%(epoch_idx+1))
    # 计算总时间
    total_time = int(time.time() - start_time)
    print('训练总耗时：', total_time)
    # 将结果保存到文件中
    dict1 = {"avg_loss":total_loss_list,
             "all_time": total_time,
             "avg_acc": total_acc_list}
    with open('./save_results/ai_rnn.json', 'w') as fw:
        fw.write(json.dumps(dict1))

    return total_loss_list, total_time, total_acc_list


# 训练lstm模型
def train_lstm():
    # 读取数据
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    # 实例化dataset数据源对象
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    # 实例化模型
    # n_letters=57, hidden_size=128,类别总数output_size=18
    my_lstm = My_LSTM(input_size=57, hidden_size=128, output_size=18)
    # 实例化损失函数对象
    my_nll_loss = nn.NLLLoss()
    # 实例化优化器对象
    my_optim = optim.Adam(my_lstm.parameters(), lr=my_lr)
    # 定义打印日志的参数
    start_time = time.time()
    total_iter_num = 0 # 当前已经训练的样本总数
    total_loss = 0  # 已经训练的损失值
    total_loss_list = [] # 每隔n个样本，保存平均损失值
    total_acc_num = 0 # 预测正确的样本个数
    total_acc_list = [] # 每隔n个样本，保存平均准确率
    # 开始训练
    for epoch_idx in range(epochs):
        # 实例化dataloader
        my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
        # 开始内部迭代数据，送入模型
        for i, (x, y) in enumerate(tqdm(my_dataloader)):
            # print(f'x--》{x.shape}')
            # print(f'y--》{y}')
            hidden, cn = my_lstm.inithidden()
            output, hn, cn = my_lstm(input=x[0],  hidden=hidden, c=cn)
            # print(f'output--》{output}') # [1, 18]
            # 计算损失
            my_loss = my_nll_loss(output, y)
            # print(f'my_loss--》{my_loss}')
            # print(f'my_loss--》{type(my_loss)}')

            # 梯度清零
            my_optim.zero_grad()
            # 反向传播
            my_loss.backward()
            # 梯度更新
            my_optim.step()

            # 统计一下已经训练样本的总个数
            total_iter_num = total_iter_num + 1

            # 统计一下已经训练样本的总损失
            total_loss = total_loss + my_loss.item()

            # 统计已经训练的样本中预测正确的个数
            i_predict_num = 1 if torch.argmax(output).item() == y.item() else 0
            total_acc_num = total_acc_num + i_predict_num
            # 每隔100次训练保存一下平均损失和准确率
            if total_iter_num % 100 == 0:
                avg_loss = total_loss / total_iter_num
                total_loss_list.append(avg_loss)

                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)
            # 每隔2000次训练打印一下日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_acc_num / total_iter_num
                temp_time = time.time() - start_time
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' %(epoch_idx+1, temp_loss, temp_time, temp_acc))
        torch.save(my_lstm.state_dict(), './save_model/ai20_lstm_%d.bin'%(epoch_idx+1))
    # 计算总时间
    total_time = int(time.time() - start_time)
    print('训练总耗时：', total_time)

    # 将结果保存到文件中
    dict1 = {"avg_loss":total_loss_list,
             "all_time": total_time,
             "avg_acc": total_acc_list}
    with open('./save_results/ai_lstm.json', 'w') as fw:
        fw.write(json.dumps(dict1))
    return total_loss_list, total_time, total_acc_list


# 训练gru模型
def train_gru():
    # 读取数据
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    # 实例化dataset数据源对象
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    # 实例化模型
    # n_letters=57, hidden_size=128,类别总数output_size=18
    my_gru = My_GRU(input_size=57, hidden_size=128, output_size=18)
    # 实例化损失函数对象
    my_nll_loss = nn.NLLLoss()
    # 实例化优化器对象
    my_optim = optim.Adam(my_gru.parameters(), lr=my_lr)
    # 定义打印日志的参数
    start_time = time.time()
    total_iter_num = 0 # 当前已经训练的样本总数
    total_loss = 0  # 已经训练的损失值
    total_loss_list = [] # 每隔n个样本，保存平均损失值
    total_acc_num = 0 # 预测正确的样本个数
    total_acc_list = [] # 每隔n个样本，保存平均准确率
    # 开始训练
    for epoch_idx in range(epochs):
        # 实例化dataloader
        my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
        # 开始内部迭代数据，送入模型
        for i, (x, y) in enumerate(tqdm(my_dataloader)):
            # print(f'x--》{x.shape}')
            # print(f'y--》{y}')
            output, hn = my_gru(input=x[0], hidden=my_gru.inithidden())
            # print(f'output--》{output}') # [1, 18]
            # 计算损失
            my_loss = my_nll_loss(output, y)
            # print(f'my_loss--》{my_loss}')
            # print(f'my_loss--》{type(my_loss)}')

            # 梯度清零
            my_optim.zero_grad()
            # 反向传播
            my_loss.backward()
            # 梯度更新
            my_optim.step()

            # 统计一下已经训练样本的总个数
            total_iter_num = total_iter_num + 1

            # 统计一下已经训练样本的总损失
            total_loss = total_loss + my_loss.item()

            # 统计已经训练的样本中预测正确的个数
            i_predict_num = 1 if torch.argmax(output).item() == y.item() else 0
            total_acc_num = total_acc_num + i_predict_num
            # 每隔100次训练保存一下平均损失和准确率
            if total_iter_num % 100 == 0:
                avg_loss = total_loss / total_iter_num
                total_loss_list.append(avg_loss)

                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)
            # 每隔2000次训练打印一下日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_acc_num / total_iter_num
                temp_time = time.time() - start_time
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' %(epoch_idx+1, temp_loss, temp_time, temp_acc))
        torch.save(my_gru.state_dict(), './save_model/ai20_gru_%d.bin'%(epoch_idx+1))
    # 计算总时间
    total_time = int(time.time() - start_time)
    print('训练总耗时：', total_time)
    # 将结果保存到文件中
    dict1 = {"avg_loss":total_loss_list,
             "all_time": total_time,
             "avg_acc": total_acc_list}
    with open('./save_results/ai_gru.json', 'w') as fw:
        fw.write(json.dumps(dict1))
    return total_loss_list, total_time, total_acc_list


def read_json(data_path):
    with open(data_path, 'r')as fr:
        rnn_results = json.loads(fr.read())
    avg_loss = rnn_results["avg_loss"]
    all_time = rnn_results["all_time"]
    avg_acc = rnn_results["avg_acc"]
    return avg_loss, all_time, avg_acc
# 绘制图像
def dm_show_results():
    # 读取rnn系列模型的训练结果（保存的日志结果）
    rnn_avg_loss, rnn_all_time, rnn_avg_acc = read_json("./save_results/ai_rnn.json")
    lstm_avg_loss, lstm_all_time, lstm_avg_acc = read_json("./save_results/ai_lstm.json")
    gru_avg_loss, gru_all_time, gru_avg_acc = read_json("./save_results/ai_gru.json")
    # 对比不同模型的损失
    plt.figure(0)
    plt.plot(rnn_avg_loss, label='RNN')
    plt.plot(lstm_avg_loss, label='LSTM', color='red')
    plt.plot(gru_avg_loss, label='GRU', color='orange')
    plt.legend(loc='upper left')
    plt.savefig('./img/loss.png')
    plt.show()
    # 对比不同模型的耗时
    plt.figure(1)
    x_data = ["RNN", "LSTM", "GRU"]
    y_data = [rnn_all_time, lstm_all_time, gru_all_time]
    plt.bar(range(len(x_data)), y_data, tick_label=x_data)
    plt.savefig('./img/time.png')
    plt.show()
    # 对比不同模型的准确率
    plt.figure(2)
    plt.plot(rnn_avg_acc, label='RNN')
    plt.plot(lstm_avg_acc, label='LSTM', color='red')
    plt.plot(gru_avg_acc, label='GRU', color='orange')
    plt.legend(loc='upper left')
    plt.savefig('./img/acc.png')
    plt.show()

if __name__ == '__main__':
   # test_dataset()
   #  get_dataloader()
   #  test_model()
   #  train_rnn()
   #  train_lstm()
   #  train_gru()
   dm_show_results()











