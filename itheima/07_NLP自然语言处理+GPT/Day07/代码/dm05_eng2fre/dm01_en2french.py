# 用于正则表达式
import re
# 用于构建网络结构和函数的torch工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# torch中预定义的优化方法工具包
import torch.optim as optim
import time
# 用于随机生成数据
import random
import matplotlib.pyplot as plt

# 指定设备：判断是否有GPU，如果有接下来可以在GPU上进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'当前的设备是--》{device}')
# 指定特殊token
SOS_token = 0 # 开始字符
EOS_token = 1 # 结束字符
# 指定句子的最大长度（一般需要提前对语料进行长度分析）
MAX_LENGTH = 10
# 指定文件路径
data_path = './data/eng-fra-v2.txt'


# 定义数据清洗函数
def normal_str(s):
    # 将字符串s进行小写表示，去除两边的空白
    s = s.lower().strip()
    # 正则表达：将.?!前面加空格替换
    s = re.sub(r'([.!?])', r' \1', s)
    # print(f's--》{s}')
    # 正则表达式: 将非大小写字母以及正常.!?标点符号都用空格来替代
    # 在正则表达式中，+ 是一个量词，表示“前面的元素可以出现一次或多次”。
    # 当你在表达式中使用 + 时，它会匹配一个或多个连续的符合条件的字符，
    # 而没有 + 时，它只会匹配零个或一个符合条件的字符。
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s

# 定义函数获取英文和法文pair对，以及英文和法文的词典

def get_data():
    # 将文档数据读到内存中
    with open(data_path,mode='r', encoding='utf-8')as fw:
        lines = fw.readlines()
    # 列表推倒式，获取my_pair-->[[english1, french1], [english2, french2]..]-->[['i m .', 'j ai ans .'], ['i m ok .', 'je vais bien .']]
    my_pairs = [[normal_str(s) for s in l.split('\t')] for l in lines]
    # 取出前4对查看
    print(len(my_pairs))
    # 打印第8000条的英文 法文数据
    # print('my_pairs[8000][0]--->', my_pairs[8000][0])
    # print('my_pairs[8000][1]--->', my_pairs[8000][1])
    # 构建字典word2index
    english_word2index = {"SOS": 0, "EOS": 1}
    english_word_n = 2
    french_word2index = {"SOS": 0, "EOS": 1}
    french_word_n = 2
    # 遍历数据
    for pair in my_pairs:
        # print(f'pair--》{pair}')
        # 构建英文词典
        for word in pair[0].split(' '):
            if word not in english_word2index:
                english_word2index[word] = english_word_n
                english_word_n += 1
                # english_word2index[word] = len(english_word2index)
        # 构建法文词典
        for word in pair[1].split(' '):
            if word not in french_word2index:
                french_word2index[word] = french_word_n
                french_word_n += 1
                # french_word2index[word] = len(french_word2index)

    # print(f'english_word2index--》{english_word2index}')
    # print(f'english_word2index--》{len(english_word2index)}')
    # print(f'french_word2index--》{french_word2index}')
    # print(f'french_word2index--》{len(french_word2index)}')
    # 得到index2word
    english_index2word = {v: k for k, v in english_word2index.items()}
    french_index2word = {v: k for k, v in french_word2index.items()}

    return english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs

english_word2index, english_index2word, english_word_n, \
    french_word2index, french_index2word, french_word_n, my_pairs = get_data()
# 构造Dataset数据源
class MyPairDataset(Dataset):
    def __init__(self, my_pairs):
        super().__init__()
        self.my_pairs = my_pairs
        # 获取样本的总个数
        self.sample_len = len(my_pairs)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):
        # index异常值处理
        index = min(max(0, index), self.sample_len-1)
        # 根据索引取出样本
        english_x = self.my_pairs[index][0]
        french_y = self.my_pairs[index][1]

        # 将原始的句子用数字进行表示
        x = [english_word2index[word] for word in english_x.split(' ')]
        x.append(EOS_token)
        tensor_x = torch.tensor(x, dtype=torch.long, device=device)

        y = [french_word2index[word] for word in french_y.split(' ')]
        y.append(EOS_token)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)
        return tensor_x, tensor_y


# 定义函数得到dataloader对象
def get_dataloader():
    # 实例化Dataset对象
    mydataset = MyPairDataset(my_pairs)

    my_dataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)

    for i, (x, y) in enumerate(my_dataloader):
        print(f'x-->{x}')
        print(f'x-->{x.shape}')
        print(f'y-->{y}')
        print(f'y-->{y.shape}')
        break



# 定义GRU的编码器模型
class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        # vocab_size--》代表去重之后单词的总个数
        # hidden_size--》代表词嵌入的维度
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # 定义Embedding层
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
        # 定义GRU层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

    def forward(self, input, hidden):
        # input--》[1, 6]-->[1, 6, 256]
        input = self.embed(input)
        # 将数据送入gru
        output, hn = self.gru(input, hidden)
        return output, hn

    def inithidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def test_encoder():
    # 实例化Dataset对象
    mydataset = MyPairDataset(my_pairs)
    my_dataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)
    # 英文单词的总个数
    vocab_size = english_word_n
    hidden_size = 256
    my_encoder_gru = EncoderGRU(vocab_size, hidden_size)
    my_encoder_gru = my_encoder_gru.to(device) # 将模型送入GPU
    for i, (x, y) in enumerate(my_dataloader):
        print(f'x-->{x}')
        print(f'x-->{x.shape}')
        h0 = my_encoder_gru.inithidden()
        output, hn = my_encoder_gru(x, h0)
        print(f'output--》{output.shape}')
        print(f'hn--》{hn.shape}')
        break


# 定义GRU的解码器模型：无Attention
class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        # vocab_size代表法文单词的总个数（去重）；hidden_size指定的词向量输出维度
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # 定义embedding层
        self.embed = nn.Embedding(vocab_size, hidden_size)
        # 定义GRU层
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # 定义linear层：输出层
        self.linear = nn.Linear(hidden_size, vocab_size)

        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        #  解码的时候是一个字符一个字符输入然后去预测的 input--》[1, 1]
        # input--》[1, 1]-->embedding-->[1, 1, hidden_size]
        input = self.embed(input)
        # 将数据送入gru模型output--》[1, 1, 256]
        output, hidden = self.gru(input, hidden)
        # 需要对output的结果输入输出层得到预测结果:results-->[1, 4345]
        results = self.linear(output[0])
        return self.softmax(results), hidden

# 测试解码器
def test_decoder():
    # 获取数据
    # 实例化Dataset对象
    mydataset = MyPairDataset(my_pairs)
    my_dataloader = DataLoader(dataset=mydataset, batch_size=1, shuffle=True)
    # 实例化编码器模型：注意对英文进行编码
    english_vocab_size = english_word_n
    hidden_size = 256
    my_encoder_gru = EncoderGRU(english_vocab_size, hidden_size)
    my_encoder_gru = my_encoder_gru.to(device)  # 将模型送入GPU
    print(f'编码器模型的架构-->{my_encoder_gru}')
    # 实例化解码器模型
    french_vocab_size = french_word_n
    hidden_size1 = 256
    my_decoder_gru = DecoderGRU(french_vocab_size, hidden_size1)
    my_decoder_gru = my_decoder_gru.to(device)  # 将模型送入GPU
    print(f'解码器模型的架构-->{my_decoder_gru}')
    # 将数据送入编码器得到编码的结果
    print('*'*80)
    for i, (x, y) in enumerate(my_dataloader):
        print(f'x-->{x}')
        print(f'y-->{y}')
        print(f'y-->{y.shape}')
        # 需要对x进行编码：一次性送入编码器
        encoder_output, hidden = my_encoder_gru(x, my_encoder_gru.inithidden())
        print(f'encoder_output-->{encoder_output.shape}')
        print(f'hidden-->{hidden.shape}')

        # 需要对y进行解码：一个字符一个字符的解码，
        # 注意：当你进行第一个字符解码的时候，我们用到的上一时间步隐藏层输出结果是编码器最后一个单词的hidden
        for i in range(y.shape[1]):
            temp = y[0][i].view(1, -1)
            output, hidden = my_decoder_gru(temp, hidden)
            print(f'output--》{output.shape}')
            print(f'hidden--》{hidden.shape}')

        break

# 定义带Attention的解码器
class AttentionDecoder(nn.Module):
    def __init__(self,vocab_size, hidden_size, dropout_p=0.1, max_len=MAX_LENGTH):
        super().__init__()
        # vocab_size-——>法文单词去重之后的总个数
        self.vocab_size = vocab_size
        # hidden_size--->单词的词嵌入维度
        self.hidden_size =hidden_size
        # dropout_p -->随机失活的概率
        self.dropout_p = dropout_p
        # 定义句子最大长度：统一句子长度
        self.max_len = max_len

        # 定义Embbedding层
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
        # 定义Dropout层
        self.dropout = nn.Dropout(p=self.dropout_p)
        # 计算注意力第一步：按照第一种计算规则， 定义一个全连接层
        self.atten = nn.Linear(self.hidden_size*2, self.max_len)
        # 计算注意力第二步：因为第一步计算过程有拼接，
        # 我们需要对Q和第一步计算的结果再次拼接，然后再线形变化，按照指定维度输出
        self.atten_combin = nn.Linear(self.hidden_size*2, self.hidden_size)
        # 定义GRU层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        # 定义输出层out
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, Q, K, V):
        # Q:代表解码器上一时间步预测的结果
        # K:代表解码器上一时间步隐藏层的输出结果
        # V:代表编码器的编码结果（每一个时间步隐藏层输出结果合并）
        # 先把Q进行升维度-->Q--》[1, 1]-->[1,1,256]
        input = self.embed(Q)
        # 进行随机失活
        embeded = self.dropout(input)
        # embeded就是query-->新Q————>要和K【1，1， 256】拼接，经过linear变化，再softmax归一化
        # atten_weight--.shape--》[1, 10]
        atten_weight = F.softmax(self.atten(torch.cat((embeded[0], K[0]), dim=-1)), dim=-1)

        # 将权重值和V进行矩阵相乘：atten_weight--.shape--》[1, 10]和V--》[10, 256]-->相乘得到atten_applied————shape-->[1, 1,256]
        atten_applied = torch.bmm(atten_weight.unsqueeze(dim=0), V.unsqueeze(dim=0))
        # 按照注意力计算步骤第二步，query和atten_applied需要再次拼接:combin_result-->[1, 512]
        combin_result = torch.cat((embeded[0], atten_applied[0]), dim=-1)
        # 按照注意力计算步骤第三步：将上述结果combin_result经过全连接层，按照指定维度输出:output-->[1, 1, 256]
        gru_input = self.atten_combin(combin_result).unsqueeze(dim=0)
        # 将上述结果进行relu激活函数
        gru_input = F.relu(gru_input)

        # 数据送入gru模型，K即使hidden；output--》[1, 1, 256]
        output, hidden = self.gru(gru_input, K)
        # 将gru模型的输出结果送入输出层:result-->【1， 4345】
        result = self.out(output[0])
        return self.softmax(result), hidden, atten_weight

    def inithidden(self):
        # return torch.zeros(1, 1, self.hidden_size).to(device=device)
        return torch.zeros(1, 1, self.hidden_size, device=device)







if __name__ == '__main__':
    # get_dataloader()
    # test_encoder()
    test_decoder()