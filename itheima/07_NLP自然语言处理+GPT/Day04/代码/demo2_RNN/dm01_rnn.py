# coding:utf-8
import torch
import torch.nn as nn
torch.manual_seed(1)
def dm_rnn_base():
    # 实例化RNN模型
    # input_size代表输入x的词嵌入维度
    # hidden_size代表rnn单元（隐藏层）输出的维度
    # num_layers代表几个rnn单元（隐藏层）
    my_rnn = nn.RNN(input_size=5, hidden_size=8, num_layers=1)

    # 准备数据
    # input的参数解析：
    # 第一个参数代表：sequence_length(句子长度)
    # 第二个参数代表：batch_size(一个批次的样本个数)
    # 第三个参数代表：input_size(embed_dim)(词嵌入维度)
    input = torch.randn(1, 3, 5)
    # h0的参数解析：
    # 第一个参数代表：num_layers代表几个rnn单元（隐藏层）
    # 第二个参数代表：batch_size(一个批次的样本个数)
    # 第三个参数代表：hidden_size代表rnn单元（隐藏层）输出的维度
    h0 = torch.randn(1, 3, 8)

    # 将数据送入模型
    output, hn = my_rnn(input, h0)
    print(f'output--》{output}')
    print(f'hn--》{hn}')

# 修改长度
def dm_rnn_length():
    # 实例化RNN模型
    # input_size代表输入x的词嵌入维度
    # hidden_size代表rnn单元（隐藏层）输出的维度
    # num_layers代表几个rnn单元（隐藏层）
    my_rnn = nn.RNN(input_size=5, hidden_size=6, num_layers=1)

    # 准备数据
    # input的参数解析：
    # 第一个参数代表：sequence_length(句子长度)
    # 第二个参数代表：batch_size(一个批次的样本个数)
    # 第三个参数代表：input_size(embed_dim)(词嵌入维度)
    input = torch.randn(8, 3, 5)
    # h0的参数解析：
    # 第一个参数代表：num_layers代表几个rnn单元（隐藏层）
    # 第二个参数代表：batch_size(一个批次的样本个数)
    # 第三个参数代表：hidden_size代表rnn单元（隐藏层）输出的维度
    h0 = torch.randn(1, 3, 6)

    # 将整个数据送入模型
    output, hn = my_rnn(input, h0)
    print(f'output--》{output}')
    print(f'hn--》{hn}')


# 修改batch_first是否等于True,如果为True,将bacth_size放在第一位，默认是False
def dm_rnn_batch():
    # 实例化RNN模型
    # input_size代表输入x的词嵌入维度
    # hidden_size代表rnn单元（隐藏层）输出的维度
    # num_layers代表几个rnn单元（隐藏层）
    my_rnn = nn.RNN(input_size=5, hidden_size=6, num_layers=1, batch_first=True)

    # 准备数据
    # input的参数解析：
    # 第一个参数代表：batch_size(一个批次的样本个数)
    # 第二个参数代表：sequence_length(句子长度)
    # 第三个参数代表：input_size(embed_dim)(词嵌入维度)
    input = torch.randn(2, 3, 5)
    # h0的参数解析：
    # 第一个参数代表：num_layers代表几个rnn单元（隐藏层）
    # 第二个参数代表：batch_size(一个批次的样本个数)
    # 第三个参数代表：hidden_size代表rnn单元（隐藏层）输出的维度
    h0 = torch.randn(1, 2, 6)

    # 将整个数据送入模型
    output, hn = my_rnn(input, h0)
    print(f'output--》{output}')
    print(f'output--》{output.shape}')
    print('*'*80)
    print(f'hn--》{hn}')
    print(f'hn--》{hn.shape}')


# 理解RNN将数据一次性送入样本和一个字符一个字符送入样本的区别和联系
def dm_rnn_oneAll():
    # 实例化RNN模型
    # input_size代表输入x的词嵌入维度
    # hidden_size代表rnn单元（隐藏层）输出的维度
    # num_layers代表几个rnn单元（隐藏层）
    my_rnn = nn.RNN(input_size=5, hidden_size=6,
                    num_layers=1, batch_first=True)

    # 准备数据
    # input的参数解析：
    # 第一个参数代表：batch_size(一个批次的样本个数)
    # 第二个参数代表：sequence_length(句子长度)
    # 第三个参数代表：input_size(embed_dim)(词嵌入维度)
    input = torch.randn(1, 3, 5)
    # print(f'input-->{input}')
    # h0的参数解析：
    # 第一个参数代表：num_layers代表几个rnn单元（隐藏层）
    # 第二个参数代表：batch_size(一个批次的样本个数)
    # 第三个参数代表：hidden_size代表rnn单元（隐藏层）输出的维度
    hidden = torch.randn(1, 1, 6)

    # 将整个数据送入模型
    output, hn = my_rnn(input, hidden)
    print(f'output--》{output}')
    print(f'output--》{output.shape}')
    print('*'*80)
    print(f'hn--》{hn}')
    print(f'hn--》{hn.shape}')
    print("--"*40)
    # 一个字符一个字符的送入模型
    for i in range(3):
        temp = input[0][i]
        # temp = input[0, i]
        # print(f'temp--》{temp}')
        temp_vc = temp.unsqueeze(dim=0).unsqueeze(dim=0)
        output, hidden = my_rnn(temp_vc, hidden)
        print(f'output-->{output}')
        print(f'hidden-->{hidden}')
        print('*'*80)


# 理解num_layers
def dm_rnn_numLayers():
    # 实例化RNN模型
    # input_size代表输入x的词嵌入维度
    # hidden_size代表rnn单元（隐藏层）输出的维度
    # num_layers代表几个rnn单元（隐藏层）
    my_rnn = nn.RNN(input_size=5, hidden_size=8, num_layers=2)

    # 准备数据
    # input的参数解析：
    # 第一个参数代表：sequence_length(句子长度)
    # 第二个参数代表：batch_size(一个批次的样本个数)
    # 第三个参数代表：input_size(embed_dim)(词嵌入维度)
    input = torch.randn(4, 3, 5)
    # h0的参数解析：
    # 第一个参数代表：num_layers代表几个rnn单元（隐藏层）
    # 第二个参数代表：batch_size(一个批次的样本个数)
    # 第三个参数代表：hidden_size代表rnn单元（隐藏层）输出的维度
    h0 = torch.randn(2, 3, 8)

    # 将数据送入模型
    output, hn = my_rnn(input, h0)
    print(f'output--》{output}')
    print(f'hn--》{hn}')





if __name__ == '__main__':
    # dm_rnn_base()
    # dm_rnn_length()
    # dm_rnn_batch()
    # dm_rnn_oneAll()
    dm_rnn_numLayers()