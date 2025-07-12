import torch
import torch.nn as nn

def dm_lstm_use():
    # 实例化对象
    # 第一个参数：input_size代表输入张量的词嵌入维度
    # 第二个参数：hidden_size代表隐藏层输出维度
    # 第三个参数：num_layers代表隐藏层的个数
    lstm = nn.LSTM(input_size=5, hidden_size=6, num_layers=2)

    # 定义输入
    # input参数说明：
    # 第一个参数：sequence_length代表句子长度
    # 第二个参数：batch_size代表一个批次的样本个数
    # 第三个参数：input_size代表输入张量的词嵌入维度
    input = torch.randn(4, 3, 5)
    # h0和c0参数说明：
    # 第一个参数：num_layers代表隐藏层的个数
    # 第二个参数：batch_size代表一个批次的样本个数
    # 第三个参数：hidden_size代表隐藏层输出维度
    h0 = torch.randn(2, 3, 6)
    c0 = torch.randn(2, 3, 6)

    # 将数据送入模型

    output, (hn, cn) = lstm(input, (h0, c0))
    print(f'output--》{output}')
    print(f'hn--》{hn}')
    print(f'cn--》{cn}')

if __name__ == '__main__':
    dm_lstm_use()
