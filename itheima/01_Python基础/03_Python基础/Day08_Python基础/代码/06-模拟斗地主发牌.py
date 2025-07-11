# 案例: 模拟斗地主发牌.
import random

# 定义变量, 表示扑克牌.
poker_dict = {}  # 键: 牌的索引, 值: 具体的牌.  规则: 牌越小, 索引越小.
poker_index = []  # 所有的 牌的索引, 我们发的是这个, 看牌是: 排序后, 根据键找值.
p1 = []  # 玩家1
p2 = []  # 玩家2
p3 = []  # 玩家3
dp = []  # 底牌


# 1. 买牌.
def get_poker():
    global poker_dict
    # 1.1 定义 花色列表.
    color_list = ['♠', '♥', '♦', '♣']
    # 1.2 定义 点数列表.
    num_list = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
    # 1.3 生成 字典, 键: 索引, 值: 牌.   规则: 牌越小, 索引越小.
    # 列表
    poker_list = [color + num for num in num_list for color in color_list]
    # 字典
    poker_dict = {i: poker_list[i] for i in range(len(poker_list))}
    # 添加大小往.
    poker_dict[52] = '小🤡'
    poker_dict[53] = '大🤡'
    # print(poker_dict)


# 2. 洗牌.
def shuffle_poker():
    global poker_index
    # 获取所有牌的索引.
    poker_index = list(poker_dict.keys())
    # print(poker_index)

    # 具体的洗牌动作.
    random.shuffle(poker_index)
    # print(poker_index)


# 3. 发牌
def send_poker():
    global p1, p2, p3, dp
    # 规则: 最后3张做底牌, 其它轮询发送.
    for i in range(len(poker_index)):  # i就是 打乱顺序后的牌的编号的 索引
        # 发送底牌
        if i >= len(poker_index) - 3:
            dp.append(poker_index[i])
        elif i % 3 == 0:
            p1.append(poker_index[i])
        elif i % 3 == 1:
            p2.append(poker_index[i])
        else:
            p3.append(poker_index[i])


# 4. 看牌.
def look_poker(player_name, player_poker_num):
    """
    根据玩家手中 牌的编号, 取 牌盒poker_dict中找 牌.
    :param player_name: 玩家名
    :param player_poker_num: 玩家手中的牌的编号.
    :return:
    """
    # 4.1 排序.
    player_poker_num.sort()
    # 4.2 玩家手中具体的牌.
    player_poker = [poker_dict[i] for i in player_poker_num]
    # 4.3 打印结果
    print(f'{player_name}的牌是: {player_poker}')


# 在main函数中调用.
if __name__ == '__main__':
    # 1. 买牌
    get_poker()
    # 2. 洗牌
    shuffle_poker()
    # 3. 发牌
    send_poker()
    # 看牌
    look_poker('刘亦菲', p1)
    look_poker('赵丽颖', p2)
    look_poker('张小二', p3)
    look_poker('底牌', dp)
