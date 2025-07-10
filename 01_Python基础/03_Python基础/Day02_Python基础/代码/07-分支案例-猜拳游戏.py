# 案例: 猜拳游戏.
# 需求: 键盘录入玩家出的手势, 且和 电脑人的手势(随机生成), 进行比较, 打印结果.


# 铺垫知识, 如何获取一个指定范围内的随机数.
# 1. 导包, 即: 通过第三方的包来实现.
import random

# 2. 通过 random.randint(a, b) 函数实现 获取随机数.
print(random.randint(1, 3))     # 包左包右, 生成 1 ~ 3之间的随机数.
print('-' * 28)

# 具体的完成 猜拳游戏案例, 规则: 石头 -> 1, 剪刀 -> 2, 布 -> 3.
# 1. 键盘录入玩家的手势编号, 并接收. 记得转成数值.
player = int(input('请输入您的手势编号, 规则: 石头(1), 剪刀(2), 布(3)  => '))

# 2. 随机生成 电脑人的 手势编号.
pc = random.randint(1, 3)
# print(f'电脑人的手势编号为: {pc}')

# 3. 比较 玩家 和 电脑人的手势编号, 并打印结果.
# 情况1: 玩家胜利.   玩家:石头, 电脑: 剪刀,    玩家:剪刀, 电脑:布,    玩家: 布, 电脑: 石头
if (player == 1 and pc == 2) or (player == 2 and pc == 3) or (player==3 and pc == 1):
    print('玩家 获得胜利')
elif pc == player:
    # 情况2: 平局.
    print('哎呀, 平局了!')
elif (player == 1 and pc == 3) or (player == 2 and pc == 1) or (player==3 and pc == 2):
    # 情况3: 电脑人 胜利
    print('电脑人 获得胜利!')
else:
    print('手势编号有误, 该轮无成绩!')
