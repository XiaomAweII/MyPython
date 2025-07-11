"""
需求:
    键盘录入1个字符串, 并接收, 统计其中每个字符的次数, 并将结果打印到控制台上.
"""

# 1. 键盘录入1个字符串,并接收.
s = input('请录入1个字符串, 我来统计每个字符的次数: ')        # 假设: aaabbc

# 2. 定义字典, 记录每个字符 及其 次数.   字符做键, 次数做值, 例如: 'a':3, 'b':2, 'A':1
# wc_dict = {}        # word count: 单词数量      hello world python sql linux

# 方式1: 分解版
# # 3. 遍历上述的字符串, 获取到每个字符, 充当字典的键.
# for key in s:     # i的值: 'a', 'a', 'a', 'b', 'b', 'c'
#     # 4. 核心: 判断字典中是否有这个键, 有就将其次数 + 1 重新存储.
#     if key in wc_dict:
#         # 例如: 'a': 2 => 'a': 3
#         wc_dict[key] = wc_dict[key] + 1
#     else:
#         # 5. 没有说明这个键是第一次出现, 就将其次数记录为1
#         # 例如: 'b', 1
#         wc_dict[key] = 1

# 方式2: 三元运算符
# 3. 遍历上述的字符串, 获取到每个字符, 充当字典的键.
# for key in s:     # i的值: 'a', 'a', 'a', 'b', 'b', 'c'
#     wc_dict[key] = wc_dict[key] + 1  if key in wc_dict else 1

# 方式3: 字典推导式
wc_dict = {key: s.count(key) for key in s}      # key是字符串中的每个字符.

# 6. 循环结束后, 打印字典即可.
print(wc_dict)