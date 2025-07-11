"""
正则规则, 如下都是 和 数量词 相关:
     ?           数量词, 代表: 前边的内容出现 0次 或者 1次
     *           数量词, 代表: 前边的内容出现 0 ~ n
     +           数量词, 代表: 前边的内容出现 1 ~ n
     {n}         数量词, 恰好n次, 多一次, 少一次都不行.
     {n,}        数量词, 至少n次, 至多无所谓
     {n,m}       数量词, 至少n次, 至多m次, 包括 n 和 m
"""
import re

# 案例: 演示正则表达式, 校验多个字符.
# \d?  要么1个\d, 要么1个\d也没有, 即: 要么1个任意的整数, 要么没有.
result = re.match('\d?it', 'ait')   # N
result = re.match('\d?it', '2it')   # Y
result = re.match('\d?it', 'it')   # Y

# 验证 *           数量词, 代表: 前边的内容出现 0 ~ n
result = re.match('\d*it', '123321it')   # Y
result = re.match('\d*it', 'it')   # Y
result = re.match('\d*it', 'ait')   # N

# 验证 +           数量词, 代表: 前边的内容出现 1 ~ n
result = re.match('\d+it', '123321it')   # Y
result = re.match('\d+it', 'it')   # N
result = re.match('\d+it', 'ait')   # N

# 验证 {n}         数量词, 恰好n次, 多一次, 少一次都不行.
result = re.match('\d{3}it', '235it')   # Y
result = re.match('\d{3}it', '2345it')   # N
result = re.match('\d{3}it', 'abcit')    # N

# 验证 {n,}       数量词, 至少n次, 至多无所谓.
result = re.match('\d{3,}it', '235it')    # Y
result = re.match('\d{3,}it', '2345it')   # Y
result = re.match('\d{3,}it', '23it')   # N
result = re.match('\d{3,}it', 'abcit')   # N

# 验证 {n,m}      数量词, 至少n次, 最多m次, 包括n 和 m
result = re.match('\d{3,5}it', '23it')    # N
result = re.match('\d{3,5}it', '234it')    # Y
result = re.match('\d{3,5}it', '2345it')    # Y
result = re.match('\d{3,5}it', '23456it')    # Y
result = re.match('\d{3,5}it', '234567it')    # N

# 打印匹配结果.
# if result != None:
if result:
    print(f'匹配到: {result.group()}')
else:
    print('未匹配到!')

# 三元写法.
print(f'匹配到: {result.group()}' if result else '未匹配到!')