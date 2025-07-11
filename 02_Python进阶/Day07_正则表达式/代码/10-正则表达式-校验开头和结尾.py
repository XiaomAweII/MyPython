"""
正则表达式的规则:
     ^  代表正则的开头
     $  代表正则的结尾
"""
import re

# 需求: 演示 正则表达式的开头 和 结尾.
# 演示: ^  代表正则的开头
result = re.match('^\dit', '1it')       # Y
result = re.match('^\dit', 'a1it')      # N

result = re.search('^\dit', '1it')       # Y
result = re.search('^\dit', 'a1it')      # Y => N

# 演示: $  代表正则的结尾
# 需求: 必须以 xyz任意1个字符 或者 任意1个数字结尾
result = re.match('it[xyz0-9]', 'it1')      # Y
result = re.match('it[xyz0-9]', 'itx')      # Y
result = re.match('it[xyz0-9]', 'itxabc')   # Y
result = re.match('it[xyz0-9]$', 'itxabc')  # N

# 扩展: 校验手机号.  规则: 1. 长度必须是11位.  2. 第2位数字必须是3,4,5,6,7,8,9    3.第1位数字必须是1.  4.必须是纯数字.
result = re.match('^1[3-9]\d{9}$', '13123456789')     # Y
result = re.match('^1[3-9]\d{9}$', '131234567890')    # N
result = re.match('^1[3-9]\d{9}$', '1312345678a')     # N
result = re.match('^1[3-9]\d{9}$', '12123456789')     # N
result = re.match('^1[3-9]\d{9}$', '26123456789')     # N

# 打印匹配结果.
# if result != None:
if result:
    print(f'匹配到: {result.group()}')
else:
    print('未匹配到!')

# 三元写法.
print(f'匹配到: {result.group()}' if result else '未匹配到!')