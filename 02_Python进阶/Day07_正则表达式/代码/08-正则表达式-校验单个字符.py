"""
正则规则:
    .           任意的1个字符, 除了\n
    \.          取消.的特殊用法, 就是1个普通的.
    a           代表1个字符a
    [abc]       代表: a,b,c中任意的1个字符
    [^abc]      代表: 除了a,b,c以外的任意1个字符

    [0-9]       代表: 任意的1个整数, 例如: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    \d          代表: 任意的1个整数, 效果同上.  \d = [0-9]
    \D          代表: 除了整数外的任意1个字符, 即: \D = [^0-9]
    \s          代表: 空白字符, 例如: 空格, tab键等...
    \S          代表: 非空白字符.
    \w          代表: 非特殊字符, 例如: 字母, 数字, 下划线(_), 汉字
    \W          代表: 特殊字符.
"""
import re

# 案例: 正则校验 单个字符.
# 演示 .           任意的1个字符, 除了\n
result = re.match('it.', 'ita')     # 可以匹配(Y)
result = re.match('it.', 'it1')     # 可以匹配(Y)
result = re.match('it.', 'it\t')     # 可以匹配(Y)
result = re.match('it.', 'it\n')     # 不可以匹配(N)

# 演示 \.          取消.的特殊用法, 就是1个普通的.
result = re.match('hm\.', 'hma')     # N
result = re.match('hm\.', 'hm.')     # Y

# 演示 [abc]       代表: a,b,c中任意的1个字符
result = re.match('[abc]hm', 'ahm')     # Y
result = re.match('[abc]hm', 'bhm')     # Y
result = re.match('[abc]hm', 'xhm')     # N

# 演示 [^abc]      代表: 除了a,b,c以外的任意1个字符
result = re.match('[^abc]hm', 'bhm')     # N
result = re.match('[^abc]hm', 'xhm')     # Y

# 演示 [0-9]       代表: 任意的1个整数, 例如: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
result = re.match('[0-9]hm', '1hm')     # Y
result = re.match('[0-9]hm', '6hm')     # Y
result = re.match('[0-9]hm', 'ahm')     # N

# 演示 \d          代表: 任意的1个整数, 效果同上.  \d = [0-9]
result = re.match('\dhm', '6hm')     # Y
result = re.match('\dhm', 'ahm')     # N

# 演示 \D          代表: 除了整数外的任意1个字符, 即: \D = [^0-9]
result = re.match('\Dhm', '6hm')     # N
result = re.match('\Dhm', 'ahm')     # Y

# 演示 \s          代表: 空白字符, 例如: 空格, tab键等...
result = re.match('\shm', '\nhm')     # Y
result = re.match('\shm', ' hm')      # Y
result = re.match('\shm', '\thm')     # Y
result = re.match('\shm', 'ahm')      # N

# 演示 \S          代表: 非空白字符.
result = re.match('\Shm', '\nhm')     # N
result = re.match('\Shm', ' hm')      # N
result = re.match('\Shm', '\thm')     # N
result = re.match('\Shm', 'ahm')      # Y

# 演示 \w          代表: 非特殊字符, 例如: 字母, 数字, 下划线(_), 汉字
result = re.match('\whm', 'ahm')      # Y
result = re.match('\whm', '1hm')      # Y
result = re.match('\whm', '_hm')      # Y
result = re.match('\whm', '爱hm')      # Y

result = re.match('\whm', '\hm')      # N
result = re.match('\whm', '@hm')      # N

# 演示 \W          代表: 特殊字符.
result = re.match('\Whm', '爱hm')      # N
result = re.match('\Whm', '\hm')      # Y
result = re.match('\Whm', '@hm')      # Y

# 打印匹配结果.
# if result != None:
if result:
    print(f'匹配到: {result.group()}')
else:
    print('未匹配到!')

# 三元写法.
print(f'匹配到: {result.group()}' if result else '未匹配到!')