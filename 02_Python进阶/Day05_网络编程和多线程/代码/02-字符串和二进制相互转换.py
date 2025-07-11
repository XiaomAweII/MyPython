"""
案例: 演示字符串 和 二进制数据 相互转换.

背景:
    网编中, 客户端 和 服务器端交互数据, 都是采用 二进制形式 实现的.
格式:
    字符串.encode(encoding='码表名')          # 编码, 字符串 => 二进制形式, 一般用: utf-8码表
    二进制字符串.decode(encoding='码表名')     # 解码, 二进制字符串 => 字符串形式, 一般用: utf-8码表
细节:
    1. 编解码时, 码表要一致, 否则可能出现: 乱码的情况.
    2. 数字, 英文字母, 特殊符号, 无论在什么码表中, 都只占 1个字节.
       中文在 GBK码表(国内常用)中 占 2个字节,  在utf-8(万国码, 统一码)码表中 占 3个字节.
    3. b'内容' 这种写法 是 二进制形式的字符串, 只针对于: 字母, 数字, 特殊符号有效, 针对于中文无效.
       即: b'abc123!@#' 可以,   b'你好' 不行.
"""

# 需求1: 演示 编码, 中文
s1 = '黑马'

# 不写码表, 默认是: utf-8
bs1 = s1.encode()                   # bytes 即: 多个字节.   b'\xe9\xbb\x91\xe9\xa9\xac'
bs2 = s1.encode(encoding='gbk')     # b'\xba\xda\xc2\xed'
bs3 = s1.encode(encoding='utf-8')   # b'\xe9\xbb\x91\xe9\xa9\xac'

print(bs1)  # b'\xe9\xbb\x91\xe9\xa9\xac'
print(bs2)  # b'\xba\xda\xc2\xed'
print(bs3)  # b'\xe9\xbb\x91\xe9\xa9\xac'

print(type(bs1))    # <class 'bytes'>
print(type(bs2))    # <class 'bytes'>
print(type(bs3))    # <class 'bytes'>
print('-' * 20)

# 需求2: 演示 编码, 数字, 字母, 特殊符号.
s2 = 'abc123!@#'
bs4 = s2.encode(encoding='gbk')
bs5 = s2.encode(encoding='utf-8')
print(bs4)  # b'abc123!@#'
print(bs5)  # b'abc123!@#'
print(type(bs4))    # <class 'bytes'>
print(type(bs5))    # <class 'bytes'>

print(type(b'abc12!@#'))    # OK
# print(type(b'你好'))         # 报错
print('-' * 20)

# 需求3: 演示解码.
bs1 = b'\xe9\xbb\x91\xe9\xa9\xac'       # 二进制形式, utf-8  黑马
bs2 = b'\xba\xda\xc2\xed'               # 二进制形式, gbk  黑马
bs3 = b'abc123!@#'                      # 啥码表都行.

s1 = bs1.decode(encoding='utf-8')
print(s1)
print(type(s1))

# s2 = bs1.decode(encoding='gbk')       # 编解码不一致, 乱码.
s2 = bs2.decode(encoding='gbk')
print(s2)
print(type(s2))

print(bs3.decode(encoding='gbk'))
print(bs3.decode(encoding='utf-8'))