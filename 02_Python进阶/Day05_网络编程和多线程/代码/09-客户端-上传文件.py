"""
案例: 文件上传, 即: 客户端给服务器端上传1个文件.

流程:
    1. 客户端 => 服务器端, 上传1个文件.
    2. 服务器端收到后, 保存到 服务器上的某个路径下.   例如: ./data/这里

客户端, 实现步骤:
    1. 创建客户端的Socket对象.
    2. 连接服务器端的 Ip地址 和 端口号.
    3. 通过 open()函数, 关联: 数据源文件的路径.
    4. (循环)读取文件中的内容, 并将其写给服务器端.
    5. 如果读取完毕, 就结束读取. 即: break
    6. 关闭客户端即可.
"""
# 导包
import socket

# 1. 创建客户端的Socket对象. 参1: IpV4规则,  参2: 流的形式传输数据.
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 2. 连接服务器端的 Ip地址 和 端口号.
client_socket.connect(('192.168.28.98', 10010))

# 3. 通过 open()函数, 关联: 数据源文件的路径.
# with open(r'd:\绕口令.txt', 'rb') as src_f:
with open(r'd:\图片\a.jpg', 'rb') as src_f:
    # 4. (循环)读取文件中的内容, 并将其写给服务器端.
    while True:
        data = src_f.read(1024)
        # 5. 如果读取完毕, 就结束读取. 即: break
        if len(data) <= 0:
            break   # 走这里, 文件读完了.
        # 走这里, 说明有数据, 将其写给服务器端.
        client_socket.send(data)

# 6. 释放资源, 关闭accept_socket.
client_socket.close()

