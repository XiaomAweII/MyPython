"""
案例: 演示TCP入门, 即: 服务器端给客户端发送1句话, 客户端收到后, 给出回执信息.

流程:
    1. 服务器端  =>  客户端发送,  'Welcome to study socket!'
    2. 客户端接收到消息后, 打印, 并给出回执信息.  '消息已收到, So Easy!'
    3. 服务器端收到 客户端的 回执信息, 打印即可.

客户端, 实现步骤:
    1. 创建客户端的Socket对象.
    2. 连接服务器端的 Ip地址 和 端口号.
    3. 接收服务器端发过来的信息(二进制信息), 记得转成 字符串, 并打印.
    4. 给 服务器端 发送1句话, 二进制形式.
    5. 释放资源, 关闭accept_socket.
"""
# 导包
import socket

# 1. 创建客户端的Socket对象. 参1: IpV4规则,  参2: 流的形式传输数据.
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 2. 连接服务器端的 Ip地址 和 端口号.
client_socket.connect(('127.0.0.1', 12306))

# 3. 接收服务器端发过来的信息(二进制信息), 记得转成 字符串, 并打印.
recv_data_bytes = client_socket.recv(1024)
recv_data = recv_data_bytes.decode(encoding='utf-8')        # 把 二进制字符串 转成 字符串.
print(f'客户端收到: {recv_data}')

# 4. 给 服务器端 发送1句话, 二进制形式.
client_socket.send('消息已收到, 有内鬼, 终止交易, Over!'.encode(encoding='utf-8'))


# 5. 释放资源, 关闭accept_socket.
client_socket.close()