"""
扩展: 长连接 和 短连接
    短连接: 目前我们写的代码都是这个, 客户端和服务器端交互一次, 然后 socket对象就销毁了.
    长连接: 客户端 和 服务器端可以多次交互, 可以手动选择在合适的时机 关闭(释放)资源. 一般适用于: 数据库的连接.
"""

# 案例: 演示 长连接, 即: 客户端不断地给服务器端发送消息, 服务器端接收消息并打印.  客户端发送 886 结束发送.

# 当前代码为: 服务器端的代码.
import socket

# 1. 创建服务器端的Socket对象.   参1: IpV4规则,  参2: 流的形式传输数据.
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 2. 绑定Ip地址 和 端口号.
server_socket.bind(('127.0.0.1', 10086))        # 127.0.0.1 代表本地回路(回环)地址, 在哪里运行, 就代表本机.
# server_socket.bind(('192.168.28.98', 12306))
# 3. 设置最大监听数(允许挂载, 挂起的数量)
server_socket.listen(20)

# 4. 具体的监听动作, 接收客户端请求, 并获取1个socket对象, 负责和该客户端的交互.
# accept_socket: 负责和客户端交互的socket对象.
# client_info:   客户端的ip信息.
accept_socket, client_info = server_socket.accept()
# print(f'客户端ip: {client_info}')

while True:
    # 5. 接收客户端发过来的回执信息(二进制信息), 记得转成 字符串, 并打印.
    # 1024表示 一次性接收客户端数据的长度(单位: 字节), 超出则无法接收.
    recv_data_bytes = accept_socket.recv(1024)
    recv_data = recv_data_bytes.decode(encoding='utf-8')        # 把 二进制字符串 转成 字符串.
    print(f'服务器端收到 {client_info} 的回执信息: {recv_data}')

    # 6. 如果接收到的消息是886, 就结束程序.
    if recv_data == '886':
        break

# 6. 释放资源, 关闭accept_socket.
accept_socket.close()     # 和客户端交互的socket, 一般要关闭.
# server_socket.close()   服务器端一般不关闭.
