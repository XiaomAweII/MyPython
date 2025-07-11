"""
案例: 文件上传, 即: 客户端给服务器端上传1个文件.

流程:
    1. 客户端 => 服务器端, 上传1个文件.
    2. 服务器端收到后, 保存到 服务器上的某个路径下.   例如: ./data/这里

服务器端, 实现步骤:
    1. 创建客户端的Socket对象.
    2. 连接服务器端的 Ip地址 和 端口号.
    3. 通过 open()函数, 关联: 目的地文件的路径.
    4. (循环)接收客户端写过来的数据, 并将其写到 目的地文件中.
    5. 如果接收完毕, 就结束即可. 即: break
    6. 关闭客户端即可.
"""

# 案例: 演示 长连接, 即: 客户端不断地给服务器端发送消息, 服务器端接收消息并打印.  客户端发送 886 结束发送.

# 当前代码为: 服务器端的代码.
import socket

# 1. 创建服务器端的Socket对象.   参1: IpV4规则,  参2: 流的形式传输数据.
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 2. 绑定Ip地址 和 端口号.
server_socket.bind(('127.0.0.1', 10010))        # 127.0.0.1 代表本地回路(回环)地址, 在哪里运行, 就代表本机.
# server_socket.bind(('192.168.28.98', 12306))
# 3. 设置最大监听数(允许挂载, 挂起的数量)
server_socket.listen(20)

# 4. 具体的监听动作, 接收客户端请求, 并获取1个socket对象, 负责和该客户端的交互.
# accept_socket: 负责和客户端交互的socket对象.
# client_info:   客户端的ip信息.
accept_socket, client_info = server_socket.accept()
# print(f'客户端ip: {client_info}')

# 5. 通过 open()函数, 关联: 目的地文件的路径.
with open('./data/hg.txt', 'wb') as dest_f:
    # 6. (循环)接收客户端写过来的数据, 并将其写到 目的地文件中.
    while True:
        # 接收客户端写过来的数据
        data = accept_socket.recv(1024)
        # 7. 如果接收完毕, 就结束即可. 即: break
        if len(data) <= 0:
            break   # 走这里, 文件读完了.
        # 走这里, 说明读取到内容了.
        dest_f.write(data)

# 8. 释放资源, 关闭accept_socket.
accept_socket.close()     # 和客户端交互的socket, 一般要关闭.
# server_socket.close()   服务器端一般不关闭.
