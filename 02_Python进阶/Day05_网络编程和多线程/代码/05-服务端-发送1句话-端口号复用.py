"""
案例: 演示TCP入门, 即: 服务器端给客户端发送1句话, 客户端收到后, 给出回执信息.

流程:
    1. 服务器端  =>  客户端发送,  'Welcome to study socket!'
    2. 客户端接收到消息后, 打印, 并给出回执信息.  '消息已收到, So Easy!'
    3. 服务器端收到 客户端的 回执信息, 打印即可.

服务器端, 实现步骤:
    1. 创建服务器端的Socket对象.
    2. 绑定Ip地址 和 端口号.
    3. 设置最大监听数(允许挂载, 挂起的数量)
    4. 具体的监听动作, 接收客户端请求, 并获取1个socket对象, 负责和该客户端的交互.
    5. 给 客户端 发送1句话, 二进制形式.
    6. 接收客户端发过来的回执信息(二进制信息), 记得转成 字符串, 并打印.
    7. 释放资源, 关闭accept_socket.

设置端口号复用:
    背景/原因:
        当服务器端关闭的时候, 端口号不会立即释放, 而是需要等待 1 ~ 2分钟才会释放.
        如何解决这个问题呢?
    思路:
        1. 端口号重用.
        2. 手动更改1个新的端口号.
    格式:
        # 参1: 代表当前的 Socket对象, 即: server_socket(服务器端Socket对象)
        # 参2: Reuse Address, 表示: 端口号重用, 这里是: 属性名.
        # 参3: True, 表示: 成立, False: 不成立, 这里是: 属性值.
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)

"""
# 当前代码为: 服务器端的代码.
import socket

# 1. 创建服务器端的Socket对象.   参1: IpV4规则,  参2: 流的形式传输数据.
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 2. 绑定Ip地址 和 端口号.
server_socket.bind(('127.0.0.1', 12306))        # 127.0.0.1 代表本地回路(回环)地址, 在哪里运行, 就代表本机.
# 3. 设置最大监听数(允许挂载, 挂起的数量)
server_socket.listen(5)
# 4. 具体的监听动作, 接收客户端请求, 并获取1个socket对象, 负责和该客户端的交互.
# accept_socket: 负责和客户端交互的socket对象.
# client_info:   客户端的ip信息.
print('server: 1')
accept_socket, client_info = server_socket.accept()
# print(f'客户端ip: {client_info}')
print('server: 2')
# 5. 给 客户端 发送1句话, 二进制形式.
accept_socket.send(b'Welcome to study socket!')

# 6. 接收客户端发过来的回执信息(二进制信息), 记得转成 字符串, 并打印.
# 1024表示 一次性接收客户端数据的长度(单位: 字节), 超出则无法接收.
recv_data_bytes = accept_socket.recv(1024)
recv_data = recv_data_bytes.decode(encoding='utf-8')        # 把 二进制字符串 转成 字符串.
print(f'服务器端收到回执信息: {recv_data}')

# 7. 释放资源, 关闭accept_socket.
accept_socket.close()     # 和客户端交互的socket, 一般要关闭.
# server_socket.close()   服务器端一般不关闭.

# 8. 设置端口号重用.
# 参1: 代表当前的 Socket对象, 即: server_socket(服务器端Socket对象)
# 参2: Reuse Address, 表示: 端口号重用, 这里是: 属性名.
# 参3: True, 表示: 成立, False: 不成立, 这里是: 属性值.
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
