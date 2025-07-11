# 需求: 键盘录入 当前目录下任意的1个文件名, 然后对该文件进行备份, 备份文件名格式为: 原文件名[备份].原后缀名, 例如:  test.txt => test[备份].txt

# 1. 提示用户录入要备份的 文件名, 并接收.       绕口令.txt,  abc.mp3.txt,   .txt
old_name = input('请录入要备份的文件名: ')

# 2. 找到 最后1个 . 的索引.
index = old_name.rfind('.')

# 3. 判断文件名是否合法.
# 3.1 如果合法, 就拷贝.
if index > 0:
    # 4. 根据原文件名, 拼接 新文件名.           绕口令.txt  => 绕口令[备份].txt
    new_name = old_name[:index] + '[备份]' + old_name[index:]
    # print(new_name)

    # 5. 正常的读写操作.
    # 5.1 打开 数据源文件.
    # old_f = open(old_name, 'r', encoding='utf-8')   # 码表性质只能拷贝 纯文本文件.
    old_f = open(old_name, 'rb')        # 二进制形式读写, 无需指定码表, 通用版读写.
    # 5.2 打开 目的地文件.
    # new_f = open(new_name, 'w', encoding='utf-8')
    new_f = open(new_name, 'wb')
    # 5.3 循环 拷贝.
    while True:
        # 5.4 每次读取 8192个 字节的数据, 存储到 data变量中.
        data = old_f.read(8192)
        # 5.5 判断, 如果没有读取到内容, 说明文件内容读取完毕, 结束拷贝.
        if len(data) <= 0:
            break
        # 5.6 走到这里, 说明读取到内容, 将读取到的数据写出到 目的地文件中.
        new_f.write(data)
    # 5.7 释放资源.
    old_f.close()
    new_f.close()
    print('备份成功!')

else:
    # 3.2 如果不合法, 就提示, 然后程序结束.
    print('您录入的路径有误, 程序结束!')