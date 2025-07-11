"""
捕获异常, 完整格式如下:
    try:
        里边写的是可能出问题的代码
    except [Exception as e]:
        出现问题后的 解决方案
    else:
        只要try中内容无问题, 就会执行这里的内容.
        只要try中有问题, 就会跳过这里的代码.
    finally:
        无论try是否有问题, 都会走这里的内容, 一般用于释放资源.
"""

# 演示 捕获异常 完整格式.
try:
    print('try 1')
    print(10 // 2)
    print('try 2')
except Exception as e:
    print(f'程序出问题了, {e}')
else:
    print('我是else, 看看我执行了吗?')
finally:
    print('我是finally, 看看我执行了吗?')

print('-' * 28)

# 案例: 拷贝文件, 加入异常处理.
try:
    src_f = open('1.txt', 'rb')
    dest_f = open('aa/bb/cc/2.txt', 'wb')
except Exception as e:
    print(e)
else:
    while True:
        data = src_f.read(1024)
        if len(data) <= 0:
            break
        dest_f.write(data)
finally:
    try:
        src_f.close()
    except Exception as e:
        print(e)

    try:
        dest_f.close()
    except Exception as e:
        print(e)

print('-' * 28)

try:
    with open('1.txt', 'rb') as src_f, open('aa/bb/cc/2.txt', 'wb') as dest_f:
        while True:
            data = src_f.read(1024)
            if len(data) <= 0:
                break
            dest_f.write(data)
except Exception as e:
    print(e)