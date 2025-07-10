"""
异常是具有传递性的, 函数内的异常 会传递给该函数的 调用者, 逐级传递, 直至这个异常被处理, 或者传递到main还不处理, 程序就会报错.
"""


# 案例: 演示异常的传递性.
def fun01():
    print('----- fun01 start-----')
    print(10 // 0)
    print('----- fun01 end-----')


def fun02():
    print('----- fun02 start-----')

    # 调用 fun01()
    fun01()
    # try:
    #     fun01()
    # except:
    #     print('除数不能为0')

    print('----- fun02 end-----')


def fun03():
    print('----- fun03 start-----')
    # 调用 fun02()
    fun02()
    print('----- fun03 end-----')


if __name__ == '__main__':
    try:
        fun03()
    except:
        print('除数不能为0')

