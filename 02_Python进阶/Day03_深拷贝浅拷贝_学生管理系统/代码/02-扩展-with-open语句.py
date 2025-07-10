"""
问: 操作文件的步骤是什么?
答:
    1. 打开文件.
    2. 读写数据.
    3. 释放资源.

扩展: with open语句, 它会在内容执行完毕后, 自动释放资源, 无需我们再次手动close了.
格式:
    with open(...) as 变量名:
        逻辑代码
特点:
    逻辑代码执行完毕后, 会自动释放 with 后边定义的 变量(对象).
"""

# 需求: 往文件1.txt中写一句话.
# # 1. 打开文件.
# dest_f = open('./1.txt', 'w', encoding='utf-8')
# # 2. 读写数据.
# dest_f.write('好好学习, 天天向上!')
# # 3. 释放资源.
# dest_f.close()


# with open写法.
# with open('./1.txt', 'w', encoding='utf-8') as dest_f, open('./1.txt', 'r', encoding='utf-8') as src_f:
with open('./1123.txt', 'w', encoding='utf-8') as dest_f:
    dest_f.write('我们来到了Python高级阶段, 大家要一起加油!')
