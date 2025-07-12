import jieba
# def fun(x):
#     return x + 2
#
# a = map(fun, [1, 2, 3])
# # print(*a)
#
# b = map(lambda x: x+2, [1, 2, 3])
# print(list(b))
# from itertools import chain
# # list1 = [1, 2, 3]
# # list2 = [2, 3, 4]
# # a = chain(list1, list2)
# # print(set(a))
# # # map(lambda x: jieba.lcut(x), train_data["sentence"])
#
# list1 = ["我爱黑马", "我爱你, 我爱黑马"]
# a = map(lambda x: jieba.lcut(x), list1)
# b = chain(*a)
# print(set(b))

# import jieba.posseg as pseg
#
# content ='交通很方便，房间小了一点，但是干净整洁，很有香港的特色，性价比较高，推荐一下哦'
# for value in pseg.lcut(content):
#     if value.flag == 'a':
#         print(value.word)

list1 = [1, 2, 3]
list2 = [2, 3, 4]
a = zip(list1,list2)
print(list(a))

list3 = [[1, 2, 3],
         [2, 3, 4, 5]]
b = zip(*list3)
print(list(b))