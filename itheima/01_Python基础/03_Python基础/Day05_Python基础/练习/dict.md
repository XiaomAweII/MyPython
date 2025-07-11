## 题目1 [综合训练]

### 题干

请实现以下代码：

```python
# 1. 定义一个空字典 my_dict

# 2. 定义一个字典 my_dict1,字典存在以下键值对,name:isaac, age:18, pi:3.14, 

# 3. 打印my_dict1和my_dict1 的类型

# 4. 输出 my_dict1中18 这个数据

# 5. 求my_dict1中元素的个数

# 6. 思考:通过字典的 key 访问和 get 方法访问,两者之间有什么不同

# 7. 修改字典my_dict1中age 的值为 19并输出

# 8. my_dict1中添加一个元素, gender: 男, 并输出打印

# 9. 删除my_dict1中 key 为 pi 的键值对(元素)

# 10. 清空字典my_dict1中的内容

# 11. 删除字典变量my_dict1

# 12. 现在是否还能使用 my_dict1 变量
```



### 训练目标

1.  字典的定义
2.  字典的增删改查操作

### 训练提示

1.  怎样定义一个字典？
2.  怎么访问字典中的元素？
3.  怎样修改字典中的元素？
4.  怎样添加元素？
5.  怎样删除元素？



### 参考方案

1.  字典的定义是 `{key: value, key: value}`,元素是键值对的格式
2.  字典不支持下标操作，可以使用 `字典[key] ` 的方式访问，也可以使用 get 方法访问
3.  修改数据使用 `字典名[key] = 新的数据值`
4.  添加数据使用`字典名[key] = 新的数据值`
5.  删除指定 key 值的元素用 `del 字典名[key]`
6.  清空字典使用 `字典名.clear()`
7.  删除字典变量使用 `del 字典名`

### 操作步骤

见题目注释

### 参考答案

```python
# 1. 定义一个空字典 my_dict
# 方法一
my_dict = {}
# 方法二
my_dict = dict()

# 2. 定义一个字典 my_dict1,字典存在以下键值对,name:isaac, age:18, pi:3.14, 
my_dict1 = {"name": "isaac", "age": 18, "pi": 3.14}

# 3. 打印my_dict1和my_dict1 的类型
print(my_dict1, type(my_dict1))

# 4. 输出 my_dict1中 18 这个数据
# 4.1 方法一, 使用字典名[key] 的方法
age = my_dict1['age']
print(age)
# 4.2 方法二, 使用get方法
age1 = my_dict1.get('age')
print(age1)

# 5. 求my_dict1中元素的个数
length = len(my_dict1)
print(length)

# 6. 思考:通过字典的 key 访问和 get 方法访问,两者之间有什么不同
# 字典[key]的方法,key值不存在,会报错
# get方法,key值不存在,不会报错

# 7. 修改字典my_dict1中age 的值为 19并输出
my_dict1['age'] = 19
print(my_dict1)

# 8. my_dict1中添加一个元素, gender: 男, 并输出打印
my_dict1['gender'] = '男'
print(my_dict1)

# 9. 删除my_dict1中 key 为 pi 的键值对(元素)
del my_dict1['pi']
print(my_dict1)
# 10. 清空字典my_dict1中的内容
my_dict1.clear()
print(my_dict1)

# 11. 删除字典变量my_dict1
del my_dict1

# 12. 现在是否还能使用 my_dict1 变量
# 不能使用,因为,已经使用del 变量名,将变量删除

```





## 题目2 [加强训练]

### 题干

现有字典dict1 = {'name':'chuanzhi','age':18}
要求：1.使用循环将字典中所有的键输出到屏幕上
    2.使用循环将字典中所有的值输出到屏幕上
    3.使用循环将字典中所有的键值对输出到屏幕上
      输出方式：  

```python
name:chuanzhi
age:18
```

### 训练目标

1. for循环的使用复习
2. 学会如何获取字典所有的键
3. 学会如何获取字典所有的值
4. 学会如何获取字典所有的键值对

### 训练提示

1. 用for来实现循环
2. 用字典.keys()来获取所有的键
3. 用字典.values()来获取所有的值
4. 用字典.items()来获取所有的键值对

### 参考方案

用for来实现循环，让字典.keys()取到所有的键并控制循环次数，依次打印出每一个值，值与键值对同理；

### 操作步骤

1. 让字典.keys()获取所有的值，将所有的值进行循环遍历，并依次print（）
2. 让字典.keys()获取所有的值，将所有的值进行循环遍历，并依次print（）
3. 让字典.items()获取所有的值，将所有的值进行循环遍历，并依次print（key, ":", value）

### 参考答案

``` python
dict1 = {'name':'chuanzhi','age':18}
for key in dict1.keys():
  print(key)
for value in dict1.values():
  print(value)
for key,value in dict1.items():
  print(key, ":", value)
```



## 题目3 [综合训练]

### 题干

有这样的一个列表,保存商品以及单价信息，如下：

```python
product=[
{"name":"电脑", "price":7000},
{"name":"鼠标", "price":30},
{"name":"usb电动小风扇", "price":20},
{"name":"遮阳伞", "price":50},
]
```

小明一共有8000块钱，他能否买下所有商品，如果能，请输出“能”，否则输出“不能”

### 训练目标

1.  列表和字典的混合使用
2.  列表的遍历
3.  字典取值

### 训练提示

1.  每个商品信息是使用字典存储的，多个商品存在放列表中，即列表嵌套字典，那该如何得到每个商品信息，即列表中字典？
2.  商品的价格是存放在字典中，使用 key `price`存储，那么该如何取到对应的 value 值呢？
3.  取到每个 value 值，如何进行累加保存总共需要的金钱？
4.  如何判断自己的钱是否够买所有的商品？

### 参考方案

1.  可以使用循环遍历列表，可以得到列表中的每一个元素，即每个商品的信息，类型为字典。
2.  可以使用`字典[key] `和`字典.get()` 方法获取字典中的值
3.  定义遍历保存商品的总价格
4.  使用判断，比较自己拥有的金钱和商品的总价格之间的关系

### 操作步骤

1. 遍历列表
2. 求商品的总价格
3. 判断自己拥有的金钱和商品的总价格之间的关系

### 参考答案

``` python
# 定义商品列表
product = [
    {"name": "电脑", "price": 7000},
    {"name": "鼠标", "price": 30},
    {"name": "usb电动小风扇", "price": 20},
    {"name": "遮阳伞", "price": 50},
]

# 拥有的金钱
money = 8000
my_sum = 0  # sum初始化为0，是用来存放总价格

# 遍历商品列表,取出每一个商品的字典
for dict1 in product:
    # dict1 为商品字典,想要取出商品价格,可以使用 字典[key] 和get() 方法
    my_sum += dict1['price']

# 判断商品的总价格和自己钱的关系
if money >= my_sum:
    print("能")
else:
    print("不能")

```

