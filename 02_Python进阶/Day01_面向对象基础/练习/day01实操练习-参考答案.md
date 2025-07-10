## 实操题



（1）定义一个手机类，能开机、能关机、可以拍照。

==参考答案：==

```python
class Phone(object):
    # 能开机、能关机、可以拍照
    def open(self):
        print("手机能正常开机啦~~~")
    
    def close(self):
        print("关机, 睡觉啦.")
    
    def take_photo(self):
        print("我超自恋,就爱自拍,哈哈哈.")

phone = Phone()
phone.open()
phone.take_photo()
phone.close()
```



（2)定义一个电脑类，并给电脑添加品牌、价格等属性，同时电脑能用于编程、看视频。

==参考答案：==

```python
class Computer:
    def programming(self):
        print("使用电脑来进行编程...")

    def look(self):
        print("业余时间看看视频放松放松...")
        
computer = Computer()
# 添加属性
computer.brand = "联想"
computer.price = 5999
print(f"品牌:{computer.brand}")
print(f"价格:{computer.price}")
computer.programming()
computer.look()
```



（3）尝试定义一个工程师类，同时使用`__init__()`初始化岗位名称、薪资金额等属性，工作内容是每天码代码，同时使用`__str__()`展示对象所拥有的所有信息。

==参考答案：==

```python
class Engineer:
    def __init__(self,name,salary):
        self.name = name
        self.salary = salary

    def work(self):
        print("爱好码代码...")

    def __str__(self):
        return f"岗位名称:{self.name},月薪:{self.salary}元."

engineer = Engineer("算法开发工程师",26000)
engineer.work()
print(engineer)
```

