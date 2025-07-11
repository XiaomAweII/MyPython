#### 今日内容大纲

* Python数据分析的优势
* Python数据分析环境搭建
* Jupyter Lab 和 Jupyter Notebook初体验
* 配置PyCharm连接Jupyter
* Numpy详解
  * 属性
  * 创建
  * 内置函数
  * 运算

---

#### 1.Python数据处理分析简介

* Python作为当下最为流行的编程语言之一
  * 可以独立完成数据分析的各种任务
  * 数据分析领域里有海量开源库
  * 机器学习/深度学习领域最热门的编程语言
  * 在爬虫，Web开发等领域均有应用
* 与Excel，PowerBI，Tableau等软件比较
  * Excel有百万行数据限制
  * PowerBI ，Tableau在处理大数据的时候速度相对较慢
  * Excel，Power BI 和Tableau 需要付费购买授权
  * Python功能远比Excel，PowerBI，Tableau等软件强大
  * Python跨平台，Windows，MacOS，Linux都可以运行
* 与R语言比较
  * Python在处理海量数据的时候比R语言效率更高
  * Python的工程化能力更强，R专注于统计与数据分析领域
  * Python在非结构化数据（文本，图像）和深度学习领域比R更有优势
  * 在数据分析相关开源社区，python相关的内容远多于R语言
* 总结
  1. Python应用广泛, 且是当下最热门的编程语言之一.
  2. Python功能强大, 且开源, 免费.
  3. Python的社区活跃度相对较高. 



#### 2.常用Python数据分析开源库介绍

* **NumPy(Numerical Python)** 
  * 它是 Python 语言的一个扩展程序库。是一个运行速度非常快的数学库.
  * 主要用于数组计算
  * 包含：
    * 一个强大的N维数组对象 ndarray
    * 广播功能函数
    * 整合 C/C++/Fortran 代码的工具
    * 线性代数、傅里叶变换、随机数生成等功能
* **Pandas**
  * Pandas是一个强大的分析结构化数据的工具集
  * 它的使用基础是Numpy（提供高性能的矩阵运算）
  * 用于数据挖掘和数据分析，同时也提供数据清洗功能
  * **Pandas利器之 Series**，是一种类似于一维数组的对象
  * **Pandas利器之 DataFrame**，是Pandas中的一个表格型的数据结构
* **Matplotlib** 
  * 它是一个功能强大的数据可视化开源Python库
  * Python中使用最多的图形绘图库
  * 可以创建静态, 动态和交互式的图表
* **Seaborn**
  * 它是一个Python数据可视化开源库, 建立在matplotlib之上，并集成了pandas的数据结构
  * Seaborn通过更简洁的API来绘制信息更丰富，更具吸引力的图像
  * 面向数据集的API，与Pandas配合使用起来比直接使用Matplotlib更方便
* **Sklearn** 
  * scikit-learn 是基于 Python 语言的机器学习工具
  * 简单高效的数据挖掘和数据分析工具
  * 可供大家在各种环境中重复使用
  * 建立在 NumPy ，SciPy 和 matplotlib 上
* **jupyter notebook**
  - **它不是开源库,** 它是一个开源Web应用程序, 可以创建和共享代码、公式、可视化图表、笔记文档
  - 是数据分析学习和开发的首选开发环境, 作用如下: 
    - 数据清理和转换
    - 数值模拟
    - 统计分析
    - 数据可视化
    - 机器学习等



#### 3.Python数据分析环境搭建-本地环境

> 主要有**本地环境** 和 **虚拟机环境**两种, 区别是: 看在哪里安装Anaconda软件.

* Anaconda介绍
  * Anaconda 是最流行的数据分析平台，全球两千多万人在使用
  * Anaconda 附带了一大批常用数据科学包
  * Anaconda 是在 conda（一个包管理器和环境管理器）上发展出来的
  * 可以帮助你在计算机上安装和管理数据分析相关包
  * 包含了虚拟环境管理工具

* Anaconda安装

  * Anaconda 可用于多个平台（ Windows、Mac OS X 和 Linux）

  * 可以在官网上下载对应平台的安装包

  * 如果计算机上已经安装了 Python，安装不会对你有任何影响

  * 下载链接为: https://www.anaconda.com/products/individual

  * 安装的过程很简单，一路下一步即可

  * 检测是否安装成功

    ![1713457290634](assets/1713457290634.png)

* Anaconda界面介绍

  ![1713457355847](assets/1713457355847.png)

  ![1713457602042](assets/1713457602042.png)

* Anaconda的命令操作

  * 安装包的命令

    ```python
    # 安装包的命令
    conda install 包名字
    pip install 报名字
    
    
    # 注意，使用pip时最好指定安装源, 参考镜像地址, 
    阿里云：https://mirrors.aliyun.com/pypi/simple/
    豆瓣：https://pypi.douban.com/simple/
    清华大学：https://pypi.tuna.tsinghua.edu.cn/simple/
    中国科学技术大学 http://pypi.mirrors.ustc.edu.cn/simple/
    
    # 完整格式如下
    pip install 包名 -i https://mirrors.aliyun.com/pypi/simple/  #通过阿里云镜像安装
    ```

  * 操作虚拟环境(沙箱)的命令

    ```python
    通过命令行创建虚拟环境
    conda create -n 虚拟环境名字 python=python版本  #创建虚拟环境
    conda activate 虚拟环境名字 #进入虚拟环境
    conda deactivate 虚拟环境名字 #退出虚拟环境
    conda remove -n 虚拟环境名字 --all  #删除虚拟环境
    conda env list   # 查看所有虚拟环境(沙箱)
    
    
    （2）克隆旧环境
     
    conda create -n new_name --clone old_nam
    ```

* 以**管理员的身份**打开 Anaconda的命令窗口, 运行jupyter lab 或者 jupyter notebook即可

  ![1713458619371](assets/1713458619371.png)

  ![1713458767547](assets/1713458767547.png)



#### 4.Jupyter Lab初体验

1. 去Linux虚拟机中, 启动 jupyter环境即可

   ![1713455451816](assets/1713455451816.png)

2. 打开浏览器, 输入上边的网址.

   ![1713455527212](assets/1713455527212.png)



#### 5.Jupyter NoteBook初体验

1. 确保你的C盘hosts文件, 配置了域名映射

   ```sql
   -- 路径为: C:\Windows\System32\drivers\etc\hosts
   
   -- 内容如下:
   192.168.88.161 node1.itcast.cn node1
   ```

2. 去Linux虚拟机中, 启动 jupyter环境即可

   ![1713452647913](assets/1713452647913.png)

3. 打开浏览器, 输入上边的网址, 新建1个 numpy文件夹

   ![1713452704836](assets/1713452704836.png)

4. 新建1个test1测试文件.

   ![1713452761548](assets/1713452761548.png)

5. 输入测试代码, 测试执行即可.

   ![1713452831051](assets/1713452831051.png)



#### 6.Jupyter NoteBook的使用

- 菜单栏中相关按钮功能介绍

  > Jupyter Notebook的代码的输入框和输出显示的结果都称之为**cell**，cell行号前的 * ，表示代码正在运行

  ![1713453480599](assets/1713453480599.png)

- 常用快捷键

  > Jupyter Notebook中分为两种模式：命令模式和编辑模式

  - 两种模式通用快捷键

    - **`Shift+Enter`，执行本单元代码，并跳转到下一单元**
    - **`Ctrl+Enter`，执行本单元代码，留在本单元**

  - 按ESC进入**命令模式**

    ![1713453817360](assets/1713453817360.png)

    - `Y`，cell切换到Code模式
    - `M`，cell切换到Markdown模式
    - `A`，在当前cell的上面添加cell
    - `B`，在当前cell的下面添加cell
    - `双击D`：删除当前cell

  - **编辑模式**：按Enter进入，或鼠标点击代码编辑框体的输入区域

    ![1713453988822](assets/1713453988822.png)

    - 撤销：`Ctrl+Z`（Mac:CMD+Z）
    - 反撤销: `Ctrl + Y`（Mac:CMD+Y）
    - 补全代码：变量、方法后跟`Tab键`
    - 为一行或多行代码添加/取消注释：`Ctrl+/`（Mac:CMD+/）
    - 代码提示: `shift + Tab`

- 使用Markdown

  - 在命令模式中，按M即可进入到Markdown编辑模式

  - 使用Markdown语法可以在代码间穿插格式化的文本作为说明文字或笔记

  - Markdown基本语法：标题和缩进

    - 代码如下:

      ![1713454391206](assets/1713454391206.png)

    - 效果图如下

      ![1713454340049](assets/1713454340049.png)



#### 7.配置PyCharm连接Anaconda

* 连接本地的Anaconda环境

  ![1713458939722](assets/1713458939722.png)

  ![1713459544631](assets/1713459544631.png)

* 连接本地的Anaconda环境

  * 确保Linux的Jupyter环境开启了

    ![1713459806683](assets/1713459806683.png)

  * 配置方式和上述步骤一样,只不过把URL地址换成 http://192.168.88.161:8888


#### 8.Numpy的ndarray的属性

* Numpy简介

  * NumPy（Numerical Python）是Python数据分析必不可少的第三方库
  * NumPy的出现一定程度上解决了Python运算性能不佳的问题，同时提供了更加精确的数据类型，使其具备了构造复杂数据类型的能力。
  * 本身是由C语言开发，是个很基础的扩展，NumPy被Python其它科学计算包作为基础包，因此理解np的数据类型对python数据分析十分重要。
  * NumPy重在数值计算，主要用于多维数组（矩阵）处理的库。用来存储和处理大型矩阵，比Python自身的嵌套列表结构要高效的多

* 重要功能如下

  1. 高性能科学计算和数据分析的基础包
  2. ndarray，多维数组，具有矢量运算能力，快速、节省空间
  3. 矩阵运算，无需循环，可完成类似Matlab中的矢量运算
  4. 用于读写磁盘数据的工具以及用于操作内存映射文件的工具

* Numpy的属性

  >  NumPy的数组类被称作ndarray，通常被称作数组。

  * ndarray.ndim

  * ndarray.shape

  * ndarray.size

  * ndarray.dtype

  * ndarray.itemsize

    > 数组的维度。这是一个指示数组在每个维度上大小的整数元组。例如一个n排 m列的矩阵，它的shape属性将是(2,3),这个元组的长度显然是秩，即维度或者ndim属性。

* 代码演示

  ```python
  # numpy属性介绍:  shape, dtype, ndim, itemsize, size
  
  # 导包
  import numpy as np
  
  # 1. 创建numpy的数组.   
  # arange(15)       等价于python的 range(15), 即: 获取 0 ~ 14的整数
  # reshape(3, 5)    把上述数据封装到 3个一维数组中, 每个一维数组的长度为: 5,  然后把三个一维数组封装成1个 二维数组.
  arr = np.arange(15).reshape(3, 5)
  
  # 2. 打印数组, 查看内容
  print(arr)
  
  # 3. 演示numpy的 属性
  print(f'数组的维度: {arr.shape}')         # (3, 5)   3个元素(一维数组), 每个元素(一维数组)又有5个元素(值)
  print(f'数组轴的个数: {arr.ndim}')        # 几维数组, 轴就是几,  2  
  print(f'数组元素类型: {arr.dtype}')       # int64 
  print(f'数组每个元素的占用字节数: {arr.itemsize}')    # 8
  print(f'数组元素个数: {arr.size}')        # 15 
  print(f'数组类型: {type(arr)}')          # <class 'numpy.ndarray'>
  
  # 4. 上述的 shape, ndim, size属性 可以 函数写法 实现.
  # 格式: np.函数名(数组)
  print(f'数组的维度: {np.shape(arr)}')         # (3, 5)   3个元素(一维数组), 每个元素(一维数组)又有5个元素(值)
  print(f'数组轴的个数: {np.ndim(arr)}')        # 几维数组, 轴就是几,  2  
  print(f'数组元素个数: {np.size(arr)}')        # 15 
  print(f'数组类型: {type(arr)}')              # <class 'numpy.ndarray'>
  ```

  ![1713460689719](assets/1713460689719.png)



#### 9.Numpy的ndarray的创建

* ndarray介绍

  * NumPy数组是一个多维的数组对象（矩阵），称为**ndarray(N-Dimensional Array)**
  * 具有矢量算术运算能力和复杂的广播能力，并具有执行速度快和节省空间的特点
  * 注意：ndarray的下标从0开始，且数组里的所有元素必须是相同类型。

* 数组形式

  ```python
  import numpy as np 
  a = np.array([2, 3, 4])
  print('数组a元素类型: ', a)
  print('数组a类型:', a.dtype)
  
  b = np.array([1.2, 3.5, 5.1])
  print('数组b类型:', b.dtype)
  ```

  ![1713461234989](assets/1713461234989.png)

* **zeros() /ones()/empty()**

  > 函数zeros创建一个全是0的数组，
  >
  > 函数ones创建一个全1的数组，
  >
  > 函数empty创建一个内容随机并且依赖于内存状态的数组。默认创建的数组类型(dtype)都是float64

  ```python
  zero1 = np.zeros((3, 4))    # 3个一维数组, 每个长度为: 4
  print('数组zero1: ', zero1)
  
  ones1 = np.ones((2, 3, 4))  # 2个二维数组, 每个二维数组有3个一维数组, 每个一维数组有4个元素1, 整体放入1个数组中
  print('数组one1: ', ones1)
  
  empty1 = np.empty((2, 3))
  print('数组empty1: ', empty1)
  
  print(zero1.dtype, ones1.dtype, empty1.dtype)
  ```

  ![1713461992065](assets/1713461992065.png)

* **arange(),** 类似 python 的 range() ，创建一个一维 ndarray 数组。

  ```python
  np_arange = np.arange(10, 20, 5,dtype=int)   # 起始, 结束, 步长, 类型
  print("arange创建np_arange:", np_arange)
  print("arange创建np_arange的元素类型:", np_arange.dtype)
  print("arange创建np_arange的类型:", type(np_arange))
  ```

  ![1713462238714](assets/1713462238714.png)

* **matrix()**,  是 ndarray 的子类，只能生成 2 维的矩阵

  ```python
  x1 = np.mat("1 2;3 4")
  print(x1)
  x2 = np.matrix("1,2;3,4")
  print(x2)
  x3 = np.matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
  print(x3)
  ```

  ![1713462358103](assets/1713462358103.png)

* **创建随机数矩阵**

  ```python
  import numpy as np
  
  # 生成指定维度大小(3行4列)的随机多维浮点型数据(二维), rand固定区间0.0 ~ 1.0
  arr = np.random.rand(3, 4)
  print(arr)
  print(type(arr))
  
  # 生成指定维度大小(3行4列)的随机多维整型数据(二维), randint()可指定区间(-1, 5)
  arr = np.random.randint(-1, 5, size=(3, 4))
  print(arr)
  print(type(arr))
  
  #生成指定维度大小(3行4列)的随机多维浮点型数据(二维), uniform()可以指定区间(-1, 5)产生-1到5之间均匀分布的样本值
  arr = np.random.uniform(-1, 5, size=(3, 4))
  print(arr)
  print(type(arr))
  ```

  ![1713462598950](assets/1713462598950.png)

* **ndarray的数据类型**

  ```python
  # 细节
  1. dtype参数，指定数组的数据类型，类型名+位数，如float64, int32
  2. astype方法，转换数组的数据类型
  
  # 初始化3行4列数组，数据类型为f1oat64
  zeros_float_arr =np.zeros((3,4),dtype=np.float64)
  print(zeros_float_arr)
  print(zeros_float_arr.dtype) # float64
  
  # astype转换数据类型，将已有的数组的数据类型转换为int32
  zeros_int_arr = zeros_float_arr.astype(np.int32)
  print(zeros_int_arr)
  print(zeros_int_arr.dtype) #int32
  ```

  ![1713462898470](assets/1713462898470.png)

* **等比数列**

  ```python
  # np.logspace 等比数列, logspace中，开始点和结束点是10的幂
  # 我们让开始点为0，结束点为0，元素个数为10，看看输出结果。
  
  a = np.logspace(0,0,10)
  
  # 输出结果
  a
  
  # 假如，我们想要改变基数，不让它以10为底数，我们可以改变base参数，将其设置为2
  
  ```

  ![1713463199873](assets/1713463199873.png)

  ![1713463421618](assets/1713463421618.png)

  ```python
  # 假如，我们想要改变基数，不让它以10为底数，我们可以改变base参数，将其设置为2
  ```

  ![1713463501987](assets/1713463501987.png)

* **等差数列**

  ```python
  # np.linspace等差数列
  # np.linspace是用于创建一个一维数组，并且是等差数列构成的一维数组，它最常用的有三个参数。
  # 第一个例子，用到三个参数，第一个参数表示起始点，第二个参数表示终止点，第三个参数表示数列的个数。
  ```

  ![1713463598854](assets/1713463598854.png)

  ![1713463694637](assets/1713463694637.png)

#### 10.Numpy的内置函数

* 基本函数

  ```python
  # 基本函数如下
  np.ceil(): 向上最接近的整数，参数是 number 或 array
  np.floor(): 向下最接近的整数，参数是 number 或 array
  np.rint(): 四舍五入，参数是 number 或 array
  np.isnan(): 判断元素是否为 NaN(Not a Number)，参数是 number 或 array
  np.multiply(): 元素相乘，参数是 number 或 array
  np.divide(): 元素相除，参数是 number 或 array
  np.abs()：元素的绝对值，参数是 number 或 array
  np.where(condition, x, y): 三元运算符，x if condition else y
  # 注意: 需要注意multiply/divide 如果是两个ndarray进行运算 shape必须一致
  
      
  # 示例代码
  arr = np.random.randn(2, 3)
  print(arr)
  print(np.ceil(arr))
  print(np.floor(arr))
  print(np.rint(arr))
  print(np.isnan(arr))
  print(np.multiply(arr, arr))
  print(np.divide(arr, arr))
  print(np.where(arr > 0, 1, -1))
  ```

  ![1713464265467](assets/1713464265467.png)

* 统计函数

  ```python
  np.mean(), np.sum()：所有元素的平均值，所有元素的和，参数是 number 或 array
  np.max(), np.min()：所有元素的最大值，所有元素的最小值，参数是 number 或 array
  np.std(), np.var()：所有元素的标准差，所有元素的方差，参数是 number 或 array
  np.argmax(), np.argmin()：最大值的下标索引值，最小值的下标索引值，参数是 number 或 array
  np.cumsum(), np.cumprod()：返回一个一维数组，每个元素都是之前所有元素的 累加和 和 累乘积，参数是 number 或 array
   # 多维数组默认统计全部维度，axis参数可以按指定轴心统计，值为0则按列统计，值为1则按行统计。
  
      
  # 实例代码
  arr = np.arange(12).reshape(3, 4)
  print(arr)
  print(np.cumsum(arr))   # 返回一个一维数组, 每个元素都是之前所有元素的 累加和
  print(np.sum(arr))      # 所有元素的和
  print(np.sum(arr, axis = 0))  #数组的按列统计和
  print(np.sum(arr, axis = 1))  #数组的按行统计和
  ```

  ![1713464414920](assets/1713464414920.png)

* 比较函数

  ```python
  假如我们想要知道矩阵a和矩阵b中所有对应元素是否相等，我们需要使用all方法，
  假如我们想要知道矩阵a和矩阵b中对应元素是否有一个相等，我们需要使用any方法。
  # np.any(): 至少有一个元素满足指定条件，返回True
  # np.all(): 所有的元素满足指定条件，返回True
  
  # 实例代码
  arr = np.random.randn(2, 3)
  print(arr)
  print(np.any(arr > 0))
  print(np.all(arr > 0))
  ```

  ![1713464541500](assets/1713464541500.png)

* 去重函数

  ```python
  # np.unique():找到唯一值并返回排序结果，类似于Python的set集合
  
  # 实例代码
  arr = np.array([[1, 2, 1], [2, 3, 4]])
  print(arr)
  print(np.unique(arr))
  ```

  ![1713464617558](assets/1713464617558.png)

* 排序函数

  ```python
  arr = np.array([1, 2, 34, 5])
  print("原数组arr:", arr)
  
  # np.sort()函数排序, 返回排序后的副本
  sortarr1 = np.sort(arr)
  print("numpy.sort()函数排序后的数组:", sortarr1)
  
  # ndarray直接调用sort, 在原数据上进行修改
  arr.sort()
  print("数组.sort()方法排序:", arr)
  ```

  ![1713464726218](assets/1713464726218.png)



#### 11.Numpy运算

* 基本运算

  ```python
  # 数组的算数运算是按照元素的。新的数组被创建并且被结果填充。
  
  # 示例代码
  import numpy as np
  
  a = np.array([20, 30, 40, 50])
  b = np.arange(4)
  c = a - b
  print("数组a:", a)
  print("数组b:", b)
  print("数组运算a-b:", c)
  ```

  ![1713465083593](assets/1713465083593.png)

  > 两个ndarray, 一个是arr_a  另一个是arr_b
  >
  > 它们俩之间进行  arr_a  + arr_b  或 arr_a  - arr_b  或 arr_a  * arr_b 这样计算的前提是 shape相同
  >
  > 计算的时候, 位置对应的元素 进行 加减乘除的计算, 计算之后得到的结果的shape 跟arr_a  /arr_b 一样

* 矩阵运算

  > `arr_a.dot(arr_b) 前提`   **arr_a 列数 = arr_b行数**

  * 场景1

    ![1713465337722](assets/1713465337722.png)

    ```python
    import numpy as np
    
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    print(a * b)
    
    print(np.multiply(a, b))
    ```

    ![1713465463292](assets/1713465463292.png)

  * 场景2

    ![1713465512782](assets/1713465512782.png)

    ​	![1713465542079](assets/1713465542079.png)

    ```python
    import numpy as np
    
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[6, 23], [-1, 7], [8, 9]])
    print(x)
    print(y)
    print(x.dot(y))
    print(np.dot(x, y))
    ```

    ![1713465621030](assets/1713465621030.png)