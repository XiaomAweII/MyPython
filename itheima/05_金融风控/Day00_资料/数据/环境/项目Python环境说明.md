# Python环境说明

项目中在Anaconda的基础上，可能还需要安装的Python库，都列举如下：

~~~shell
#1.安装pyecharts库，用来画图（了解）
pip install pyecharts -i https://pypi.tuna.tsinghua.edu.cn/simple
#2.安装pydotplus库
pip install pydotplus -i https://pypi.tuna.tsinghua.edu.cn/simple
#3.安装graphviz库
pip install graphviz -i https://pypi.tuna.tsinghua.edu.cn/simple
#4.安装Boruta库
pip install Boruta -i https://pypi.tuna.tsinghua.edu.cn/simple
#5.安装toad库（注意版本）
pip install toad==0.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
#6.安装LightGBM库
pip install lightgbm -i https://pypi.tuna.tsinghua.edu.cn/simple/
#7.安装xgboost库
pip install xgboost -i https://pypi.tuna.tsinghua.edu.cn/simple/
#8.安装pyod库
pip install pyod -i https://pypi.tuna.tsinghua.edu.cn/simple/
~~~

注意：

graphviz库安装完后，不一定能正常显示图。可能需要配置graphviz的环境变量，同时安装`xxx-graphviz-xxx.exe`安装包。

