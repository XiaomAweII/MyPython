"""
## 逻辑回归案例
# 1.导入依赖包
# 2.加载数据及数据预处理
# 2.1 数据处理，缺失值
# 2.2 获取特征和目标值
# 2.3 数据划分
# 3.特征工程(标准化)
# 4.模型训练
# 5.模型预测和评估
"""

# 1.导入依赖包
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# 2.加载数据及数据预处理
data =pd.read_csv('breast-cancer-wisconsin.csv')
# print(data.info)
# 2.1 数据处理，缺失值
data =data.replace(to_replace='?',value=np.NAN)
data=data.dropna()

# 2.2 获取特征和目标值
X = data.iloc[:,1:-1]
y = data['Class']

# 2.3 数据划分
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=22)

# 3.特征工程(标准化)
pre =StandardScaler()
x_train=pre.fit_transform(x_train)
x_test=pre.transform(x_test)

# 4.模型训练
model=LogisticRegression()
model.fit(x_train,y_train)

# 5.模型预测和评估
y_predict =model.predict(x_test)
print(y_predict)
print(accuracy_score(y_test,y_predict))
