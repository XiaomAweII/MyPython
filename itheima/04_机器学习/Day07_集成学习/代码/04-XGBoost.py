"""
## XGBoost
# 1.导入依赖包
# 2.数据读取及数据预处理
# 2.1 数据获取
# 2.2 数据预处理
# 2.3 数据集划分
# 3.模型训练
# 4.模型预测
# 5.模型评估
"""

# 1.导入依赖包
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn.ensemble import GradientBoostingClassifier

# # 2.数据读取及数据预处理
# # 2.1 数据获取
# data =pd.read_csv('./data/红酒品质分类.csv')
# print(data.head())
# # 2.2 数据预处理
# x = data.iloc[:,:-1]
# y = data.iloc[:,-1]-3
# # 2.3 数据集划分
# x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.2)

# pd.concat([x_train,y_train],axis=1).to_csv('红酒品质分类_train.csv')
# pd.concat([x_test,y_test],axis=1).to_csv('红酒品质分类_test.csv')

# 2.数据读取及数据预处理
# 2.1 数据获取
train_data = pd.read_csv('红酒品质分类_train.csv')
test_data = pd.read_csv('红酒品质分类_test.csv')
# 2.2 数据预处理
x_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]
class_weight = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

# 3.模型训练
model = XGBClassifier(n_estimators=5, objective='multi:softmax')
# model =GradientBoostingClassifier(n_estimators=5)
model.fit(x_train, y_train, sample_weight=class_weight)

# 4.模型预测
y_pre = model.predict(x_test)

# 5.模型评估
print(classification_report(y_test, y_pre))
