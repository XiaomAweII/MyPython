"""
## 随机森林
# 1.导入依赖包
# 2.读取数据
# 3.数据处理
# 4.模型训练
# 4.1 决策树
# 4.2 随机森林
# 4.3 网格搜索交叉验证
# 5.模型评估
# 5.1 决策树
# 5.2 随机森林
# 5.3 网格搜索交叉验证
"""

# 1.导入依赖包
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 2.读取数据
data = pd.read_csv('./data/titanic/train.csv')
print(data.head())
print(data.info)

# 3.数据处理
x = data[['Pclass', 'Sex', 'Age']].copy()
y = data['Survived'].copy()
print(x.head(10))
x['Age'].fillna(x['Age'].mean(), inplace=True)
print(x.head(10))
x = pd.get_dummies(x)
print(x.head(10))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 4.模型训练
# 4.1 决策树
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)

# 4.2 随机森林
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# 4.3 网格搜索交叉验证
params = {'n_estimators': [10, 20], 'max_depth': [2, 3, 4, 5]}
model = GridSearchCV(estimator=rf, param_grid=params, cv=3)
model.fit(x_train, y_train)
print(model.best_estimator_)

rfs = RandomForestClassifier(max_depth=4, n_estimators=10)
rfs.fit(x_train, y_train)

# 5.模型评估
# 5.1 决策树
print(tree.score(x_test, y_test))

# 5.2 随机森林
print(rf.score(x_test, y_test))

# 5.3 网格搜索交叉验证
print(rfs.score(x_test, y_test))
