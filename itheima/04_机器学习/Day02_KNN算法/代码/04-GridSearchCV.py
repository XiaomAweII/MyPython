"""
## 交叉验证+网格搜索
# 1.导入工具包
# 2.加载数据
# 3.数据集划分
# 4.特征预处理
# 5.模型实例化+交叉验证+网格搜索
# 6.模型训练
# 7.模型预测
# 8.模型评估
"""


# 1.导入工具包
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 2.加载数据
data = load_iris()

# 3.数据集划分
x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.2,random_state=22)

# 4.特征预处理
pre = StandardScaler()
x_train=pre.fit_transform(x_train)
x_test=pre.transform(x_test)

# 5.模型实例化+交叉验证+网格搜索
model = KNeighborsClassifier(n_neighbors=1)
paras_grid = {'n_neighbors':[4,5,7,9]}
# estimator =GridSearchCV(estimator=model,param_grid=paras_grid,cv=4)
# estimator.fit(x_train,y_train)

# print(estimator.best_score_)
# print(estimator.best_estimator_)
# print(estimator.cv_results_)

model = KNeighborsClassifier(n_neighbors=7)
# 6.模型训练
model.fit(x_train,y_train)
x = [[5.1, 3.5, 1.4, 0.2]]
x=pre.transform(x)

# 7.模型预测
y_prdict=model.predict(x_test)

# 8.模型评估
print(accuracy_score(y_test,y_prdict))

