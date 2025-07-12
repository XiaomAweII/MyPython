"""
## RBF参数测试
# 1.导入依赖包
# 2.获取数据
# 3.构建函数
# 4.实验测试【0.5,1.0,100,0.1】,模型训练
"""

# 1.导入依赖包
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from plot_util import plot_decision_boundary

# 2.获取数据
x, y = make_moons(noise=0.15)
plt.scatter(x[y == 0, 0], x[y == 0, 1])
plt.scatter(x[y == 1, 0], x[y == 1, 1])
plt.show()

# 3.构建函数
def RBFsvm(gamma=0.1):
    return Pipeline(
        [
            ('std_scalar', StandardScaler()),
            ('svc', SVC(kernel='rbf', gamma=gamma))
        ]
    )


## 4.实验测试【0.5,1.0,100,0.1】,模型训练
# svc1 = RBFsvm(0.5)
# x_std=svc1['std_scalar'].fit_transform(x)
# svc1.fit(x_std,y)
# plot_decision_boundary(svc1,axis=[-3,3,-3,3])
# plt.scatter(x_std[y==0,0],x_std[y==0,1])
# plt.scatter(x_std[y==1,0],x_std[y==1,1])
# plt.show()

svc1 = RBFsvm(1.0)
svc1.fit(x, y)
plot_decision_boundary(svc1, axis=[-1.5, 2.5, -1, 1.5])
plt.scatter(x[y == 0, 0], x[y == 0, 1])
plt.scatter(x[y == 1, 0], x[y == 1, 1])
plt.show()

svc2 = RBFsvm(100)
svc2.fit(x, y)
plot_decision_boundary(svc2, axis=[-1.5, 2.5, -1, 1.5])
plt.scatter(x[y == 0, 0], x[y == 0, 1])
plt.scatter(x[y == 1, 0], x[y == 1, 1])
plt.show()

svc3 = RBFsvm(0.1)
svc3.fit(x, y)
plot_decision_boundary(svc3, axis=[-1.5, 2.5, -1, 1.5])
plt.scatter(x[y == 0, 0], x[y == 0, 1])
plt.scatter(x[y == 1, 0], x[y == 1, 1])
plt.show()