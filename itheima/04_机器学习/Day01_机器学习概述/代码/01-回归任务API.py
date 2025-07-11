"""
## （一）回归任务预测
# 1.导入依赖包
# 2.准备数据
# 3.实例化 线性回归模型
# 4.模型训练
# 5.模型预测

x = [[80, 86], [82, 80], [85, 78], [90, 90], [86, 82], [82, 90], [78, 80], [92, 94]]
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]
"""
# 1.导入依赖包
from sklearn.linear_model import LinearRegression


def dm01_Regression_pred():
    # 2.准备数据 平时成绩 期末成绩 最终成绩
    x = [[80, 86], [82, 80], [85, 78], [90, 90], [86, 82], [82, 90], [78, 80], [92, 94]]
    y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]
    # 3.实例化 线性回归模型
    estimator = LinearRegression()
    print('estimator-->', estimator)

    # 4.模型训练
    # 打印 线性回归模型参数 coef_ intercept_
    estimator.fit(x, y)
    print('estimator.coef_-->', estimator.coef_)
    print('estimator.intercept_-->', estimator.intercept_)

    # 5.模型预测
    mypred = estimator.predict([[90, 80]])
    print('mypred-->', mypred)


"""
## （二）回归任务保存和加载模型
# 1.导入依赖包
# 2.准备数据
# 3.实例化 线性回归模型
# 4.模型训练
# 5.模型预测
# 6.模型保存 joblib.dump(estimator, xxpath)
# 7.模型加载 joblib.load(xxpath)
# 8.模型预测

x = [[80, 86], [82, 80], [85, 78], [90, 90], [86, 82], [82, 90], [78, 80], [92, 94]]
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

"""
# 1.导入依赖包
import joblib


def dm02_Regression_save_load():
    # 2.准备数据 平时成绩 期末成绩 最终成绩
    x = [[80, 86], [82, 80], [85, 78], [90, 90], [86, 82], [82, 90], [78, 80], [92, 94]]
    y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

    # 3.实例化 线性回归模型
    estimator = LinearRegression()
    print('estimator-->', estimator)

    # 4.模型训练
    estimator.fit(x, y)

    # 5.模型预测
    mypred = estimator.predict([[90, 80]])
    print('mypred-->', mypred)

    # 6.模型保存
    print('\n模型保存和模型重新加载')
    joblib.dump(estimator, './model/mylrmodel01.bin')

    # 7.模型加载
    myestimator2 = joblib.load('./model/mylrmodel01.bin')
    print('myestimator2-->', myestimator2)

    # 8.模型预测
    mypred2 = myestimator2.predict([[90, 80]])
    print('mypred2-->', mypred2)


if __name__ == '__main__':
    # （一）回归任务预测
    dm01_Regression_pred()
    # （二）回归任务保存和加载模型
    # dm02_Regression_save_load()
