"""
## FeatureExtraction
# 1.导入依赖包
# 2.1 低方差过滤法
# 2.2 PCA主成分分析
# 2.3 相关系数法
"""

# 1.导入依赖包
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# 2.1 低方差过滤法
data = pd.read_csv('垃圾邮件分类数据.csv')
print(data.shape)

transform = VarianceThreshold(threshold=0.1)
x = transform.fit_transform(data)
print(x.shape)


# 2.2 PCA主成分分析
x, y = load_iris(return_X_y=True)
print(x)
pca1 = PCA(n_components=0.95)
print(pca1.fit_transform(x))
pca1 = PCA(n_components=3)
print(pca1.fit_transform(x))

# 2.3 相关系数法
x, y = load_iris(return_X_y=True)
x1 = x[:, 2]
x2 = x[:, 1]

print(pearsonr(x1, x2))
print(spearmanr(x1, x2))