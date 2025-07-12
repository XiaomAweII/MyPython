"""
# 1.导入依赖包
# 2.构建数据:真实值,预测值
# 3.1 混淆矩阵
# 3.2 精确率
# 3.3 召回率
# 3.4 f1-score

y_true = ['恶性','恶性','恶性','恶性','恶性','恶性','良性','良性','良性','良性']
y_pre_A = ['恶性','恶性','恶性','良性','良性','良性','良性','良性','良性','良性']
y_pre_B = ['恶性','恶性','恶性','恶性','恶性','恶性','恶性','恶性','恶性','良性']
"""

# 1.导入依赖包
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd

# 2.构建数据:真实值,预测值
y_true = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性']
y_pre_A = ['恶性', '恶性', '恶性', '良性', '良性', '良性', '良性', '良性', '良性', '良性']
y_pre_B = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性']

# 3.1 混淆矩阵
A = confusion_matrix(y_true, y_pre_A, labels=['恶性', '良性'])
print(pd.DataFrame(A, columns=['恶性(正例)', '良性(反例)'], index=['恶性(正例)', '良性(反例)']))
# B = confusion_matrix(y_true, y_pre_B, labels=['恶性', '良性'])
# print(pd.DataFrame(B, columns=['恶性(正例)', '良性(反例)'], index=['恶性(正例)', '良性(反例)']))

# 3.2 精确率
print(precision_score(y_true, y_pre_A, pos_label='恶性'))
print(precision_score(y_true, y_pre_B, pos_label='恶性'))

# 3.3 召回率
print(recall_score(y_true, y_pre_A, pos_label='恶性'))
print(recall_score(y_true, y_pre_B, pos_label='恶性'))

# 3.4 f1-score
print(f1_score(y_true, y_pre_A, pos_label='恶性'))
print(f1_score(y_true, y_pre_B, pos_label='恶性'))
