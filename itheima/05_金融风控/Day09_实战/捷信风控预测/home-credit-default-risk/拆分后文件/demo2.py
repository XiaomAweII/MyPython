import pandas as pd
import glob

# 获取所有拆分的小文件，按文件名排序确保顺序正确
file_list = sorted(glob.glob('small_file_*.csv'))

# 读取所有小文件并合并成一个大的DataFrame
df_list = [pd.read_csv(file) for file in file_list]
merged_df = pd.concat(df_list, ignore_index=True)

# 保存合并后的大文件
merged_df.to_csv('merged_installments_payments.csv', index=False)