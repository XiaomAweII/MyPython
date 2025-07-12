import pandas as pd

chunk_size = 100000  # 每个文件的行数
file_count = 1

for chunk in pd.read_csv('installments_payments.csv', chunksize=chunk_size):
    chunk.to_csv(f'small_file_{file_count}.csv', index=False)
    file_count += 1