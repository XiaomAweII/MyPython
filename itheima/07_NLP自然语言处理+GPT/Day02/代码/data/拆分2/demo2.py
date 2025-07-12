import os

def merge_binary_files(input_prefix, output_file):
    """
    合并拆分的二进制文件。

    :param input_prefix: 拆分文件的前缀，如 'part_'
    :param output_file: 合并后输出的文件名
    """
    file_count = 1
    with open(output_file, 'wb') as out_f:
        while True:
            part_file = f'{input_prefix}{file_count}'
            if not os.path.exists(part_file):
                break
            with open(part_file, 'rb') as in_f:
                out_f.write(in_f.read())
            file_count += 1

# 使用示例
merge_binary_files('small_part_', 'fil9')