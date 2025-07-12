def split_binary_file(file_path, chunk_size=1024 * 1024 * 10, output_prefix='part_'):
    """
    按字节大小拆分二进制文件。

    :param file_path: 原始文件路径
    :param chunk_size: 每个小文件的字节大小，默认10MB
    :param output_prefix: 输出小文件的前缀
    """
    file_count = 1
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            with open(f'{output_prefix}{file_count}', 'wb') as out_f:
                out_f.write(chunk)
            file_count += 1


# 使用示例
split_binary_file('ai20_fil9.bin', chunk_size=10 * 1024 * 1024, output_prefix='small_part_')