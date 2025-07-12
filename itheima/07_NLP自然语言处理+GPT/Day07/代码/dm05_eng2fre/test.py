import torch
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "mps"
# print(device)

# import re
#
# a = "我是超人.!"
# print(re.sub(r'([.!])', r'aaa\1', a))
# list1 = [1, 2,3]
# for i, value in enumerate(list1, start=10):
#     print(i, value)

# import torch
#
# # 解码参数1 encode_output_c [10,256]
# x = torch.randn(1, 6)
# encode_output = torch.randn(1, 6, 8)
# print(encode_output)
# encode_output_c = torch.zeros(10, 8)
# print('*'*80)
# print(f'encode_output_c-》{encode_output_c}')
# for idx in range(x.shape[1]):
#     print(encode_output_c[idx])
#     encode_output_c[idx] = encode_output[0, idx]
# print(encode_output_c)
import random
print(random.random())