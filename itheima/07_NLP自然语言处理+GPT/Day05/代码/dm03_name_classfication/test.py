# import torch.nn as nn
# import torch
# b = nn.LogSoftmax(dim=-1)
# a = torch.tensor([[1.0, 2.0, 0.0],
#                   [1.0, 2.0, 0.0]]) # [2, 3]
#
# print(b(a))
import json
# list1 = [1, 2, 3]
# a = 4
# list2 = [2, 3, 4]
# dict1 = {"loss": list1,
#          "time": a,
#          "acc": list2}
# with open('a.json', 'w')as fw:
#     fw.write(json.dumps(dict1))

with open('a.json', 'r')as fr:
    results = fr.read()
    print(type(results))
    print(f'results-->{results}')
dict1 = json.loads(results)
print(type(dict1))
print(dict1)
print(dict1["loss"])




