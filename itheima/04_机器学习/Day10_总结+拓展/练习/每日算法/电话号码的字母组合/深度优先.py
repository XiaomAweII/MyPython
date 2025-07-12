def letterCombinationsDFS(digits):
    # 如果输入的字符串为空，则直接返回空列表
    if not digits:
        return []

    # 创建一个字典，用于存储每个数字对应的字母
    phone_map = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }

    # 定义一个辅助函数来进行深度优先搜索
    def dfs(index, path):
        # 当路径的长度等于输入字符串的长度时，说明已经找到了一组完整的组合
        if len(path) == len(digits):
            # 将当前的路径添加到结果列表中
            # path 是一个列表，包含当前组合中的所有字母，需要将它们连接成一个字符串
            result.append(''.join(path))
            return

        # 获取当前索引对应的数字
        for letter in phone_map[digits[index]]:
            # 将当前字母添加到路径中
            path.append(letter)
            # 递归地对下一个数字进行搜索
            dfs(index + 1, path)
            # 回溯，移除最后一个字母
            path.pop()

    # 初始化结果列表
    result = []
    # 从第一个数字开始进行深度优先搜索
    dfs(0, [])
    # 返回所有找到的组合
    return result

if __name__ == '__main__':
    # 从用户那里获取输入
    digits = input('请输入数字：')

    print("DFS Result:")
    print(letterCombinationsDFS(digits))






'''
# 导入 copy 模块，用于深拷贝列表
import copy

def letterCombinations(digits):
    # 创建一个字典，用于存储每个数字对应的字母
    conversion = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }

    # 如果输入的字符串为空，则直接返回空列表
    if len(digits) == 0:
        return []

    # 初始化一个列表，其中包含一个空字符串
    product = ['']

    # 遍历输入字符串中的每一个数字
    for k in digits:
        # 创建一个临时列表，用于存放当前层级的所有组合
        tmp = []

        # 对当前的组合列表进行扩展
        for i in product:
            # 对于当前数字对应的每一个字母
            for j in conversion[k]:
                # 将当前组合加上新的字母，并将结果添加到临时列表中
                tmp.append(i + j)

        # 使用深拷贝将临时列表赋值给 product
        # 这里使用深拷贝是为了避免引用同一对象的问题，但在这个上下文中其实不需要深拷贝
        product = copy.deepcopy(tmp)

    # 返回所有找到的组合
    return product

if __name__ == '__main__':
    # 从用户那里获取输入
    digits = input()
    # 调用函数并打印结果
    r = letterCombinations(digits)
    print(r)
'''