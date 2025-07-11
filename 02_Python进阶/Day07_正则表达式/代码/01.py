# 思路二：哈希表（HashMap）
def twoSum_hash_map(nums, target):
    """
    使用一个哈希表来存储数组中的元素和它们的索引。
    遍历数组，对于每个元素，计算其与目标值的差值，然后检查该差值是否已经在哈希表中。
    如果在，则返回对应的索引和当前元素的索引；如果不在，则将当前元素和它的索引存入哈希表中。

    参数:
    nums (List[int]): 整数数组
    target (int): 目标值

    返回:  
    List[int]: 包含两个整数索引的列表，这两个索引对应的元素之和等于目标值
    如果没有找到则返回 None
    """
    num_dict = {}  # 创建一个空的哈希表
    for i, num in enumerate(nums):
        complement = target - num  # 计算补数
        if complement in num_dict:
            return [num_dict[complement], i]  # 返回补数的索引和当前元素的索引
        num_dict[num] = i  # 将当前元素和它的索引存入哈希表中
    return None  # 如果没有找到则返回None


# 示例
if __name__ == '__main__':
    nums = [2, 7, 11, 15]
    target = 9
    print(twoSum_hash_map(nums, target))  # 输出: [0, 1]