# 执行操作使频率分数最大
#
# 给你一个下标从 0 开始的整数数组 nums 和一个整数 k 。
# 你可以对数组执行 至多 k 次操作：
# 从数组中选择一个下标 i ，将 nums[i] 增加 或者 减少 1 。
# 最终数组的频率分数定义为数组中众数的 频率 。
# 请你返回你可以得到的 最大 频率分数。
# 众数指的是数组中出现次数最多的数。一个元素的频率指的是数组中这个元素的出现次数。

# 示例 1：

# 输入：nums = [1,2,6,4], k = 3
# 输出：3
# 解释：我们可以对数组执行以下操作：
# - 选择 i = 0 ，将 nums[0] 增加 1 。得到数组 [2,2,6,4] 。
# - 选择 i = 3 ，将 nums[3] 减少 1 ，得到数组 [2,2,6,3] 。
# - 选择 i = 3 ，将 nums[3] 减少 1 ，得到数组 [2,2,6,2] 。
# 元素 2 是最终数组中的众数，出现了 3 次，所以频率分数为 3 。
# 3 是所有可行方案里的最大频率分数。
# 示例 2：

# 输入：nums = [1,4,4,2,4], k = 0
# 输出：3
# 解释：我们无法执行任何操作，所以得到的频率分数是原数组中众数的频率 3 。
# 代码如下:


def maxFrequencyScore(nums, k):

    nums.sort()
    # 数组的副本
    copy_sum = nums
    # 对nums数组做累加和
    for i in range(1, len(nums)):
        copy_sum[i] += copy_sum[i - 1]

    # 作用是把从 i 到 r 的子数组内所有的元素变为中位数所需要的操作次数是否在 k次内
    def check(l, r):
        if r - l == 0:
            return True
        m = (l + r) // 2
        f_left = nums[m] * (m - l + 1) - (copy_sum[m] - copy_sum[l - 1] if l else copy_sum[m])
        f_right = (copy_sum[r] - copy_sum[m - 1] if m else copy_sum[r]) - nums[m] * (r - m + 1)
        return f_left + f_right <= k

    l = 0

    freq = 1
    for r in range(len(nums)):
        while not check(l, r):
            l += 1
        freq = max(freq, r - l + 1)
    return freq


# 测试
if __name__ == '__main__':
    nums1 = [1, 2, 3, 4, 5, 6]
    nums2 = [2, 4, 3, 7, 6, 5, 3, 8, 7, 5]
    nums3 = [1, 2, 6, 4]
    k = 4
    result = maxFrequencyScore(nums1, k)
    print(result)  # 输出应为最大频率分数
