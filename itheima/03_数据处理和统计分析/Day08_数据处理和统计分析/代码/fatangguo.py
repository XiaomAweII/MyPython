def candy(ratings):
    # 先初始给每一个孩子一个糖果
    ls = [1]*len(ratings)
    # 从左往右遍历比较大小
    for i in range(len(ratings)-1):
        # 如果相邻后一个数更大，那后一个孩子就要比前一个孩子多一个糖果
        if ratings[i+1] > ratings[i]:
            ls[i+1]=ls[i]+1
        elif ratings[i+1] == ratings[i]:
            ls[i + 1] = ls[i]
    # 从右往左遍历比较大小
    for i in range(len(ratings)-1, 0, -1):
        if ratings[i-1] > ratings[i]:
            # 如果相邻前一个数更大，那前一个孩子就要比后一个孩子多一个糖果，取左右遍历得到的最大数即满足左右循环且最小糖果数
            ls[i-1]=max(ls[i-1], ls[i]+1)
        elif ratings[i-1] == ratings[i]:
            ls[i-1]=ls[i]
    print(ls)
    return sum(ls)
if __name__ == '__main__':
    ratings = [1, 3, 5, 4, 3, 2, 1, 7]
    print(candy(ratings))
