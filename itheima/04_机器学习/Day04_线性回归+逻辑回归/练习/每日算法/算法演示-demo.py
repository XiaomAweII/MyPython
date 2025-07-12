def roman_to_int(s):
    # 定义一个字典，将罗马数字字符映射到对应的整数值,定义的都是转换后得值
    map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

    # 初始化一个变量来存储转换后的整数值
    number = 0

    # 遍历罗马数字字符串中的每个字符（除了最后一个）,因为如果进行比较的话,他没有右侧的值,该越界了
    for i in range(len(s) - 1):
        # 如果当前字符代表的值小于下一个字符代表的值，则从总数中减去当前字符的值
        # 这是因为在罗马数字中，较小的数字在较大的数字左边时表示减法
        if map[s[i]] < map[s[i + 1]]:     #如果当前项小于右边的这一项
            number -= map[s[i]]
        else:
            # 否则，将当前字符的值加到总数上
            number += map[s[i]]

            # 加上最后一个字符的值，因为它没有下一个字符来进行比较
    # 这一步是必要的，因为循环只遍历到倒数第二个字符
    number += map[s[-1]]

    # 返回计算得到的整数值
    return number


if __name__ == '__main__':
    print(roman_to_int("III"))  # 输出: 3
    print(roman_to_int("IV"))  # 输出: 4
    print(roman_to_int("IX"))  # 输出: 9
    print(roman_to_int("LVIII"))  # 输出: 58
    print(roman_to_int("MCMXCIV"))  # 输出: 1994  1000-100+1000-10+100-1          +5
