class Stack(object):
    def __init__(self):
        # 通过列表模拟栈空间
        self.stack_list = []

    def push(self, item):
        """
        入栈
        :param item: 入栈元素
        :return:
        """
        self.stack_list.append(item)

    def pop(self):
        """
        出栈
        :return: 栈顶元素
        """
        return self.stack_list.pop()

    def peek(self):
        """
        窥视栈顶元素
        :return: 栈顶元素  or None
        """
        if len(self.stack_list) > 0:
            return self.stack_list[-1]

    def size(self):
        """
        查看栈空间长度
        :return: 栈空间长度
        """
        return len(self.stack_list)

    def is_empty(self):
        """
        判断栈是否为空
        :return: True or False
        """
        return len(self.stack_list) == 0

    def show_stack_list(self):
        """
        查看栈空间所有元素
        :return: 栈空间所有元素
        """
        return self.stack_list

# 括号映射字典
blacket_dict = {
    '(':')',
    '[':']',
    '{':'}'
}

def handle_valid_blacket(blacket_str):
    """
    处理有效括号的函数
    :param blacket_str: 括号字符串
    :return: True or False
    """
    # 判断括号字符串如果为空,则不走下面的逻辑
    if len(blacket_str) == 0:
        return False
    # 填充左括号的栈
    stack = Stack()
    # 校验完合法的右括号后,填充剩下的非法右括号的栈
    rubbish_stack = Stack()
    # 遍历输入括号字符串
    for blacket_item in blacket_str:
        # 判断如果是左括号就填充进stack
        if blacket_item in blacket_dict.keys():
            stack.push(blacket_item)
        # 判断栈顶元素和右括号是否匹配,如果匹配则弹出栈顶的左括号
        if stack.size() > 0 and blacket_item == blacket_dict[stack.peek()]:
            stack.pop()
        # 走完上面的右括号校验后,如果是非法的右括号则弹进垃圾栈
        elif blacket_item in blacket_dict.values():
            rubbish_stack.push(blacket_item)
    print(stack.show_stack_list())
    print(rubbish_stack.show_stack_list())
    # 如果填充左括号的栈和填充非法右括号的垃圾栈都为空,则是合法的有效括号字符串
    return stack.is_empty() and rubbish_stack.is_empty()

if __name__ == '__main__':
    # stack = Stack()
    # print(stack.size())
    # print(stack.is_empty())
    # stack.push('lulu')
    # stack.push('julia')
    # print(stack.show_stack_list())
    # print(stack.peek())
    # stack.pop()
    # print(stack.size())
    # print(stack.show_stack_list())
    str1 = '((()))'
    str2 = '{[()]()()]}'
    str3 = ']'
    str4 = ']}}'
    print(handle_valid_blacket(str1))
    print(handle_valid_blacket(str2))
    print(handle_valid_blacket(str3))
    print(handle_valid_blacket(str4))

