"""ikfabrik 核心 API（初始未实现，仅标准库）。"""


class Fabrik:
    def __init__(self, lengths, iters=16, tol=1e-6):
        """构造"""
        raise NotImplementedError

    def solve(self, root, target):
        """求解"""
        raise NotImplementedError

    def joints(self):
        """关节位置"""
        raise NotImplementedError
