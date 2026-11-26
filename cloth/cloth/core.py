"""cloth 核心 API（初始未实现，仅标准库）。"""


class Cloth:
    def __init__(self, w, h, spacing, stiffness):
        """构造"""
        raise NotImplementedError

    def step(self, dt, colliders):
        """步进"""
        raise NotImplementedError

    def max_stretch(self):
        """最大伸长"""
        raise NotImplementedError
