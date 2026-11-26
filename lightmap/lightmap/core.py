"""lightmap 核心 API（初始未实现，仅标准库）。"""


class Lightmap:
    def __init__(self, tex, bounds):
        """构造"""
        raise NotImplementedError

    def uv(self, pos):
        """位置到 UV"""
        raise NotImplementedError

    def sample(self, pos):
        """采样"""
        raise NotImplementedError
