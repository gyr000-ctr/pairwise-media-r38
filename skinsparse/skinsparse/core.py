"""skinsparse 核心 API（初始未实现，仅标准库）。"""


class SkinSparse:
    def compress(self, weights, k):
        """压缩"""
        raise NotImplementedError

    def decompress(self, data):
        """还原"""
        raise NotImplementedError

    def error(self):
        """误差"""
        raise NotImplementedError
