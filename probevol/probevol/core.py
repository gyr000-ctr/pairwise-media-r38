"""probevol 核心 API（初始未实现，仅标准库）。"""


class ProbeVolume:
    def __init__(self, res, probes):
        """构造"""
        raise NotImplementedError

    def weights(self, p):
        """权重"""
        raise NotImplementedError

    def sample(self, p, dirn):
        """采样"""
        raise NotImplementedError
