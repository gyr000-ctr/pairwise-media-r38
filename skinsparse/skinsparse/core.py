"""skinsparse —— 稀疏蒙皮压缩（仅使用 Python 标准库）。

功能概览
--------
``SkinSparse.compress(weights, k)``
    对每个顶点仅保留权重最大的 ``k`` 个骨骼，并把保留的权重归一化到
    总和 1（浮点误差 <= 1e-9），随后量化为 8 位整数（0..255）。
    返回 ``(indices, quantized)`` 二元组：

    * ``indices``  —— 每个顶点一个骨骼索引列表（按权重降序、权重相同时按
      索引升序，保证结果确定）；
    * ``quantized`` —— 与 ``indices`` 逐项对应的 8 位量化权重（0..255）。

    两个表按顶点一一对齐，且每个顶点的索引个数与权重个数严格相等。

``SkinSparse.decompress(data)``
    把 ``compress`` 的返回值反量化为 ``[[float, ...], ...]``；每个顶点
    的权重之和为 255/255 = 1（浮点误差约 1e-16）。

``SkinSparse.error()``
    返回最近一次 ``compress`` 中“被丢弃权重”的相对误差上界
    ``max_vertex((总权重 - 保留权重) / 总权重)``，类型为 ``float``。
    未执行过压缩时返回 ``0.0``。

量化方案
--------
归一化权重 ``w_i`` 先放大 255 倍，向下取整得到 ``q_i``，再把剩余的
``255 - sum(q_i)`` 个单位按小数部分从大到小逐个分配（最大余数法）。
这样得到的整数满足 ``sum(q_i) == 255``，且每个分量满足
``|q_i / 255 - w_i| <= 1/255``，即“8 位量化后反量化误差 <= 1/255”。

非法输入
--------
所有错误都派生自 :class:`SkinSparseError`（同时是 ``ValueError``）：

* :class:`InvalidKError`     —— 非法 k（非整数、bool、<= 0）；
* :class:`InvalidWeightsError` —— 空输入、空行、非数值、NaN/Inf、负数、
  行权重全为 0 等；
* :class:`InvalidDataError`  —— ``decompress`` 收到结构不合法的数据。

因此调用方既可以捕获具体的子类来区分错误类型，也可以像普通
``ValueError`` 一样统一处理。权重必须是 ``int``/``float``（``bool``
按 Python 数值语义视作 0/1）。
"""

from __future__ import annotations

import heapq
import math

__all__ = [
    "SkinSparse",
    "SkinSparseError",
    "InvalidKError",
    "InvalidWeightsError",
    "InvalidDataError",
]

#: 8 位量化的最大整数值。
_QUANT_MAX = 255
_INV_QUANT_MAX = 1.0 / _QUANT_MAX
_INF = float("inf")


class SkinSparseError(ValueError):
    """skinsparse 所有非法输入错误的基类（兼容 ValueError）。"""


class InvalidKError(SkinSparseError):
    """非法的 k：必须为 ``int``（非 bool）且 ``k >= 1``。"""


class InvalidWeightsError(SkinSparseError):
    """非法的权重输入。"""


class InvalidDataError(SkinSparseError):
    """非法的压缩数据（``decompress`` 的输入）。"""


def _is_real_number(value):
    """判断 value 能否按实数处理（bool 视作 0/1，数值语义与 C 级路径一致）。"""
    return isinstance(value, (int, float))


def _quantize_scaled(scaled):
    """把“已放大 255 倍”的非负权重列表量化为 8 位整数。

    使用最大余数法（largest remainder）：先向下取整，再把剩余的
    ``255 - sum(q)`` 个单位按小数部分从大到小分配（并列时索引小者
    优先，保证确定性）。结果满足 ``sum(q) == 255``，且每个分量满足
    ``|q/255 - w| <= 1/255``。

    返回 ``(quant, max_err)``：``quant`` 为整数列表，``max_err`` 为
    该顶点实际出现的最大反量化误差（相对 ``scaled/255``）。
    """
    n = len(scaled)
    quant = list(map(int, scaled))  # 非负，int() 即 floor
    if n == 0:
        return quant, 0.0
    units = _QUANT_MAX - sum(quant)

    if units <= 0:
        # units == 0：误差即最大小数部分。units < 0 是浮点极端兜底，
        # 正常不会发生，直接逐项计算实际误差。
        if units == 0:
            worst = max(map(float.__sub__, scaled, quant))
            return quant, worst * _INV_QUANT_MAX
        fracs = [(scaled[i] - quant[i], -i) for i in range(n)]
        fracs.sort()
        for t in range(min(-units, n)):
            j = -fracs[t][1]
            if quant[j] > 0:
                quant[j] -= 1
        worst = max(map(abs, map(float.__sub__, quant, scaled)))
        return quant, worst * _INV_QUANT_MAX

    # 需要额外 +1 的单位数等于“小数部分之和”，故 0 <= units < n。
    # 键为 (小数部分, -索引)：降序即“小数部分大者优先，并列时索引小者
    # 优先”，保证确定性。
    fracs = sorted(
        zip(map(float.__sub__, scaled, quant), range(0, -n, -1)),
        reverse=True,
    )
    if units <= n:
        for t in range(units):
            quant[-fracs[t][1]] += 1
        # 被 +1 的最坏误差 = 1 - 最小被调整的小数部分；
        # 未被调整的最坏误差 = 最大未调整的小数部分。
        worst = 1.0 - fracs[units - 1][0]
        if units < n and fracs[units][0] > worst:
            worst = fracs[units][0]
        return quant, worst * _INV_QUANT_MAX

    # 浮点极端兜底（正常不会发生）：轮转分配，误差需逐项计算。
    for t in range(units):
        quant[-fracs[t % n][1]] += 1
    worst = max(map(abs, map(float.__sub__, quant, scaled)))
    return quant, worst * _INV_QUANT_MAX


class SkinSparse:
    """稀疏蒙皮压缩器。

    同一实例可重复使用；``error()`` 报告最近一次 ``compress`` 的结果。
    实例不持有对输入列表的引用，也不会修改输入。
    """

    #: 8 位量化的最大值（类属性，便于外部引用）。
    QUANT_MAX = _QUANT_MAX

    def __init__(self):
        self._error = 0.0
        self._quant_error = 0.0
        self._k = None
        self._discarded = []
        self._indices = None
        self._quantized = None

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_k(k):
        if isinstance(k, bool) or not isinstance(k, int):
            raise InvalidKError(
                "k 必须是正整数（int，非 bool），收到 {!r}".format(k)
            )
        if k <= 0:
            raise InvalidKError("k 必须 >= 1，收到 {}".format(k))
        return k

    @staticmethod
    def _to_row(row, vi):
        """把单个顶点的权重行规整为 ``list``（不做数值校验）。"""
        if isinstance(row, (str, bytes)):
            raise InvalidWeightsError(
                "顶点 {} 的权重行必须是数值序列".format(vi)
            )
        try:
            return list(row)
        except TypeError:
            raise InvalidWeightsError(
                "顶点 {} 的权重行必须是数值序列，收到 {!r}".format(
                    vi, type(row).__name__
                )
            )

    @staticmethod
    def _check_row(clean, vi):
        """逐元素校验一行权重；失败时给出精确位置与原因。"""
        if len(clean) == 0:
            raise InvalidWeightsError("顶点 {} 的权重行为空".format(vi))
        total = 0.0
        for bi, value in enumerate(clean):
            if not _is_real_number(value):
                raise InvalidWeightsError(
                    "顶点 {} 骨骼 {} 的权重不是数值：{!r}".format(
                        vi, bi, value
                    )
                )
            try:
                f = float(value)
            except (OverflowError, ValueError):
                raise InvalidWeightsError(
                    "顶点 {} 骨骼 {} 的权重不是有限数：{!r}".format(
                        vi, bi, value
                    )
                )
            # 拒绝式比较同时排除 NaN、Inf 与负数。
            if not (f >= 0.0 and f < _INF):
                raise InvalidWeightsError(
                    "顶点 {} 骨骼 {} 的权重不是有限非负数：{!r}".format(
                        vi, bi, value
                    )
                )
            total += f
        if not (total > 0.0):
            raise InvalidWeightsError(
                "顶点 {} 的权重总和必须大于 0（全为 0 无法归一化）".format(vi)
            )
        return total

    @classmethod
    def _validate_weights(cls, weights):
        """规整输入并校验，返回 ``(rows, totals)``。

        采用“C 级快速通道 + 失败回溯”：正常输入只用 ``min`` 与
        ``math.fsum`` 各扫一遍；任何可疑情况回退到逐元素校验，
        抛出带顶点/骨骼下标的精确错误。
        """
        if isinstance(weights, (str, bytes)):
            raise InvalidWeightsError("weights 必须是“顶点权重行”的序列")
        if not isinstance(weights, (list, tuple)):
            # 允许其他可迭代容器（例如生成器），但拒绝标量。
            try:
                weights = list(weights)
            except TypeError:
                raise InvalidWeightsError(
                    "weights 必须是“顶点权重行”的序列，收到 {!r}".format(
                        type(weights).__name__
                    )
                )
        if len(weights) == 0:
            raise InvalidWeightsError("weights 不能为空（至少需要一个顶点）")

        rows = []
        totals = []
        for vi, row in enumerate(weights):
            clean = cls._to_row(row, vi)
            try:
                if not clean:
                    raise ValueError("empty row")
                low = min(clean)
                total = math.fsum(clean)
            except (TypeError, ValueError, OverflowError):
                cls._check_row(clean, vi)  # 抛出精确错误
                raise InvalidWeightsError("顶点 {} 的权重非法".format(vi))
            if not (low >= 0.0 and total > 0.0 and total < _INF):
                cls._check_row(clean, vi)  # 抛出精确错误
                raise InvalidWeightsError(
                    "顶点 {} 的权重非法（NaN/Inf/负数或总和为 0）".format(vi)
                )
            rows.append(clean)
            totals.append(total)
        return rows, totals

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    def compress(self, weights, k):
        """保留每个顶点权重最大的 ``k`` 个骨骼并量化为 8 位。

        参数
        ----
        weights : 序列的序列（数值）
            每个元素是一个顶点的骨骼权重行；权重必须有限且非负，
            行和必须大于 0。
        k : int
            每个顶点保留的骨骼数上限（``>= 1``）。若顶点骨骼数少于
            ``k``，则保留全部骨骼。

        返回
        ----
        (indices, quantized) : tuple
            ``indices[i]`` 为顶点 i 保留的骨骼索引（降序），
            ``quantized[i]`` 为与之对应的 8 位量化权重，两者长度相等，
            且各自不超过 ``k``。

        异常
        ----
        InvalidKError : k 非法（非 int、bool、<= 0）。
        InvalidWeightsError : 权重为空/含非法值/全 0 等。
        """
        # 先校验 k（便宜且与数据无关），保证非法 k 总是得到 InvalidKError，
        # 而不会被权重错误掩盖；随后再校验权重。
        k = self._validate_k(k)
        rows, totals = self._validate_weights(weights)

        all_indices = []
        all_quant = []
        discarded_ratios = []
        max_quant_error = 0.0

        for vi in range(len(rows)):
            row = rows[vi]
            total = totals[vi]
            n = len(row)

            # 取权重最大的 k 个骨骼。k < n 时用 heapq.nlargest（只维护
            # k 个元素，避免全排序）；其并列取舍与
            # sorted(..., reverse=True) 的稳定行为一致（相等权重时保留
            # 索引较小者），因此结果确定。k >= n 时直接全排序。
            if k < n:
                order = heapq.nlargest(k, range(n), key=row.__getitem__)
            else:
                order = sorted(range(n), key=row.__getitem__, reverse=True)

            picked = [row[i] for i in order]
            kept_total = math.fsum(picked)

            # 归一化并把权重放大 255 倍（等价于先归一化再乘 255，
            # 但少一次除法）：sum(scaled) == 255，供 8 位量化使用。
            scaled = [w * _QUANT_MAX / kept_total for w in picked]

            quant, q_err = _quantize_scaled(scaled)
            if q_err > max_quant_error:
                max_quant_error = q_err

            # 理论误差上界：被丢弃权重占原权重的比例。
            discarded_ratios.append((total - kept_total) / total)

            all_indices.append(order)
            all_quant.append(quant)

        self._k = k
        self._indices = all_indices
        self._quantized = all_quant
        self._discarded = discarded_ratios
        self._error = max(discarded_ratios) if discarded_ratios else 0.0
        self._quant_error = max_quant_error

        return (all_indices, all_quant)

    def decompress(self, data):
        """把 ``compress`` 的结果反量化为浮点权重。

        返回 ``[[float, ...], ...]``：每个顶点一组权重 ``q / 255``，
        总和为 1（浮点误差约 1e-16）。结构不合法时抛
        :class:`InvalidDataError`（例如索引表/权重表长度不一致）。
        """
        # 快速路径：compress 自己返回的就是 (list[list[int]], list[list[int]])，
        # 此时整表长度一致、每行整型且 0..255，可直接向量化转换。
        if (
            isinstance(data, tuple)
            and len(data) == 2
            and type(data[0]) is list
            and type(data[1]) is list
            and len(data[0]) == len(data[1])
        ):
            indices, quantized = data
            try:
                if all(
                    type(x) is int and 0 <= x <= _QUANT_MAX
                    for row in quantized
                    for x in row
                ) and all(
                    type(i) is int and i >= 0
                    for row in indices
                    for i in row
                ) and all(
                    len(a) == len(b)
                    for a, b in zip(indices, quantized)
                ):
                    scale = _INV_QUANT_MAX
                    return [[x * scale for x in row] for row in quantized]
            except TypeError:
                pass  # 行不是可迭代对象，走通用路径报精确错误
        return self._decompress_checked(data)

    @staticmethod
    def _decompress_checked(data):
        """通用（带完整校验）的反量化路径。"""
        if isinstance(data, (str, bytes)):
            raise InvalidDataError("data 必须是 (indices, quantized) 二元组")
        try:
            indices, quantized = data
        except (TypeError, ValueError):
            raise InvalidDataError("data 必须是 (indices, quantized) 二元组")
        try:
            indices = list(indices)
            quantized = list(quantized)
        except TypeError:
            raise InvalidDataError("indices/quantized 必须是可迭代序列")
        if len(indices) != len(quantized):
            raise InvalidDataError(
                "索引表与权重表长度不一致：{} vs {}".format(
                    len(indices), len(quantized)
                )
            )

        out = []
        for vi, (idx_row, q_row) in enumerate(zip(indices, quantized)):
            try:
                idx_row = list(idx_row)
                q_row = list(q_row)
            except TypeError:
                raise InvalidDataError(
                    "顶点 {} 的索引/量化权重必须是可迭代序列".format(vi)
                )
            if len(idx_row) != len(q_row):
                raise InvalidDataError(
                    "顶点 {} 的索引数与权重数不一致：{} vs {}".format(
                        vi, len(idx_row), len(q_row)
                    )
                )
            row = []
            for bi, q in enumerate(q_row):
                if not _is_real_number(q):
                    raise InvalidDataError(
                        "顶点 {} 第 {} 个量化权重不是数值：{!r}".format(
                            vi, bi, q
                        )
                    )
                q = float(q)
                if not (q >= 0.0 and q <= _QUANT_MAX):  # 同时排除 NaN
                    raise InvalidDataError(
                        "顶点 {} 第 {} 个量化权重超出 8 位范围：{!r}".format(
                            vi, bi, q
                        )
                    )
                row.append(q * _INV_QUANT_MAX)
            for bi, i in enumerate(idx_row):
                if not _is_real_number(i):
                    raise InvalidDataError(
                        "顶点 {} 第 {} 个骨骼索引不是数值：{!r}".format(
                            vi, bi, i
                        )
                    )
                if not (i >= 0 and i < _INF):  # 同时排除 NaN/Inf/负数
                    raise InvalidDataError(
                        "顶点 {} 第 {} 个骨骼索引非法：{!r}".format(
                            vi, bi, i
                        )
                    )
            out.append(row)
        return out

    def error(self):
        """返回最近一次压缩中“被丢弃权重”的相对误差上界。

        定义为 ``max_vertex((原权重和 - 保留权重和) / 原权重和)``；
        尚未压缩时为 ``0.0``。返回类型为 ``float``。
        """
        return float(self._error)

    # ------------------------------------------------------------------
    # 只读诊断属性（便于调用方报告/检查，不影响主 API）
    # ------------------------------------------------------------------
    @property
    def quant_error(self):
        """最近一次压缩中实际出现的最大量化-反量化误差。"""
        return float(self._quant_error)

    @property
    def discarded(self):
        """最近一次压缩中每个顶点的被丢弃权重比例（新列表）。"""
        return list(self._discarded)

    @property
    def k(self):
        """最近一次压缩使用的 k（未压缩时为 ``None``）。"""
        return self._k
