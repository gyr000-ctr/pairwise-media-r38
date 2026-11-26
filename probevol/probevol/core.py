"""probevol：光照探针体积（SH-L2 三线性插值采样），仅使用标准库。

约定
----
* 探针网格按 ``(x, y, z)`` 存放，x 变化最快，全局索引为
  ``ix + res[0] * (iy + res[1] * iz)``。
* 每个探针是一组 SH-L2 的 9 个实球谐系数，顺序采用 ACN（Ambisonics
  Channel Numbering）：

  =====  ==========================================  ==============
  索引   基函数                                       常数
  =====  ==========================================  ==============
  0      Y00                                         0.28209479
  1      Y1-1  = C1 * y                               0.48860251
  2      Y10  = C1 * z                               0.48860251
  3      Y11  = C1 * x                               0.48860251
  4      Y2-2 = C2 * x * y                          1.09254843
  5      Y2-1 = C2 * y * z                          1.09254843
  6      Y20  = C3 * (3 z^2 - 1)                     0.31539157
  7      Y21  = C2 * x * z                          1.09254843
  8      Y22  = C2H * (x^2 - y^2)                    0.54627422
  =====  ==========================================  ==============

* 三线性权重以 ``(x, y, z)`` 的低位为角点索引（k 的 bit0 -> x，
  bit1 -> y，bit2 -> z），权重恒非负且和为 1。
"""

import math

__all__ = [
    "ProbeVolume",
    "InvalidResolution",
    "InvalidProbes",
    "sh_basis",
    "SH_COEFF_COUNT",
]

SH_COEFF_COUNT = 9
SH_L1_COEFF_COUNT = 4

_PI = math.pi
# 实球谐常数（与上表一致，由 π 精确导出避免手写截断误差）
_C0 = 0.5 / math.sqrt(_PI)              # Y00
_C1 = 0.5 * math.sqrt(3.0 / _PI)        # Y1*
_C2 = 0.5 * math.sqrt(15.0 / _PI)       # Y2±1, Y2±2
_C2H = 0.25 * math.sqrt(15.0 / _PI)     # Y22
_C3 = 0.25 * math.sqrt(5.0 / _PI)       # Y20

_isfinite = math.isfinite
_sqrt = math.sqrt


class InvalidResolution(ValueError):
    """分辨率非法（无法构成探针体积）。"""


class InvalidProbes(ValueError):
    """探针数据非法（数量、维度或数值不合法）。"""


def sh_basis(direction):
    """把单位方向向量重建为 SH-L2 的 9 个基函数值（ACN 顺序）。"""
    x, y, z = direction[0], direction[1], direction[2]
    return [
        _C0,
        _C1 * y,
        _C1 * z,
        _C1 * x,
        _C2 * x * y,
        _C2 * y * z,
        _C3 * (3.0 * z * z - 1.0),
        _C2 * x * z,
        _C2H * (x * x - y * y),
    ]


def _vec3(v, what):
    """校验并转换为长度为 3 的有限浮点向量。"""
    if isinstance(v, (str, bytes)):
        raise ValueError("%s必须是长度为 3 的数值序列，收到字符串" % what)
    try:
        a, b, c = v
    except TypeError:
        raise ValueError("%s必须是长度为 3 的数值序列" % what) from None
    except ValueError:
        raise ValueError("%s必须是长度为 3 的数值序列" % what) from None
    try:
        fa = float(a)
        fb = float(b)
        fc = float(c)
    except (TypeError, ValueError):
        raise ValueError("%s的分量必须为实数" % what) from None
    if not (_isfinite(fa) and _isfinite(fb) and _isfinite(fc)):
        raise ValueError("%s的分量必须为有限值" % what)
    return fa, fb, fc


def _corner_index(c, span):
    """由夹紧后的坐标得到单元左下角索引与小数偏移，返回 (int, float)。

    坐标 c 已夹紧到 [0, span]（span = res - 1）。当 c 恰在末端角点时
    取最后一个单元、偏移为 1.0，保证角点精确。
    """
    i = int(c)                      # c >= 0，等价于 floor
    if i > span - 1:
        i = span - 1
    return i, c - i


class ProbeVolume:
    """三线性插值的 SH-L2 探针体积。"""

    __slots__ = ("_res", "_span", "_probes", "_count")

    def __init__(self, res, probes):
        # --- 分辨率校验 -------------------------------------------------
        if isinstance(res, (str, bytes)):
            raise InvalidResolution("分辨率必须是长度为 3 的整数序列")
        try:
            rx, ry, rz = res
        except TypeError:
            raise InvalidResolution("分辨率必须是长度为 3 的元组/列表") from None
        except ValueError:
            raise InvalidResolution(
                "分辨率必须恰好包含 3 个分量，收到 %r" % (res,)
            ) from None

        dims = []
        for name, value in (("x", rx), ("y", ry), ("z", rz)):
            # bool 是 int 的子类，但它显然不是合法分辨率
            if isinstance(value, bool) or not isinstance(value, int):
                raise InvalidResolution(
                    "分辨率分量 %s 必须为整数，收到 %r" % (name, value)
                )
            if value < 2:
                raise InvalidResolution(
                    "分辨率分量 %s 必须 >= 2（至少两个探针才能形成一个格子），"
                    "收到 %r" % (name, value)
                )
            dims.append(value)
        self._res = (dims[0], dims[1], dims[2])
        self._span = (dims[0] - 1, dims[1] - 1, dims[2] - 1)

        # --- 探针校验 ---------------------------------------------------
        if isinstance(probes, (str, bytes)):
            raise InvalidProbes("探针数据必须是序列")
        try:
            n = len(probes)
        except TypeError:
            # 允许一次性可迭代对象（生成器等）
            try:
                probes = tuple(probes)
            except TypeError:
                raise InvalidProbes("探针数据必须是可迭代序列") from None
            n = len(probes)
        expected = dims[0] * dims[1] * dims[2]
        if n != expected:
            raise InvalidProbes(
                "探针数量与分辨率不匹配：期望 %d 个（%d×%d×%d），收到 %d 个"
                % (expected, dims[0], dims[1], dims[2], n)
            )

        built = []
        for idx, probe in enumerate(probes):
            coeffs = _vec3_probe(probe, idx)
            built.append(coeffs)
        self._probes = tuple(built)
        self._count = expected

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def weights(self, p):
        """返回 8 个角点的三线性权重（非负、和为 1、与角点一一对应）。

        角点顺序同 ``sample`` 的混合顺序：k 的 bit0/1/2 分别对应
        x/y/z 方向的右下角。
        """
        if isinstance(p, (str, bytes)):
            raise ValueError("位置必须是长度为 3 的数值序列，收到字符串")
        try:
            px, py, pz = p
        except TypeError:
            raise ValueError("位置必须是长度为 3 的数值序列") from None
        except ValueError:
            raise ValueError("位置必须恰好包含 3 个分量") from None
        px = float(px)
        py = float(py)
        pz = float(pz)
        if not (_isfinite(px) and _isfinite(py) and _isfinite(pz)):
            raise ValueError("位置分量必须为有限值")

        span = self._span

        if px < 0.0:
            px = 0.0
        elif px > span[0]:
            px = span[0]
        ix = int(px)
        if ix > span[0] - 1:
            ix = span[0] - 1
        fx = px - ix

        if py < 0.0:
            py = 0.0
        elif py > span[1]:
            py = span[1]
        iy = int(py)
        if iy > span[1] - 1:
            iy = span[1] - 1
        fy = py - iy

        if pz < 0.0:
            pz = 0.0
        elif pz > span[2]:
            pz = span[2]
        iz = int(pz)
        if iz > span[2] - 1:
            iz = span[2] - 1
        fz = pz - iz

        ux = 1.0 - fx
        uy = 1.0 - fy
        uz = 1.0 - fz

        w0 = ux * uy * uz
        w1 = fx * uy * uz
        w2 = ux * fy * uz
        w3 = fx * fy * uz
        w4 = ux * uy * fz
        w5 = fx * uy * fz
        w6 = ux * fy * fz
        w7 = fx * fy * fz
        return [w0, w1, w2, w3, w4, w5, w6, w7]

    def sample(self, p, dirn):
        """先按系数做三线性插值，再按方向重建 SH-L2 的辐射值。"""
        px, py, pz = _vec3(p, "位置")
        dx, dy, dz = _vec3(dirn, "方向")

        length_sq = dx * dx + dy * dy + dz * dz
        if length_sq == 0.0:
            raise ValueError("方向向量不能为零向量")
        inv = 1.0 / _sqrt(length_sq)
        dx *= inv
        dy *= inv
        dz *= inv

        # --- 三线性权重（内联，热路径） ---------------------------------
        span = self._span

        if px < 0.0:
            px = 0.0
        elif px > span[0]:
            px = span[0]
        ix = int(px)
        if ix > span[0] - 1:
            ix = span[0] - 1
        fx = px - ix

        if py < 0.0:
            py = 0.0
        elif py > span[1]:
            py = span[1]
        iy = int(py)
        if iy > span[1] - 1:
            iy = span[1] - 1
        fy = py - iy

        if pz < 0.0:
            pz = 0.0
        elif pz > span[2]:
            pz = span[2]
        iz = int(pz)
        if iz > span[2] - 1:
            iz = span[2] - 1
        fz = pz - iz

        ux = 1.0 - fx
        uy = 1.0 - fy
        uz = 1.0 - fz

        w0 = ux * uy * uz
        w1 = fx * uy * uz
        w2 = ux * fy * uz
        w3 = fx * fy * uz
        w4 = ux * uy * fz
        w5 = fx * uy * fz
        w6 = ux * fy * fz
        w7 = fx * fy * fz

        # --- 8 个角点的全局索引（x 最快） ------------------------------
        q = self._probes
        rx, ry, _rz = self._res
        base = ix + rx * (iy + ry * iz)
        q0 = q[base]
        q1 = q[base + 1]
        q2 = q[base + rx]
        q3 = q[base + rx + 1]
        s_xy = rx * ry
        q4 = q[base + s_xy]
        q5 = q[base + s_xy + 1]
        q6 = q[base + s_xy + rx]
        q7 = q[base + s_xy + rx + 1]

        # 先按系数插值（线性组合），再重建——与“先重建再插值”等价
        c0 = q0[0] * w0 + q1[0] * w1 + q2[0] * w2 + q3[0] * w3 \
            + q4[0] * w4 + q5[0] * w5 + q6[0] * w6 + q7[0] * w7
        c1 = q0[1] * w0 + q1[1] * w1 + q2[1] * w2 + q3[1] * w3 \
            + q4[1] * w4 + q5[1] * w5 + q6[1] * w6 + q7[1] * w7
        c2 = q0[2] * w0 + q1[2] * w1 + q2[2] * w2 + q3[2] * w3 \
            + q4[2] * w4 + q5[2] * w5 + q6[2] * w6 + q7[2] * w7
        c3 = q0[3] * w0 + q1[3] * w1 + q2[3] * w2 + q3[3] * w3 \
            + q4[3] * w4 + q5[3] * w5 + q6[3] * w6 + q7[3] * w7
        c4 = q0[4] * w0 + q1[4] * w1 + q2[4] * w2 + q3[4] * w3 \
            + q4[4] * w4 + q5[4] * w5 + q6[4] * w6 + q7[4] * w7
        c5 = q0[5] * w0 + q1[5] * w1 + q2[5] * w2 + q3[5] * w3 \
            + q4[5] * w4 + q5[5] * w5 + q6[5] * w6 + q7[5] * w7
        c6 = q0[6] * w0 + q1[6] * w1 + q2[6] * w2 + q3[6] * w3 \
            + q4[6] * w4 + q5[6] * w5 + q6[6] * w6 + q7[6] * w7
        c7 = q0[7] * w0 + q1[7] * w1 + q2[7] * w2 + q3[7] * w3 \
            + q4[7] * w4 + q5[7] * w5 + q6[7] * w6 + q7[7] * w7
        c8 = q0[8] * w0 + q1[8] * w1 + q2[8] * w2 + q3[8] * w3 \
            + q4[8] * w4 + q5[8] * w5 + q6[8] * w6 + q7[8] * w7

        # --- SH-L2 重建（ACN 顺序，与 sh_basis 一致） ------------------
        y = dy
        z = dz
        x = dx
        return (
            c0 * _C0
            + c1 * (_C1 * y)
            + c2 * (_C1 * z)
            + c3 * (_C1 * x)
            + c4 * (_C2 * x * y)
            + c5 * (_C2 * y * z)
            + c6 * (_C3 * (3.0 * z * z - 1.0))
            + c7 * (_C2 * x * z)
            + c8 * (_C2H * (x * x - y * y))
        )


def _vec3_probe(probe, idx):
    """校验单个探针的 SH 系数，返回归一化后的 9 元组。

    允许 SH-L1（4 系数）与 SH-L2（9 系数）；SH-L1 的高阶系数补 0，
    因此“补零后重建”与直接按 4 个基函数重建结果逐位一致。
    """
    if isinstance(probe, (str, bytes)):
        raise InvalidProbes("第 %d 个探针必须是数值序列，收到字符串" % idx)
    try:
        n = len(probe)
    except TypeError:
        try:
            probe = tuple(probe)
        except TypeError:
            raise InvalidProbes("第 %d 个探针必须是数值序列" % idx) from None
        n = len(probe)
    if n not in (SH_L1_COEFF_COUNT, SH_COEFF_COUNT):
        raise InvalidProbes(
            "第 %d 个探针应含 %d 个（SH-L2）或 %d 个（SH-L1）系数，收到 %d 个"
            % (idx, SH_COEFF_COUNT, SH_L1_COEFF_COUNT, n)
        )
    out = []
    for j, value in enumerate(probe):
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise InvalidProbes(
                "第 %d 个探针的第 %d 个系数不是实数：%r" % (idx, j, value)
            ) from None
        if not _isfinite(f):
            raise InvalidProbes(
                "第 %d 个探针的第 %d 个系数不是有限值：%r" % (idx, j, value)
            )
        out.append(f)
    out.extend([0.0] * (SH_COEFF_COUNT - n))
    return tuple(out)
