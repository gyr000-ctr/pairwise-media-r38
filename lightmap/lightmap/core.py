"""lightmap 核心 API（仅使用 Python 标准库）。

坐标约定：
- ``tex`` 为行优先的二维序列 ``tex[row][col]``，``u`` 对应列、``v`` 对应行，
  ``tex[0][0]`` 位于 UV 原点 ``(0, 0)``。
- UV 域固定为 ``[0, 1] x [0, 1]``，纹理网格点间距为 ``1/(w-1)``、``1/(h-1)``。
- 越界采样采用 clamp-to-edge：先把 UV 夹到 ``[0, 1]`` 再取网格，
  边界外的插值恒等于边缘像素，因此接缝处不会出现黑边。
- 世界位置与 UV 之间为互逆的仿射映射。
"""


class Lightmap:
    __slots__ = (
        "_tex", "_w", "_h",
        "_x0", "_y0", "_x1", "_y1",
        "_inv_dx", "_inv_dy",
        "_gxu", "_gyv",
    )

    def __init__(self, tex, bounds):
        # ---- 校验并拍平纹理（拷贝一份，调用方后续修改不影响本对象）----
        try:
            nrows = len(tex)
        except TypeError:
            raise ValueError("tex 必须为二维序列")
        if nrows == 0:
            raise ValueError("tex 不能为空")
        try:
            ncols = len(tex[0])
        except TypeError:
            raise ValueError("tex 必须为二维序列")
        if ncols == 0:
            raise ValueError("tex 行不能为空")

        flat = []
        append = flat.append
        for row in tex:
            if len(row) != ncols:
                raise ValueError("tex 必须是矩形（各行长度一致）")
            for value in row:
                try:
                    append(float(value))
                except (TypeError, ValueError):
                    raise ValueError("tex 元素必须为数值")

        # ---- 校验世界包围盒 ----
        try:
            (x0, y0), (x1, y1) = bounds
            x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
        except (TypeError, ValueError):
            raise ValueError("bounds 必须为 ((x0, y0), (x1, y1)) 形式")
        # NaN 比较恒为 False，会自然落入下面的校验
        if not (x0 < x1 and y0 < y1):
            raise ValueError("bounds 必须满足 x0 < x1 且 y0 < y1")

        self._tex = flat
        self._w = ncols
        self._h = nrows
        self._x0, self._y0 = x0, y0
        self._x1, self._y1 = x1, y1
        inv_dx = 1.0 / (x1 - x0)
        inv_dy = 1.0 / (y1 - y0)
        self._inv_dx = inv_dx
        self._inv_dy = inv_dy
        # 世界坐标 -> 纹理网格坐标（跳过显式 UV 中间步，供采样热路径使用）
        self._gxu = (ncols - 1) * inv_dx
        self._gyv = (nrows - 1) * inv_dy

    # ------------------------------------------------------------------
    # 坐标映射
    # ------------------------------------------------------------------
    @staticmethod
    def _nan(x, y):
        return x != x or y != y

    def uv(self, pos):
        """世界位置 -> UV，返回 ``(u, v)`` 浮点元组。"""
        x, y = pos
        x, y = float(x), float(y)
        if self._nan(x, y):
            raise ValueError("坐标不能为 NaN")
        return (
            (x - self._x0) * self._inv_dx,
            (y - self._y0) * self._inv_dy,
        )

    def pos(self, uv):
        """UV -> 世界位置，为 :meth:`uv` 的逆映射，返回 ``(x, y)`` 浮点元组。"""
        u, v = uv
        u, v = float(u), float(v)
        if self._nan(u, v):
            raise ValueError("UV 不能为 NaN")
        return (
            self._x0 + u * (self._x1 - self._x0),
            self._y0 + v * (self._y1 - self._y0),
        )

    # ------------------------------------------------------------------
    # 双线性采样（clamp-to-edge）
    # ------------------------------------------------------------------
    def sample(self, pos):
        x, y = pos
        if x != x or y != y:
            raise ValueError("坐标不能为 NaN")

        tex = self._tex
        w = self._w
        h = self._h
        # 世界 -> 纹理网格坐标（内联仿射映射，省去一次 uv() 调用）
        fx = (x - self._x0) * self._gxu
        fy = (y - self._y0) * self._gyv

        # 越界按 clamp；网格坐标合法范围为 [0, w-1] / [0, h-1]
        gxmax = w - 1
        gymax = h - 1
        if fx < 0.0:
            fx = 0.0
        elif fx > gxmax:
            fx = gxmax
        if fy < 0.0:
            fy = 0.0
        elif fy > gymax:
            fy = gymax

        ix = int(fx)
        iy = int(fy)
        tx = fx - ix
        ty = fy - iy

        # 边缘处相邻索引折回自身（权重乘以 0，结果恒为边缘像素 -> 无黑边）
        ix1 = ix + 1
        if ix1 == w:
            ix1 = ix
        iy1 = iy + 1
        if iy1 == h:
            iy1 = iy

        row0 = iy * w
        row1 = iy1 * w
        c00 = tex[row0 + ix]
        c10 = tex[row0 + ix1]
        c01 = tex[row1 + ix]
        c11 = tex[row1 + ix1]

        top = c00 + (c10 - c00) * tx
        bottom = c01 + (c11 - c01) * tx
        return top + (bottom - top) * ty
