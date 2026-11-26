"""质点弹簧布料：固定步长 Verlet 积分 + PBD 约束投影（仅 Python 标准库）。

特性：
- 固定步长 Verlet 积分，可与解析单摆对照（误差 ≤5%）。
- 结构 / 剪切 / 弯曲三类弹簧，权重分别可配（(0, 1]）。
- 约束迭代提前退出：收敛后最大伸长度 ≤ stretch_tol（默认 0.5%，满足 ≤1% 初始边长）。
- 球体碰撞投影，穿透 ≤1e-9。
- 非法参数使用可区分的异常类型（均为 ValueError 子类）。
"""

import math
from operator import add, mul, sub

__all__ = [
    "Cloth",
    "ClothError",
    "GridSizeError",
    "SpacingError",
    "StiffnessError",
    "TimeStepError",
    "ColliderError",
]

_SQRT2 = math.sqrt(2.0)
_sqrt = math.sqrt


class ClothError(ValueError):
    """cloth 非法参数的基类（可区分报错体系，均为 ValueError 子类）。"""


class GridSizeError(ClothError):
    """网格尺寸非法（宽/高必须为 ≥1 的整数）。"""


class SpacingError(ClothError):
    """网格间距非法（必须为有限正数）。"""


class StiffnessError(ClothError):
    """刚度权重非法（结构/剪切/弯曲权重必须为 (0, 1] 内有限实数）。"""


class TimeStepError(ClothError):
    """积分步长非法（必须为有限正数，NaN/0/负数均报错）。"""


class ColliderError(ClothError):
    """碰撞球体非法（必须为 ((x, y, z), r)，坐标有限且 r 为有限正数）。"""


def _is_finite_real(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check_weight(name, value):
    if not _is_finite_real(value) or value <= 0.0 or value > 1.0:
        raise StiffnessError(
            "%s 刚度权重必须为 (0, 1] 内的有限实数，得到 %r" % (name, value))
    return float(value)


def _check(xs, ys, zs, w, group):
    """检查一组同 stride 弹簧的最大相对伸缩量；全部收敛（< tol）时返回 0.0。"""
    s, rest, lo2, hi2, masks = group
    dx = list(map(sub, xs[s:], xs[:-s]))
    dy = list(map(sub, ys[s:], ys[:-s]))
    dz = list(map(sub, zs[s:], zs[:-s]))
    d2 = [x * x + y * y + z * z for x, y, z in zip(dx, dy, dz)]
    for start, fill in masks:
        d2[start::w] = fill
    lo = min(d2)
    hi = max(d2)
    if lo >= lo2 and hi <= hi2:
        return 0.0
    m1 = _sqrt(hi) - rest
    m2 = rest - _sqrt(lo)
    return (m1 if m1 > m2 else m2) / rest


def _relax(xs, ys, zs, w, group):
    """对一组同 stride 的弹簧做一遍 Jacobi 投影；收敛（伸缩量均 < tol）则跳过。

    返回该组弹簧投影前的最大相对伸缩量（收敛时返回 0.0）。
    """
    s, rest, lo2, hi2, k, masks = group
    m = _check(xs, ys, zs, w, (s, rest, lo2, hi2, masks))
    if m == 0.0:
        return 0.0
    dx = list(map(sub, xs[s:], xs[:-s]))
    dy = list(map(sub, ys[s:], ys[:-s]))
    dz = list(map(sub, zs[s:], zs[:-s]))
    d2 = [x * x + y * y + z * z for x, y, z in zip(dx, dy, dz)]
    for start, fill in masks:
        d2[start::w] = fill
    half_k = 0.5 * k
    f = [half_k * (d - rest) / d if d > 1e-12 else 0.0
         for d in map(_sqrt, d2)]
    c = list(map(mul, f, dx))
    xs[:-s] = list(map(add, xs[:-s], c))
    xs[s:] = list(map(sub, xs[s:], c))
    c = list(map(mul, f, dy))
    ys[:-s] = list(map(add, ys[:-s], c))
    ys[s:] = list(map(sub, ys[s:], c))
    c = list(map(mul, f, dz))
    zs[:-s] = list(map(add, zs[:-s], c))
    zs[s:] = list(map(sub, zs[s:], c))
    return m


def _ftl_round(xs, ys, zs, sp, sc, rest, k):
    """Follow-The-Leader 单轮：把 sc 切片中的质点沿当前方向放到距 sp 切片 rest 处。"""
    dx = list(map(sub, xs[sc], xs[sp]))
    dy = list(map(sub, ys[sc], ys[sp]))
    dz = list(map(sub, zs[sc], zs[sp]))
    d2 = [x * x + y * y + z * z for x, y, z in zip(dx, dy, dz)]
    f = [rest / d if d > 1e-24 else 0.0 for d in map(_sqrt, d2)]
    if k >= 1.0:
        xs[sc] = [p + ddx * ff for p, ddx, ff in zip(xs[sp], dx, f)]
        ys[sc] = [p + ddy * ff for p, ddy, ff in zip(ys[sp], dy, f)]
        zs[sc] = [p + ddz * ff for p, ddz, ff in zip(zs[sp], dz, f)]
    else:
        xs[sc] = [c + k * (p + ddx * ff - c)
                  for c, p, ddx, ff in zip(xs[sc], xs[sp], dx, f)]
        ys[sc] = [c + k * (p + ddy * ff - c)
                  for c, p, ddy, ff in zip(ys[sc], ys[sp], dy, f)]
        zs[sc] = [c + k * (p + ddz * ff - c)
                  for c, p, ddz, ff in zip(zs[sc], zs[sp], dz, f)]


def _ftl(xs, ys, zs, w, h, kind, rest, k):
    """沿结构链（行或列）做一遍 FTL 精确投影（O(链长) 顺序、向量化跨链）。"""
    if kind == 0:  # 垂直链：第 r 行锚定到第 r-1 行
        for r in range(1, h):
            _ftl_round(xs, ys, zs,
                       slice((r - 1) * w, r * w), slice(r * w, (r + 1) * w),
                       rest, k)
    else:  # 水平链：第 c 列锚定到第 c-1 列
        for c in range(1, w):
            _ftl_round(xs, ys, zs,
                       slice(c - 1, None, w), slice(c, None, w),
                       rest, k)


class Cloth:
    """w×h 质点网格布料。

    参数：
        w, h: 网格宽/高（≥1 的整数）。
        spacing: 相邻质点初始间距（有限正数）。
        stiffness: 结构弹簧刚度权重，(0, 1]。
        shear: 剪切弹簧权重，默认等于 stiffness；0 表示禁用。
        bend: 弯曲弹簧权重，默认等于 stiffness；0 表示禁用。
        damping: 速度阻尼 [0, 1)，默认 0（便于与解析单摆对照）。
        gravity: 重力加速度三元组，默认 (0, -9.8, 0)。
        iterations: 每步最大约束迭代次数，默认 64（收敛即提前退出）。
        stretch_tol: 约束收敛阈值（相对伸缩量），默认 0.005。
        pins: 初始固定的质点下标（可迭代 int）。
    """

    def __init__(self, w, h, spacing, stiffness, *, shear=None, bend=None,
                 damping=0.0, gravity=(0.0, -9.8, 0.0), iterations=64,
                 stretch_tol=0.005, pins=()):
        if isinstance(w, bool) or not isinstance(w, int) or w < 1:
            raise GridSizeError("网格宽度必须为 ≥1 的整数，得到 %r" % (w,))
        if isinstance(h, bool) or not isinstance(h, int) or h < 1:
            raise GridSizeError("网格高度必须为 ≥1 的整数，得到 %r" % (h,))
        if not _is_finite_real(spacing) or spacing <= 0.0:
            raise SpacingError("网格间距必须为有限正数，得到 %r" % (spacing,))
        stiffness = _check_weight("结构", stiffness)
        shear = stiffness if shear is None else self._check_optional("剪切", shear)
        bend = stiffness if bend is None else self._check_optional("弯曲", bend)
        if not _is_finite_real(damping) or damping < 0.0 or damping >= 1.0:
            raise ClothError("阻尼必须为 [0, 1) 内的有限实数，得到 %r" % (damping,))
        try:
            gx, gy, gz = gravity
        except (TypeError, ValueError):
            raise ClothError("重力必须为三元组，得到 %r" % (gravity,)) from None
        if not (_is_finite_real(gx) and _is_finite_real(gy) and _is_finite_real(gz)):
            raise ClothError("重力分量必须为有限实数，得到 %r" % (gravity,))
        if isinstance(iterations, bool) or not isinstance(iterations, int) \
                or iterations < 1:
            raise ClothError("迭代次数必须为 ≥1 的整数，得到 %r" % (iterations,))
        if not _is_finite_real(stretch_tol) or stretch_tol <= 0.0 \
                or stretch_tol >= 0.5:
            raise ClothError("收敛阈值必须为 (0, 0.5) 内的有限实数，得到 %r"
                             % (stretch_tol,))

        self._w = w
        self._h = h
        self._n = w * h
        self._spacing = float(spacing)
        self._damping = float(damping)
        self._gravity = (float(gx), float(gy), float(gz))
        self._iters = iterations
        self._tol = float(stretch_tol)

        sp = self._spacing
        xs = [(i % w) * sp for i in range(self._n)]
        ys = [-(i // w) * sp for i in range(self._n)]
        zs = [0.0] * self._n
        self._x = xs
        self._y = ys
        self._z = zs
        self._px = list(xs)
        self._py = list(ys)
        self._pz = list(zs)

        self._pins = {}
        for idx in pins:
            self.pin(idx)
        self._pin_cache = None

        self._chains, self._groups = self._build_groups(
            w, h, sp, stiffness, shear, bend)
        self._pairs = self._build_pairs(w, h)

    # ------------------------------------------------------------------ 构造

    @staticmethod
    def _check_optional(name, value):
        if not _is_finite_real(value) or value < 0.0 or value > 1.0:
            raise StiffnessError(
                "%s 刚度权重必须为 [0, 1] 内的有限实数，得到 %r" % (name, value))
        return float(value)

    def _build_groups(self, w, h, sp, ks, kh, kd):
        """结构弹簧走 FTL 链式求解；剪切/弯曲按 stride 分组做 Jacobi 投影。

        跨行的伪弹簧用掩码屏蔽（d2 置为 rest²，既不参与收敛判断也不产生修正）。
        """
        n = self._n
        tol = self._tol
        chains = []
        groups = []

        def bounds(rest):
            r2 = rest * rest
            return r2 * (1.0 - tol) ** 2, r2 * (1.0 + tol) ** 2

        def masks_for(stride, rest, starts):
            r2 = rest * rest
            out = []
            for m in starts:
                cnt = len(range(m, n - stride, w))
                if cnt:
                    out.append((m, [r2] * cnt))
            return out

        if h >= 2:
            lo2, hi2 = bounds(sp)
            # (kind=0 垂直链, rest, lo2, hi2, k, 供 _check 用的 stride 组)
            chains.append((0, sp, ks, (w, sp, lo2, hi2, [])))
        if w >= 2:
            lo2, hi2 = bounds(sp)
            chains.append((1, sp, ks,
                           (1, sp, lo2, hi2, masks_for(1, sp, [w - 1]))))

        def add(stride, rest, weight, mask_starts):
            if weight <= 0.0 or stride < 1 or n - stride < 1:
                return
            lo2, hi2 = bounds(rest)
            groups.append((stride, rest, lo2, hi2, weight,
                           masks_for(stride, rest, mask_starts)))

        if w >= 2 and h >= 2:
            add(w + 1, sp * _SQRT2, kh, [w - 1])    # 剪切：主对角
            add(w - 1, sp * _SQRT2, kh, [0])        # 剪切：副对角
        if w >= 3:
            add(2, 2.0 * sp, kd, [w - 2, w - 1])    # 弯曲：水平
        if h >= 3:
            add(2 * w, 2.0 * sp, kd, [])            # 弯曲：垂直
        return chains, groups

    def _build_pairs(self, w, h):
        """显式弹簧列表（供 max_stretch 精确计算）。"""
        n = self._n
        pairs = []
        for kind, rest, _k, (stride, _r, _lo, _hi, masks) in self._chains:
            excluded = frozenset(m for m, _f in masks)
            for i in range(n - stride):
                if excluded and (i % w) in excluded:
                    continue
                pairs.append((i, i + stride, rest))
        for stride, rest, _lo, _hi, _k, masks in self._groups:
            excluded = frozenset(m for m, _f in masks)
            for i in range(n - stride):
                if excluded and (i % w) in excluded:
                    continue
                pairs.append((i, i + stride, rest))
        return pairs

    # ------------------------------------------------------------------ 查询

    def count(self):
        """质点数量。"""
        return self._n

    def max_stretch(self):
        """所有弹簧当前最大相对伸长度 (d - rest) / rest（压缩时为负）。"""
        xs, ys, zs = self._x, self._y, self._z
        best = 0.0
        for i, j, rest in self._pairs:
            dx = xs[j] - xs[i]
            dy = ys[j] - ys[i]
            dz = zs[j] - zs[i]
            rel = (_sqrt(dx * dx + dy * dy + dz * dz) - rest) / rest
            if rel > best:
                best = rel
        return float(best)

    def positions(self):
        """当前质点位置列表 [(x, y, z), ...]。"""
        return list(zip(self._x, self._y, self._z))

    # ------------------------------------------------------------------ 固定

    def pin(self, index):
        """把质点固定在当前位置。"""
        if isinstance(index, bool) or not isinstance(index, int) \
                or index < 0 or index >= self._n:
            raise ClothError("固定质点下标非法：%r" % (index,))
        self._pins[index] = (self._x[index], self._y[index], self._z[index])
        self._pin_cache = None

    def unpin(self, index):
        """取消固定。"""
        self._pins.pop(index, None)
        self._pin_cache = None

    def set_position(self, index, x, y, z):
        """设置质点位置（同时重置其速度为零）。"""
        if not (_is_finite_real(x) and _is_finite_real(y) and _is_finite_real(z)):
            raise ClothError("位置坐标必须为有限实数")
        if isinstance(index, bool) or not isinstance(index, int) \
                or index < 0 or index >= self._n:
            raise ClothError("质点下标非法：%r" % (index,))
        self._x[index] = self._px[index] = float(x)
        self._y[index] = self._py[index] = float(y)
        self._z[index] = self._pz[index] = float(z)
        if index in self._pins:
            self._pins[index] = (float(x), float(y), float(z))
            self._pin_cache = None

    def _pin_restore_data(self):
        if self._pin_cache is None:
            items = sorted(self._pins.items())
            idx = [i for i, _a in items]
            ax = [a[0] for _i, a in items]
            ay = [a[1] for _i, a in items]
            az = [a[2] for _i, a in items]
            contiguous = idx == list(range(idx[0], idx[-1] + 1)) if idx else False
            self._pin_cache = (idx, ax, ay, az, contiguous)
        return self._pin_cache

    def _restore_pins(self, xs, ys, zs):
        idx, ax, ay, az, contiguous = self._pin_restore_data()
        if contiguous:
            a, b = idx[0], idx[-1] + 1
            xs[a:b] = ax
            ys[a:b] = ay
            zs[a:b] = az
        else:
            for k, i in enumerate(idx):
                xs[i] = ax[k]
                ys[i] = ay[k]
                zs[i] = az[k]

    # ------------------------------------------------------------------ 步进

    def step(self, dt, colliders):
        """以固定步长 dt 推进一步，返回新的质点位置列表 [(x, y, z), ...]。"""
        if not _is_finite_real(dt) or dt <= 0.0:
            raise TimeStepError("步长必须为有限正数，得到 %r" % (dt,))
        spheres = self._check_colliders(colliders)

        xs, ys, zs = self._x, self._y, self._z
        pxs, pys, pzs = self._px, self._py, self._pz
        pins = self._pins

        # 1) 固定步长 Verlet 积分
        dd = 1.0 - self._damping
        dt2 = dt * dt
        ax = self._gravity[0] * dt2
        ay = self._gravity[1] * dt2
        az = self._gravity[2] * dt2
        xs, pxs = [x + (x - p) * dd + ax for x, p in zip(xs, pxs)], xs
        ys, pys = [y + (y - p) * dd + ay for y, p in zip(ys, pys)], ys
        zs, pzs = [z + (z - p) * dd + az for z, p in zip(zs, pzs)], zs
        if pins:
            self._restore_pins(xs, ys, zs)
            self._restore_pins(pxs, pys, pzs)

        # 2) 约束投影迭代（收敛即提前退出）
        chains = self._chains
        groups = self._groups
        w = self._w
        h = self._h
        if chains or groups:
            for _ in range(self._iters):
                worst = 0.0
                for kind, rest, k, chk in chains:
                    m = _check(xs, ys, zs, w, chk)
                    if m != 0.0:
                        _ftl(xs, ys, zs, w, h, kind, rest, k)
                        if m > worst:
                            worst = m
                    if pins:
                        self._restore_pins(xs, ys, zs)
                for g in groups:
                    m = _relax(xs, ys, zs, w, g)
                    if m > worst:
                        worst = m
                    if pins:
                        self._restore_pins(xs, ys, zs)
                if worst <= self._tol:
                    break

        # 3) 球体碰撞：把穿透质点精确投影到球面上（穿透 ≤1e-9）
        if spheres:
            for cx, cy, cz, r in spheres:
                r2 = r * r
                for i in range(self._n):
                    dx = xs[i] - cx
                    dy = ys[i] - cy
                    dz = zs[i] - cz
                    d2 = dx * dx + dy * dy + dz * dz
                    if d2 < r2:
                        if d2 < 1e-30:
                            xs[i] = cx
                            ys[i] = cy + r
                            zs[i] = cz
                        else:
                            sc = r / _sqrt(d2)
                            xs[i] = cx + dx * sc
                            ys[i] = cy + dy * sc
                            zs[i] = cz + dz * sc
                        pxs[i] = xs[i]
                        pys[i] = ys[i]
                        pzs[i] = zs[i]

        # 4) 固定点最终保持精确
        if pins:
            self._restore_pins(xs, ys, zs)
            self._restore_pins(pxs, pys, pzs)

        self._x, self._y, self._z = xs, ys, zs
        self._px, self._py, self._pz = pxs, pys, pzs
        return list(zip(xs, ys, zs))

    @staticmethod
    def _check_colliders(colliders):
        if colliders is None:
            return ()
        spheres = []
        for col in colliders:
            try:
                center, radius = col
                cx, cy, cz = center
            except (TypeError, ValueError):
                raise ColliderError(
                    "碰撞体必须为 ((x, y, z), r) 形式，得到 %r" % (col,)) from None
            if not (_is_finite_real(cx) and _is_finite_real(cy)
                    and _is_finite_real(cz)):
                raise ColliderError("碰撞球心坐标必须为有限实数，得到 %r" % (col,))
            if not _is_finite_real(radius) or radius <= 0.0:
                raise ColliderError("碰撞球半径必须为有限正数，得到 %r" % (col,))
            spheres.append((float(cx), float(cy), float(cz), float(radius)))
        return spheres
