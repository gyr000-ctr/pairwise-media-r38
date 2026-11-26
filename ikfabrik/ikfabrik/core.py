"""ikfabrik: FABRIK inverse kinematics (standard library only).

Public API
----------
``Fabrik(lengths, iters=16, tol=1e-6)`` builds a solver for a serial chain of
fixed-length segments.  ``solve(root, target)`` returns the joint positions as
a list of tuples.  The tip lands on ``target`` when the target is reachable and
the chain is stretched straight towards it otherwise.  ``joints()`` returns the
pose of the most recent solve.

Each solve builds a geometrically exact starting pose, then runs the classic
FABRIK backward/forward refinement passes until the tip is within ``tol`` of
the target or ``iters`` passes have been used.  Every pass re-derives each
joint at its exact segment length from its neighbour, so segment lengths are
preserved to floating point precision (they are never integrated).

Reachability
------------
For a chain whose remaining links sum to ``S`` and whose longest remaining link
is ``m``, the tip can be placed anywhere at a distance from the chain base in
the interval ``[max(0, 2*m - S), S]`` (the reachable annulus).  Targets outside
that annulus are unreachable and produce a straight chain.

Invalid arguments raise subclasses of :class:`FabrikError` (itself a
``ValueError``), so each kind of bad parameter is distinguishable.
"""

from __future__ import annotations

import math

__all__ = [
    "Fabrik",
    "FabrikError",
    "InvalidLengthsError",
    "InvalidIterationsError",
    "InvalidToleranceError",
    "InvalidPointError",
]


class FabrikError(ValueError):
    """Base class for all invalid ikfabrik parameters."""


class InvalidLengthsError(FabrikError):
    """``lengths`` is empty, non-numeric, non-finite or non-positive."""


class InvalidIterationsError(FabrikError):
    """``iters`` is not a positive integer."""


class InvalidToleranceError(FabrikError):
    """``tol`` is not a positive finite number."""


class InvalidPointError(FabrikError):
    """A root/target point is malformed, non-finite or dimension-mismatched."""


def _is_real(value):
    """True for ``int``/``float`` values, excluding ``bool``."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_lengths(lengths):
    if isinstance(lengths, (str, bytes, bytearray)):
        raise InvalidLengthsError(
            "lengths must be an iterable of positive finite numbers, "
            "not a string"
        )
    try:
        values = list(lengths)
    except TypeError:
        raise InvalidLengthsError(
            "lengths must be an iterable of positive finite numbers"
        ) from None
    if not values:
        raise InvalidLengthsError(
            "lengths must not be empty (a chain needs at least one segment)"
        )
    parsed = []
    for index, value in enumerate(values):
        if not _is_real(value):
            raise InvalidLengthsError(
                "lengths[%d] must be a real number, got %s"
                % (index, type(value).__name__)
            )
        length = float(value)
        if not math.isfinite(length) or length <= 0.0:
            raise InvalidLengthsError(
                "lengths[%d] must be finite and > 0, got %r" % (index, value)
            )
        parsed.append(length)
    cumulative = [0.0]
    total = 0.0
    for length in parsed:
        total += length
        if not math.isfinite(total):
            raise InvalidLengthsError("sum of lengths overflows a float")
        cumulative.append(total)
    return tuple(parsed), tuple(cumulative)


def _parse_iters(iters):
    if isinstance(iters, bool):
        raise InvalidIterationsError("iters must be an integer >= 1, got bool")
    if isinstance(iters, float):
        if not math.isfinite(iters) or iters != int(iters):
            raise InvalidIterationsError(
                "iters must be an integer >= 1, got %r" % (iters,)
            )
        iters = int(iters)
    elif not isinstance(iters, int):
        raise InvalidIterationsError(
            "iters must be an integer >= 1, got %s" % (type(iters).__name__,)
        )
    if iters < 1:
        raise InvalidIterationsError("iters must be >= 1, got %d" % (iters,))
    return iters


def _parse_tol(tol):
    if not _is_real(tol):
        raise InvalidToleranceError(
            "tol must be a real number > 0, got %s" % (type(tol).__name__,)
        )
    tol = float(tol)
    if not math.isfinite(tol) or tol <= 0.0:
        raise InvalidToleranceError("tol must be finite and > 0, got %r" % (tol,))
    return tol


def _parse_point(point, name):
    if isinstance(point, (str, bytes, bytearray)):
        raise InvalidPointError(
            "%s must be an iterable of finite real numbers" % (name,)
        )
    try:
        values = list(point)
    except TypeError:
        raise InvalidPointError(
            "%s must be an iterable of finite real numbers" % (name,)
        ) from None
    if not values:
        raise InvalidPointError("%s must have at least one coordinate" % (name,))
    parsed = []
    for index, value in enumerate(values):
        if not _is_real(value):
            raise InvalidPointError(
                "%s[%d] must be a real number, got %s"
                % (name, index, type(value).__name__)
            )
        coordinate = float(value)
        if not math.isfinite(coordinate):
            raise InvalidPointError(
                "%s[%d] must be finite, got %r" % (name, index, value)
            )
        parsed.append(coordinate)
    return tuple(parsed)


def _perp(axis):
    """A deterministic unit vector perpendicular to the unit vector ``axis``."""
    dimension = len(axis)
    # Orthogonalise the coordinate direction least aligned with ``axis`` so the
    # subtraction never cancels catastrophically.
    best = 0
    best_value = abs(axis[0])
    for index in range(1, dimension):
        value = abs(axis[index])
        if value < best_value:
            best = index
            best_value = value
    projection = axis[best]
    vec = [-projection * axis[k] for k in range(dimension)]
    vec[best] += 1.0
    norm = math.sqrt(sum(component * component for component in vec))
    if norm == 0.0:  # pragma: no cover - only reachable via a bad unit vector
        vec = [0.0] * dimension
        vec[(best + 1) % dimension] = 1.0
        return vec
    return [component / norm for component in vec]


class Fabrik:
    """A FABRIK solver for a serial chain of fixed-length segments."""

    __slots__ = (
        "_lengths",
        "_cumulative",
        "_iters",
        "_tol",
        "_joints",
        "_suffix_sum",
        "_suffix_rmin",
        "_min_reach",
    )

    def __init__(self, lengths, iters=16, tol=1e-6):
        """Create a solver.

        Args:
            lengths: positive finite segment lengths; at least one segment
                (hence at least two joints).
            iters: maximum number of FABRIK refinement passes per solve
                (integer >= 1).
            tol: convergence threshold on the tip-to-target distance (> 0).
        """
        self._lengths, self._cumulative = _parse_lengths(lengths)
        self._iters = _parse_iters(iters)
        self._tol = _parse_tol(tol)
        self._joints = []

        # Reachable annulus of every suffix of the chain, in O(n) time.
        count = len(self._lengths)
        suffix_sum = [0.0] * (count + 1)
        suffix_max = [0.0] * (count + 1)
        for index in range(count - 1, -1, -1):
            suffix_sum[index] = suffix_sum[index + 1] + self._lengths[index]
            suffix_max[index] = max(suffix_max[index + 1], self._lengths[index])
        self._suffix_sum = tuple(suffix_sum)
        self._suffix_rmin = tuple(
            max(0.0, 2.0 * suffix_max[index] - suffix_sum[index])
            for index in range(count + 1)
        )
        self._min_reach = self._suffix_rmin[0]

    # ------------------------------------------------------------------
    # read-only configuration
    @property
    def lengths(self):
        """Segment lengths as a tuple of floats."""
        return self._lengths

    @property
    def iters(self):
        """Configured maximum iteration count."""
        return self._iters

    @property
    def tol(self):
        """Configured convergence tolerance."""
        return self._tol

    @property
    def total_length(self):
        """Sum of all segment lengths."""
        return self._cumulative[-1]

    # ------------------------------------------------------------------
    def solve(self, root, target):
        """Solve the chain from ``root`` towards ``target``.

        Returns a list of joint positions (tuples of floats) starting exactly
        at ``root``.  Reachable targets are met to within the configured
        tolerance; unreachable ones leave the chain straight.

        The solver keeps no input state between calls other than the pose
        cached for :meth:`joints`; identical arguments always give identical
        output.
        """
        start = _parse_point(root, "root")
        goal = _parse_point(target, "target")
        if len(start) != len(goal):
            raise InvalidPointError(
                "root has %d dimensions but target has %d"
                % (len(start), len(goal))
            )
        if len(start) == 2:
            points = self._solve_2d(start[0], start[1], goal[0], goal[1])
        elif len(start) == 1:
            points = self._solve_1d(start[0], goal[0])
        else:
            points = self._solve_nd(start, goal)
        self._joints = points
        return list(points)

    def joints(self):
        """Joint positions from the most recent :meth:`solve` call."""
        return list(self._joints)

    # ------------------------------------------------------------------
    # 2-D fast path (the common case)
    def _solve_2d(self, root_x, root_y, target_x, target_y):
        lengths = self._lengths
        count = len(lengths)
        total = self._cumulative[-1]

        delta_x = target_x - root_x
        delta_y = target_y - root_y
        distance = math.hypot(delta_x, delta_y)
        if distance > 0.0:
            dir_x = delta_x / distance
            dir_y = delta_y / distance
        else:
            dir_x = 1.0
            dir_y = 0.0

        # Unreachable: beyond full extension, or inside the fold hole.
        if distance >= total or distance < self._min_reach:
            return [
                (root_x + dir_x * offset, root_y + dir_y * offset)
                for offset in self._cumulative
            ]

        # Exact seed: place each joint at its segment length from the previous
        # one, choosing the distance to the target that the remaining suffix
        # can still span.  The final joint snaps onto the target exactly.
        points = [(root_x, root_y)]
        point_x = root_x
        point_y = root_y
        suffix_rmin = self._suffix_rmin
        suffix_sum = self._suffix_sum
        for index in range(count - 1):
            length = lengths[index]
            vec_x = target_x - point_x
            vec_y = target_y - point_y
            rho = math.hypot(vec_x, vec_y)
            if rho == 0.0:
                point_x += length
                points.append((point_x, point_y))
                continue
            unit_x = vec_x / rho
            unit_y = vec_y / rho
            wanted = abs(rho - length)
            remaining_min = suffix_rmin[index + 1]
            if wanted < remaining_min:
                wanted = remaining_min
            elif wanted > suffix_sum[index + 1]:
                wanted = suffix_sum[index + 1]
            cos_angle = (
                rho * rho + length * length - wanted * wanted
            ) / (2.0 * rho * length)
            if cos_angle > 1.0:
                cos_angle = 1.0
            elif cos_angle < -1.0:
                cos_angle = -1.0
            sin_sq = 1.0 - cos_angle * cos_angle
            sin_angle = math.sqrt(sin_sq) if sin_sq > 0.0 else 0.0
            point_x += length * (cos_angle * unit_x - sin_angle * unit_y)
            point_y += length * (cos_angle * unit_y + sin_angle * unit_x)
            points.append((point_x, point_y))
        points.append((target_x, target_y))
        return self._refine_2d(points, root_x, root_y, target_x, target_y)

    def _refine_2d(self, points, root_x, root_y, target_x, target_y):
        """FABRIK backward/forward passes, honouring ``iters`` and ``tol``."""
        tol_sq = self._tol * self._tol
        lengths = self._lengths
        count = len(lengths)
        error_x = points[-1][0] - target_x
        error_y = points[-1][1] - target_y
        best = points
        best_error = error_x * error_x + error_y * error_y
        if best_error <= tol_sq:
            return points
        for _ in range(self._iters):
            # Backward pass: anchor the tip on the target.
            anchor_x = target_x
            anchor_y = target_y
            points[count] = (target_x, target_y)
            for index in range(count - 1, -1, -1):
                joint_x, joint_y = points[index]
                vec_x = joint_x - anchor_x
                vec_y = joint_y - anchor_y
                span = math.hypot(vec_x, vec_y)
                length = lengths[index]
                if span > 0.0:
                    scale = length / span
                    joint_x = anchor_x + vec_x * scale
                    joint_y = anchor_y + vec_y * scale
                else:
                    joint_x = anchor_x + length
                    joint_y = anchor_y
                points[index] = (joint_x, joint_y)
                anchor_x = joint_x
                anchor_y = joint_y

            # Forward pass: pin the base back to the root.
            anchor_x = root_x
            anchor_y = root_y
            points[0] = (root_x, root_y)
            for index in range(1, count + 1):
                joint_x, joint_y = points[index]
                vec_x = joint_x - anchor_x
                vec_y = joint_y - anchor_y
                span = math.hypot(vec_x, vec_y)
                length = lengths[index - 1]
                if span > 0.0:
                    scale = length / span
                    joint_x = anchor_x + vec_x * scale
                    joint_y = anchor_y + vec_y * scale
                else:
                    joint_x = anchor_x + length
                    joint_y = anchor_y
                points[index] = (joint_x, joint_y)
                anchor_x = joint_x
                anchor_y = joint_y

            error_x = anchor_x - target_x
            error_y = anchor_y - target_y
            error_sq = error_x * error_x + error_y * error_y
            if error_sq < best_error:
                best_error = error_sq
                best = list(points)
            if error_sq <= tol_sq:
                return points
        return best

    # ------------------------------------------------------------------
    # Generic N-D path.
    def _solve_nd(self, root, target):
        dimension = len(root)
        lengths = self._lengths
        count = len(lengths)
        total = self._cumulative[-1]

        delta = [target[i] - root[i] for i in range(dimension)]
        distance = math.sqrt(sum(component * component for component in delta))
        if distance > 0.0:
            direction = [component / distance for component in delta]
        else:
            direction = [1.0] + [0.0] * (dimension - 1)

        if distance >= total or distance < self._min_reach:
            return [
                tuple(
                    root[i] + direction[i] * offset for i in range(dimension)
                )
                for offset in self._cumulative
            ]

        points = [tuple(root)]
        point = list(root)
        suffix_rmin = self._suffix_rmin
        suffix_sum = self._suffix_sum
        # A fallback direction used only when a joint sits exactly on the
        # target (rho == 0), where the target direction is undefined.
        fallback = _perp(direction)
        for index in range(count - 1):
            length = lengths[index]
            vec = [target[i] - point[i] for i in range(dimension)]
            rho = math.sqrt(sum(component * component for component in vec))
            if rho == 0.0:
                point = [
                    point[i] + length * fallback[i] for i in range(dimension)
                ]
                points.append(tuple(point))
                continue
            unit = [component / rho for component in vec]
            wanted = abs(rho - length)
            remaining_min = suffix_rmin[index + 1]
            if wanted < remaining_min:
                wanted = remaining_min
            elif wanted > suffix_sum[index + 1]:
                wanted = suffix_sum[index + 1]
            cos_angle = (rho * rho + length * length - wanted * wanted) / (
                2.0 * rho * length
            )
            if cos_angle > 1.0:
                cos_angle = 1.0
            elif cos_angle < -1.0:
                cos_angle = -1.0
            sin_sq = 1.0 - cos_angle * cos_angle
            sin_angle = math.sqrt(sin_sq) if sin_sq > 0.0 else 0.0
            perp = _perp(unit)
            point = [
                point[i]
                + length * (cos_angle * unit[i] + sin_angle * perp[i])
                for i in range(dimension)
            ]
            points.append(tuple(point))
        points.append(tuple(target))
        return self._refine_nd(points, root, target)

    def _refine_nd(self, points, root, target):
        tol_sq = self._tol * self._tol
        dimension = len(root)
        lengths = self._lengths
        count = len(lengths)

        def error_sq(tip):
            return sum((tip[i] - target[i]) ** 2 for i in range(dimension))

        best = points
        best_error = error_sq(points[-1])
        if best_error <= tol_sq:
            return points
        for _ in range(self._iters):
            # Backward pass.
            anchor = list(target)
            points[count] = tuple(target)
            for index in range(count - 1, -1, -1):
                vec = [points[index][i] - anchor[i] for i in range(dimension)]
                span = math.sqrt(
                    sum(component * component for component in vec)
                )
                length = lengths[index]
                if span > 0.0:
                    scale = length / span
                    joint = [anchor[i] + vec[i] * scale for i in range(dimension)]
                else:
                    joint = list(anchor)
                    joint[0] += length
                points[index] = tuple(joint)
                anchor = joint
            # Forward pass.
            anchor = list(root)
            points[0] = tuple(root)
            for index in range(1, count + 1):
                vec = [points[index][i] - anchor[i] for i in range(dimension)]
                span = math.sqrt(
                    sum(component * component for component in vec)
                )
                length = lengths[index - 1]
                if span > 0.0:
                    scale = length / span
                    joint = [anchor[i] + vec[i] * scale for i in range(dimension)]
                else:
                    joint = list(anchor)
                    joint[0] += length
                points[index] = tuple(joint)
                anchor = joint
            current = error_sq(anchor)
            if current < best_error:
                best_error = current
                best = list(points)
            if current <= tol_sq:
                return points
        return best

    # ------------------------------------------------------------------
    def _solve_1d(self, root_x, target_x):
        """Best-effort 1-D solve: a collinear chain can only reach discrete
        offsets, so each joint takes the sign that keeps the remaining suffix
        able to span the target."""
        lengths = self._lengths
        count = len(lengths)
        total = self._cumulative[-1]
        delta = target_x - root_x
        direction = 1.0 if delta >= 0.0 else -1.0
        if abs(delta) >= total or abs(delta) < self._min_reach:
            return [(root_x + direction * offset,) for offset in self._cumulative]

        suffix_rmin = self._suffix_rmin
        suffix_sum = self._suffix_sum
        points = [(root_x,)]
        point_x = root_x
        for index in range(count - 1):
            length = lengths[index]
            remaining_min = suffix_rmin[index + 1]
            remaining_max = suffix_sum[index + 1]
            rho = target_x - point_x
            distance = abs(rho)
            candidates = (rho - length, rho + length)
            choice = None
            for candidate in candidates:
                if remaining_min <= abs(candidate) <= remaining_max:
                    if choice is None or abs(candidate) < abs(choice):
                        choice = candidate
            if choice is None:
                choice = candidates[0]
            step = length if choice >= point_x - target_x else -length
            point_x += step
            points.append((point_x,))
        points.append((target_x,))
        return self._refine_nd(points, (root_x,), (target_x,))
