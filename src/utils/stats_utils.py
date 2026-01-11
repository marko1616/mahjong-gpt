import math
from statistics import NormalDist


class RunningStats:
    """Welford online mean/variance."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # sum of squares of diffs

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self) -> float:
        # unbiased sample variance
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0


def z_from_confidence(ci_confidence: float) -> float:
    """Two-sided normal z for given confidence, e.g. 0.95 -> 1.9599..."""
    if not (0.0 < ci_confidence < 1.0):
        raise ValueError("ci_confidence must be in (0, 1)")
    alpha = 1.0 - float(ci_confidence)
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def normal_mean_bounds(
    mean: float, var: float, n: int, z: float
) -> tuple[float, float, float]:
    """Return (low, high, se) for mean under normal approx."""
    if n <= 1 or var <= 0.0 or z <= 0.0:
        return mean, mean, 0.0
    se = math.sqrt(var / n)
    return mean - z * se, mean + z * se, se
