"""
Tail percentiles for numeric, discrete and date forecasts.

The forecasters give percentiles 10..90. forecasting-tools then puts every
open bound at the halfway percentile (5th / 95th) when the bound lies beyond
the outer declared percentile. So about 5% of the mass lands outside each
open bound, and the rest of the outer 10% is spread evenly up to the bound,
however far away it is. On a wide question range that moves probability far
from where the forecasters expect the answer.

We add a 2nd and a 98th percentile, extrapolated from a two-piece logistic
fitted to each forecaster's own median and outer percentiles. The 10-90 core
stays exactly as declared; only the outer 10% on each side is shaped.

Skipped for log-scaled questions, for a tail that would cross a closed bound,
and whenever anything looks off: the original percentiles are then used.
"""
from __future__ import annotations

import logging
import math

from forecasting_tools import DateQuestion, NumericDistribution, NumericQuestion, Percentile

logger = logging.getLogger(__name__)

TAIL_PERCENTILE = 0.02
# forecasting-tools rejects declared values more than 2x the range beyond a bound.
_MAX_OVERSHOOT = 1.5


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _median(points: list[Percentile]) -> float | None:
    for a, b in zip(points, points[1:]):
        if a.percentile <= 0.5 <= b.percentile:
            if b.percentile == a.percentile:
                return a.value
            return a.value + (b.value - a.value) * (0.5 - a.percentile) / (
                b.percentile - a.percentile
            )
    return None


def add_tail_percentiles(
    percentiles: list[Percentile],
    question: NumericQuestion | DateQuestion,
    tail: float = TAIL_PERCENTILE,
) -> list[Percentile]:
    try:
        if getattr(question, "zero_point", None) is not None:
            return percentiles
        points = sorted(percentiles, key=lambda p: p.percentile)
        low, high = points[0], points[-1]
        if not (tail < low.percentile < 0.5 < high.percentile < 1.0 - tail):
            return percentiles
        median = _median(points)
        if median is None:
            return percentiles

        base = NumericDistribution.from_question(points, question)
        lower, upper = base.lower_bound, base.upper_bound
        span = upper - lower
        if span <= 0:
            return percentiles

        result = list(points)
        lower_scale = (median - low.value) / -_logit(low.percentile)
        if lower_scale > 0:
            value = max(median + lower_scale * _logit(tail), lower - _MAX_OVERSHOOT * span)
            if value < low.value and (base.open_lower_bound or value > lower):
                result.insert(0, Percentile(percentile=tail, value=value))
        upper_scale = (high.value - median) / _logit(high.percentile)
        if upper_scale > 0:
            value = min(median + upper_scale * _logit(1.0 - tail), upper + _MAX_OVERSHOOT * span)
            if value > high.value and (base.open_upper_bound or value < upper):
                result.append(Percentile(percentile=1.0 - tail, value=value))

        if len(result) == len(points):
            return percentiles
        NumericDistribution.from_question(result, question).get_cdf()
        return result
    except Exception as exc:  # never block a forecast over the tails
        logger.warning(f"Tail percentiles skipped for {getattr(question, 'page_url', '?')}: {exc}")
        return percentiles
