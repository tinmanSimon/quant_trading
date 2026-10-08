"""Recent slope changes, independent level shifts, and short counter-moves."""

from dataclasses import dataclass, replace
import math

import numpy as np

from .scoring import Scorer


@dataclass(frozen=True)
class TailInfo:
    status: str = "No tail change detected"
    start: int | None = None
    retracement: float = 0.0
    fit_improvement: float = 0.0
    slope_change: float = 0.0
    level_change: float = 0.0
    promoted_start: int | None = None


def diagnose(start: int, scorer: Scorer) -> TailInfo:
    config, n = scorer.config, scorer.n
    full = scorer.score(start)
    values = scorer.values[start:]
    count = len(values)
    extreme = int(np.argmax(full.direction * values))
    movement = max(0.0, full.direction * float(values[extreme] - values[0]))
    retreat = max(0.0, full.direction * float(values[extreme] - values[-1]))
    retracement = retreat / max(movement, 1e-12)
    base = TailInfo(retracement=retracement)
    young_length = count - 1 - extreme
    shock = retreat / (full.sigma * math.sqrt(max(1, young_length)))
    if 0 < young_length < config.min_tail_length and shock >= config.shock_threshold:
        base = TailInfo("Abrupt counter-move; insufficient observations", start + extreme + 1, retracement)
    minimum = max(config.min_tail_length, config.min_segment_length)
    maximum = min(int(count * config.max_tail_fraction), count - config.min_segment_length)
    total_cost = float(scorer.costs.fit(start, n)[0])
    if maximum < minimum or total_cost <= 0:
        return base
    splits = n - np.arange(minimum, maximum + 1)
    left_cost = scorer.costs.fit(start, splits)[0]
    right_cost = scorer.costs.fit(splits, n)[0]
    improvements = np.clip((total_cost - left_cost - right_cost) / total_cost, 0, 1)
    eligible = improvements >= config.tail_fit_threshold
    if not eligible.any():
        return base
    prefix_slopes = scorer.prefix_slopes(start)
    choices = []
    opposite = []
    clear_tails = []
    # An opposite robust slope needs an opposite or tied pair-ordering count
    # (an even-pair median may be nonzero when positive/negative counts tie).
    # Sorting by fitting gain permits exact early exits: after the best meaningful and
    # promotable splits are known, only possible opposite tails can supersede
    # them. This avoids expensive median work on irrelevant later splits.
    eligible_indices = np.flatnonzero(eligible)
    order = eligible_indices[np.argsort(-improvements[eligible_indices], kind="stable")]
    could_oppose = ((scorer._ordering[splits] * full.direction <= 0)
                   & (scorer._range[splits] >= 1e-13) & scorer.can_reach_clarity[splits])
    remaining_opposite = int(could_oppose[order].sum())
    for index in order:
        split, gain = splits[index], improvements[index]
        possible_opposite = bool(could_oppose[index])
        remaining_opposite -= int(possible_opposite)
        if choices and not scorer.can_reach_clarity[split]:
            continue
        if choices and clear_tails and not possible_opposite:
            if not remaining_opposite:
                break
            continue
        split = int(split)
        prefix_n, tail_n = split - start, n - split
        prefix_slope = float(prefix_slopes[prefix_n])
        tail = scorer.score(split)
        slope_change = abs(tail.slope - prefix_slope) * math.sqrt(tail_n) / full.sigma
        prefix_intercept = scorer.prefix_intercept(start, split, prefix_slope)
        level_change = abs(tail.intercept - (prefix_intercept + prefix_slope * prefix_n)) / full.sigma
        if slope_change < config.tail_slope_threshold and level_change < config.tail_level_threshold:
            continue
        accepted = not scorer.rejection(split)
        slows = abs(tail.slope) <= config.slowing_ratio * abs(prefix_slope)
        if accepted and tail.direction != full.direction:
            status = ("Broader direction materially challenged" if retracement >= config.retracement_threshold
                      or tail_n / count >= config.countertrend_fraction else "Contains a local countertrend")
        elif slows:
            status = "Recent movement slowed or paused"
        elif slope_change >= config.tail_slope_threshold:
            status = "Recent slope changed"
        else:
            status = "Recent level shift"
        if (level_change >= config.tail_level_threshold and slows
                and not (accepted and tail.direction != full.direction)):
            status = "Level shift; recent movement slowed or paused"
        info = TailInfo(status, split, retracement, float(gain), slope_change, level_change,
                        split if accepted else None)
        choices.append(info)
        if accepted:
            clear_tails.append(info)
            if tail.direction != full.direction:
                opposite.append(info)
                break
        if choices and clear_tails and not remaining_opposite:
            break
    if not choices:
        return base
    # Prefer the strongest clear opposite tail, as specified, otherwise the
    # largest fitting improvement. A different clear tail can still be added
    # when the most explanatory change itself fails the trend acceptance gates.
    chosen = max(opposite or choices, key=lambda item: item.fit_improvement)
    if chosen.promoted_start is None and clear_tails:
        chosen = replace(chosen, promoted_start=max(clear_tails, key=lambda item: item.fit_improvement).start)
    if base.status.startswith("Abrupt") and not opposite:
        return base
    return chosen
