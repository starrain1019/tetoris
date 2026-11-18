"""Hold-aware candidate features shared by training and the exported worker."""
from __future__ import annotations

from time import perf_counter

if __package__:
    from . import afterstate as base
    from . import placement_rules as rules
else:
    import importlib.util
    from pathlib import Path
    _spec = importlib.util.spec_from_file_location("base_features", Path(__file__).with_name("afterstate.py"))
    base = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(base)
    _spec = importlib.util.spec_from_file_location("placement_rules", Path(__file__).with_name("placement_rules.py"))
    rules = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(rules)

PIECES = base.PIECES
SHAPES = base.SHAPES
ENGINE_MIN_X = base.ENGINE_MIN_X
ACTION_SIZE = 160
MODEL_FORMAT = "afterstate-hold-spin-v3"
RULES_VERSION = rules.RULES_VERSION
FEATURE_NAMES = (base.FEATURE_NAMES[:26]
                 + tuple(f"next_current_{p}" for p in PIECES)
                 + tuple(f"next_preview_{p}" for p in PIECES)
                 + tuple(f"hold_{p}" for p in PIECES) + ("hold_empty",)
                 + ("attack", "spin", "mini", "all_clear", "combo", "b2b", "b2b_active",
                    "route_ticks", "gravity"))
FEATURE_SIZE = len(FEATURE_NAMES)


def uses_hold(action):
    return (int(action) // 40) % 2 == 1


def branch(piece, hold, queue, use_hold):
    """Piece to place, resulting hold, and queue before locking that piece."""
    queue = list(queue)
    if not use_hold:
        return piece, hold, queue
    if hold is None:
        if not queue:
            raise ValueError("Empty hold requires a preview piece")
        return queue[0], piece, queue[1:]
    return hold, piece, queue


def geometry_mask(piece, hold, queue, can_hold=True):
    mask = [False] * ACTION_SIZE
    for use_hold in (False, True):
        if use_hold and (not can_hold or (hold is None and not queue)):
            continue
        active, _, _ = branch(piece, hold, queue, use_hold)
        for rotation, cells in enumerate(SHAPES[active]):
            for left in range(10 - max(dx for dx, _ in cells)):
                mask[40 * use_hold + rotation * 10 + left] = True
    return mask


def iter_candidates(rows, piece, hold, queue, can_hold=True, *, start=None, tick=0, gravity=None,
                    combo=-1, b2b=0, b2b_active=False, incoming=0, deadline=None):
    """Yield executable candidates in action order, with an optional wall deadline.

    Ordinary placements for both hold branches come before extra T routes, so a
    worker can retain a scored move even when it must stop surface exploration.
    Training leaves deadline unset and retains the complete candidate set.
    """
    rows = rules.padded(rows)
    start = rules.spawn(rows, piece) if start is None else start
    if start is None:
        return
    def candidate(action, path, data):
        active, after_hold, remaining, prefix, prefix_states, upcoming, context = data
        path = rules.Route(prefix + path.commands, prefix_states + path.states, path.landing)
        outcome = rules.lock(rows, active, path.landing, upcoming[0])
        lock_tick = tick + len(path.commands) - 1
        attack = rules.attack_for(outcome, combo, b2b, b2b_active, lock_tick, incoming)
        combat = [attack.sent / 20, float(outcome.spin), float(outcome.mini), float(outcome.all_clear),
                  attack.combo / 20, attack.b2b / 20, float(attack.b2b_active),
                  len(path.commands) / 160, rules.speed_at(lock_tick, tick, gravity) / 20]
        features = base.features(outcome.rows, outcome.cleared, outcome.top_out, upcoming[0]) + context + combat
        return action, features, (active, after_hold, remaining, path, outcome)

    extra_branches = []
    for use_hold in (False, True):
        if deadline is not None and perf_counter() >= deadline:
            return
        if use_hold and (not can_hold or (hold is None and not queue)):
            continue
        active, after_hold, remaining = branch(piece, hold, queue, use_hold)
        state, route_tick, route_gravity = start, tick, gravity
        prefix, prefix_states = (), ()
        if use_hold:
            state = rules.spawn(rows, active)  # Hold never enables Clutch Clear.
            if state is None:
                continue
            state, _, _ = rules.frame(rows, active, state, 0, rules.speed_at(tick, tick, gravity))
            prefix, prefix_states = (8,), (state,)
            route_tick += 1
            route_gravity = rules.speed_at(route_tick, tick, gravity)
        upcoming = (remaining + [None, None])[:2]
        context = ([float(p == upcoming[1]) for p in PIECES]
                   + [float(p == after_hold) for p in PIECES] + [float(after_hold is None)])
        ordinary = rules.reachable(rows, active, state, route_tick, route_gravity, deadline=deadline)
        data = active, after_hold, remaining, prefix, prefix_states, upcoming, context
        extra_branches.append((use_hold, state, route_tick, route_gravity, ordinary, data))
        for placement, path in ordinary.items():
            if deadline is not None and perf_counter() >= deadline:
                return
            yield candidate(40 * use_hold + placement, path, data)
    for use_hold, state, route_tick, route_gravity, ordinary, data in extra_branches:
        # Leave time to score the partial search's results before returning.
        search_deadline = None if deadline is None else deadline - 0.003
        if search_deadline is not None and perf_counter() >= search_deadline:
            return
        routes = rules.spin_routes(rows, data[0], state, ordinary, route_tick, route_gravity,
                                   deadline=search_deadline)
        for index, path in enumerate(routes):
            if deadline is not None and perf_counter() >= deadline:
                return
            yield candidate(80 + 40 * use_hold + index, path, data)


def planned_candidates(rows, piece, hold, queue, can_hold=True, **kwargs):
    """All frame-validated features/plans, indexed by the stable action slots."""
    result = [[0.0] * FEATURE_SIZE for _ in range(ACTION_SIZE)]
    mask, plans = [False] * ACTION_SIZE, {}
    for action, features, plan in iter_candidates(rows, piece, hold, queue, can_hold, **kwargs):
        result[action], mask[action], plans[action] = features, True, plan
    return result, mask, plans


def candidates_from_rows(rows, piece, hold, queue, can_hold=True, **kwargs):
    features, mask, _ = planned_candidates(rows, piece, hold, queue, can_hold, **kwargs)
    return features, mask


def legal_mask(piece, hold, queue, can_hold=True, *, rows=None, **kwargs):
    # Empty-board defaults retain the public helper's original call signature.
    return candidates_from_rows([0] * 40 if rows is None else rows, piece, hold, queue, can_hold, **kwargs)[1]


def candidates(board, piece, hold, queue, can_hold=True, **kwargs):
    return candidates_from_rows(rules.board_rows(board), piece, hold, queue, can_hold, **kwargs)
