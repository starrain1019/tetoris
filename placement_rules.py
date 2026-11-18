"""Dependency-free movement/lock subset of the pinned competition engine.

SRS tables in srs.py are copied from the kit and checked against it by
differential tests. Surface search includes soft drops and final rotations.
"""
from __future__ import annotations

from collections import namedtuple, deque
import math
from time import perf_counter

if __package__:
    from .srs import SHAPES, kicks
else:
    import importlib.util
    from pathlib import Path
    _spec = importlib.util.spec_from_file_location("placement_srs", Path(__file__).with_name("srs.py"))
    _srs = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_srs)
    SHAPES, kicks = _srs.SHAPES, _srs.kicks

HEIGHT = 40
RULES_VERSION = "spin-attack-40-v3"
Motion = namedtuple("Motion", "x y rotation counter lock_ticks lock_resets last_rotation kick", defaults=(0.0, 0, 0, False, 0))
Route = namedtuple("Route", "commands states landing")
Outcome = namedtuple("Outcome", "rows cleared top_out spawn occupied spin mini all_clear", defaults=(False, False, False))
Attack = namedtuple("Attack", "raw sent combo b2b b2b_active")
MIN_X = {p: tuple(min(x for x, _ in s) for s in shapes) for p, shapes in SHAPES.items()}
CANONICAL = {p: ((0,) * 4 if p == "O" else (0, 1, 0, 1) if p in "ISZ" else (0, 1, 2, 3))
             for p in SHAPES}


def _collision_shape(shape):
    left, bottom = min(x for x, _ in shape), min(y for _, y in shape)
    width, height = max(x for x, _ in shape) - left + 1, max(y for _, y in shape) - bottom + 1
    masks = tuple(sum(1 << (x - left) for x, cy in shape if cy == y + bottom) for y in range(height))
    return left, bottom, width, masks


COLLISION = {p: tuple(_collision_shape(shape) for shape in shapes) for p, shapes in SHAPES.items()}


def board_rows(board):
    rows = [sum(1 << x for x, cell in enumerate(row) if cell) for row in board[:HEIGHT]]
    return rows + [0] * (HEIGHT - len(rows))


def padded(rows):
    return list(rows[:HEIGHT]) + [0] * max(0, HEIGHT - len(rows))


def cells(piece, state):
    return tuple((state.x + dx, state.y + dy) for dx, dy in SHAPES[piece][state.rotation])


def valid(rows, piece, x, y, rotation):
    left, bottom, width, masks = COLLISION[piece][rotation]
    x, y = x + left, y + bottom
    if x < 0 or x + width > 10 or y < 0 or y + len(masks) > HEIGHT:
        return False
    for offset, mask in enumerate(masks):
        if rows[y + offset] & (mask << x):
            return False
    return True


def spawn(rows, piece, allow_clutch=False):
    y = 20
    if allow_clutch:
        while y < HEIGHT - 4 and not valid(rows, piece, 3, y, 0):
            y += 1
    return Motion(3, y, 0) if valid(rows, piece, 3, y, 0) else None


def gravity_at(tick):
    return min(20.0, 1.0 / 60.0 + max(0, tick - 1200) / 60.0 * 0.035)


def speed_at(tick, start_tick, gravity):
    # Use the engine's exact arithmetic for its published curve. Subtracting
    # and re-adding the starting value can shift a fractional fall by one tick.
    if gravity is None or gravity == gravity_at(start_tick):
        return gravity_at(tick)
    return min(20.0, gravity + gravity_at(tick) - gravity_at(start_tick))


def frame(rows, piece, state, action, gravity):
    """Apply one official tick, returning (motion, locked, command_succeeded)."""
    x, y, rotation, counter, lock_ticks, resets, last_rotation, kick = state

    def moved(nx, ny, nr):
        nonlocal x, y, rotation, lock_ticks, resets, last_rotation
        x, y, rotation = nx, ny, nr
        last_rotation = False
        if not valid(rows, piece, x, y - 1, rotation) and resets < 15:
            lock_ticks = 0
            resets += 1

    success = True
    if action in (1, 2):
        nx = x + (-1 if action == 1 else 1)
        success = valid(rows, piece, nx, y, rotation)
        if success:
            moved(nx, y, rotation)
    elif action in (5, 6, 7):
        nr = (rotation + {5: 1, 6: -1, 7: 2}[action]) % 4
        success = False
        for index, (dx, dy) in enumerate(kicks(piece, rotation, nr)):
            if valid(rows, piece, x + dx, y + dy, nr):
                moved(x + dx, y + dy, nr)
                last_rotation, kick = True, index
                success = True
                break
    elif action == 4:
        initial_y = y
        while valid(rows, piece, x, y - 1, rotation):
            y -= 1
        if y != initial_y:
            last_rotation = False
            if resets < 15:
                lock_ticks = 0
                resets += 1
        return Motion(x, y, rotation, counter, lock_ticks, resets, last_rotation, kick), True, True
    counter += gravity * (20.0 if action == 3 else 1.0)
    while counter >= 1.0 and valid(rows, piece, x, y - 1, rotation):
        moved(x, y - 1, rotation)
        counter -= 1.0
    if not valid(rows, piece, x, y - 1, rotation):
        lock_ticks += 1
    else:
        lock_ticks = 0
    return Motion(x, y, rotation, counter, lock_ticks, resets, last_rotation, kick), lock_ticks >= 30, success


def spin_type(rows, piece, state):
    if piece != "T" or not state.last_rotation:
        return False, False
    def occupied(dx, dy):
        x, y = state.x + 1 + dx, state.y + 1 + dy
        return x < 0 or x >= 10 or y < 0 or y >= HEIGHT or bool(rows[y] & (1 << x))
    if sum(occupied(dx, dy) for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1))) < 3:
        return False, False
    fronts = (((-1, 1), (1, 1)), ((1, -1), (1, 1)),
              ((-1, -1), (1, -1)), ((-1, -1), (-1, 1)))[state.rotation]
    mini = sum(occupied(dx, dy) for dx, dy in fronts) < 2 and state.kick != 4
    return True, mini


def attack_for(outcome, combo=-1, b2b=0, b2b_active=False, tick=0, incoming=0):
    """Pinned engine arithmetic, including its actual B2B boundaries/order.

    Single-board training uses incoming=0. Export prediction can cancel the
    currently public incoming count; subsequent opponent attacks are unknown.
    """
    lines = outcome.cleared
    if not lines:
        return Attack(0, 0, -1, b2b, b2b_active)
    combo += 1
    base = ((0, 2, 4, 6, 10) if outcome.spin and not outcome.mini else (0, 0, 1, 2, 4))[min(lines, 4)]
    bonus = 0
    if b2b_active:
        bonus = next((i for i, upper in enumerate((1, 3, 8, 24, 67, 185, 504, 1370, 3725))
                      if b2b <= upper), 9)
    weight, level = base + bonus, max(combo - 1, 0)
    raw = weight + level * weight // 4 if weight else (math.floor(math.log(1 + 1.25 * level)) if level >= 2 else 0)
    difficult = outcome.spin or lines == 4 or outcome.all_clear
    if difficult:
        b2b, b2b_active = (b2b + 1 if b2b_active else 0), True
    elif b2b_active:
        b2b, b2b_active = 0, False
    raw += 10 * outcome.all_clear
    multiplier = 1 + 0.25 * min(4, max(0, tick) // 1800)
    return Attack(raw, math.ceil(max(0, raw - incoming) * multiplier), combo, b2b, b2b_active)


def lock(rows, piece, landing, next_piece):
    """Clear the entire internal board before checking next spawn / Clutch Clear."""
    occupied = cells(piece, landing)
    spin, mini = spin_type(rows, piece, landing)
    result = list(rows)
    for x, y in occupied:
        if not (0 <= x < 10 and 0 <= y < HEIGHT):
            return Outcome(result, 0, True, None, occupied)
        result[y] |= 1 << x
    cleared = sum(row == 1023 for row in result)
    result = [row for row in result if row != 1023] + [0] * cleared
    upcoming = spawn(result, next_piece, allow_clutch=cleared > 0) if next_piece else None
    return Outcome(result, cleared, next_piece is not None and upcoming is None, upcoming, occupied,
                   spin, mini, bool(cleared and not any(result)))


def slot(piece, state):
    return CANONICAL[piece][state.rotation] * 10 + state.x + MIN_X[piece][state.rotation]


def route(rows, piece, start, rotation, left, tick=0, gravity=None, move_first=False):
    """Only return a route if every command succeeds and its actual lock matches."""
    if start is None or not valid(rows, piece, start.x, start.y, start.rotation):
        return None
    target = rotation * 10 + left
    state, commands, states = start, [], []
    for offset in range(min(16, max(0, 36000 - tick))):
        current_left = state.x + MIN_X[piece][state.rotation]
        turn = 0 if CANONICAL[piece][state.rotation] == rotation else (rotation - state.rotation) % 4
        delta = left - current_left
        if delta and (move_first or not turn):
            action = 2 if delta > 0 else 1
        elif turn:
            action = {1: 5, 2: 7, 3: 6}[turn]
        else:
            action = 4
        state, locked, success = frame(rows, piece, state, action, speed_at(tick + offset, tick, gravity))
        if not success:
            return None
        commands.append(action)
        states.append(state)
        if locked:
            return Route(tuple(commands), tuple(states), state) if slot(piece, state) == target else None
    return None


def reachable(rows, piece, start, tick=0, gravity=None, *, deadline=None):
    plans = {}
    for rotation in sorted(set(CANONICAL[piece])):
        shape = SHAPES[piece][rotation]
        width = max(x for x, _ in shape) - min(x for x, _ in shape) + 1
        for left in range(11 - width):
            if deadline is not None and perf_counter() >= deadline:
                return plans
            plan = route(rows, piece, start, rotation, left, tick, gravity)
            if plan is None:
                plan = route(rows, piece, start, rotation, left, tick, gravity, move_first=True)
            if plan is not None:
                plans[rotation * 10 + left] = plan
    return plans


def spin_routes(rows, piece, start, ordinary, tick=0, gravity=None, *, deadline=None):
    """Bounded T surface search; every returned command is frame-validated.

    Keep ordinary slots separate: a tuck can share rotation/column with a
    different landing height, and the same cells can have different spin flags.
    Limits bound CPU/memory, so this is deliberately not exhaustive reachability.
    """
    if piece != "T" or not ordinary:
        return []
    horizon = min(160, 36000 - tick)
    speeds = tuple(speed_at(tick + offset, tick, gravity) for offset in range(horizon))
    queue, visited, found = deque(), set(), {}
    ordinary_keys = {(tuple(sorted(cells(piece, p.landing))), spin_type(rows, piece, p.landing)) for p in ordinary.values()}
    def key(state):
        return (state.x, state.y, state.rotation, state.last_rotation, state.kick if state.last_rotation else 0)
    def enqueue(state, commands, states, depth):
        signature = key(state)
        if signature not in visited and len(commands) < horizon:
            commands, states = tuple(commands), tuple(states)
            visited.add(signature)
            queue.append((state, commands, states, depth))
            landing, _, _ = frame(rows, piece, state, 4, 0)
            landed = (tuple(sorted(cells(piece, landing))), spin_type(rows, piece, landing))
            if landed not in ordinary_keys:
                candidate = Route(commands + (4,), states + (landing,), landing)
                if landed not in found or len(candidate.commands) < len(found[landed].commands):
                    found[landed] = candidate
    # Align in the air, then descend using real soft-drop ticks. Retain seeds
    # at the floor and within two rows, where a rotation can enter an overhang.
    for plan in ordinary.values():
        commands, states = list(plan.commands[:-1]), list(plan.states[:-1])
        state = states[-1] if states else start
        if plan.commands[-1] != 4:
            continue
        while len(commands) < horizon - 1:
            if deadline is not None and perf_counter() >= deadline:
                break
            # This seed only falls vertically. The already validated hard-drop
            # landing gives the exact first obstruction, including overhangs.
            if state.y <= plan.landing.y + 2:
                enqueue(state, commands, states, 0)
            if state.y == plan.landing.y:
                break
            state, locked, _ = frame(rows, piece, state, 3, speeds[len(commands)])
            commands.append(3)
            states.append(state)
            if locked:
                break
    queue = deque(sorted(queue, key=lambda item: (item[0].y, len(item[1]))))
    for _ in range(256):
        if not queue or (deadline is not None and perf_counter() >= deadline):
            break
        state, commands, states, depth = queue.popleft()
        if depth >= 3 or len(commands) >= horizon - 2:
            continue
        for action in (5, 6, 7, 1, 2, 3):
            after, locked, success = frame(rows, piece, state, action, speeds[len(commands)])
            if not success or locked:
                continue
            next_commands, next_states = commands + (action,), states + (after,)
            # Treat descent by one row as one search edge, even at 1/60 G.
            # Preserve every individual tick in the exported command sequence.
            while (action == 3 and after.y == state.y and len(next_commands) < horizon - 1
                   and valid(rows, piece, after.x, after.y - 1, after.rotation)):
                after, locked, _ = frame(rows, piece, after, 3, speeds[len(next_commands)])
                next_commands, next_states = next_commands + (3,), next_states + (after,)
                if locked:
                    break
            if not locked:
                enqueue(after, next_commands, next_states, depth + 1)
    def priority(path):
        outcome = lock(rows, piece, path.landing, None)
        return (-int(outcome.spin), -outcome.cleared, int(outcome.mini), len(path.commands),
                path.landing.rotation, path.landing.x, path.landing.y)
    return sorted(found.values(), key=priority)[:40]


def reconcile_rows(predicted, visible, max_garbage=16, exact_shift=None):
    """Track hidden rows using only past own locks and public visible rows.

    Garbage enters only after a non-clearing lock. Infer its upward shift from
    public incoming/attack counters when available, otherwise visible rows.
    If ambiguous, union hidden occupancy to avoid assuming free
    space. This cannot reveal information absent from the public observation.
    """
    visible = list(visible[:20])
    matches = []
    shifts = (exact_shift,) if exact_shift is not None else range(min(16, max_garbage) + 1)
    for shift in shifts:
        if predicted[:20 - shift] == visible[shift:]:
            matches.append(([0] * shift + list(predicted))[:HEIGHT])
    hidden = list(predicted[20:]) if not matches else [0] * 20
    for match in matches:
        for y in range(20, HEIGHT):
            hidden[y - 20] |= match[y]
    return visible + hidden
