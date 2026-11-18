"""Hold-aware placement Agent with dependency-free CPU inference."""
from __future__ import annotations

import importlib.util
import json
import math
from operator import mul
from pathlib import Path
from time import perf_counter

_spec = importlib.util.spec_from_file_location("hold_features", Path(__file__).with_name("hold_afterstate.py"))
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)


def _dot(left, right):
    return sum(map(mul, left, right))


# Python 3.12+ performs the dot product in C; older workers retain a stdlib-only
# fallback. No NumPy/PyTorch import or extra worker memory is needed.
_dot = getattr(math, "sumprod", _dot)


class Agent:
    # Official wall limit is 50ms including JSON/IPC and process scheduling.
    # Full searches are normally faster; stop early to leave response headroom.
    planning_budget_seconds = 0.030

    def __init__(self):
        model = json.loads(Path(__file__).with_name("model.json").read_text(encoding="utf-8"))
        if model.get("model_format") != _core.MODEL_FORMAT:
            raise ValueError("Incompatible hold-aware model")
        self.weights = [(model[f"w{i}"], model[f"b{i}"]) for i in range(3)]
        self.reset(0, 0)

    def reset(self, seed, player):
        self.last_piece_count = -1
        self.rows = [0] * 40
        self.path = None
        self.path_index = 0
        self.expected = None
        self.pending_rows = None
        self.pending_garbage_cap = 0
        self.lock_observation = None
        self.fallback = False
        self.b2b_active = False
        self.pending_b2b_active = None

    def _score(self, features):
        value = features
        for index, (weight, bias) in enumerate(self.weights):
            value = [_dot(value, row) + offset
                     for row, offset in zip(weight, bias)]
            if index < 2:
                value = [max(0.0, item) for item in value]
        return value[0]

    def _plan(self, observation):
        deadline = (None if self.planning_budget_seconds is None else
                    perf_counter() + self.planning_budget_seconds)
        me = observation["self"]
        start = _core.rules.Motion(int(me["x"]), int(me["y"]), int(me["rotation"]))
        candidates = _core.iter_candidates(
            self.rows, me["piece"], me.get("hold"), me["next"], me.get("can_hold", True),
            start=start, tick=observation["tick"], gravity=observation.get("gravity"),
            combo=me.get("combo", -1), b2b=me.get("b2b", 0),
            b2b_active=self.b2b_active or me.get("b2b", 0) > 0, incoming=me.get("incoming", 0),
            deadline=deadline)
        best_plan, best_score = None, float("-inf")
        for _, features, plan in candidates:
            score = self._score(features)
            if best_plan is None or score > best_score:
                best_plan, best_score = plan, score
        if best_plan is None:
            self.fallback = True
            return
        self.active, self.after_hold, _, self.path, self.outcome = best_plan
        self.path_index = 0

    @staticmethod
    def _signature(me):
        return (me["piece"], me.get("hold"), int(me["x"]), int(me["y"]), int(me["rotation"]))

    def _remember_lock(self, observation, outcome):
        self.pending_rows = outcome.rows
        me = observation["self"]
        self.pending_b2b_active = _core.rules.attack_for(
            outcome, me.get("combo", -1), me.get("b2b", 0), self.b2b_active,
            observation["tick"], me.get("incoming", 0)).b2b_active
        cap = observation.get("sudden_death", {}).get("garbage_cap", 16)
        self.pending_garbage_cap = 0 if outcome.cleared else min(cap, observation["self"].get("incoming", 0))
        self.lock_observation = observation

    def _garbage_shift(self, observation):
        before = self.lock_observation
        if before is None or observation["tick"] != before["tick"] + 1:
            return None
        me, old = observation["self"], before["self"]
        if me["lines"] > old["lines"]:
            return 0  # Any clear blocks garbage entry, independent of attack cancellation.
        # A non-clearing lock sends/cancels zero attack. Account for the other
        # player's attack on this very tick, which is enqueued after both locks.
        attack = observation["opponent"]["attack"] - before["opponent"]["attack"]
        shift = old["incoming"] + attack - me["incoming"]
        return shift if 0 <= shift <= self.pending_garbage_cap else None

    def _hard_drop(self, observation):
        # Unexpected movement/hold rejection invalidates hidden motion counters.
        # Hard drop needs no counter estimate and cannot repeat a blocked command.
        me = observation["self"]
        state = _core.rules.Motion(int(me["x"]), int(me["y"]), int(me["rotation"]))
        if _core.rules.valid(self.rows, me["piece"], state.x, state.y, state.rotation):
            landing, _, _ = _core.rules.frame(self.rows, me["piece"], state, 4, 0)
            outcome = _core.rules.lock(self.rows, me["piece"], landing, me["next"][0])
            self._remember_lock(observation, outcome)
        return 4

    def act(self, observation):
        me = observation["self"]
        if not me["alive"]:
            return 0
        visible = _core.base.board_rows(me["board"])
        if me["pieces"] != self.last_piece_count:
            if self.pending_b2b_active is not None:
                self.b2b_active = self.pending_b2b_active
                self.pending_b2b_active = None
            if self.pending_rows is not None:
                self.rows = _core.rules.reconcile_rows(self.pending_rows, visible, self.pending_garbage_cap,
                                                       self._garbage_shift(observation))
            else:
                self.rows = visible + self.rows[20:]
            self.last_piece_count = me["pieces"]
            self.expected = self.pending_rows = self.path = None
            self.fallback = False
            self._plan(observation)
        elif (visible != self.rows[:20] or self.expected is None
              or self.expected != (observation["tick"], self._signature(me))):
            self.rows[:20] = visible
            self.fallback = True
        if self.fallback or self.path is None or self.path_index >= len(self.path.commands):
            return self._hard_drop(observation)
        action = self.path.commands[self.path_index]
        state = self.path.states[self.path_index]
        self.path_index += 1
        self.expected = (observation["tick"] + 1,
                         (self.active, self.after_hold, state.x, state.y, state.rotation))
        if self.path_index == len(self.path.commands):
            self._remember_lock(observation, self.outcome)
        return action


def create_agent():
    return Agent()
