"""Shared, dependency-free placement features for learning and submission.

Rows are bottom-up 10-bit integers. A candidate is a vertical hard drop;
neither the features nor the action space contains hold.
"""
from __future__ import annotations

PIECES = "IJLOSTZ"
ACTION_SIZE = 40
FEATURE_NAMES = tuple(
    [f"height_{x}" for x in range(10)]
    + [f"holes_{x}" for x in range(10)]
    + ["aggregate_height", "max_height", "holes", "bumpiness", "cleared", "top_out"]
    + [f"next_{piece}" for piece in PIECES]
)
FEATURE_SIZE = len(FEATURE_NAMES)
MODEL_FORMAT = "afterstate-v1"

# Normalized versions of the lab/engine shapes (leftmost cell x=0, lowest y=0).
SHAPES = {
    "I": (((0, 0), (1, 0), (2, 0), (3, 0)), ((0, 0), (0, 1), (0, 2), (0, 3))),
    "J": (((0, 1), (0, 0), (1, 0), (2, 0)), ((0, 2), (1, 2), (0, 1), (0, 0)),
          ((0, 1), (1, 1), (2, 1), (2, 0)), ((1, 2), (1, 1), (0, 0), (1, 0))),
    "L": (((2, 1), (0, 0), (1, 0), (2, 0)), ((0, 2), (0, 1), (0, 0), (1, 0)),
          ((0, 1), (1, 1), (2, 1), (0, 0)), ((0, 2), (1, 2), (1, 1), (1, 0))),
    "O": (((0, 1), (1, 1), (0, 0), (1, 0)),),
    "S": (((1, 1), (2, 1), (0, 0), (1, 0)), ((0, 2), (0, 1), (1, 1), (1, 0))),
    "T": (((1, 1), (0, 0), (1, 0), (2, 0)), ((0, 2), (0, 1), (1, 1), (0, 0)),
          ((0, 1), (1, 1), (2, 1), (1, 0)), ((1, 2), (0, 1), (1, 1), (1, 0))),
    "Z": (((0, 1), (1, 1), (1, 0), (2, 0)), ((1, 2), (0, 1), (1, 1), (0, 0))),
}
ENGINE_MIN_X = {"I": (0, 2, 0, 1), "J": (0, 1, 0, 0), "L": (0, 1, 0, 0),
                "O": (1, 1, 1, 1), "S": (0, 1, 0, 0), "T": (0, 1, 0, 0), "Z": (0, 1, 0, 0)}


def board_rows(board) -> list[int]:
    return [sum(1 << x for x, cell in enumerate(row) if cell) for row in board[:20]]


def column_stats(rows: list[int]) -> tuple[list[int], list[int]]:
    columns = [0] * 10
    for y, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << y
            row ^= bit
    heights = [column.bit_length() for column in columns]
    return heights, [height - column.bit_count() for height, column in zip(heights, columns)]


def drop(rows: list[int], cells: tuple, left: int, heights=None):
    """Return (rows after clearing, cleared, top_out, occupied piece cells)."""
    if heights is None:
        heights, _ = column_stats(rows)
    bottoms = {}
    for dx, dy in cells:
        bottoms[dx] = min(dy, bottoms.get(dx, dy))
    y = max(heights[left + dx] - dy for dx, dy in bottoms.items())
    occupied = tuple((left + dx, y + dy) for dx, dy in cells)
    top_out = any(cy >= 20 for _, cy in occupied)
    placed = rows.copy()
    for cx, cy in occupied:
        if cy < 20:
            placed[cy] |= 1 << cx
    cleared = 0 if top_out else sum(row == 1023 for row in placed)
    if cleared:
        placed = [row for row in placed if row != 1023] + [0] * cleared
    return placed, cleared, top_out, occupied


def features(rows: list[int], cleared: int, top_out: bool, next_piece: str | None) -> list[float]:
    heights, holes = column_stats(rows)
    return ([h / 20.0 for h in heights] + [h / 20.0 for h in holes]
            + [sum(heights) / 200.0, max(heights) / 20.0, sum(holes) / 200.0,
               sum(abs(a - b) for a, b in zip(heights, heights[1:])) / 180.0,
               cleared / 4.0, float(top_out)]
            + [float(piece == next_piece) for piece in PIECES])


def candidates_from_rows(rows: list[int], piece: str, next_piece: str | None):
    result = [[0.0] * FEATURE_SIZE for _ in range(ACTION_SIZE)]
    mask = [False] * ACTION_SIZE
    heights, _ = column_stats(rows)
    for rotation, cells in enumerate(SHAPES[piece]):
        width = max(dx for dx, _ in cells) + 1
        for left in range(11 - width):
            action = rotation * 10 + left
            after, cleared, top_out, _ = drop(rows, cells, left, heights)
            result[action] = features(after, cleared, top_out, next_piece)
            mask[action] = True
    return result, mask


def candidates(board, piece: str, next_piece: str | None):
    return candidates_from_rows(board_rows(board), piece, next_piece)
