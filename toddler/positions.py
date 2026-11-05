"""Step 3: positions. A clock hand turns 45 degrees per position; its (x, y) is added to the token vector."""
import numpy as np

POSITION_COLUMNS = ["px", "py"]


def position_vector(position, step_degrees=45):
    angle = np.radians((position - 1) * step_degrees)
    return np.round([np.cos(angle), np.sin(angle)], 10) + 0.0  # rounding removes float dust like 6e-17


def with_positions(grid):
    """Token vectors (n x 8) -> input vectors (n x 10): the meaning columns, then the 2 position columns."""
    n, width = grid.shape
    token_part = np.hstack([grid, np.zeros((n, 2))])
    position_part = np.hstack([np.zeros((n, width)), np.array([position_vector(p) for p in range(1, n + 1)])])
    return token_part + position_part
