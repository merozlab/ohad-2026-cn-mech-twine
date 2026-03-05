"""
Journal-aware figure sizing.
All sizes in inches.
"""

COLUMN_WIDTH = 3.5
DOUBLE_COLUMN_WIDTH = 7.0

FIGURE_STYLES = {

    # 1-row wide figure (Fig1 style)
    "wide_1x3": {
        "figsize": (DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * 0.2),
        "wspace": 0.3,
        "hspace": 0.3,
        "margins": dict(left=0.025, right=0.975, top=0.975, bottom=0.025),
    },

    # Main figure (Fig3 / Fig4)
    "main_2x3": {
        "figsize": (DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH * 0.7),
        "wspace": 0.35,
        "hspace": 0.35,
        "margins": dict(left=0.075, right=0.95, top=0.95, bottom=0.075),
    },

    # Single-column panel
    "single_column": {
        "figsize": (COLUMN_WIDTH, COLUMN_WIDTH * 0.8),
        "wspace": 0.3,
        "hspace": 0.3,
        "margins": dict(left=0.15, right=0.95, top=0.9, bottom=0.15),
    },
}
