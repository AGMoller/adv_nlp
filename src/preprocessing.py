import re
from typing import Dict, List

import numpy as np
import regex


def remove_trailing_newlines(text: str) -> str:
    """Remove trailing newlines from text."""
    return re.sub(r"\n+", " ", text)


def select_subset(
    data: List[Dict],
    coverage: float = 1.0,
    density: float = np.float("inf"),
    compression: float = np.float("inf"),
) -> List[Dict]:
    """Select a subset of data based on a threshold on coverage, density and compression."""
    return [
        d
        for d in data
        if d["coverage"] <= coverage
        and d["density"] <= density
        and d["compression"] <= compression
    ]


if __name__ == "__main__":

    string = "Hello\n\nworld\n."
    print(repr(string))
    print()
    print(repr(remove_trailing_newlines(string)))
