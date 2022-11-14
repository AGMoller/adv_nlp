import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, data: Iterable[Dict]):
    with open(path, "w") as f:
        for line in data:
            f.write(f"{json.dumps(line)}\n")


def read_jsonl(path: Path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
