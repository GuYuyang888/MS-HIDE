
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Union

import numpy as np


                                                                             
            
                                                                             

def parse_matpower_case(filepath: Union[str, Path]) -> Dict[str, Union[float, np.ndarray]]:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"MATPOWER case file not found: {filepath}")

    raw = filepath.read_text(encoding="utf-8")

                                                                      
                                                                         
                                                                              
    cleaned = _strip_comments(raw)

    result: Dict[str, Union[float, np.ndarray]] = {}

                                                                          
    for match in re.finditer(
        r"mpc\.(\w+)\s*=\s*([0-9eE.+\-]+)\s*;", cleaned
    ):
        key, value = match.group(1), match.group(2)
        result[key] = float(value)

                                                                          
                                                                    
    for match in re.finditer(
        r"mpc\.(\w+)\s*=\s*\[\s*(.*?)\s*\]\s*;", cleaned, re.DOTALL
    ):
        key = match.group(1)
        body = match.group(2)
        matrix = _parse_matrix_body(body)
        if matrix is not None:
            result[key] = matrix

                                                                          
    _required = ("baseMVA", "bus", "gen", "branch")
    missing = [k for k in _required if k not in result]
    if missing:
        raise ValueError(
            f"MATPOWER file {filepath.name} is missing required sections: "
            f"{', '.join(missing)}"
        )

    return result


                                                                             
                  
                                                                             

def _strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
                                                            
                                                                  
                                                                         
        idx = _find_comment_start(line)
        if idx is not None:
            line = line[:idx]
        lines.append(line)
    return "\n".join(lines)


def _find_comment_start(line: str) -> Union[int, None]:
    in_string = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_string:
            in_string = True
        elif ch == "'" and in_string:
            in_string = False
        elif ch == "%" and not in_string:
            return i
    return None


def _parse_matrix_body(body: str) -> Union[np.ndarray, None]:
                                                                    
                                                            
    raw_rows = body.replace("\n", ";").split(";")

    rows = []
    for raw in raw_rows:
        raw = raw.strip()
        if not raw:
            continue
        tokens = raw.split()
                                                         
        try:
            row = [float(t) for t in tokens]
        except ValueError:
            return None
        rows.append(row)

    if not rows:
        return None

    return np.array(rows, dtype=np.float64)
