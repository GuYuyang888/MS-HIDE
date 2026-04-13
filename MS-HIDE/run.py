#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

                                                               
                                                                             
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ms_hide.run import main


if __name__ == "__main__":
    main()
