"""Legacy entry point of the end-to-end probe; runs the same R1 harness as
``run_end_to_end_eval_gpu.py`` (one implementation, so the two can no longer drift).

The submitted version of this file differed from the GPU script only in ``--device``,
``--temperature`` and ``--benchmark-aware`` (default aware); it also called
``torch.cuda.is_available()`` without importing torch, so its local backend could not run.
All of those options exist in the shared harness (``--device`` is an alias of
``--protected-device``). The benchmark mode now defaults to agnostic, the main
configuration of the revision; pass ``--benchmark-aware`` for the submitted Table 3 setting.
See ``run_end_to_end_eval_gpu.py --help``.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_end_to_end_eval_gpu import main  # noqa: E402

if __name__ == "__main__":
    main()
