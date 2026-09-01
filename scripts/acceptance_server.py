from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

from gemmafischer.web import create_app

if __name__ == "__main__":
    uvicorn.run(
        create_app(node_budget=5_000, history_path=Path(sys.argv[2])),
        host="127.0.0.1",
        port=int(sys.argv[1]),
        log_level="warning",
    )
