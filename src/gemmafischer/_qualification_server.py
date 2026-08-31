from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from .web import create_app


def main() -> None:
    """Run the isolated loopback server used by runtime qualification."""
    port = int(os.environ["GEMMAFISCHER_QUALIFICATION_PORT"])
    token = os.environ["GEMMAFISCHER_QUALIFICATION_TOKEN"]
    history_path = Path(os.environ["GEMMAFISCHER_QUALIFICATION_HISTORY"])
    node_budget = int(os.environ["GEMMAFISCHER_QUALIFICATION_NODES"])
    engine_path = os.environ.get("GEMMAFISCHER_QUALIFICATION_ENGINE")
    uvicorn.run(
        create_app(
            engine_path=engine_path,
            node_budget=node_budget,
            capability_token=token,
            history_path=history_path,
        ),
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
