from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' package is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.api.main import app  # noqa: E402 — re-exported for uvicorn main:app
from src.core.config import get_settings  # noqa: E402


def main() -> None:
    """Launch the FastAPI ASGI application with Uvicorn server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
    )


if __name__ == "__main__":
    main()

