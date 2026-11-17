"""Shared Portfolio123 API client for this project.

Credentials are read from environment variables, or from a `.env` file in the
project root (gitignored):

    P123_API_ID=12345
    P123_API_KEY=your-key-here

Usage:

    from p123_client import get_client

    with get_client() as client:
        df = client.data_universe({...}, to_pandas=True)
        preds = client.aifactor_predict(predictor_id, {...}, to_pandas=True)

Check that the credentials work (requests an auth token; doesn't run a data query):

    python -m p123_client

Most API calls use your account's API credits.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def get_credentials() -> tuple[str, str]:
    _load_dotenv()
    api_id = os.environ.get("P123_API_ID", "").strip()
    api_key = os.environ.get("P123_API_KEY", "").strip()
    if not api_id or not api_key:
        raise RuntimeError(
            "P123_API_ID and P123_API_KEY are not set. Add them to the .env file in the "
            f"project root ({PROJECT_ROOT / '.env'}) or export them in your shell."
        )
    return api_id, api_key


def get_client():
    """Return an authenticated-on-first-use p123api.Client."""
    import p123api

    api_id, api_key = get_credentials()
    return p123api.Client(api_id=api_id, api_key=api_key)


def main() -> int:
    try:
        client = get_client()
        client.auth()
    except Exception as exc:  # noqa: BLE001 - report any failure plainly
        print(f"P123 API check failed: {exc}", file=sys.stderr)
        return 1
    print("P123 API credentials OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
