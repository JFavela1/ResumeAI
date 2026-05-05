"""Run the web UI with: python -m resumeai.web --reload"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="ResumeAI web server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Dev-only auto-reload",
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "resumeai.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
