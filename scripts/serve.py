"""Run the ECG Monitor service.

Usage:
    python scripts/serve.py
    python scripts/serve.py --port 8080 --data-dir data/raw/mitdb
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data-dir", default="data/raw/mitdb")
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Install the service extras:")
        print("    pip install -e .[server]")
        return 1

    from ecgmon.app.server import create_app

    app = create_app(data_dir=args.data_dir, out_dir=args.out_dir)
    print(f"ECG Monitor on http://{args.host}:{args.port}  (docs at /docs)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
