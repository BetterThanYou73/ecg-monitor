"""Cache PhysioNet records locally.

PhysioNet returns intermittent 502s under load, so every file is retried with
backoff. Caching matters beyond convenience: a benchmark that silently drops
records when the network hiccups reports a different number each run, which
makes it useless for tracking whether a change helped.

Usage:
    python scripts/fetch_data.py                      # default benchmark set
    python scripts/fetch_data.py --db mitdb --all
    python scripts/fetch_data.py --records 100 106 --db mitdb
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _configure_tls() -> None:
    bundle = Path(__file__).resolve().parents[1] / "configs" / "ca-bundle.pem"
    if bundle.exists():
        os.environ.setdefault("REQUESTS_CA_BUNDLE", str(bundle))
        os.environ.setdefault("SSL_CERT_FILE", str(bundle))


_configure_tls()

import wfdb  # noqa: E402  (import after TLS env is set)

from ecgmon.io.wfdb_loader import DEFAULT_DB  # noqa: E402

# Extensions making up one MIT-BIH record.
RECORD_EXTS = [".hea", ".dat", ".atr"]

DEFAULT_RECORDS = ["100", "103", "106", "108", "119", "203", "208", "222"]

ALL_MITDB = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def is_cached(record: str, target: Path) -> bool:
    """True when every file of the record is already on disk and non-empty."""
    return all(
        (target / f"{record}{ext}").exists() and (target / f"{record}{ext}").stat().st_size > 0
        for ext in RECORD_EXTS
    )


def fetch_record(record: str, db: str, target: Path, attempts: int = 6) -> bool:
    """Download one record, retrying with exponential backoff."""
    if is_cached(record, target):
        print(f"  {record}: cached")
        return True

    for attempt in range(1, attempts + 1):
        try:
            wfdb.dl_database(db, str(target), records=[record])
            if is_cached(record, target):
                print(f"  {record}: downloaded")
                return True
            print(f"  {record}: incomplete after download (attempt {attempt})")
        except Exception as exc:
            print(f"  {record}: attempt {attempt}/{attempts} failed "
                  f"({type(exc).__name__})")
        if attempt < attempts:
            time.sleep(min(2 ** attempt, 20))

    print(f"  {record}: GAVE UP")
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", nargs="+", default=None)
    p.add_argument("--all", action="store_true", help="all 48 MIT-BIH records")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--data-dir", default="data/raw")
    args = p.parse_args(argv)

    records = ALL_MITDB if args.all else (args.records or DEFAULT_RECORDS)
    target = Path(args.data_dir) / args.db
    target.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {len(records)} record(s) from {args.db} into {target.resolve()}")
    ok = [r for r in records if fetch_record(r, args.db, target)]

    print(f"\n{len(ok)}/{len(records)} records available.")
    missing = sorted(set(records) - set(ok))
    if missing:
        print(f"Missing: {' '.join(missing)}")
        print("PhysioNet returns 502s under load; re-run to pick up the rest.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
