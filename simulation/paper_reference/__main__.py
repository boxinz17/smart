"""Verify the portable paper-results bundle; PDF checks are optional."""
import argparse
import json
from pathlib import Path

from . import DEFAULT_REFERENCE, verify_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--manuscript-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_reference(args.reference, manuscript_root=args.manuscript_root), indent=2))


if __name__ == "__main__":
    main()
