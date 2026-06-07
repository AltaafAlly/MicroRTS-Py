#!/usr/bin/env python3
"""Print AI-vs-UTT alignment notes (see AI_UTT_ALIGNMENT_REFERENCE.txt)."""

from pathlib import Path

def main() -> None:
    p = Path(__file__).resolve().parent / "AI_UTT_ALIGNMENT_REFERENCE.txt"
    print(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
