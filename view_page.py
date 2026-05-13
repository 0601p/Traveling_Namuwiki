from __future__ import annotations

import argparse

from environment import NamuwikiEnvironment


def print_page(env: NamuwikiEnvironment, title: str, raw_chars: int) -> None:
    if title not in env.graph:
        print(f"[not found] {title!r}")
        return

    actions = env.actions(title)
    raw = env.raw(title)

    print(f"\n=== {title} ===")
    print(f"[actions] ({len(actions)})")
    for a in actions:
        print(f"  - {a}")

    print(f"\n[raw] ({len(raw)} chars)")
    if not raw:
        print("  (empty — was the environment loaded with load_raw=True?)")
    else:
        snippet = raw if raw_chars <= 0 else raw[:raw_chars]
        print(snippet)
        if raw_chars > 0 and len(raw) > raw_chars:
            print(f"... ({len(raw) - raw_chars} more chars)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Namuwiki page (actions + raw).")
    parser.add_argument("titles", nargs="*", help="Titles to inspect. If empty, enters interactive mode.")
    parser.add_argument("--no-raw", action="store_true", help="Skip loading raw text (faster).")
    parser.add_argument("--raw-chars", type=int, default=500, help="Chars of raw to print (0 = all).")
    args = parser.parse_args()

    print("Loading environment...")
    env = NamuwikiEnvironment.from_dataset(load_raw=not args.no_raw)
    print(f"Loaded {len(env.graph)} pages, {len(env.raws)} with raw text.")

    if args.titles:
        for title in args.titles:
            print_page(env, title, args.raw_chars)
        return

    print("\nInteractive mode — enter a title (empty line / Ctrl+C to quit).")
    while True:
        try:
            title = input("title> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not title:
            break
        print_page(env, title, args.raw_chars)


if __name__ == "__main__":
    main()
